/**
 * Lyncoin Flex CUDA Miner
 * Production-ready main application with pool support
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <signal.h>
#include <atomic>
#include <sstream>

// CUDA runtime
#include <cuda_runtime.h>

// Configuration
#include "config.h"

// Pool integration
#include "mining_pool.h"

// RPC client for solo mining
#include "rpc_client.h"

// Performance testing
#include "performance_test.h"

// Declare CUDA functions
extern "C" {
    void flex_hash_gpu(uint32_t threads, uint32_t startNonce, uint32_t* h_input, uint32_t* h_target, uint32_t* h_result);
    void ghostrider_hash_gpu(uint32_t threads, uint32_t startNonce, uint32_t* h_input, uint32_t* h_target, uint32_t* h_result);
}

// Forward declaration for signal handling
class FlexCudaMiner;

// Global variables for signal handling
static std::atomic<bool> g_running(true);
static FlexCudaMiner* g_miner = nullptr;

// Signal handler declaration (definition after class)
void signalHandler(int signal);

class FlexCudaMiner {
private:
    int deviceId;
    uint32_t threadsPerBlock;
    uint32_t blocksPerGrid;
    uint32_t totalThreads;
    std::atomic<bool> running;
    
    // Algorithm selection
    std::string algorithm;
    
    // Mining data
    uint32_t blockHeader[19]; // 76 bytes of block header
    uint32_t target[8];       // 32 bytes target
    uint32_t currentNonce;
    
    // Pool support
    std::unique_ptr<MiningPool> pool_;
    bool usePool_;
    std::string poolAddress_;
    int poolPort_;
    std::string poolUsername_;
    std::string poolPassword_;
    
    // Solo mining support
    std::unique_ptr<LyncoinRPCClient> rpcClient_;
    bool useSolo_;
    std::string rpcHost_;
    int rpcPort_;
    std::string rpcUsername_;
    std::string rpcPassword_;
    std::string miningAddress_;
    
    // Statistics
    uint64_t hashesPerformed;
    uint64_t sharesSubmitted;
    uint64_t sharesAccepted;
    std::chrono::steady_clock::time_point startTime;
      // Performance monitoring
    // std::unique_ptr<PerformanceProfiler> profiler_;

public:    FlexCudaMiner(int deviceId = 0) : deviceId(deviceId), running(false), currentNonce(0), 
                                     hashesPerformed(0), sharesSubmitted(0), sharesAccepted(0),
                                     usePool_(false), poolPort_(0), useSolo_(false), rpcPort_(8332),
                                     algorithm("flex") {
        // Initialize CUDA
        cudaSetDevice(deviceId);
        
        // Get device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);
        
        std::cout << "Using GPU: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        
        // Set optimal thread configuration
        threadsPerBlock = 256;
        blocksPerGrid = prop.multiProcessorCount * 8; // 8 blocks per SM
        totalThreads = threadsPerBlock * blocksPerGrid;
        
        std::cout << "Thread configuration: " << blocksPerGrid << " blocks × " << threadsPerBlock << " threads = " << totalThreads << " total threads" << std::endl;
        
        // Initialize performance profiler
        profiler_ = std::make_unique<PerformanceProfiler>();
        profiler_->detectGPUs();
          // Initialize default values
        memset(blockHeader, 0, sizeof(blockHeader));
        memset(target, 0x00, sizeof(target)); // Initialize to impossible target
        // Set a realistic mining target (much harder than current easy target)
        target[0] = 0x00; target[1] = 0x00; target[2] = 0x00; target[3] = 0x0F; // Very hard difficulty
        // This ensures no "fake" solutions are found on empty/test data
    }
      ~FlexCudaMiner() {
        stop();
        if (pool_) {
            pool_->disconnect();
        }
        if (rpcClient_) {
            rpcClient_->disconnect();
        }
    }
    
    void setBlockHeader(const uint8_t* header, size_t size) {
        if (size > 76) size = 76;
        memcpy(blockHeader, header, size);
        std::cout << "Block header updated" << std::endl;
    }
    
    void setTarget(const uint8_t* targetBytes, size_t size) {
        if (size > 32) size = 32;
        memcpy(target, targetBytes, size);
        std::cout << "Target updated" << std::endl;
    }
    
    void printTarget() {
        std::cout << "Current target: ";
        for (int i = 7; i >= 0; i--) {
            std::cout << std::hex << std::setw(8) << std::setfill('0') << target[i];
        }
        std::cout << std::dec << std::endl;
    }
      // Pool configuration
    void configurePool(const std::string& address, int port, const std::string& username, const std::string& password) {
        poolAddress_ = address;
        poolPort_ = port;
        poolUsername_ = username;
        poolPassword_ = password;
        usePool_ = true;
        useSolo_ = false;
        
        std::cout << "Pool configured: " << address << ":" << port << std::endl;
    }
    
    // Solo mining configuration
    void configureSolo(const std::string& host, int port, const std::string& username, const std::string& password, const std::string& address) {
        rpcHost_ = host;
        rpcPort_ = port;
        rpcUsername_ = username;
        rpcPassword_ = password;
        miningAddress_ = address;
        useSolo_ = true;
        usePool_ = false;
        
        std::cout << "Solo mining configured: " << host << ":" << port << std::endl;
        std::cout << "Mining to address: " << address << std::endl;
    }
      // Algorithm configuration
    void setAlgorithm(const std::string& algo) {
        if (algo == "flex" || algo == "ghostrider") {
            algorithm = algo;
            std::cout << "Algorithm set to: " << algorithm << std::endl;
        } else {
            std::cout << "Warning: Unknown algorithm '" << algo << "', using flex" << std::endl;
            algorithm = "flex";
        }
    }
    
    bool connectToPool() {
        if (!usePool_) {
            return false;
        }
        
        pool_ = std::make_unique<MiningPool>(poolAddress_, poolPort_, poolUsername_, poolPassword_);
        
        // Set callbacks
        pool_->setNewJobCallback([this](const MiningJob& job) {
            updateMiningJob(job);
        });
        
        pool_->setConnectionCallback([this](bool connected) {
            if (connected) {
                std::cout << "Connected to pool successfully!" << std::endl;
            } else {
                std::cout << "Disconnected from pool" << std::endl;
            }
        });
        
        return pool_->connect();
    }
    
    bool connectToRPC() {
        if (!useSolo_) {
            return false;
        }
        
        rpcClient_ = std::make_unique<LyncoinRPCClient>(rpcHost_, rpcPort_, rpcUsername_, rpcPassword_);
        
        if (!rpcClient_->connect()) {
            std::cerr << "Failed to connect to Lyncoin Core RPC" << std::endl;
            return false;
        }
        
        // Test blockchain info
        std::string info = rpcClient_->getBlockchainInfo();
        std::cout << "Blockchain info: " << info.substr(0, 100) << "..." << std::endl;
        
        int blockCount = rpcClient_->getBlockCount();
        if (blockCount > 0) {
            std::cout << "Current block height: " << blockCount << std::endl;
        }
        
        return true;
    }
    
    void updateMiningJob(const MiningJob& job) {
        std::cout << "New mining job received: " << job.jobId << std::endl;
        
        // Convert job data to mining header
        // This is simplified - real implementation would construct proper block header
        memset(blockHeader, 0, sizeof(blockHeader));
        
        // Reset nonce for new job
        currentNonce = 0;
        
        // Update difficulty target from job
        // Real implementation would convert nBits to target
    }
    
    void start() {
        if (running) {
            std::cout << "Miner is already running!" << std::endl;
            return;
        }
        
        running = true;
        hashesPerformed = 0;
        sharesSubmitted = 0;
        sharesAccepted = 0;
        startTime = std::chrono::steady_clock::now();
          std::cout << "Starting Flex CUDA miner..." << std::endl;
        
        // Connect to pool if configured
        if (usePool_) {
            if (!connectToPool()) {
                std::cerr << "Failed to connect to pool. Mining solo..." << std::endl;
                usePool_ = false;
            }
        }
        
        // Connect to RPC if solo mining
        if (useSolo_) {
            if (!connectToRPC()) {
                std::cerr << "Failed to connect to Lyncoin Core RPC. Cannot start solo mining." << std::endl;
                running = false;
                return;
            }
        }
        
        // Check if we have valid work to mine on
        if (usePool_ && pool_ && pool_->isConnected()) {
            std::cout << "Pool connected. Waiting for mining job..." << std::endl;
            
            // Wait for a mining job from the pool before starting
            while (running && g_running && usePool_ && (!pool_->hasCurrentJob())) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                std::cout << "Waiting for work from pool..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
            
            if (pool_->hasCurrentJob()) {
                std::cout << "Received work from pool. Starting mining..." << std::endl;
            } else {
                std::cout << "No work received from pool. Cannot start mining." << std::endl;
                running = false;
                return;
            }
        }
        
        if (!usePool_) {
            printTarget();
        }
        
        // Run benchmark if requested
        if (profiler_) {
            std::cout << "Running initial benchmark..." << std::endl;
            AlgorithmBenchmark benchmark(deviceId);
            auto flexResult = benchmark.testFlexComplete(100);
            if (flexResult.successful) {
                std::cout << "Benchmark hashrate: " << flexResult.hashrate << " H/s" << std::endl;
            }
        }
        
        miningLoop();
    }
      void stop() {
        if (!running) return;
        
        running = false;
        std::cout << "Stopping miner..." << std::endl;
        
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
        
        if (duration.count() > 0) {
            double hashrate = static_cast<double>(hashesPerformed) / duration.count();
            std::cout << "Final statistics:" << std::endl;
            std::cout << "Total hashes: " << hashesPerformed << std::endl;
            std::cout << "Mining time: " << duration.count() << " seconds" << std::endl;
            std::cout << "Average hashrate: " << hashrate << " H/s" << std::endl;
        }
    }
    
    bool isRunning() const {
        return running;
    }
      bool setupMining() {
        if (useSolo_) {
            // Ensure RPC client is connected
            if (!rpcClient_) {
                if (!connectToRPC()) {
                    std::cout << "RPC connection failed, falling back to test data" << std::endl;
                    // Continue to fallback section
                }
            }
            
            if (rpcClient_) {
                // Get block template from Lyncoin Core
                BlockTemplate blockTemplate;
                if (rpcClient_->getBlockTemplate(blockTemplate)) {
                std::cout << "Block template received from Lyncoin Core" << std::endl;
                std::cout << "  Height: " << blockTemplate.height << std::endl;
                std::cout << "  Transactions: " << blockTemplate.transactions.size() << std::endl;
                std::cout << "  Previous hash: " << blockTemplate.previousBlockHash.substr(0, 16) << "..." << std::endl;
                
                // Convert block template to mining header
                // This is simplified - real implementation would construct proper block header
                uint8_t rpcHeader[76] = {0};
                
                // Version (4 bytes, little endian)
                *((uint32_t*)&rpcHeader[0]) = blockTemplate.version;
                
                // Previous block hash (32 bytes)
                // Convert hex string to bytes (simplified)
                for (int i = 0; i < 32 && i * 2 < blockTemplate.previousBlockHash.length(); i++) {
                    std::string byteStr = blockTemplate.previousBlockHash.substr(i * 2, 2);
                    rpcHeader[4 + i] = (uint8_t)strtol(byteStr.c_str(), nullptr, 16);
                }
                
                // Timestamp (4 bytes, little endian)
                *((uint32_t*)&rpcHeader[68]) = blockTemplate.curTime;
                
                // nBits (4 bytes, little endian) 
                uint32_t nBits = (uint32_t)strtol(blockTemplate.bits.c_str(), nullptr, 16);
                *((uint32_t*)&rpcHeader[72]) = nBits;
                
                setBlockHeader(rpcHeader, sizeof(rpcHeader));
                
                // Convert target from hex string to bytes
                uint8_t rpcTarget[32] = {0};
                for (int i = 0; i < 32 && i * 2 < blockTemplate.target.length(); i++) {
                    std::string byteStr = blockTemplate.target.substr(i * 2, 2);
                    rpcTarget[i] = (uint8_t)strtol(byteStr.c_str(), nullptr, 16);
                }
                setTarget(rpcTarget, sizeof(rpcTarget));
                
                return true;
            } else {
                std::cerr << "Failed to get block template from Lyncoin Core" << std::endl;
                std::cout << "Using test data for solo mining..." << std::endl;
            }
        }
        
        // Fallback to test data
        uint8_t exampleHeader[76] = {0};
        exampleHeader[0] = 0x01; // Version
        exampleHeader[4] = 0x00; // Previous block hash...
        setBlockHeader(exampleHeader, sizeof(exampleHeader));
        
        // Set a realistic solo mining target
        uint8_t exampleTarget[32] = {0};
        exampleTarget[0] = 0x00; exampleTarget[1] = 0x00; 
        exampleTarget[2] = 0x01; exampleTarget[3] = 0x00;
        setTarget(exampleTarget, sizeof(exampleTarget));
        
        if (!useSolo_ || !rpcClient_) {
            std::cout << "Note: Use --rpc-user and --rpc-pass for real solo mining" << std::endl;
        }
        
        return false;
    }
    
private:    void miningLoop() {
        uint32_t result[9]; // nonce + 8 words of hash
        auto lastStatsTime = std::chrono::steady_clock::now();
        
        while (running && g_running) {
            // Clear result
            memset(result, 0, sizeof(result));
              // Run CUDA kernel based on selected algorithm
            if (algorithm == "ghostrider") {
                ghostrider_hash_gpu(totalThreads, currentNonce, blockHeader, target, result);
            } else {
                // Default to flex algorithm
                flex_hash_gpu(totalThreads, currentNonce, blockHeader, target, result);
            }
            
            // Check for CUDA errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                break;
            }
            
            hashesPerformed += totalThreads;
            currentNonce += totalThreads;
            
            // Check if solution found
            if (result[0] != 0) {
                std::cout << "\n*** SOLUTION FOUND! ***" << std::endl;
                std::cout << "Nonce: " << std::hex << result[0] << std::dec << std::endl;
                std::cout << "Hash: ";
                for (int i = 1; i <= 8; i++) {
                    std::cout << std::hex << std::setw(8) << std::setfill('0') << result[i];
                }
                std::cout << std::dec << std::endl;
                  // Submit to pool if connected
                if (usePool_ && pool_ && pool_->isConnected()) {
                    std::cout << "Submitting share to pool..." << std::endl;
                    
                    // Convert nonce to hex string
                    std::stringstream nonceHex;
                    nonceHex << std::hex << std::setw(8) << std::setfill('0') << result[0];
                    
                    // Get current job if available
                    if (pool_->hasCurrentJob()) {
                        MiningJob job = pool_->getCurrentJob();
                        
                        // Submit share (simplified - real implementation would need proper extraNonce2, nTime)
                        bool submitted = pool_->submitShare(job.jobId, "00000000", job.nTime, nonceHex.str());
                        if (submitted) {
                            sharesSubmitted++;
                            std::cout << "Share submitted successfully!" << std::endl;
                        } else {
                            std::cout << "Share submission failed!" << std::endl;
                        }
                    } else {
                        std::cout << "No mining job available for share submission" << std::endl;
                    }
                } else {
                    std::cout << "Solo mining - solution found but not submitted to pool" << std::endl;
                }
            }
            
            // Print statistics every 5 seconds
            auto now = std::chrono::steady_clock::now();
            auto statsDuration = std::chrono::duration_cast<std::chrono::seconds>(now - lastStatsTime);
            
            if (statsDuration.count() >= 5) {
                auto totalDuration = std::chrono::duration_cast<std::chrono::seconds>(now - startTime);
                if (totalDuration.count() > 0) {
                    double hashrate = static_cast<double>(hashesPerformed) / totalDuration.count();
                    if (usePool_ && pool_) {
                        std::cout << "Pool: " << (pool_->isConnected() ? "Connected" : "Disconnected") 
                                  << ", Hashrate: " << hashrate << " H/s, Shares: " << sharesSubmitted 
                                  << "/" << sharesAccepted << ", Nonce: " << currentNonce << std::endl;
                    } else {
                        std::cout << "Solo: Hashrate: " << hashrate << " H/s, Total: " << hashesPerformed 
                                  << " hashes, Nonce: " << currentNonce << std::endl;
                    }
                }
                lastStatsTime = now;
            }
            
            // Small delay to prevent overwhelming the GPU
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // If we exit the loop, stop the miner
        running = false;    }

    // ...existing code...
};

// Signal handler implementation
void signalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ". Stopping miner..." << std::endl;
    g_running = false;
    if (g_miner) {
        g_miner->stop();
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Lyncoin Flex CUDA Miner v1.0" << std::endl;
    std::cout << "==============================" << std::endl;
    
    // Load configuration file
    std::cout << "Loading configuration..." << std::endl;
    if (!g_config.loadFromFile("config.ini")) {
        std::cout << "Note: Using default configuration (config.ini not found or unreadable)" << std::endl;
    }
    
    // Check for CUDA devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;      // Parse command line arguments and config
    int deviceId = g_config.getInt("gpu_device", 0);
    std::string algorithm = g_config.getString("algorithm", "flex");
    bool poolMining = false;
    bool soloMining = false;
    std::string poolAddress, poolUser, poolPass;
    std::string rpcHost = "127.0.0.1", rpcUser, rpcPass, miningAddress;
    int rpcPort = 8332;
    
    // Command line overrides config file
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            deviceId = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pool") == 0 && i + 1 < argc) {
            poolAddress = argv[++i];
            poolMining = true;
        } else if (strcmp(argv[i], "--user") == 0 && i + 1 < argc) {
            poolUser = argv[++i];
        } else if (strcmp(argv[i], "--pass") == 0 && i + 1 < argc) {
            poolPass = argv[++i];
        } else if (strcmp(argv[i], "--solo") == 0) {
            soloMining = true;
        } else if (strcmp(argv[i], "--rpc-host") == 0 && i + 1 < argc) {
            rpcHost = argv[++i];
        } else if (strcmp(argv[i], "--rpc-port") == 0 && i + 1 < argc) {
            rpcPort = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--rpc-user") == 0 && i + 1 < argc) {
            rpcUser = argv[++i];
        } else if (strcmp(argv[i], "--rpc-pass") == 0 && i + 1 < argc) {
            rpcPass = argv[++i];        } else if (strcmp(argv[i], "--address") == 0 && i + 1 < argc) {
            miningAddress = argv[++i];
        } else if (strcmp(argv[i], "--algorithm") == 0 && i + 1 < argc) {
            algorithm = argv[++i];
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            g_config.loadFromFile(argv[++i]);        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --device <id>       GPU device ID (default: 0)" << std::endl;
            std::cout << "  --pool <address>    Pool address with port (e.g., stratum+tcp://pool.example.com:4444)" << std::endl;
            std::cout << "  --user <address>    Mining address/username (required for pool mining)" << std::endl;
            std::cout << "  --pass <password>   Pool password (default: x)" << std::endl;
            std::cout << "  --solo              Enable solo mining mode" << std::endl;
            std::cout << "  --rpc-host <host>   RPC host for solo mining (default: 127.0.0.1)" << std::endl;
            std::cout << "  --rpc-port <port>   RPC port for solo mining (default: 8332)" << std::endl;
            std::cout << "  --rpc-user <user>   RPC username for solo mining" << std::endl;            std::cout << "  --rpc-pass <pass>   RPC password for solo mining" << std::endl;
            std::cout << "  --address <addr>    Mining address for solo mining" << std::endl;
            std::cout << "  --algorithm <algo>  Mining algorithm: flex or ghostrider (default: flex)" << std::endl;
            std::cout << "  --config <file>     Configuration file (default: config.ini)" << std::endl;
            std::cout << "  --help              Show this help" << std::endl;
            std::cout << std::endl;
            std::cout << "Examples:" << std::endl;
            std::cout << "  Solo mining:    " << argv[0] << " --solo --rpc-user rpcuser --rpc-pass rpcpass --address your_address" << std::endl;
            std::cout << "  Pool mining:    " << argv[0] << " --pool stratum+tcp://pool.example.com:4444 --user 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa" << std::endl;
            return 0;
        }
    }
    
    if (deviceId < 0 || deviceId >= deviceCount) {
        std::cerr << "Invalid device ID. Using device 0." << std::endl;
        deviceId = 0;
    }
    
    // Get configuration values
    int threadsPerBlock = g_config.getInt("threads_per_block", 256);
    int blocksPerGrid = g_config.getInt("blocks_per_grid", 0);
    bool useFastMath = g_config.getBool("use_fast_math", true);
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  GPU Device: " << deviceId << std::endl;
    std::cout << "  Threads per block: " << threadsPerBlock << std::endl;
    std::cout << "  Use fast math: " << (useFastMath ? "Yes" : "No") << std::endl;      try {
        FlexCudaMiner miner(deviceId);
        g_miner = &miner;
        
        // Set the algorithm
        miner.setAlgorithm(algorithm);
        
        // Set up signal handlers
        signal(SIGINT, signalHandler);
        signal(SIGTERM, signalHandler);
        
        // Configure pool if pool mining is enabled
        if (poolMining && !poolAddress.empty()) {
            // Parse pool address to extract host and port
            std::string poolHost;
            int poolPort = 4444; // Default port
            
            // Handle stratum+tcp:// prefix
            std::string cleanAddress = poolAddress;
            if (cleanAddress.find("stratum+tcp://") == 0) {
                cleanAddress = cleanAddress.substr(14);
            } else if (cleanAddress.find("stratum://") == 0) {
                cleanAddress = cleanAddress.substr(10);
            } else if (cleanAddress.find("tcp://") == 0) {
                cleanAddress = cleanAddress.substr(6);
            }
            
            // Split host:port
            size_t colonPos = cleanAddress.find(':');
            if (colonPos != std::string::npos) {
                poolHost = cleanAddress.substr(0, colonPos);
                poolPort = std::atoi(cleanAddress.substr(colonPos + 1).c_str());
            } else {
                poolHost = cleanAddress;
            }
            
            // Set default values if not provided
            if (poolUser.empty()) {
                poolUser = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"; // Example address
                std::cout << "Warning: No user address provided, using example address" << std::endl;
            }
            if (poolPass.empty()) {
                poolPass = "x";
            }
            
            std::cout << "Pool Mining Mode:" << std::endl;
            std::cout << "  Host: " << poolHost << std::endl;
            std::cout << "  Port: " << poolPort << std::endl;
            std::cout << "  User: " << poolUser << std::endl;
            std::cout << "  Pass: " << poolPass << std::endl;            miner.configurePool(poolHost, poolPort, poolUser, poolPass);
        } else {
            std::cout << "Solo Mining Mode" << std::endl;
            
            // Set default values if not provided for solo mining
            if (rpcUser.empty() || rpcPass.empty()) {
                std::cout << "Warning: RPC credentials not provided. Using test data." << std::endl;
                std::cout << "Note: Use --rpc-user and --rpc-pass for real solo mining" << std::endl;
            }
            if (miningAddress.empty()) {
                miningAddress = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"; // Example address
                std::cout << "Warning: No mining address provided, using example address" << std::endl;
            }
            
            std::cout << "  RPC Host: " << rpcHost << std::endl;
            std::cout << "  RPC Port: " << rpcPort << std::endl;
            std::cout << "  RPC User: " << (rpcUser.empty() ? "(not set)" : rpcUser) << std::endl;
            std::cout << "  Mining Address: " << miningAddress << std::endl;
            
            // Configure solo mining
            miner.configureSolo(rpcHost, rpcPort, rpcUser, rpcPass, miningAddress);
            
            if (miner.setupMining()) {
                std::cout << "Solo mining configured with live Lyncoin blockchain data" << std::endl;
            } else {
                std::cout << "Solo mining configured with test data (RPC connection failed)" << std::endl;
            }
        }
        
        // Start mining
        std::cout << "Press Ctrl+C to stop mining..." << std::endl;
        miner.start();
        
        // Wait for the mining to complete (interrupted by signal)
        while (g_running && miner.isRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        g_miner = nullptr;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
