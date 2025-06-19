/**
 * Performance Testing Framework for Flex CUDA Miner
 * Comprehensive benchmarking and optimization tools
 */

#ifndef PERFORMANCE_TEST_H
#define PERFORMANCE_TEST_H

#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <map>
#include <functional>

struct GPUInfo {
    int deviceId;
    std::string name;
    int computeCapability[2]; // major, minor
    size_t totalMemory;
    size_t freeMemory;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    int maxBlocksPerSM;
    int warpSize;
    int memoryBusWidth;
    int memoryClockRate;
    int coreClockRate;
};

struct HashTestResult {
    std::string algorithm;
    double hashrate; // H/s
    double averageTime; // milliseconds per hash
    double minTime;
    double maxTime;
    uint64_t totalHashes;
    uint32_t iterations;
    bool successful;
    std::string errorMessage;
};

struct MemoryTestResult {
    std::string testName;
    double bandwidth; // GB/s
    double latency; // microseconds
    bool successful;
};

struct ThermalTestResult {
    double initialTemp;
    double maxTemp;
    double averageTemp;
    double powerDraw; // Watts
    std::chrono::seconds duration;
};

class PerformanceProfiler {
private:
    std::vector<GPUInfo> gpuInfo_;
    std::map<std::string, std::chrono::steady_clock::time_point> timers_;
    std::map<std::string, std::vector<double>> measurements_;
    
public:
    // GPU Detection and Info
    bool detectGPUs();
    std::vector<GPUInfo> getGPUInfo() const { return gpuInfo_; }
    GPUInfo getGPUInfo(int deviceId) const;
    
    // Timing functions
    void startTimer(const std::string& name);
    double stopTimer(const std::string& name); // Returns milliseconds
    void addMeasurement(const std::string& category, double value);
    
    // Statistics
    double getAverage(const std::string& category) const;
    double getMin(const std::string& category) const;
    double getMax(const std::string& category) const;
    double getStdDev(const std::string& category) const;
    
    // Generate reports
    std::string generateReport() const;
    void saveReport(const std::string& filename) const;
};

class AlgorithmBenchmark {
private:
    PerformanceProfiler profiler_;
    int deviceId_;
    
public:
    AlgorithmBenchmark(int deviceId = 0) : deviceId_(deviceId) {}
    
    // Individual algorithm tests
    HashTestResult testKeccak512(uint32_t iterations = 10000);
    HashTestResult testBlake512(uint32_t iterations = 10000);
    HashTestResult testBMW512(uint32_t iterations = 10000);
    HashTestResult testGroestl512(uint32_t iterations = 10000);
    HashTestResult testAllAlgorithms(uint32_t iterations = 10000);
    
    // Full Flex algorithm test
    HashTestResult testFlexComplete(uint32_t iterations = 1000);
    
    // Memory bandwidth tests
    MemoryTestResult testGlobalMemoryBandwidth();
    MemoryTestResult testSharedMemoryBandwidth();
    MemoryTestResult testTextureMemoryBandwidth();
    
    // Generate comprehensive report
    std::string generateBenchmarkReport();
};

class OptimizationTuner {
private:
    int deviceId_;
    
public:
    OptimizationTuner(int deviceId = 0) : deviceId_(deviceId) {}
    
    // Thread configuration optimization
    struct ThreadConfig {
        int threadsPerBlock;
        int blocksPerGrid;
        double expectedHashrate;
    };
    
    std::vector<ThreadConfig> findOptimalThreadConfig();
    ThreadConfig autoTuneThreads(uint32_t testIterations = 1000);
    
    // Memory optimization
    bool testCoalescedMemoryAccess();
    double measureMemoryThroughput();
    
    // Algorithm-specific optimizations
    void optimizeForComputeCapability(int major, int minor);
    std::map<std::string, std::string> getOptimizationFlags() const;
};

class StressTest {
private:
    int deviceId_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> totalHashes_;
    std::chrono::steady_clock::time_point startTime_;
    
public:
    StressTest(int deviceId = 0) : deviceId_(deviceId), running_(false), totalHashes_(0) {}
    
    // Long-duration stability test
    void startStressTest(std::chrono::minutes duration);
    void stopStressTest();
    
    // Thermal monitoring
    ThermalTestResult runThermalTest(std::chrono::minutes duration);
    
    // Error detection
    bool runErrorDetectionTest(uint32_t iterations = 100000);
    
    // Memory stability
    bool runMemoryStabilityTest(std::chrono::minutes duration);
    
    // Power consumption analysis
    double measurePowerConsumption();
    
    // Statistics
    double getCurrentHashrate() const;
    uint64_t getTotalHashes() const { return totalHashes_; }
    std::chrono::seconds getRuntime() const;
};

class ComparisonBenchmark {
public:
    struct ComparisonResult {
        std::string configuration;
        double hashrate;
        double efficiency; // H/W
        double temperature;
        double memoryUsage;
    };
    
    // Compare different thread configurations
    std::vector<ComparisonResult> compareThreadConfigurations();
    
    // Compare with CPU implementation
    ComparisonResult compareCPUvsGPU();
    
    // Compare different GPUs
    std::vector<ComparisonResult> compareMultipleGPUs();
    
    // Generate comparison charts (CSV format)
    std::string generateComparisonCSV(const std::vector<ComparisonResult>& results);
};

// Utility functions for external use
namespace PerformanceUtils {
    // CUDA error checking
    bool checkCudaError(const std::string& operation);
    
    // GPU monitoring
    double getGPUTemperature(int deviceId);
    double getGPUPowerDraw(int deviceId);
    size_t getGPUMemoryUsage(int deviceId);
    
    // System info
    std::string getSystemInfo();
    std::string getCUDAVersion();
    std::string getDriverVersion();
    
    // Validation
    bool validateHashOutput(const std::vector<uint8_t>& input, const std::vector<uint8_t>& expectedHash);
    bool crossValidateGPUvsCPU(const std::string& algorithm, const std::vector<uint8_t>& input);
}

#endif // PERFORMANCE_TEST_H
