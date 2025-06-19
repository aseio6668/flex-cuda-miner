/**
 * Mining Pool Integration for Lyncoin Flex CUDA Miner
 * Implements Stratum protocol for pool mining
 */

#ifndef MINING_POOL_H
#define MINING_POOL_H

#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <unistd.h>
    #define SOCKET int
    #define INVALID_SOCKET -1
    #define SOCKET_ERROR -1
    #define closesocket close
#endif

struct MiningJob {
    std::string jobId;
    std::string previousHash;
    std::string coinbase1;
    std::string coinbase2;
    std::vector<std::string> merkleBranches;
    std::string version;
    std::string nBits;
    std::string nTime;
    bool cleanJobs;
    uint32_t extraNonce2Size;
    uint32_t extraNonce1;
};

struct StratumMessage {
    std::string id;
    std::string method;
    std::vector<std::string> params;
    std::string error;
    bool isResponse;
};

class MiningPool {
private:
    SOCKET socket_;
    std::string poolAddress_;
    int poolPort_;
    std::string username_;
    std::string password_;
    
    std::atomic<bool> connected_;
    std::atomic<bool> running_;
    std::thread networkThread_;
    std::thread heartbeatThread_;
    
    mutable std::mutex jobMutex_;
    MiningJob currentJob_;
    bool hasJob_;
    
    std::mutex callbackMutex_;
    std::function<void(const MiningJob&)> newJobCallback_;
    std::function<void(bool)> connectionCallback_;
    
    uint32_t messageId_;
    std::chrono::steady_clock::time_point lastActivity_;
    
public:
    MiningPool(const std::string& address, int port, const std::string& username, const std::string& password);
    ~MiningPool();
    
    // Connection management
    bool connect();
    void disconnect();
    bool isConnected() const { return connected_; }
    
    // Job management
    bool hasCurrentJob() const;
    MiningJob getCurrentJob();
    bool submitShare(const std::string& jobId, const std::string& extraNonce2, 
                     const std::string& nTime, const std::string& nonce);
    
    // Callbacks
    void setNewJobCallback(std::function<void(const MiningJob&)> callback);
    void setConnectionCallback(std::function<void(bool)> callback);
    
    // Statistics
    std::string getPoolInfo() const;
    double getLastPingTime() const;

private:
    // Network operations
    void networkLoop();
    void heartbeatLoop();
    bool sendMessage(const std::string& message);
    std::string receiveMessage();
    
    // Stratum protocol
    bool authorize();
    bool subscribe();
    StratumMessage parseMessage(const std::string& json);
    std::string createMessage(const std::string& method, const std::vector<std::string>& params, int id = -1);
    
    // Message handlers
    void handleNotification(const StratumMessage& msg);
    void handleResponse(const StratumMessage& msg);
    void handleMiningNotify(const std::vector<std::string>& params);
    void handleMiningSetDifficulty(const std::vector<std::string>& params);
    
    // Utility functions
    std::string generateExtraNonce2(uint32_t size);
    std::vector<uint8_t> hexToBytes(const std::string& hex);
    std::string bytesToHex(const std::vector<uint8_t>& bytes);
    uint32_t getNextMessageId() { return ++messageId_; }
    
    // Socket operations
    bool initializeSocket();
    void cleanupSocket();
};

// Pool manager for handling multiple pools
class PoolManager {
private:
    std::vector<std::unique_ptr<MiningPool>> pools_;
    std::atomic<int> activePipelineIndex_;
    std::mutex poolMutex_;
    
public:
    void addPool(const std::string& address, int port, const std::string& username, const std::string& password);
    bool connectToPool(int index = -1); // -1 for auto-select
    void disconnectAll();
    
    MiningPool* getActivePool();
    bool submitShare(const std::string& jobId, const std::string& extraNonce2, 
                     const std::string& nTime, const std::string& nonce);
    
    void setCallbacks(std::function<void(const MiningJob&)> newJobCallback,
                      std::function<void(bool)> connectionCallback);
    
    std::vector<std::string> getPoolStatuses() const;
};

#endif // MINING_POOL_H
