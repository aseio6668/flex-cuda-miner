/**
 * Mining Pool Implementation
 * Stratum protocol support for pool mining
 */

#include "mining_pool.h"

#ifdef SIMPLE_JSON_PARSER
    #include "simple_json.h"
#else
    #include <json/json.h>
#endif

#include <iostream>
#include <sstream>
#include <random>
#include <iomanip>

MiningPool::MiningPool(const std::string& address, int port, const std::string& username, const std::string& password)
    : poolAddress_(address), poolPort_(port), username_(username), password_(password),
      socket_(INVALID_SOCKET), connected_(false), running_(false), hasJob_(false), messageId_(1) {
    
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
}

MiningPool::~MiningPool() {
    disconnect();
#ifdef _WIN32
    WSACleanup();
#endif
}

bool MiningPool::connect() {
    if (connected_) {
        return true;
    }
    
    std::cout << "Connecting to pool: " << poolAddress_ << ":" << poolPort_ << std::endl;
    
    if (!initializeSocket()) {
        std::cerr << "Failed to initialize socket" << std::endl;
        return false;
    }
    
    // Connect to pool
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(poolPort_);
    
    struct hostent* host = gethostbyname(poolAddress_.c_str());
    if (!host) {
        std::cerr << "Failed to resolve pool address: " << poolAddress_ << std::endl;
        cleanupSocket();
        return false;
    }
    
    memcpy(&serverAddr.sin_addr, host->h_addr_list[0], host->h_length);
    
    if (::connect(socket_, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        std::cerr << "Failed to connect to pool" << std::endl;
        cleanupSocket();
        return false;
    }
    
    connected_ = true;
    running_ = true;
    lastActivity_ = std::chrono::steady_clock::now();
    
    // Start network and heartbeat threads
    networkThread_ = std::thread(&MiningPool::networkLoop, this);
    heartbeatThread_ = std::thread(&MiningPool::heartbeatLoop, this);
    
    // Perform stratum handshake
    if (!subscribe() || !authorize()) {
        std::cerr << "Failed to complete stratum handshake" << std::endl;
        disconnect();
        return false;
    }
    
    std::cout << "Successfully connected to pool!" << std::endl;
    
    // Notify connection callback
    {
        std::lock_guard<std::mutex> lock(callbackMutex_);
        if (connectionCallback_) {
            connectionCallback_(true);
        }
    }
    
    return true;
}

void MiningPool::disconnect() {
    if (!connected_) {
        return;
    }
    
    std::cout << "Disconnecting from pool..." << std::endl;
    
    running_ = false;
    connected_ = false;
    
    // Close socket to break network operations
    cleanupSocket();
    
    // Wait for threads to finish
    if (networkThread_.joinable()) {
        networkThread_.join();
    }
    if (heartbeatThread_.joinable()) {
        heartbeatThread_.join();
    }
    
    hasJob_ = false;
    
    // Notify connection callback
    {
        std::lock_guard<std::mutex> lock(callbackMutex_);
        if (connectionCallback_) {
            connectionCallback_(false);
        }
    }
    
    std::cout << "Disconnected from pool" << std::endl;
}

bool MiningPool::hasCurrentJob() const {
    std::lock_guard<std::mutex> lock(jobMutex_);
    return hasJob_;
}

MiningJob MiningPool::getCurrentJob() {
    std::lock_guard<std::mutex> lock(jobMutex_);
    return currentJob_;
}

bool MiningPool::submitShare(const std::string& jobId, const std::string& extraNonce2, 
                            const std::string& nTime, const std::string& nonce) {
    if (!connected_) {
        return false;
    }
    
    std::vector<std::string> params = {
        "\"" + username_ + "\"",
        "\"" + jobId + "\"",
        "\"" + extraNonce2 + "\"",
        "\"" + nTime + "\"",
        "\"" + nonce + "\""
    };
    
    std::string message = createMessage("mining.submit", params, getNextMessageId());
    
    std::cout << "Submitting share: " << nonce << std::endl;
    
    return sendMessage(message);
}

void MiningPool::setNewJobCallback(std::function<void(const MiningJob&)> callback) {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    newJobCallback_ = callback;
}

void MiningPool::setConnectionCallback(std::function<void(bool)> callback) {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    connectionCallback_ = callback;
}

std::string MiningPool::getPoolInfo() const {
    std::ostringstream ss;
    ss << "Pool: " << poolAddress_ << ":" << poolPort_ 
       << ", Connected: " << (connected_ ? "Yes" : "No")
       << ", Job: " << (hasJob_ ? "Yes" : "No");
    return ss.str();
}

double MiningPool::getLastPingTime() const {
    auto now = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastActivity_);
    return diff.count();
}

bool MiningPool::initializeSocket() {
    socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_ == INVALID_SOCKET) {
        return false;
    }
    
    // Set socket options
    int flag = 1;
#ifdef _WIN32
    setsockopt(socket_, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(flag));
#else
    setsockopt(socket_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
#endif
    
    return true;
}

void MiningPool::cleanupSocket() {
    if (socket_ != INVALID_SOCKET) {
        closesocket(socket_);
        socket_ = INVALID_SOCKET;
    }
}

void MiningPool::networkLoop() {
    std::cout << "Network thread started" << std::endl;
    
    while (running_ && connected_) {
        std::string message = receiveMessage();
        if (message.empty()) {
            if (running_) {
                std::cerr << "Lost connection to pool" << std::endl;
                connected_ = false;
            }
            break;
        }
        
        lastActivity_ = std::chrono::steady_clock::now();
        
        try {
            StratumMessage msg = parseMessage(message);
            
            if (msg.method.empty()) {
                handleResponse(msg);
            } else {
                handleNotification(msg);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing message: " << e.what() << std::endl;
        }
    }
    
    std::cout << "Network thread stopped" << std::endl;
}

void MiningPool::heartbeatLoop() {
    while (running_ && connected_) {
        std::this_thread::sleep_for(std::chrono::seconds(30));
        
        if (!running_ || !connected_) break;
        
        // Check if we've received data recently
        auto now = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::seconds>(now - lastActivity_);
        
        if (diff.count() > 120) { // 2 minutes timeout
            std::cerr << "Pool timeout - reconnecting..." << std::endl;
            connected_ = false;
            break;
        }
    }
    
    std::cout << "Heartbeat thread stopped" << std::endl;
}

bool MiningPool::sendMessage(const std::string& message) {
    if (socket_ == INVALID_SOCKET) {
        return false;
    }
    
    std::string fullMessage = message + "\n";
    int totalSent = 0;
    int messageLen = fullMessage.length();
    
    while (totalSent < messageLen) {
        int sent = send(socket_, fullMessage.c_str() + totalSent, messageLen - totalSent, 0);
        if (sent == SOCKET_ERROR) {
            return false;
        }
        totalSent += sent;
    }
    
    return true;
}

std::string MiningPool::receiveMessage() {
    if (socket_ == INVALID_SOCKET) {
        return "";
    }
    
    std::string buffer;
    char tempBuffer[1024];
    
    while (true) {
        int received = recv(socket_, tempBuffer, sizeof(tempBuffer) - 1, 0);
        if (received <= 0) {
            return "";
        }
        
        tempBuffer[received] = '\0';
        buffer += tempBuffer;
        
        // Look for complete messages (ended with \n)
        size_t newlinePos = buffer.find('\n');
        if (newlinePos != std::string::npos) {
            std::string message = buffer.substr(0, newlinePos);
            buffer = buffer.substr(newlinePos + 1);
            return message;
        }
    }
}

bool MiningPool::subscribe() {
    std::vector<std::string> params = {"\"Lyncoin-Flex-Miner/1.0\""};
    std::string message = createMessage("mining.subscribe", params, getNextMessageId());
    
    return sendMessage(message);
}

bool MiningPool::authorize() {
    std::vector<std::string> params = {"\"" + username_ + "\"", "\"" + password_ + "\""};
    std::string message = createMessage("mining.authorize", params, getNextMessageId());
    
    return sendMessage(message);
}

StratumMessage MiningPool::parseMessage(const std::string& json) {
    StratumMessage msg;
    
    try {
        Json::Value root;
        Json::Reader reader;
        
        if (!reader.parse(json, root)) {
            std::cerr << "Failed to parse JSON: " << json << std::endl;
            return msg;
        }
        
        if (root.isMember("method")) {
            // Notification
            msg.method = root["method"].asString();
            msg.isResponse = false;
            
            if (root.isMember("params") && root["params"].isArray()) {
                for (size_t i = 0; i < root["params"].size(); ++i) {
                    msg.params.push_back(root["params"][i].asString());
                }
            }
        } else {
            // Response
            msg.isResponse = true;
            if (root.isMember("id")) {
                if (root["id"].isInt()) {
                    msg.id = std::to_string(root["id"].asInt());
                } else {
                    msg.id = root["id"].asString();
                }
            }
            if (root.isMember("error") && !root["error"].isNull()) {
                msg.error = root["error"].asString();
            }        }
    } catch (const std::exception& e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
    }
    
    return msg;
}

std::string MiningPool::createMessage(const std::string& method, const std::vector<std::string>& params, int id) {
    Json::Value root;
    
    if (id >= 0) {
        root["id"] = id;
    } else {
        root["id"] = Json::Value::null;
    }
    
    root["method"] = method;
    
    Json::Value paramsArray = Json::makeArray();
    for (const auto& param : params) {
        // Check if param is already JSON string
        if (param.front() == '"' && param.back() == '"') {
            paramsArray.append(param.substr(1, param.length() - 2));
        } else {
            paramsArray.append(param);
        }    }
    root["params"] = paramsArray;
    
    Json::StreamWriterBuilder builder;
    // builder["indentation"] = "";  // Simplified - not needed for basic implementation
    return Json::writeString(builder, root);
}

void MiningPool::handleNotification(const StratumMessage& msg) {
    if (msg.method == "mining.notify") {
        handleMiningNotify(msg.params);
    } else if (msg.method == "mining.set_difficulty") {
        handleMiningSetDifficulty(msg.params);
    }
}

void MiningPool::handleResponse(const StratumMessage& msg) {
    if (!msg.error.empty()) {
        std::cerr << "Pool error: " << msg.error << std::endl;
    }
}

void MiningPool::handleMiningNotify(const std::vector<std::string>& params) {
    if (params.size() < 8) {
        std::cerr << "Invalid mining.notify parameters" << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(jobMutex_);
    
    currentJob_.jobId = params[0];
    currentJob_.previousHash = params[1];
    currentJob_.coinbase1 = params[2];
    currentJob_.coinbase2 = params[3];
    
    // Parse merkle branches
    currentJob_.merkleBranches.clear();
    // Merkle branches would be in params[4] as JSON array
    
    currentJob_.version = params[5];
    currentJob_.nBits = params[6];
    currentJob_.nTime = params[7];
    currentJob_.cleanJobs = (params.size() > 8 && params[8] == "true");
    
    hasJob_ = true;
    
    std::cout << "New job received: " << currentJob_.jobId << std::endl;
    
    // Notify callback
    {
        std::lock_guard<std::mutex> callbackLock(callbackMutex_);
        if (newJobCallback_) {
            newJobCallback_(currentJob_);
        }
    }
}

void MiningPool::handleMiningSetDifficulty(const std::vector<std::string>& params) {
    if (params.size() > 0) {
        std::cout << "Difficulty updated: " << params[0] << std::endl;
    }
}

std::string MiningPool::generateExtraNonce2(uint32_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    
    std::ostringstream ss;
    for (uint32_t i = 0; i < size; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << dis(gen);
    }
    
    return ss.str();
}

std::vector<uint8_t> MiningPool::hexToBytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byteString = hex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(strtol(byteString.c_str(), nullptr, 16));
        bytes.push_back(byte);
    }
    return bytes;
}

std::string MiningPool::bytesToHex(const std::vector<uint8_t>& bytes) {
    std::ostringstream ss;
    for (uint8_t byte : bytes) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
    }
    return ss.str();
}
