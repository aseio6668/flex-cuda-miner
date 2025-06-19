/**
 * JSON-RPC Client for Lyncoin Core Communication
 * Enables solo mining by connecting to Lyncoin Core node
 */

#ifndef RPC_CLIENT_H
#define RPC_CLIENT_H

#include <string>
#include <vector>
#include <memory>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <winhttp.h>
    #pragma comment(lib, "ws2_32.lib")
    #pragma comment(lib, "winhttp.lib")
#else
    #include <curl/curl.h>
#endif

struct BlockTemplate {
    std::string previousBlockHash;
    std::vector<std::string> transactions;
    std::string coinbaseValue;
    std::string target;
    std::string bits;
    int height;
    int version;
    uint32_t curTime;
    std::string workId;
    uint32_t nonceRange;
};

struct RPCResponse {
    bool success;
    std::string result;
    std::string error;
    int errorCode;
};

class HTTPClient {
private:
    std::string host_;
    int port_;
    std::string username_;
    std::string password_;
    
#ifdef _WIN32
    HINTERNET hSession_;
    HINTERNET hConnect_;
#else
    CURL* curl_;
#endif

public:
    HTTPClient(const std::string& host, int port, const std::string& username, const std::string& password);
    ~HTTPClient();
    
    bool initialize();
    void cleanup();
    
    RPCResponse sendRequest(const std::string& method, const std::vector<std::string>& params);
    
private:
    std::string createJSONRequest(const std::string& method, const std::vector<std::string>& params, int id = 1);
    RPCResponse parseJSONResponse(const std::string& response);
    
#ifdef _WIN32
    std::string sendHTTPRequest(const std::string& jsonData);
    std::string base64Encode(const std::string& input);
    std::wstring stringToWString(const std::string& str);
#else
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data);
#endif
};

class LyncoinRPCClient {
private:
    std::unique_ptr<HTTPClient> httpClient_;
    std::string host_;
    int port_;
    std::string username_;
    std::string password_;
    bool connected_;
    
public:
    LyncoinRPCClient(const std::string& host, int port, const std::string& username, const std::string& password);
    ~LyncoinRPCClient();
    
    // Connection management
    bool connect();
    void disconnect();
    bool isConnected() const { return connected_; }
    
    // Blockchain info
    int getBlockCount();
    double getDifficulty();
    double getNetworkHashrate();
    std::string getBestBlockHash();
    
    // Mining operations
    bool getBlockTemplate(BlockTemplate& blockTemplate);
    bool submitBlock(const std::string& blockHex);
    
    // Wallet operations
    std::vector<std::string> listAccounts();
    std::string getNewAddress(const std::string& account = "");
    double getBalance(const std::string& account = "");
    
    // Testing
    bool testConnection();
    std::string getBlockchainInfo();
    
private:
    bool parseBlockTemplate(const std::string& jsonResponse, BlockTemplate& blockTemplate);
    std::string constructBlockHex(const BlockTemplate& blockTemplate, uint32_t nonce, const std::string& coinbase);
};

#endif // RPC_CLIENT_H
