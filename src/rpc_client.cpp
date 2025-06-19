/**
 * JSON-RPC Client Implementation
 * Connects to cryptocurrency daemon for solo mining
 */

#include "rpc_client.h"
#include "simple_json.h"
#include <iostream>
#include <sstream>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

// HTTPClient Implementation
HTTPClient::HTTPClient(const std::string& host, int port, const std::string& username, const std::string& password)
    : host_(host), port_(port), username_(username), password_(password) {
#ifdef _WIN32
    hSession_ = nullptr;
    hConnect_ = nullptr;
#else
    curl_ = nullptr;
#endif
}

HTTPClient::~HTTPClient() {
    cleanup();
}

bool HTTPClient::initialize() {
#ifdef _WIN32
    hSession_ = WinHttpOpen(L"Lyncoin-Flex-Miner/1.0", 
                           WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                           WINHTTP_NO_PROXY_NAME, 
                           WINHTTP_NO_PROXY_BYPASS, 0);
    
    if (!hSession_) {
        std::cerr << "Failed to initialize WinHTTP session" << std::endl;
        return false;
    }
    
    // Convert host to wide string
    int hostLen = MultiByteToWideChar(CP_UTF8, 0, host_.c_str(), -1, NULL, 0);
    std::wstring wHost(hostLen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, host_.c_str(), -1, &wHost[0], hostLen);
    
    hConnect_ = WinHttpConnect(hSession_, wHost.c_str(), port_, 0);
    if (!hConnect_) {
        std::cerr << "Failed to connect to " << host_ << ":" << port_ << std::endl;
        return false;
    }
    
    return true;
#else
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_ = curl_easy_init();
    return (curl_ != nullptr);
#endif
}

void HTTPClient::cleanup() {
#ifdef _WIN32
    if (hConnect_) {
        WinHttpCloseHandle(hConnect_);
        hConnect_ = nullptr;
    }
    if (hSession_) {
        WinHttpCloseHandle(hSession_);
        hSession_ = nullptr;
    }
#else
    if (curl_) {
        curl_easy_cleanup(curl_);
        curl_ = nullptr;
    }
    curl_global_cleanup();
#endif
}

std::string HTTPClient::createJSONRequest(const std::string& method, const std::vector<std::string>& params, int id) {
    // Create a simple JSON-RPC 1.0 request manually
    std::ostringstream json;
    json << "{";
    json << "\"jsonrpc\":\"1.0\",";
    json << "\"id\":" << id << ",";
    json << "\"method\":\"" << method << "\",";
    json << "\"params\":[";
    
    for (size_t i = 0; i < params.size(); ++i) {
        if (i > 0) json << ",";
        
        // Check if parameter is already a JSON object/array
        if (params[i].front() == '{' || params[i].front() == '[') {
            json << params[i];  // Don't quote JSON objects/arrays
        } else {
            json << "\"" << params[i] << "\"";  // Quote strings
        }
    }
    
    json << "]}";
    
    std::string result = json.str();
    std::cout << "Sending JSON-RPC request: " << result << std::endl;
    return result;
}

RPCResponse HTTPClient::parseJSONResponse(const std::string& response) {
    RPCResponse rpcResponse;
    rpcResponse.success = false;
    rpcResponse.errorCode = 0;
    
    std::cout << "Received RPC response: " << response.substr(0, 200) << std::endl;
    
    try {
        // Simple JSON parsing for basic responses
        if (response.find("\"error\"") != std::string::npos && response.find("null") == std::string::npos) {
            // Extract error message
            size_t msgStart = response.find("\"message\":");
            if (msgStart != std::string::npos) {
                msgStart = response.find("\"", msgStart + 10);
                size_t msgEnd = response.find("\"", msgStart + 1);
                if (msgStart != std::string::npos && msgEnd != std::string::npos) {
                    rpcResponse.error = response.substr(msgStart + 1, msgEnd - msgStart - 1);
                }
            }
            
            // Extract error code
            size_t codeStart = response.find("\"code\":");
            if (codeStart != std::string::npos) {
                codeStart += 7;
                size_t codeEnd = response.find_first_of(",}", codeStart);
                if (codeEnd != std::string::npos) {
                    std::string codeStr = response.substr(codeStart, codeEnd - codeStart);
                    rpcResponse.errorCode = std::stoi(codeStr);
                }
            }
            return rpcResponse;
        }
        
        // Extract result
        size_t resultStart = response.find("\"result\":");
        if (resultStart != std::string::npos) {
            resultStart += 9;
            // Find the end of the result value
            size_t resultEnd = response.find(",\"error\"", resultStart);
            if (resultEnd == std::string::npos) {
                resultEnd = response.find(",\"id\"", resultStart);
            }
            if (resultEnd == std::string::npos) {
                resultEnd = response.find("}", resultStart);
            }
            
            if (resultEnd != std::string::npos) {
                rpcResponse.result = response.substr(resultStart, resultEnd - resultStart);
                rpcResponse.success = true;
            }
        }
        
    } catch (const std::exception& e) {
        rpcResponse.error = std::string("JSON parsing error: ") + e.what();
    }
    
    return rpcResponse;
}

#ifdef _WIN32
std::string HTTPClient::sendHTTPRequest(const std::string& jsonData) {
    if (!hConnect_) {
        std::cerr << "HTTP connection not established" << std::endl;
        return "";
    }
    
    std::cout << "Sending HTTP request to: " << host_ << ":" << port_ << std::endl;
    
    HINTERNET hRequest = WinHttpOpenRequest(hConnect_, L"POST", L"/",
                                          NULL, WINHTTP_NO_REFERER,
                                          WINHTTP_DEFAULT_ACCEPT_TYPES,
                                          0);
    
    if (!hRequest) {
        DWORD error = GetLastError();
        std::cerr << "WinHttpOpenRequest failed with error: " << error << std::endl;
        return "";
    }
      // Prepare headers
    std::wstring headers = L"Content-Type: application/json\r\n";
    
    // Add basic authentication (Base64 encoded)
    std::string auth = username_ + ":" + password_;
    std::string authBase64 = base64Encode(auth);
    std::wstring authHeader = L"Authorization: Basic " + stringToWString(authBase64) + L"\r\n";
    headers += authHeader;
      // Send request
    bool result = WinHttpSendRequest(hRequest,
                                   headers.c_str(), -1,
                                   (LPVOID)jsonData.c_str(), jsonData.length(),
                                   jsonData.length(), 0);
    
    if (!result) {
        DWORD error = GetLastError();
        std::cerr << "WinHttpSendRequest failed with error: " << error << std::endl;
        WinHttpCloseHandle(hRequest);
        return "";
    }
    
    std::string response;
    if (!WinHttpReceiveResponse(hRequest, NULL)) {
        DWORD error = GetLastError();
        std::cerr << "WinHttpReceiveResponse failed with error: " << error << std::endl;
        WinHttpCloseHandle(hRequest);
        return "";
    }
    
    // Get HTTP status code
    DWORD statusCode = 0;
    DWORD statusCodeSize = sizeof(statusCode);
    WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                       WINHTTP_HEADER_NAME_BY_INDEX, &statusCode, &statusCodeSize, WINHTTP_NO_HEADER_INDEX);
    
    if (statusCode != 200) {
        std::cerr << "HTTP request failed with status code: " << statusCode << std::endl;
        if (statusCode == 401) {
            std::cerr << "Authentication failed - check RPC username and password" << std::endl;
        }
        WinHttpCloseHandle(hRequest);
        return "";
    }
    
    DWORD size = 0;
    DWORD downloaded = 0;
    
    do {
        size = 0;
        if (!WinHttpQueryDataAvailable(hRequest, &size)) {
            break;
        }
        
        std::vector<char> buffer(size + 1);
        ZeroMemory(&buffer[0], size + 1);
        
        if (!WinHttpReadData(hRequest, &buffer[0], size, &downloaded)) {
            break;
        }
        
        response.append(&buffer[0], downloaded);
        
    } while (size > 0);
    
    WinHttpCloseHandle(hRequest);
    return response;
}
#else
size_t HTTPClient::WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data) {
    size_t totalSize = size * nmemb;
    data->append((char*)contents, totalSize);
    return totalSize;
}
#endif

RPCResponse HTTPClient::sendRequest(const std::string& method, const std::vector<std::string>& params) {
    RPCResponse response;
    response.success = false;
    
    std::string jsonRequest = createJSONRequest(method, params);
    
#ifdef _WIN32
    std::string httpResponse = sendHTTPRequest(jsonRequest);
    if (httpResponse.empty()) {
        response.error = "Failed to send HTTP request";
        return response;
    }
    return parseJSONResponse(httpResponse);
#else
    if (!curl_) {
        response.error = "CURL not initialized";
        return response;
    }
    
    std::string responseData;
    std::string url = "http://" + host_ + ":" + std::to_string(port_) + "/";
    
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, jsonRequest.c_str());
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &responseData);
    
    // Basic authentication
    std::string userpass = username_ + ":" + password_;
    curl_easy_setopt(curl_, CURLOPT_USERPWD, userpass.c_str());
    
    // Headers
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
    
    CURLcode res = curl_easy_perform(curl_);
    curl_slist_free_all(headers);
    
    if (res != CURLE_OK) {
        response.error = "CURL error: " + std::string(curl_easy_strerror(res));
        return response;
    }
    
    return parseJSONResponse(responseData);
#endif
}

// LyncoinRPCClient Implementation
LyncoinRPCClient::LyncoinRPCClient(const std::string& host, int port, const std::string& username, const std::string& password)
    : host_(host), port_(port), username_(username), password_(password), connected_(false) {
    httpClient_ = std::make_unique<HTTPClient>(host, port, username, password);
}

LyncoinRPCClient::~LyncoinRPCClient() {
    disconnect();
}

bool LyncoinRPCClient::connect() {
    if (connected_) {
        return true;
    }
    
    std::cout << "Connecting to cryptocurrency daemon RPC at " << host_ << ":" << port_ << std::endl;
    
    if (!httpClient_->initialize()) {
        std::cerr << "Failed to initialize HTTP client" << std::endl;
        return false;
    }
    
    // Test connection
    if (!testConnection()) {
        std::cerr << "Failed to connect to cryptocurrency daemon RPC" << std::endl;
        return false;
    }
    
    connected_ = true;
    std::cout << "Successfully connected to cryptocurrency daemon!" << std::endl;
    return true;
}

void LyncoinRPCClient::disconnect() {
    if (httpClient_) {
        httpClient_->cleanup();
    }
    connected_ = false;
}

bool LyncoinRPCClient::testConnection() {
    std::cout << "Testing RPC connection..." << std::endl;
    RPCResponse response = httpClient_->sendRequest("getblockcount", {});
    if (!response.success) {
        std::cerr << "RPC test failed: " << response.error << std::endl;
        if (response.errorCode != 0) {
            std::cerr << "Error code: " << response.errorCode << std::endl;
        }
    } else {
        std::cout << "RPC connection test successful!" << std::endl;
    }
    return response.success;
}

int LyncoinRPCClient::getBlockCount() {
    RPCResponse response = httpClient_->sendRequest("getblockcount", {});
    if (response.success) {
        try {
            return std::stoi(response.result);
        } catch (...) {
            return -1;
        }
    }
    return -1;
}

bool LyncoinRPCClient::getBlockTemplate(BlockTemplate& blockTemplate) {
    std::vector<std::string> params = {"{\"rules\":[\"segwit\"]}"};
    RPCResponse response = httpClient_->sendRequest("getblocktemplate", params);
    
    if (!response.success) {
        std::cerr << "Failed to get block template: " << response.error << std::endl;
        return false;
    }
    
    return parseBlockTemplate(response.result, blockTemplate);
}

bool LyncoinRPCClient::parseBlockTemplate(const std::string& jsonResponse, BlockTemplate& blockTemplate) {
    try {
        Json::Value root;
        Json::Reader reader;
        
        if (!reader.parse(jsonResponse, root)) {
            std::cerr << "Failed to parse block template JSON" << std::endl;
            return false;
        }
        
        blockTemplate.version = root["version"].asInt();
        blockTemplate.previousBlockHash = root["previousblockhash"].asString();
        blockTemplate.height = root["height"].asInt();
        blockTemplate.curTime = root["curtime"].asInt();
        blockTemplate.bits = root["bits"].asString();
        blockTemplate.target = root["target"].asString();
        blockTemplate.coinbaseValue = root["coinbasevalue"].asString();
        
        // Parse transactions
        if (root.isMember("transactions") && root["transactions"].isArray()) {
            for (size_t i = 0; i < root["transactions"].size(); ++i) {
                blockTemplate.transactions.push_back(root["transactions"][i]["data"].asString());
            }
        }
        
        std::cout << "Block template received: height=" << blockTemplate.height 
                  << ", txns=" << blockTemplate.transactions.size() << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing block template: " << e.what() << std::endl;
        return false;
    }
}

std::string LyncoinRPCClient::getBlockchainInfo() {
    RPCResponse response = httpClient_->sendRequest("getblockchaininfo", {});
    return response.success ? response.result : response.error;
}

#ifdef _WIN32
std::string HTTPClient::base64Encode(const std::string& input) {
    static const char base64_chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::string result;
    int val = 0, valb = -6;
    for (unsigned char c : input) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            result.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) result.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (result.size() % 4) result.push_back('=');
    return result;
}

std::wstring HTTPClient::stringToWString(const std::string& str) {
    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, NULL, 0);
    std::wstring wstr(size, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], size);
    return wstr;
}
#endif
