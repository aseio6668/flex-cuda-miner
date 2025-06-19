# Solo Mining Implementation Plan

## Overview

This document outlines the implementation plan for adding solo mining support to the Flex CUDA Miner. Currently, the miner supports pool mining via Stratum protocol. Solo mining requires RPC communication with a Lyncoin Core node.

## Current Status

✅ **Completed:**
- Pool mining with Stratum protocol
- Command-line argument parsing
- CUDA mining kernels (all 14 Flex algorithms)
- Performance monitoring and statistics

⚠️ **Needed for Solo Mining:**
- JSON-RPC client for Lyncoin Core communication
- Block template handling (getblocktemplate)
- Block submission (submitblock)
- Work management and difficulty updates

## Implementation Steps

### Phase 1: RPC Client Foundation

#### 1.1 Create RPC Client Classes
**Files to create:**
- `src/rpc_client.h`
- `src/rpc_client.cpp`

**Key classes:**
```cpp
class JSONRPCClient {
public:
    bool connect(const std::string& host, int port, const std::string& user, const std::string& pass);
    std::string call(const std::string& method, const std::vector<std::string>& params);
    bool isConnected() const;
};

class LyncoinRPCClient {
public:
    bool initialize(const std::string& host, int port, const std::string& user, const std::string& pass);
    bool getBlockTemplate(BlockTemplate& tmpl);
    bool submitBlock(const std::string& blockHex);
    double getNetworkHashrate();
    int getBlockCount();
    double getDifficulty();
};
```

#### 1.2 Block Template Structure
```cpp
struct BlockTemplate {
    std::string previousBlockHash;
    std::vector<std::string> transactions;
    std::string coinbaseValue;
    std::string target;
    std::string bits;
    int height;
    int version;
    uint32_t curTime;
    std::string coinbaseAux;
    std::string workId;
};
```

### Phase 2: Solo Mining Manager

#### 2.1 Create Solo Mining Manager
**Files to create:**
- `src/solo_mining.h`
- `src/solo_mining.cpp`

**Key functionality:**
```cpp
class SoloMiningManager {
private:
    LyncoinRPCClient rpcClient_;
    BlockTemplate currentTemplate_;
    uint32_t currentNonce_;
    std::chrono::steady_clock::time_point lastTemplateUpdate_;
    
public:
    bool initialize(const SoloMiningConfig& config);
    bool getWork(uint32_t* blockHeader, uint32_t* target);
    bool submitWork(uint32_t nonce, const uint8_t* hash);
    bool needsNewWork();
    void updateBlockTemplate();
};
```

#### 2.2 Work Construction
- Build block header from template
- Calculate merkle root
- Set proper difficulty target
- Handle nonce incrementing

### Phase 3: Integration with Main Application

#### 3.1 Update Command Line Arguments
Add to `main.cpp`:
```cpp
// New solo mining arguments
else if (strcmp(argv[i], "--solo") == 0) {
    soloMining = true;
} else if (strcmp(argv[i], "--rpc-host") == 0 && i + 1 < argc) {
    rpcHost = argv[++i];
} else if (strcmp(argv[i], "--rpc-port") == 0 && i + 1 < argc) {
    rpcPort = std::atoi(argv[++i]);
} else if (strcmp(argv[i], "--rpc-user") == 0 && i + 1 < argc) {
    rpcUser = argv[++i];
} else if (strcmp(argv[i], "--rpc-pass") == 0 && i + 1 < argc) {
    rpcPass = argv[++i];
} else if (strcmp(argv[i], "--address") == 0 && i + 1 < argc) {
    miningAddress = argv[++i];
}
```

#### 3.2 Update FlexCudaMiner Class
Add solo mining support:
```cpp
class FlexCudaMiner {
private:
    // Existing members...
    std::unique_ptr<SoloMiningManager> soloManager_;
    bool useSoloMining_;
    
public:
    void configureSolo(const std::string& host, int port, const std::string& user, 
                      const std::string& pass, const std::string& address);
    bool connectToNode();
    // Update mining loop to handle solo mining
};
```

#### 3.3 Update Mining Loop
Modify `miningLoop()` to handle both pool and solo mining:
```cpp
void miningLoop() {
    while (running && g_running) {
        // Get work (either from pool or solo manager)
        if (useSoloMining_) {
            if (!soloManager_->getWork(blockHeader, target)) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
        }
        
        // Run CUDA kernel (unchanged)
        flex_hash_gpu(totalThreads, currentNonce, blockHeader, target, result);
        
        // Submit work
        if (result[0] != 0) {
            if (useSoloMining_) {
                if (soloManager_->submitWork(result[0], reinterpret_cast<uint8_t*>(&result[1]))) {
                    std::cout << "*** BLOCK FOUND! Block submitted to network ***" << std::endl;
                }
            } else {
                // Existing pool submission code
            }
        }
    }
}
```

### Phase 4: Dependencies and Build System

#### 4.1 Add HTTP Client Library
**Option 1: Use curl (recommended)**
```cmake
# CMakeLists.txt
find_package(CURL REQUIRED)
target_link_libraries(flex-cuda-miner ${CURL_LIBRARIES})
```

**Option 2: Simple HTTP implementation**
- Create minimal HTTP client for JSON-RPC calls
- Windows: WinHTTP
- Linux: libcurl or raw sockets

#### 4.2 Update CMakeLists.txt
```cmake
# Add new source files
set(SOURCES
    # Existing sources...
    src/rpc_client.cpp
    src/solo_mining.cpp
)

# Add includes for curl
if(CURL_FOUND)
    target_include_directories(flex-cuda-miner PRIVATE ${CURL_INCLUDE_DIRS})
    target_link_libraries(flex-cuda-miner ${CURL_LIBRARIES})
endif()
```

### Phase 5: Testing and Validation

#### 5.1 Unit Tests
Create tests for:
- RPC client connectivity
- Block template parsing
- Work construction
- Block submission

#### 5.2 Integration Tests
- Test with local Lyncoin testnet
- Validate block template handling
- Test network connectivity

#### 5.3 Performance Testing
- Compare solo vs pool mining performance
- Validate share/block submission timing
- Test long-running stability

## Implementation Timeline

**Week 1: RPC Foundation**
- Implement JSON-RPC client
- Create Lyncoin RPC wrapper
- Basic connectivity testing

**Week 2: Solo Mining Manager**
- Block template handling
- Work construction and submission
- Integration with existing mining loop

**Week 3: Integration and Testing**
- Command-line interface updates
- Comprehensive testing
- Documentation updates

**Week 4: Production Deployment**
- Performance optimization
- Final testing with live network
- User guides and documentation

## Dependencies Required

### Build Dependencies
```bash
# Windows
vcpkg install curl

# Ubuntu/Debian
sudo apt install libcurl4-openssl-dev

# CentOS/RHEL
sudo yum install libcurl-devel
```

### Runtime Dependencies
- Lyncoin Core node (4.0.0+)
- Network connectivity to node
- Proper RPC configuration

## Testing Environment Setup

### 1. Local Testnet
```bash
# Start Lyncoin Core in testnet mode
lyncoind -testnet -rpcuser=test -rpcpassword=test -rpcallowip=127.0.0.1
```

### 2. Regtest Mode (Recommended for Development)
```bash
# Start in regtest mode for instant block generation
lyncoind -regtest -rpcuser=test -rpcpassword=test -rpcallowip=127.0.0.1 -gen=1
```

### 3. Test Mining
```bash
# Test solo mining on regtest
flex-cuda-miner.exe --solo --rpc-host 127.0.0.1 --rpc-port 18332 --rpc-user test --rpc-pass test --address bcrt1qtest...
```

## Success Criteria

✅ **RPC Client**: Successfully connects to Lyncoin Core and executes calls
✅ **Block Templates**: Correctly receives and parses block templates
✅ **Work Construction**: Builds valid block headers for mining
✅ **Block Submission**: Successfully submits found blocks to network
✅ **Integration**: Seamless switching between pool and solo mining modes
✅ **Performance**: Solo mining performance matches pool mining
✅ **Stability**: Long-running operation without memory leaks or crashes

## Current Usage (Temporary)

Until solo mining is implemented, users should:

1. **Pool Mining**: Use existing implementation
```bash
flex-cuda-miner.exe --pool stratum+tcp://pool.address:port --user address --pass x
```

2. **Manual Solo Mining**: Use existing mining software with Lyncoin Core
3. **Hybrid Approach**: Contribute to pools that support Lyncoin

---

**Priority**: Medium (pool mining already functional)
**Complexity**: Medium (RPC client + block template handling)
**Impact**: High (enables true decentralized mining)
