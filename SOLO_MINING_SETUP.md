# Lyncoin Solo Mining Setup Guide

This guide explains how to set up solo mining with the Flex CUDA Miner, mining directly to your wallet through a Lyncoin Core node.

## Overview

Solo mining means you mine directly to your own wallet without using a mining pool. You'll need:
1. **Lyncoin Core Node** - The official Lyncoin daemon/wallet
2. **Flex CUDA Miner** - This GPU miner 
3. **Mining Configuration** - RPC connection between miner and node

## Step 1: Install and Configure Lyncoin Core

### Download Lyncoin Core
1. Download the latest Lyncoin Core from: https://downloads.lyncoin.com/lyncoin-core/
2. Or build from source using the `lyncoin-4.0.0` folder in your workspace

### Build Lyncoin Core from Source (Advanced)
```bash
cd c:\Users\aseio\source\misc\lyncoin\lyncoin-4.0.0

# Windows (using Visual Studio)
cd build_msvc
msbuild bitcoin.sln /p:Configuration=Release

# Linux
./autogen.sh
./configure
make
```

### Configure Lyncoin Core for Mining

Create a configuration file `lyncoin.conf` in your Lyncoin data directory:

**Windows**: `%APPDATA%\Lyncoin\lyncoin.conf`
**Linux**: `~/.lyncoin/lyncoin.conf`

```ini
# lyncoin.conf - Solo Mining Configuration

# RPC Settings (for miner connection)
server=1
rpcuser=lyncoin_miner
rpcpassword=your_secure_password_here
rpcallowip=127.0.0.1
rpcport=8332

# Mining Settings
gen=1
genproclimit=0

# Network Settings
listen=1
daemon=1

# Wallet Settings (optional - for GUI)
disablewallet=0
```

### Start Lyncoin Core
```bash
# Command line (daemon mode)
lyncoind

# Or with GUI
lyncoin-qt
```

Wait for the blockchain to sync completely (this may take several hours).

## Step 2: Configure the Flex CUDA Miner for Solo Mining

### Create Solo Mining Configuration

Update your `config.ini` file:

```ini
[general]
# GPU Settings
gpu_device=0
threads_per_block=256
blocks_per_grid=0
use_fast_math=true

# Solo Mining Settings (RPC to local node)
mining_mode=solo
rpc_host=127.0.0.1
rpc_port=8332
rpc_user=lyncoin_miner
rpc_password=your_secure_password_here

# Mining Address (your wallet address)
mining_address=your_lyncoin_address_here

# Performance Settings
benchmark_on_start=true
auto_tune=true
```

### Get Your Mining Address

In Lyncoin Core wallet:
```bash
# Command line
lyncoin-cli getnewaddress "mining"

# Or use the GUI: Receive -> Request Payment
```

## Step 3: Modify the Miner for Solo Mining Support

Currently, the miner supports pool mining. For solo mining, you need to add RPC support to connect to your Lyncoin node.

### Required Modifications

The miner needs these additions:

1. **JSON-RPC Client** - To communicate with Lyncoin Core
2. **GetBlockTemplate** - To get work from the node
3. **SubmitBlock** - To submit solved blocks
4. **Work Management** - To handle block templates and targets

### Example Solo Mining Command

Once implemented, you would run:

```bash
# Solo mining to your wallet
flex-cuda-miner.exe --solo --rpc-host 127.0.0.1 --rpc-port 8332 --rpc-user lyncoin_miner --rpc-pass your_password --address your_lyncoin_address

# Current pool mining (already working)
flex-cuda-miner.exe --pool stratum+tcp://pool.com:4444 --user your_address --pass x
```

## Step 4: Implementation Plan for Solo Mining

To add solo mining support to the current miner, these files need updates:

### 1. Add RPC Client (`src/rpc_client.h` and `src/rpc_client.cpp`)

```cpp
class LyncoinRPCClient {
public:
    bool connect(const std::string& host, int port, const std::string& user, const std::string& pass);
    bool getBlockTemplate(BlockTemplate& blockTemplate);
    bool submitBlock(const std::string& blockHex);
    double getNetworkHashrate();
    int getBlockCount();
};
```

### 2. Update Main Application (`src/main.cpp`)

Add command-line options:
- `--solo` - Enable solo mining mode
- `--rpc-host` - RPC host (default: 127.0.0.1)
- `--rpc-port` - RPC port (default: 8332)  
- `--rpc-user` - RPC username
- `--rpc-pass` - RPC password
- `--address` - Mining address

### 3. Work Management (`src/solo_mining.h` and `src/solo_mining.cpp`)

```cpp
class SoloMiningManager {
public:
    bool initialize(const std::string& rpcHost, int rpcPort, const std::string& user, const std::string& pass);
    bool getWork(uint32_t* blockHeader, uint32_t* target);
    bool submitWork(uint32_t nonce, const uint8_t* hash);
    void updateDifficulty();
};
```

## Current Status

✅ **Pool Mining**: Fully implemented and working
⚠️ **Solo Mining**: Requires RPC client implementation

The current miner successfully connects to mining pools using the Stratum protocol. To add solo mining, the RPC client components listed above need to be implemented.

## Testing Solo Mining

### 1. Test RPC Connection
```bash
# Test if your node is accepting RPC connections
curl -u lyncoin_miner:your_password -d '{"jsonrpc":"1.0","id":"test","method":"getblockcount","params":[]}' -H 'content-type: text/plain;' http://127.0.0.1:8332/
```

### 2. Test Block Template
```bash
# Get a block template for mining
curl -u lyncoin_miner:your_password -d '{"jsonrpc":"1.0","id":"test","method":"getblocktemplate","params":[{"rules":["segwit"]}]}' -H 'content-type: text/plain;' http://127.0.0.1:8332/
```

## Security Notes

- Use a strong, unique RPC password
- Only allow RPC connections from localhost (127.0.0.1)
- Keep your Lyncoin Core wallet encrypted
- Backup your wallet regularly
- Monitor your node for security updates

## Troubleshooting

### "Connection refused" Error
- Ensure Lyncoin Core is running
- Check `lyncoin.conf` RPC settings
- Verify firewall settings

### "401 Unauthorized" Error  
- Check RPC username/password
- Verify `rpcuser` and `rpcpassword` in config

### "Method not found" Error
- Ensure you're using compatible Lyncoin Core version
- Check if mining is enabled (`gen=1`)

### Blockchain Not Syncing
- Check internet connection
- Verify Lyncoin Core is fully updated
- Allow time for initial sync (can take hours)

## Performance Tips

- **Solo vs Pool**: Solo mining finds blocks less frequently but keeps full rewards
- **Difficulty**: Solo mining competes against the entire network
- **Profitability**: Consider electricity costs vs potential rewards
- **Hardware**: More powerful GPUs increase your chances

## Next Steps

1. ✅ Current: Use pool mining with existing implementation
2. ⚠️ Future: Implement RPC client for solo mining support
3. 🔄 Alternative: Use existing mining pool that supports Lyncoin

---

**Note**: Solo mining is most profitable with significant hashrate. Consider pool mining for more consistent rewards.
