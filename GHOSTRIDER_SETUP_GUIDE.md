# GhostRider Mining Setup Guide

## Supported Coins

This miner supports GhostRider algorithm coins. Here are configuration examples for popular coins:

### Raptoreum (RTM)
```ini
# config.ini
algorithm=ghostrider
coin_name=Raptoreum
default_rpc_port=10226
```

**Command line:**
```bash
# Solo mining
./flex-cuda-miner --algorithm ghostrider --rpc-port 10226 --rpc-user your_rpc_user --rpc-pass your_rpc_pass --address your_rtm_address --solo

# Pool mining  
./flex-cuda-miner --algorithm ghostrider --pool stratum+tcp://rtm-pool.example.com:4444 --user your_rtm_address
```

### Salvium (SAL)
```ini
# config.ini
algorithm=ghostrider
coin_name=Salvium
default_rpc_port=18384
```

**Command line:**
```bash
# Solo mining
./flex-cuda-miner --algorithm ghostrider --coin Salvium --rpc-port 18384 --rpc-user your_rpc_user --rpc-pass your_rpc_pass --address your_sal_address --solo

# Pool mining
./flex-cuda-miner --algorithm ghostrider --coin Salvium --pool stratum+tcp://sal-pool.example.com:4444 --user your_sal_address
```

### Other GhostRider Coins
For any other coin using GhostRider algorithm:

```bash
# Solo mining (replace values as needed)
./flex-cuda-miner --algorithm ghostrider --coin "YourCoinName" --rpc-port YOUR_PORT --rpc-user your_rpc_user --rpc-pass your_rpc_pass --address your_address --solo

# Pool mining
./flex-cuda-miner --algorithm ghostrider --coin "YourCoinName" --pool stratum+tcp://your-pool.com:port --user your_address
```

## Common RPC Ports

| Coin | Default Port |
|------|--------------|
| Raptoreum (RTM) | 10226 |
| Salvium (SAL) | 18384 |
| Bitcoin (BTC) | 8332 |
| Litecoin (LTC) | 9332 |

## Setup Steps

1. **Install the coin's daemon** (for solo mining)
   - Download and install the official wallet/daemon
   - Configure RPC access in the daemon's config file
   - Start the daemon and let it sync

2. **Configure the miner**
   - Edit `config.ini` with your coin settings
   - Or use command line parameters

3. **Start mining**
   ```bash
   # Solo mining with daemon
   ./flex-cuda-miner --algorithm ghostrider --solo --rpc-user YOUR_USER --rpc-pass YOUR_PASS --address YOUR_ADDRESS
   
   # Pool mining  
   ./flex-cuda-miner --algorithm ghostrider --pool stratum+tcp://pool.address:port --user YOUR_ADDRESS
   ```

## Daemon Configuration Examples

### Raptoreum daemon config (raptoreum.conf):
```ini
rpcuser=your_rpc_username
rpcpassword=your_secure_rpc_password
rpcbind=127.0.0.1
rpcport=10226
rpcallowip=127.0.0.1
server=1
daemon=1
```

### Generic daemon config:
```ini
rpcuser=your_rpc_username
rpcpassword=your_secure_rpc_password  
rpcbind=127.0.0.1
rpcport=YOUR_COIN_PORT
rpcallowip=127.0.0.1
server=1
daemon=1
```

## Performance Tips

1. **GPU Selection**: Use `--device 0` to select specific GPU
2. **Algorithm Verification**: Make sure the coin actually uses GhostRider
3. **Pool Selection**: Choose pools with low latency to your location
4. **Monitor Temperature**: GhostRider can be intensive, watch GPU temps

## Troubleshooting

**RPC Connection Failed:**
- Verify daemon is running and synced
- Check RPC credentials in daemon config
- Ensure RPC port is correct
- Check firewall settings

**Low Hashrate:**
- Update GPU drivers
- Ensure adequate power supply
- Check GPU memory availability
- Try different thread configurations

**Pool Connection Issues:**
- Verify pool address and port
- Check if pool supports GhostRider
- Try different pool servers
- Verify wallet address format
