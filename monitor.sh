#!/bin/bash
# Lyncoin Flex CUDA Miner - Production Monitoring Script
# =====================================================

# Configuration
MINER_EXECUTABLE="./build/bin/flex-cuda-miner"
LOG_FILE="miner.log"
RESTART_LOG="restart.log"
MAX_RESTARTS=5
RESTART_WINDOW=3600  # 1 hour
CHECK_INTERVAL=30    # 30 seconds

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
restart_count=0
restart_times=()

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RESTART_LOG"
}

# Function to check if miner is running
is_miner_running() {
    pgrep -f "flex-cuda-miner" > /dev/null 2>&1
}

# Function to get miner PID
get_miner_pid() {
    pgrep -f "flex-cuda-miner" | head -1
}

# Function to check GPU temperature
check_gpu_temp() {
    if command -v nvidia-smi &> /dev/null; then
        temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
        if [ "$temp" -gt 85 ]; then
            log_with_timestamp "WARNING: GPU temperature high: ${temp}°C"
            return 1
        fi
    fi
    return 0
}

# Function to check hashrate from log
check_hashrate() {
    if [ -f "$LOG_FILE" ]; then
        # Look for recent hashrate in log (last 2 minutes)
        recent_hashrate=$(tail -100 "$LOG_FILE" | grep -E "Hash rate|hashrate" | tail -1)
        if [ -n "$recent_hashrate" ]; then
            echo -e "${GREEN}✓${NC} Recent hashrate: $recent_hashrate"
            return 0
        else
            log_with_timestamp "WARNING: No recent hashrate found in logs"
            return 1
        fi
    fi
    return 1
}

# Function to clean old restart times
clean_restart_times() {
    current_time=$(date +%s)
    new_times=()
    for time in "${restart_times[@]}"; do
        if [ $((current_time - time)) -lt $RESTART_WINDOW ]; then
            new_times+=("$time")
        fi
    done
    restart_times=("${new_times[@]}")
}

# Function to start miner
start_miner() {
    log_with_timestamp "Starting miner..."
    
    # Check if config exists
    if [ ! -f "config.ini" ]; then
        log_with_timestamp "ERROR: config.ini not found"
        return 1
    fi
    
    # Start miner in background
    nohup "$MINER_EXECUTABLE" > "$LOG_FILE" 2>&1 &
    sleep 5
    
    if is_miner_running; then
        local pid=$(get_miner_pid)
        log_with_timestamp "Miner started successfully (PID: $pid)"
        return 0
    else
        log_with_timestamp "ERROR: Failed to start miner"
        return 1
    fi
}

# Function to stop miner
stop_miner() {
    log_with_timestamp "Stopping miner..."
    pkill -f "flex-cuda-miner"
    sleep 3
    
    if is_miner_running; then
        log_with_timestamp "Force killing miner..."
        pkill -9 -f "flex-cuda-miner"
        sleep 2
    fi
    
    if ! is_miner_running; then
        log_with_timestamp "Miner stopped successfully"
        return 0
    else
        log_with_timestamp "ERROR: Failed to stop miner"
        return 1
    fi
}

# Function to restart miner
restart_miner() {
    clean_restart_times
    
    if [ ${#restart_times[@]} -ge $MAX_RESTARTS ]; then
        log_with_timestamp "ERROR: Too many restarts in the last hour (${#restart_times[@]}/$MAX_RESTARTS)"
        log_with_timestamp "Stopping monitoring to prevent restart loop"
        exit 1
    fi
    
    restart_times+=($(date +%s))
    
    log_with_timestamp "Restarting miner (attempt $((${#restart_times[@]}))/$MAX_RESTARTS)..."
    
    stop_miner
    sleep 5
    
    if start_miner; then
        log_with_timestamp "Miner restarted successfully"
        return 0
    else
        log_with_timestamp "ERROR: Failed to restart miner"
        return 1
    fi
}

# Signal handlers
cleanup() {
    echo ""
    log_with_timestamp "Monitoring script interrupted"
    if is_miner_running; then
        echo "Stopping miner..."
        stop_miner
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# Main monitoring loop
echo "========================================"
echo "Lyncoin Flex CUDA Miner Monitor v1.0"
echo "========================================"
echo ""
echo "Monitoring configuration:"
echo "  Executable: $MINER_EXECUTABLE"
echo "  Log file: $LOG_FILE"
echo "  Check interval: ${CHECK_INTERVAL}s"
echo "  Max restarts: $MAX_RESTARTS per hour"
echo ""

log_with_timestamp "Monitoring started"

# Initial start if not running
if ! is_miner_running; then
    echo -e "${YELLOW}Miner not running, starting...${NC}"
    start_miner
fi

while true; do
    if is_miner_running; then
        pid=$(get_miner_pid)
        echo -e "${GREEN}✓${NC} Miner running (PID: $pid)"
        
        # Check GPU temperature
        check_gpu_temp
        
        # Check hashrate
        check_hashrate
        
        # Check memory usage
        if command -v nvidia-smi &> /dev/null; then
            memory_usage=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1 | tr ',' '/')
            echo -e "${BLUE}ℹ${NC} GPU Memory: ${memory_usage} MB"
        fi
        
    else
        echo -e "${RED}✗${NC} Miner not running"
        log_with_timestamp "Miner process not found, attempting restart..."
        
        if ! restart_miner; then
            log_with_timestamp "Failed to restart miner, waiting before retry..."
            sleep 60
        fi
    fi
    
    echo "Waiting ${CHECK_INTERVAL}s for next check..."
    echo ""
    sleep $CHECK_INTERVAL
done
