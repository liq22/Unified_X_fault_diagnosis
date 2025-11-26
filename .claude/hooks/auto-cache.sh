#!/bin/bash
#
# Claude Code Simple Auto-Cache Hook
# Automatically starts simplified cache system (timestamp + content only)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CACHE_SCRIPT="$PROJECT_DIR/src/scripts/cache/start_cache.py"
CACHE_PID_FILE="$PROJECT_DIR/src/dev/cache/cache.pid"

# Function to start cache system
start_cache() {
    if [ -f "$CACHE_PID_FILE" ]; then
        local existing_pid=$(cat "$CACHE_PID_FILE")
        if ps -p "$existing_pid" > /dev/null 2>&1; then
            echo "Cache system already running with PID: $existing_pid"
            return 0
        else
            rm -f "$CACHE_PID_FILE"
        fi
    fi

    # Start cache system in background
    cd "$PROJECT_DIR"
    python "$CACHE_SCRIPT" --daemon > src/dev/cache/cache.log 2>&1 &
    local cache_pid=$!
    
    # Save PID for later cleanup
    echo "$cache_pid" > "$CACHE_PID_FILE"
    
    echo "✅ Simple auto-cache system started with PID: $cache_pid"
    echo "📊 Simple caching enabled (timestamp + content only):"
    echo "   - Claude thinking processes"
    echo "   - Research sessions"
    echo "   - Agent executions"
    echo "📁 Cache directory: $PROJECT_DIR/src/dev/cache/"
    
    return 0
}

# Function to stop cache system
stop_cache() {
    if [ -f "$CACHE_PID_FILE" ]; then
        local cache_pid=$(cat "$CACHE_PID_FILE")
        if ps -p "$cache_pid" > /dev/null 2>&1; then
            kill "$cache_pid"
            echo "🛑 Cache system stopped (PID: $cache_pid)"
        fi
        rm -f "$CACHE_PID_FILE"
    else
        echo "No cache system PID file found"
    fi
}

# Function to check cache status
status_cache() {
    if [ -f "$CACHE_PID_FILE" ]; then
        local cache_pid=$(cat "$CACHE_PID_FILE")
        if ps -p "$cache_pid" > /dev/null 2>&1; then
            echo "✅ Cache system running (PID: $cache_pid)"
            return 0
        else
            echo "❌ Cache PID file exists but process not running"
            rm -f "$CACHE_PID_FILE"
            return 1
        fi
    else
        echo "❌ Cache system not running"
        return 1
    fi
}

# Handle different commands
case "${1:-start}" in
    start)
        start_cache
        ;;
    stop)
        stop_cache
        ;;
    restart)
        stop_cache
        start_cache
        ;;
    status)
        status_cache
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo "  start   - Start auto-cache system"
        echo "  stop    - Stop auto-cache system"  
        echo "  restart - Restart auto-cache system"
        echo "  status  - Check cache system status"
        exit 1
        ;;
esac