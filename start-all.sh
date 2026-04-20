#!/bin/bash

################################################################################
# DepressoSpeech - Start All Services
# This script starts Frontend, Backend, and ML Model services
#
# Usage:
#   ./start-all.sh                    # Start with dependency check
#   ./start-all.sh --no-deps          # Start without installing dependencies
#   ./start-all.sh --kill-only        # Only kill existing processes
#
# Services will run on:
#   - Frontend: http://localhost:5173
#   - Backend: http://localhost:8000
#   - ML Model: http://localhost:8001
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/Depression-UI"
BACKEND_DIR="$SCRIPT_DIR/backend"
MODEL_DIR="$SCRIPT_DIR/Model"

FRONTEND_PORT=5173
BACKEND_PORT=8000
MODEL_PORT=8001

INSTALL_DEPS=true
KILL_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-deps)
            INSTALL_DEPS=false
            shift
            ;;
        --kill-only)
            KILL_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-deps] [--kill-only]"
            exit 1
            ;;
    esac
done

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}➜ $1${NC}"
}

# Function to check if a port is in use
is_port_in_use() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    elif fuser $port/tcp >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to get PIDs by port (tries lsof then fuser)
get_pids_by_port() {
    local port=$1
    local pids
    pids=$(lsof -i :$port -t 2>/dev/null | tr '\n' ' ')
    if [ -z "$pids" ]; then
        pids=$(fuser $port/tcp 2>/dev/null | tr -s ' ')
    fi
    echo "$pids"
}

# Function to kill process on port
kill_port() {
    local port=$1
    if is_port_in_use $port; then
        print_info "Port $port is in use. Killing process..."
        local pids=$(get_pids_by_port $port)
        if [ ! -z "$pids" ]; then
            kill -9 $pids 2>/dev/null || true
            sleep 1
            if is_port_in_use $port; then
                print_error "Failed to kill process on port $port"
                return 1
            else
                print_success "Killed process on port $port"
                return 0
            fi
        fi
    else
        print_success "Port $port is free"
        return 0
    fi
}

## Function to check NodeJS installation
check_nodejs() {
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed"
        echo "Please install Node.js from https://nodejs.org/"
        return 1
    fi
    print_success "Node.js found: $(node --version)"
    return 0
}

# Function to check Python installation
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        echo "Please install Python 3"
        return 1
    fi
    print_success "Python 3 found: $(python3 --version)"
    return 0
}

# Function to install npm dependencies
install_npm_deps() {
    print_header "Installing Frontend Dependencies (npm)"
    if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
        cd "$FRONTEND_DIR"
        print_info "Running: npm install"
        npm install --legacy-peer-deps
        print_success "Frontend dependencies installed"
        cd "$SCRIPT_DIR"
    else
        print_success "Frontend dependencies already installed"
    fi
}

# Function to install Python dependencies for Backend
install_backend_deps() {
    print_header "Installing Backend Dependencies (pip)"

    # Check if already installed (simple check using pip show)
    if pip show fastapi > /dev/null 2>&1; then
        print_success "Backend dependencies already installed"
    else
        cd "$BACKEND_DIR"
        print_info "Running: pip install -r requirements.txt"
        pip install -r requirements.txt
        print_success "Backend dependencies installed"
        cd "$SCRIPT_DIR"
    fi
}

# Function to install Python dependencies for Model (uses Model venv)
install_model_deps() {
    print_header "Installing ML Model Dependencies (pip)"

    local PIP_CMD="pip"
    if [ -f "$MODEL_DIR/.venv/bin/pip" ]; then
        PIP_CMD="$MODEL_DIR/.venv/bin/pip"
    fi

    if $PIP_CMD show slowapi > /dev/null 2>&1; then
        print_success "ML Model dependencies already installed"
    else
        cd "$MODEL_DIR"
        print_info "Running: pip install -r requirements.txt (in Model venv)"
        $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        $PIP_CMD install -r requirements.txt
        print_success "ML Model dependencies installed"
        cd "$SCRIPT_DIR"
    fi
}

# Function to start frontend
start_frontend() {
    print_header "Starting Frontend (Vite Dev Server)"
    cd "$FRONTEND_DIR"
    print_info "Frontend will run on: http://localhost:$FRONTEND_PORT"
    npm run dev > /tmp/frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd "$SCRIPT_DIR"
    sleep 2
}

# Function to start backend
start_backend() {
    print_header "Starting Backend (FastAPI - MindScope)"
    print_info "Backend will run on: http://localhost:$BACKEND_PORT"
    (cd "$BACKEND_DIR" && python3 -m uvicorn main:app --host 0.0.0.0 --port $BACKEND_PORT --reload > /tmp/backend.log 2>&1) &
    BACKEND_PID=$!
    sleep 2
}

# Function to start model server (activates Model venv)
start_model() {
    print_header "Starting ML Model Server (DepressoSpeech API)"
    print_info "ML Model Server will run on: http://localhost:$MODEL_PORT"

    if [ -f "$MODEL_DIR/.venv/bin/activate" ]; then
        print_info "Activating Model virtual environment..."
        (cd "$MODEL_DIR" && source .venv/bin/activate && python scripts/serve.py --port $MODEL_PORT > /tmp/model_serve.log 2>&1) &
    else
        print_info "No Model venv found, using system Python..."
        (cd "$MODEL_DIR" && python3 scripts/serve.py --port $MODEL_PORT > /tmp/model_serve.log 2>&1) &
    fi
    MODEL_PID=$!
    sleep 5
}

# Function to check service status (with retries for slow starters)
check_service_status() {
    local service=$1
    local port=$2
    local log_file=$3
    local retries=5

    for i in $(seq 1 $retries); do
        if is_port_in_use $port; then
            print_success "$service is running on port $port"
            return 0
        fi
        [ $i -lt $retries ] && sleep 2
    done

    print_error "$service failed to start on port $port"
    if [ -f "$log_file" ]; then
        echo -e "\n${YELLOW}Last 10 lines from log:${NC}"
        tail -10 "$log_file"
    fi
    return 1
}

# Function to show final status
show_final_status() {
    print_header "Service Status"

    echo -e "${BLUE}Frontend (Vite)${NC}"
    check_service_status "Frontend" "$FRONTEND_PORT" "/tmp/frontend.log" || true

    echo ""
    echo -e "${BLUE}Backend (FastAPI - MindScope)${NC}"
    check_service_status "Backend" "$BACKEND_PORT" "/tmp/backend.log" || true

    echo ""
    echo -e "${BLUE}ML Model Server (DepressoSpeech API)${NC}"
    check_service_status "ML Model" "$MODEL_PORT" "/tmp/model_serve.log" || true

    echo ""
    print_header "Access Points"
    echo -e "${GREEN}Frontend:        ${NC}http://localhost:$FRONTEND_PORT"
    echo -e "${GREEN}Backend API:     ${NC}http://localhost:$BACKEND_PORT/api/v1"
    echo -e "${GREEN}Backend Docs:    ${NC}http://localhost:$BACKEND_PORT/docs"
    echo -e "${GREEN}Model API:       ${NC}http://localhost:$MODEL_PORT"
    echo -e "${GREEN}Model Docs:      ${NC}http://localhost:$MODEL_PORT/docs"

    echo ""
    print_header "Log Files"
    echo -e "Frontend:  /tmp/frontend.log"
    echo -e "Backend:   /tmp/backend.log"
    echo -e "Model:     /tmp/model_serve.log"

    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}\n"
}

################################################################################
# Main Script
################################################################################

print_header "DepressoSpeech - Start All Services"

# Check if running from correct directory
if [ ! -d "$FRONTEND_DIR" ] || [ ! -d "$BACKEND_DIR" ] || [ ! -d "$MODEL_DIR" ]; then
    print_error "Not running from DepressoSpeech root directory"
    print_error "Frontend dir: $FRONTEND_DIR"
    print_error "Backend dir: $BACKEND_DIR"
    print_error "Model dir: $MODEL_DIR"
    exit 1
fi

print_success "Running from: $SCRIPT_DIR"

# Kill existing processes on ports
print_header "Checking and Clearing Ports"
kill_port $FRONTEND_PORT || exit 1
kill_port $BACKEND_PORT || exit 1
kill_port $MODEL_PORT || exit 1

# Exit if kill-only flag was set
if [ "$KILL_ONLY" = true ]; then
    print_success "All ports cleared. Exiting."
    exit 0
fi

# Disable exit-on-error for service startup (background processes)
set +e

# Check prerequisites
print_header "Checking Prerequisites"
check_nodejs || exit 1
check_python || exit 1

# Install dependencies if needed
if [ "$INSTALL_DEPS" = true ]; then
    install_npm_deps
    install_backend_deps
    install_model_deps
else
    print_info "Skipping dependency installation (--no-deps flag set)"
fi

# Start all services
print_header "Starting Services"
start_frontend
start_backend
start_model

# Show status
show_final_status

# Keep script running to handle signals
trap 'print_error "Shutting down..."; kill $FRONTEND_PID $BACKEND_PID $MODEL_PID 2>/dev/null; exit 0' INT TERM

# Wait for all background processes
wait
