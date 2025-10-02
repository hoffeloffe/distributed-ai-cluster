#!/bin/bash
# Raspberry Pi Cluster Setup Script
# This script configures a Raspberry Pi for the distributed AI cluster

set -e

echo "🚀 Setting up Raspberry Pi for Distributed AI Cluster..."

# Configuration variables
CLUSTER_CONFIG_DIR="/home/pi/distributed-ai-cluster/config"
MODEL_DIR="/home/pi/distributed-ai-cluster/models"
LOG_DIR="/home/pi/distributed-ai-cluster/logs"
SERVICE_USER="pi"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*"
    exit 1
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    error "Please run this script as the pi user, not root"
fi

# Update system packages
log "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
log "Installing Python dependencies..."
sudo apt install -y python3-pip python3-venv python3-dev

# Install AI/ML libraries
log "Installing AI/ML libraries..."
pip3 install tensorflow==2.13.0
pip3 install numpy pandas matplotlib pillow
pip3 install asyncio-mqtt paho-mqtt
pip3 install aiohttp aiofiles
pip3 install psutil GPUtil

# Install system utilities
log "Installing system utilities..."
sudo apt install -y htop iotop nethogs
sudo apt install -y git curl wget
sudo apt install -y stress-ng # For load testing

# Create project directories
log "Creating project directories..."
mkdir -p "$CLUSTER_CONFIG_DIR"
mkdir -p "$MODEL_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "/home/pi/distributed-ai-cluster/src"

# Set proper permissions
sudo chown -R pi:pi "/home/pi/distributed-ai-cluster"
chmod +x "/home/pi/distributed-ai-cluster"

# Configure network optimizations
log "Configuring network optimizations..."
sudo tee /etc/sysctl.d/99-network-optimization.conf > /dev/null <<EOF
# Network optimization for distributed AI cluster
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.core.netdev_max_backlog = 5000
net.unix.max_dgram_qlen = 1000
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.ipv4.tcp_congestion_control = bbr
EOF

sudo sysctl -p /etc/sysctl.d/99-network-optimization.conf

# Install and configure Coral TPU support (optional)
read -p "Do you want to install Coral TPU support? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Installing Coral TPU support..."
    sudo apt install -y libedgetpu1-std
    sudo apt install -y python3-tflite-runtime
    pip3 install tflite-runtime
    log "Coral TPU support installed"
fi

# Overclock CPU for better performance (optional)
read -p "Do you want to overclock the CPU? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    warn "Overclocking may reduce hardware lifespan and increase power consumption"
    sudo tee /boot/config.txt > /dev/null <<EOF
# Overclock settings for better AI performance
arm_freq=2000
gpu_freq=600
over_voltage=6
EOF
    log "CPU overclocked to 2.0GHz"
fi

# Create startup service
log "Creating systemd service..."
sudo tee /etc/systemd/system/distributed-ai-cluster.service > /dev/null <<EOF
[Unit]
Description=Distributed AI Raspberry Pi Cluster
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/distributed-ai-cluster
ExecStart=/home/pi/distributed-ai-cluster/venv/bin/python3 src/cluster_node.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable distributed-ai-cluster.service

# Create virtual environment for Python dependencies
log "Creating Python virtual environment..."
python3 -m venv /home/pi/distributed-ai-cluster/venv
source /home/pi/distributed-ai-cluster/venv/bin/activate
pip install --upgrade pip

# Install project dependencies in virtual environment
log "Installing project dependencies in virtual environment..."
/home/pi/distributed-ai-cluster/venv/bin/pip install tensorflow numpy asyncio-mqtt aiohttp psutil

log "Setup complete! 🎉"
echo ""
echo "Next steps:"
echo "1. Copy your project files to /home/pi/distributed-ai-cluster/"
echo "2. Configure cluster settings in config/cluster_config.json"
echo "3. Download your AI model to models/ directory"
echo "4. Run: sudo systemctl start distributed-ai-cluster"
echo "5. Check status: sudo systemctl status distributed-ai-cluster"
echo ""
echo "For troubleshooting, check logs with:"
echo "sudo journalctl -u distributed-ai-cluster -f"
