#!/usr/bin/env bash
# ============================================================================
# One-Command AWS EC2 (Ubuntu 22.04 / 24.04) Deployment Script
# Usage on EC2: bash deploy/aws_ec2_setup.sh
# ============================================================================
set -euo pipefail

echo "=== 1. Configuring 4GB Swapfile (prevents OOM during initial ONNX quantization) ==="
if [ ! -f /swapfile ]; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "=== 2. Installing Docker & Docker Compose ==="
if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER" || true
fi

echo "=== 3. Detecting AWS EC2 Public IP for Automatic HTTPS (nip.io) ==="
EC2_IP=$(curl -s http://checkip.amazonaws.com || echo "127.0.0.1")
DASHED_IP=$(echo "$EC2_IP" | tr '.' '-')
export PUBLIC_DOMAIN="${PUBLIC_DOMAIN:-${DASHED_IP}.nip.io}"

echo "=== 4. Building & Starting Voice & Pronunciation Engine ==="
sudo PUBLIC_DOMAIN="$PUBLIC_DOMAIN" docker compose -f docker-compose.aws.yml up -d --build

echo ""
echo "============================================================================"
echo "✅ Deployment Complete!"
echo "🔗 HTTPS URL (Microphone Enabled): https://${PUBLIC_DOMAIN}"
echo "🔗 Direct HTTP Port:               http://${EC2_IP}:8000"
echo "📊 View Logs:                      sudo docker compose -f docker-compose.aws.yml logs -f"
echo "============================================================================"
