#!/usr/bin/env bash
# setup-pi.sh — Raspberry Pi 一鍵部署花城 Discord Bot
# 用法: bash setup-pi.sh
set -euo pipefail

# ── 顏色 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 系統檢查 ──
info "檢查系統環境..."

ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" ]]; then
    error "此腳本僅支援 aarch64 (Raspberry Pi OS 64-bit)，偵測到: $ARCH"
fi

RAM_MB=$(free -m | awk '/^Mem:/{print $2}')
if (( RAM_MB < 3800 )); then
    warn "偵測到 ${RAM_MB}MB RAM — PyTorch 可能需要 4GB+，建議使用 4GB/8GB 型號"
fi

info "系統: $(uname -s) $ARCH, RAM: ${RAM_MB}MB"

# ── 專案目錄 ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
info "工作目錄: $SCRIPT_DIR"

# ── 系統依賴 ──
info "安裝系統依賴..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-venv python3-dev \
    libopenblas-dev libffi-dev \
    git curl

# ── Python venv ──
if [[ ! -d "venv" ]]; then
    info "建立 Python 虛擬環境..."
    python3 -m venv venv
else
    info "虛擬環境已存在，跳過建立"
fi

source venv/bin/activate
pip install --upgrade pip setuptools wheel -q

# ── PyTorch (CPU for aarch64) ──
# Pi 4 Cortex-A72 不支援新版 PyTorch 的 SIMD 指令，鎖定 2.6.0
info "安裝 PyTorch 2.6.0 (CPU)..."
pip install 'torch==2.6.0+cpu' 'torchvision==0.21.0' --index-url https://download.pytorch.org/whl/cpu -q

# ── Python 依賴 ──
info "安裝 Python 依賴..."
pip install -r requirements.txt -q

# ── 驗證安裝 ──
info "驗證 PyTorch..."
python3 -c "import torch; print(f'  PyTorch {torch.__version__} on {torch.device(\"cpu\")}')"
info "驗證 discord.py..."
python3 -c "import discord; print(f'  discord.py {discord.__version__}')"

# ── .env ──
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    warn ".env 已從 .env.example 建立，請編輯填入 token:"
    warn "  nano $SCRIPT_DIR/.env"
    warn "  (至少需要 DISCORD_TOKEN 和 GEMINI_API_KEY)"
else
    info ".env 已存在，跳過"
fi

# ── systemd service ──
info "設定 systemd service..."
SERVICE_FILE="$SCRIPT_DIR/openclaw.service"
sed -e "s|__USER__|$USER|g" \
    -e "s|__DIR__|$SCRIPT_DIR|g" \
    "$SCRIPT_DIR/openclaw.service.template" > "$SERVICE_FILE"

sudo cp "$SERVICE_FILE" /etc/systemd/system/openclaw.service
sudo systemctl daemon-reload
sudo systemctl enable openclaw
info "openclaw.service 已安裝並啟用 (開機自動啟動)"

# ── Ollama (選裝) ──
echo ""
read -rp "$(echo -e "${YELLOW}是否安裝 Ollama？(未來可跑本地 LLM) [y/N]: ${NC}")" install_ollama
if [[ "${install_ollama,,}" == "y" ]]; then
    info "安裝 Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    info "Ollama 安裝完成！可用 'ollama run llama3.2:1b' 測試"
else
    info "跳過 Ollama 安裝"
fi

# ── 完成 ──
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  花城 Bot 部署完成！${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  下一步:"
echo "  1. 編輯 .env 填入 token"
echo "     nano $SCRIPT_DIR/.env"
echo ""
echo "  2. 啟動服務"
echo "     sudo systemctl start openclaw"
echo ""
echo "  3. 查看 log"
echo "     journalctl -u openclaw -f"
echo ""
echo "  手動測試:"
echo "     source venv/bin/activate"
echo "     python3 bot.py"
echo ""
