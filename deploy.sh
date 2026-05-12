#!/usr/bin/env bash
# OpenClaw 自動部署腳本
# 用法: bash deploy.sh
# GitHub webhook 或 GitHub Actions 觸發時自動執行

set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_NAME="openclaw"
LOG_FILE="$APP_DIR/deploy.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

cd "$APP_DIR"

log "=== 開始部署 ==="

# 1. 拉取最新程式碼
log "git pull..."
git fetch origin
git reset --hard origin/main
log "git pull 完成: $(git log --oneline -1)"

# 2. 安裝/更新依賴（若 requirements.txt 有變動）
if git diff HEAD~1 --name-only | grep -q "requirements.txt"; then
    log "requirements.txt 有變動，更新依賴..."
    pip install -r requirements.txt --quiet
    log "依賴更新完成"
else
    log "requirements.txt 無變動，跳過依賴安裝"
fi

# 3. 重啟服務
if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    log "重啟 systemd 服務 $SERVICE_NAME..."
    sudo systemctl restart "$SERVICE_NAME"
    sleep 2
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log "服務已成功重啟"
    else
        log "ERROR: 服務重啟失敗！"
        sudo systemctl status "$SERVICE_NAME" --no-pager | tee -a "$LOG_FILE"
        exit 1
    fi
else
    log "systemd 服務不存在或未啟用，嘗試直接重啟..."
    # 嘗試用 PID 檔案或 pkill 方式重啟
    if pgrep -f "python.*bot.py" > /dev/null; then
        log "停止現有 bot 程序..."
        pkill -f "python.*bot.py" || true
        sleep 2
    fi
    log "啟動 bot..."
    nohup python3 bot.py >> "$APP_DIR/bot.log" 2>&1 &
    log "bot 已在背景啟動 (PID: $!)"
fi

log "=== 部署完成 ==="
