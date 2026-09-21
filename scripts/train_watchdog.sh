#!/usr/bin/env bash
# 训练看门狗 (watchdog)
#
# 目的：让你不必手动反复检查训练状态。出问题时它会主动弹桌面通知。
#
# 监控三类异常：
#   1. 训练进程异常退出（崩溃 / 被 OOM 杀掉 / 被守卫停止）
#   2. 训练卡死（日志超过 STALL_MIN 分钟没有更新）
#   3. GPU 温度过高
# 另外在训练正常完成时也会通知。
#
# 用法：
#   bash logs/train_watchdog.sh <训练PID> <期望总轮数>
#
# 环境变量：
#   STALL_MIN      日志静止多久算卡死（默认 12 分钟）
#   CHECK_INTERVAL 检查间隔秒数（默认 60）
#   TEMP_LIMIT     温度告警阈值（默认 80 °C）

set -u

TARGET_PID="${1:-}"
TOTAL_EPOCHS="${2:-0}"
STALL_MIN="${STALL_MIN:-12}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
TEMP_LIMIT="${TEMP_LIMIT:-80}"

LOG="logs/train_pusht_seed42.log"
STATUS="logs/training_status.txt"
WLOG="logs/watchdog.log"

[ -n "$TARGET_PID" ] || { echo "用法: bash $0 <训练PID> <总轮数>" >&2; exit 1; }

mkdir -p logs

log() { echo "$(date '+%F %T') $*" >> "$WLOG"; }

notify() {
  # $1=标题 $2=正文 $3=urgency(low/normal/critical)
  local title="$1" body="$2" urgency="${3:-normal}"
  log "NOTIFY [$urgency] $title | $body"
  if command -v notify-send >/dev/null 2>&1; then
    notify-send -u "$urgency" -a "DP 训练" "$title" "$body" 2>/dev/null || true
  fi
}

current_epoch() {
  tr '\r' '\n' < "$LOG" 2>/dev/null \
    | grep -oE "Training epoch [0-9]+" | tail -1 | grep -oE "[0-9]+" || echo 0
}

gpu_temp() {
  nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || echo 0
}

log "看门狗启动 | PID=$TARGET_PID | 目标轮数=$TOTAL_EPOCHS | 卡死阈值=${STALL_MIN}min"
notify "训练已启动" "PID $TARGET_PID，目标 $TOTAL_EPOCHS 轮。出问题会自动通知你。" "low"

last_change=$(stat -c %Y "$LOG" 2>/dev/null || date +%s)
temp_warned=0
mem_warned=0

while kill -0 "$TARGET_PID" 2>/dev/null; do
  sleep "$CHECK_INTERVAL"

  ep=$(current_epoch)
  now=$(date +%s)

  # ── 日志活性检测（防卡死）──────────────────────────────
  cur_change=$(stat -c %Y "$LOG" 2>/dev/null || echo "$now")
  if [ "$cur_change" != "$last_change" ]; then
    last_change=$cur_change
  else
    idle_min=$(( (now - last_change) / 60 ))
    if [ "$idle_min" -ge "$STALL_MIN" ]; then
      notify "⚠️ 训练可能卡死" "日志已 ${idle_min} 分钟没有更新（停在 epoch ${ep}）。建议检查：logs/train_pusht_seed42.log" "critical"
      last_change=$now   # 避免重复轰炸
    fi
  fi

  # ── 温度告警（不停止训练，只提醒）──────────────────────
  t=$(gpu_temp)
  if [ "$t" -gt "$TEMP_LIMIT" ] 2>/dev/null; then
    if [ "$temp_warned" -eq 0 ]; then
      notify "⚠️ GPU 温度偏高" "当前 ${t}°C（阈值 ${TEMP_LIMIT}°C），epoch ${ep}。守卫会在 85°C 时停训练。" "normal"
      temp_warned=1
    fi
  else
    temp_warned=0
  fi

  # ── 内存告警（不停止训练，只提醒）──────────────────────
  # 判据用「内存+swap 总余量」：只看物理内存会在 checkpoint 保存时误报。
  m_avail=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
  m_swap=$(awk '/^SwapFree:/{print int($2/1024)}' /proc/meminfo)
  m_head=$(( m_avail + m_swap ))
  if [ "$m_head" -lt "$((${MEM_WARN_MB:-4000}))" ]; then
    if [ "$mem_warned" -eq 0 ]; then
      notify "⚠️ 内存余量偏低" "内存+swap 总余量 ${m_head}MB（可用内存 ${m_avail}MB）。建议关掉 Firefox 等占用较大的程序。守卫会在 1200MB 时停训练。" "normal"
      mem_warned=1
    fi
  else
    mem_warned=0
  fi
done

# ── 训练已退出：判断是正常完成还是异常 ──────────────────────
ep=$(current_epoch)
if [ "$TOTAL_EPOCHS" -gt 0 ] && [ "$ep" -ge "$((TOTAL_EPOCHS - 1))" ]; then
  notify "✅ 训练完成" "已完成 ${ep}/${TOTAL_EPOCHS} 轮。可以提取评估指标并更新仓库了。" "normal"
  result="COMPLETED"
else
  notify "❌ 训练意外停止" "停在 epoch ${ep}/${TOTAL_EPOCHS}。请查看 logs/watchdog.log 与 logs/resource_guard.log 判断原因。" "critical"
  result="CRASHED"
fi

{
  echo "状态:      $result"
  echo "结束时间:  $(date '+%F %T')"
  echo "到达轮数:  ${ep} / ${TOTAL_EPOCHS}"
  echo "训练 PID:  $TARGET_PID"
  echo "--- 最近 5 条守卫记录 ---"
  tail -5 logs/resource_guard.log 2>/dev/null
} > "$STATUS"

log "看门狗结束 | $result | epoch=${ep}/${TOTAL_EPOCHS}"
