#!/usr/bin/env bash
# 资源守卫 (resource guard)
#
# 作用：不改变训练本身，只是在一旁盯着内存和 GPU 温度。
#       在系统可用内存低于阈值时，**先于内核 OOM Killer 主动停止训练**。
#
# 为什么需要它：
#   内核 OOM Killer 的选择是不确定的。本机曾发生过训练进程和 VS Code
#   被同一次 OOM 一起杀掉的情况（见 journalctl -k 的 oom-kill 记录）。
#   主动停止训练可以把损失限制在「丢失上次 checkpoint 之后的进度」，
#   而不是「VS Code 崩溃、工作环境被打断」。
#
# 用法：
#   bash logs/resource_guard.sh <训练进程PID>
#
# 阈值可用环境变量覆盖：
#   MIN_AVAIL_MB  最低可用内存（默认 1500 MB）
#   MAX_GPU_TEMP  最高 GPU 温度（默认 85 °C）

set -u

TARGET_PID="${1:-}"
if [ -z "$TARGET_PID" ]; then
  echo "用法: bash $0 <训练进程PID>" >&2
  exit 1
fi

MIN_AVAIL_MB="${MIN_AVAIL_MB:-300}"          # MemAvailable 硬下限（逼近内核 OOM）
MIN_HEADROOM_MB="${MIN_HEADROOM_MB:-1200}"   # MemAvailable + SwapFree 总余量下限
MAX_GPU_TEMP="${MAX_GPU_TEMP:-85}"
INTERVAL="${INTERVAL:-30}"
LOG="logs/resource_guard.log"

mkdir -p logs

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

notify() {
  log "NOTIFY: $1 | $2"
  command -v notify-send >/dev/null 2>&1 && \
    notify-send -u "${3:-critical}" -a "DP 训练" "$1" "$2" 2>/dev/null || true
}

log "守卫启动 | 监控 PID=$TARGET_PID | 总余量阈值=${MIN_HEADROOM_MB}MB | 内存硬下限=${MIN_AVAIL_MB}MB | 温度阈值=${MAX_GPU_TEMP}C"

while kill -0 "$TARGET_PID" 2>/dev/null; do
  # MemAvailable 才是内核认为「能拿来用」的量（free 的 free 列会低估）
  avail=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
  # 空闲 swap：保存 checkpoint 时的瞬时峰值靠它吸收，必须计入余量
  swapfree=$(awk '/^SwapFree:/{print int($2/1024)}' /proc/meminfo)
  headroom=$(( avail + swapfree ))
  # 训练进程自身 RSS（MB），取自 /proc/PID/status（不含共享页，比 ps 的 RSS 准）
  rss=$(awk '/^VmRSS:/{print int($2/1024)}' "/proc/$TARGET_PID/status" 2>/dev/null || echo 0)
  # GPU 温度
  temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  temp=${temp:-0}
  # 训练到第几轮了（便于评估中断时的损失）
  ep=$(tr '\r' '\n' < logs/train_pusht_seed42.log 2>/dev/null \
       | grep -oE "Training epoch [0-9]+" | tail -1 | grep -oE "[0-9]+")
  ep=${ep:-0}

  log "avail=${avail}MB swap_free=${swapfree}MB headroom=${headroom}MB rss=${rss}MB gpu=${temp}C epoch=${ep}"

  # 判据 1：总余量（内存+swap）耗尽 —— 再涨就要 OOM
  if [ "$headroom" -lt "$MIN_HEADROOM_MB" ]; then
    log "⚠️  总余量 ${headroom}MB < ${MIN_HEADROOM_MB}MB —— 主动停止训练，保护系统（epoch=${ep}）"
    notify "训练被守卫停止（内存不足）" "内存+swap 总余量仅 ${headroom}MB，低于阈值 ${MIN_HEADROOM_MB}MB。已停在 epoch ${ep}，你的 VS Code 未受影响。" critical
    kill -TERM "$TARGET_PID" 2>/dev/null
    sleep 5
    kill -KILL "$TARGET_PID" 2>/dev/null
    log "训练已停止"
    exit 0
  fi

  # 判据 2：物理内存本身逼近枯竭（保护前台程序如 VS Code 不被内核 OOM 选中）
  if [ "$avail" -lt "$MIN_AVAIL_MB" ]; then
    log "⚠️  可用内存 ${avail}MB < ${MIN_AVAIL_MB}MB —— 主动停止训练（epoch=${ep}）"
    notify "训练被守卫停止（物理内存不足）" "可用内存仅 ${avail}MB。已停在 epoch ${ep}，你的 VS Code 未受影响。" critical
    kill -TERM "$TARGET_PID" 2>/dev/null
    exit 0
  fi

  if [ "$temp" -gt "$MAX_GPU_TEMP" ] 2>/dev/null; then
    log "⚠️  GPU 温度 ${temp}C > ${MAX_GPU_TEMP}C —— 主动停止训练（当前 epoch=${ep}）"
    notify "训练被守卫停止（GPU 过热）" "GPU 温度 ${temp}C 超过阈值 ${MAX_GPU_TEMP}C。已停在 epoch ${ep}。" critical
    kill -TERM "$TARGET_PID" 2>/dev/null
    exit 0
  fi

  sleep "$INTERVAL"
done

log "目标进程 $TARGET_PID 已退出，守卫结束"
