#!/usr/bin/env bash
# 训练期间的健康监控：每 60 秒记录一次硬件状态到 CSV。
#
# 用法（后台运行）：
#   nohup bash logs/monitor_health.sh > /dev/null 2>&1 &
#
# 目的：
#   1. 留下温度/功耗/内存/磁盘的时间序列，便于事后判断是否有过热或资源耗尽
#   2. 可用内存低于阈值时在日志里告警，便于及时介入
#
# 开销：每 60 秒执行一次 nvidia-smi + free + df，可忽略不计。

set -uo pipefail

OUT="${1:-logs/health.csv}"
INTERVAL="${2:-60}"
# 可用内存低于此值（MB）时告警
RAM_WARN_MB=800

if [[ ! -f "$OUT" ]]; then
  echo "timestamp,gpu_temp_c,gpu_power_w,gpu_mem_used_mib,gpu_util_pct,ram_avail_mb,swap_used_mb,disk_avail_gb" > "$OUT"
fi

while true; do
  ts=$(date '+%Y-%m-%d %H:%M:%S')

  gpu=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  gpu_temp=$(cut -d, -f1 <<<"$gpu")
  gpu_power=$(cut -d, -f2 <<<"$gpu")
  gpu_mem=$(cut -d, -f3 <<<"$gpu")
  gpu_util=$(cut -d, -f4 <<<"$gpu")

  ram_avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  # 注意：不要用 `free | awk '/Swap/'`——中文 locale 下输出是「交换：」，匹配不到。
  # 直接读 /proc/meminfo 更稳。
  swap_used=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print int((t-f)/1024)}' /proc/meminfo)
  disk_avail=$(df -BG / | awk 'NR==2{gsub("G","",$4); print $4}')

  echo "$ts,$gpu_temp,$gpu_power,$gpu_mem,$gpu_util,$ram_avail,$swap_used,$disk_avail" >> "$OUT"

  if (( ram_avail < RAM_WARN_MB )); then
    echo "[$ts] ⚠️  可用内存仅 ${ram_avail} MB（阈值 ${RAM_WARN_MB} MB），有 OOM 风险" >> "${OUT%.csv}.alerts.log"
  fi
  if (( ${gpu_temp:-0} > 85 )); then
    echo "[$ts] ⚠️  GPU 温度 ${gpu_temp}°C 偏高（降频阈值约 87-90°C）" >> "${OUT%.csv}.alerts.log"
  fi
  if (( disk_avail < 5 )); then
    echo "[$ts] ⚠️  磁盘仅剩 ${disk_avail} GB" >> "${OUT%.csv}.alerts.log"
  fi

  sleep "$INTERVAL"
done
