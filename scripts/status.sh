#!/usr/bin/env bash
# 训练状态速查
#
# 用法（在 diffusion_policy 目录下）：
#   bash logs/status.sh
#
# 不需要记住任何 PID，直接告诉你现在什么情况。

cd "$(dirname "$0")/.." || exit 1

PID=$(cat logs/train.pid 2>/dev/null)
TOTAL=800
LOG=logs/train_pusht_seed42.log

echo "════════════════════════════════════════════════════"
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
  echo "  状态       ✅ 训练进行中"
else
  echo "  状态       ❌ 训练已停止（看下面的「最近状态文件」）"
fi
echo "════════════════════════════════════════════════════"

# 进度
if [ -f "$LOG" ]; then
  ep=$(tr '\r' '\n' < "$LOG" | grep -oE "Training epoch [0-9]+" | tail -1 | grep -oE "[0-9]+")
  ep=${ep:-0}
  phase=$(tr '\r' '\n' < "$LOG" | grep -vE "^\s*$" | tail -1 | cut -c1-70)
  pct=$(( ep * 100 / TOTAL ))
  echo "  进度       epoch ${ep} / ${TOTAL}  (${pct}%)"
  echo "  当前阶段   ${phase}"
  bars=$(( pct / 4 ))
  printf "  进度条     ["
  printf '█%.0s' $(seq 1 $bars 2>/dev/null) 2>/dev/null
  printf '░%.0s' $(seq 1 $((25 - bars)) 2>/dev/null) 2>/dev/null
  echo "]"
fi

# 已完成几次评估：从结构化指标文件里数「非 null 的 test/mean_score」
METRICS=data/outputs/pusht_seed42/logs.json.txt
if [ -f "$METRICS" ]; then
  n_eval=$(grep -o '"test/mean_score": [0-9.]*' "$METRICS" | grep -vc null)
  echo "  评估记录   ${n_eval} 次已完成（每 50 轮一次，官方 n_test=50）"
  if [ "$n_eval" -gt 0 ]; then
    best=$(grep -o '"test/mean_score": [0-9.]*' "$METRICS" | grep -v null \
           | awk '{print $2}' | sort -rn | head -1)
    echo "  当前最佳   test/mean_score = ${best}"
  fi
else
  echo "  评估记录   尚未写入"
fi

echo
echo "── 资源 ────────────────────────────────────────────"
free -h | awk 'NR==2{printf "  内存       %s 可用 / %s 总量\n", $7, $2}'
free -h | awk 'NR==3{printf "  swap       %s 可用 / %s 总量\n", $4, $2}'
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader 2>/dev/null | awk -F, '{printf "  GPU       %s显存 | %s利用率 | %s°C | %s\n", $1, $2, $3, $4}'
df -h / | awk 'NR==2{printf "  磁盘       %s 可用 / %s 总量 (%s)\n", $4, $2, $5}'
[ -n "$PID" ] && kill -0 "$PID" 2>/dev/null && \
  ps -o rss --no-headers -p "$PID" | awk '{printf "  训练进程   %.2f GB RSS\n", $1/1048576}'

echo
echo "── 守护进程 ────────────────────────────────────────"
for p in "resource_guard.sh" "train_watchdog.sh"; do
  pgrep -f "$p" >/dev/null && printf "  ✅ %-22s 运行中\n" "${p%.sh}" || printf "  ❌ %-22s 未运行\n" "${p%.sh}"
done
pgrep -f "systemd-inhibit" >/dev/null && echo "  ✅ 防挂起锁              生效中" || echo "  ❌ 防挂起锁              未运行（合盖/空闲会冻结训练）"

echo
echo "── 最近状态 ────────────────────────────────────────"
tail -2 logs/training_status.txt 2>/dev/null || echo "  （训练尚未结束，无状态文件）"
tail -1 logs/resource_guard.log 2>/dev/null | sed 's/^/  守卫: /'
tail -1 logs/watchdog.log 2>/dev/null | sed 's/^/  看门狗: /'
echo
echo "  详细日志   logs/train_pusht_seed42.log"
echo "  指标文件   data/outputs/pusht_seed42/logs.json.txt"
echo "════════════════════════════════════════════════════"
