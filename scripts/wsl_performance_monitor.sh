#!/bin/bash
# WSL2 性能监控脚本 - 用于大模型训练和推理
# 实时监控CPU、内存、GPU、磁盘使用情况

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# 清屏函数
clear_screen() {
    clear
    echo -e "${BOLD}${CYAN}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║          WSL2 性能监控面板 - AI/ML 训练专用                   ║"
    echo "║          按 Ctrl+C 退出监控                                    ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 获取内存使用情况
get_memory_info() {
    local total=$(free -g | awk '/^Mem:/{print $2}')
    local used=$(free -g | awk '/^Mem:/{print $3}')
    local free=$(free -g | awk '/^Mem:/{print $4}')
    local available=$(free -g | awk '/^Mem:/{print $7}')
    local cached=$(free -g | awk '/^Mem:/{print $6}')
    local swap_total=$(free -g | awk '/^Swap:/{print $2}')
    local swap_used=$(free -g | awk '/^Swap:/{print $3}')

    local mem_percent=$((used * 100 / total))
    local swap_percent=0
    if [ $swap_total -gt 0 ]; then
        swap_percent=$((swap_used * 100 / swap_total))
    fi

    # 内存颜色标识
    local mem_color=$GREEN
    if [ $mem_percent -gt 80 ]; then
        mem_color=$RED
    elif [ $mem_percent -gt 60 ]; then
        mem_color=$YELLOW
    fi

    echo -e "${BOLD}${BLUE}━━━ 内存使用情况 ━━━${NC}"
    echo -e "  总内存:   ${BOLD}${total}GB${NC}"
    echo -e "  已使用:   ${mem_color}${used}GB (${mem_percent}%)${NC}"
    echo -e "  可用:     ${GREEN}${available}GB${NC}"
    echo -e "  缓存:     ${cached}GB"
    echo -e "  Swap:     ${swap_used}GB / ${swap_total}GB ${YELLOW}(${swap_percent}%)${NC}"

    if [ $swap_percent -gt 30 ]; then
        echo -e "  ${RED}⚠ 警告: Swap使用率高，建议增加物理内存${NC}"
    fi
}

# 获取CPU使用情况
get_cpu_info() {
    local cpu_count=$(nproc)
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | xargs)
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')

    local cpu_color=$GREEN
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        cpu_color=$RED
    elif (( $(echo "$cpu_usage > 60" | bc -l) )); then
        cpu_color=$YELLOW
    fi

    echo -e "\n${BOLD}${BLUE}━━━ CPU 使用情况 ━━━${NC}"
    echo -e "  CPU核心:  ${BOLD}${cpu_count}${NC}"
    echo -e "  使用率:   ${cpu_color}${cpu_usage}%${NC}"
    echo -e "  负载:     ${load_avg}"
}

# 获取GPU使用情况
get_gpu_info() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "\n${BOLD}${BLUE}━━━ GPU 使用情况 (NVIDIA) ━━━${NC}"

        # 获取GPU信息
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        local gpu_mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        local gpu_mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        local gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        local gpu_power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null | head -1)

        # 计算显存使用百分比
        local gpu_mem_percent=0
        if [ -n "$gpu_mem_total" ] && [ $gpu_mem_total -gt 0 ]; then
            gpu_mem_percent=$((gpu_mem_used * 100 / gpu_mem_total))
        fi

        # GPU利用率颜色
        local gpu_util_color=$GREEN
        if [ $gpu_util -gt 90 ]; then
            gpu_util_color=$GREEN  # 高利用率是好事
        elif [ $gpu_util -lt 20 ]; then
            gpu_util_color=$YELLOW  # 低利用率可能有问题
        fi

        # 显存颜色
        local gpu_mem_color=$GREEN
        if [ $gpu_mem_percent -gt 90 ]; then
            gpu_mem_color=$RED
        elif [ $gpu_mem_percent -gt 70 ]; then
            gpu_mem_color=$YELLOW
        fi

        # 温度颜色
        local temp_color=$GREEN
        if [ $gpu_temp -gt 80 ]; then
            temp_color=$RED
        elif [ $gpu_temp -gt 70 ]; then
            temp_color=$YELLOW
        fi

        echo -e "  GPU型号:  ${BOLD}${gpu_name}${NC}"
        echo -e "  利用率:   ${gpu_util_color}${gpu_util}%${NC}"
        echo -e "  显存:     ${gpu_mem_color}${gpu_mem_used}MB / ${gpu_mem_total}MB (${gpu_mem_percent}%)${NC}"
        echo -e "  温度:     ${temp_color}${gpu_temp}°C${NC}"
        echo -e "  功耗:     ${gpu_power}"

        if [ $gpu_mem_percent -gt 95 ]; then
            echo -e "  ${RED}⚠ 警告: 显存即将耗尽，可能导致OOM${NC}"
        fi

        if [ $gpu_temp -gt 85 ]; then
            echo -e "  ${RED}⚠ 警告: GPU温度过高，注意散热${NC}"
        fi
    else
        echo -e "\n${BOLD}${BLUE}━━━ GPU 使用情况 ━━━${NC}"
        echo -e "  ${YELLOW}nvidia-smi 未找到或GPU不可用${NC}"
    fi
}

# 获取磁盘使用情况
get_disk_info() {
    echo -e "\n${BOLD}${BLUE}━━━ 磁盘使用情况 ━━━${NC}"

    # 主分区
    local disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    local disk_used=$(df -h / | awk 'NR==2 {print $3}')
    local disk_total=$(df -h / | awk 'NR==2 {print $2}')

    local disk_color=$GREEN
    if [ $disk_usage -gt 90 ]; then
        disk_color=$RED
    elif [ $disk_usage -gt 80 ]; then
        disk_color=$YELLOW
    fi

    echo -e "  根分区:   ${disk_color}${disk_used} / ${disk_total} (${disk_usage}%)${NC}"

    # 用户目录
    if [ -d "/home/$USER" ]; then
        local home_usage=$(df -h /home/$USER | awk 'NR==2 {print $5}' | sed 's/%//')
        local home_used=$(df -h /home/$USER | awk 'NR==2 {print $3}')
        local home_total=$(df -h /home/$USER | awk 'NR==2 {print $2}')
        echo -e "  用户目录: ${home_used} / ${home_total} (${home_usage}%)"
    fi

    # Windows分区
    if [ -d "/mnt/c" ]; then
        local c_usage=$(df -h /mnt/c 2>/dev/null | awk 'NR==2 {print $5}' | sed 's/%//')
        local c_used=$(df -h /mnt/c 2>/dev/null | awk 'NR==2 {print $3}')
        local c_total=$(df -h /mnt/c 2>/dev/null | awk 'NR==2 {print $2}')
        if [ -n "$c_usage" ]; then
            echo -e "  C盘:      ${c_used} / ${c_total} (${c_usage}%)"
        fi
    fi

    if [ $disk_usage -gt 95 ]; then
        echo -e "  ${RED}⚠ 警告: 磁盘空间不足，建议清理${NC}"
    fi
}

# 获取网络使用情况
get_network_info() {
    echo -e "\n${BOLD}${BLUE}━━━ 网络统计 ━━━${NC}"

    # 获取主网络接口
    local interface=$(ip route | grep default | awk '{print $5}' | head -1)

    if [ -n "$interface" ]; then
        local rx_bytes=$(cat /sys/class/net/$interface/statistics/rx_bytes 2>/dev/null || echo 0)
        local tx_bytes=$(cat /sys/class/net/$interface/statistics/tx_bytes 2>/dev/null || echo 0)

        # 转换为人类可读格式
        local rx_mb=$((rx_bytes / 1024 / 1024))
        local tx_mb=$((tx_bytes / 1024 / 1024))

        echo -e "  接口:     ${interface}"
        echo -e "  接收:     ${GREEN}${rx_mb} MB${NC}"
        echo -e "  发送:     ${MAGENTA}${tx_mb} MB${NC}"
    fi
}

# 获取训练进程信息
get_training_processes() {
    echo -e "\n${BOLD}${BLUE}━━━ 训练相关进程 ━━━${NC}"

    # 查找Python训练进程
    local python_procs=$(ps aux | grep -E "(python|python3)" | grep -v grep | grep -v "wsl_performance_monitor" | head -5)

    if [ -n "$python_procs" ]; then
        echo "$python_procs" | while IFS= read -r line; do
            local pid=$(echo "$line" | awk '{print $2}')
            local cpu=$(echo "$line" | awk '{print $3}')
            local mem=$(echo "$line" | awk '{print $4}')
            local cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf $i" "; print ""}' | cut -c 1-50)

            echo -e "  PID ${BOLD}${pid}${NC}: CPU ${cpu}%, MEM ${mem}% - ${cmd}"
        done
    else
        echo -e "  ${YELLOW}未检测到活跃的训练进程${NC}"
    fi
}

# 性能建议
get_performance_tips() {
    echo -e "\n${BOLD}${BLUE}━━━ 性能提示 ━━━${NC}"

    local tips=()

    # 检查内存
    local mem_percent=$(free | awk '/^Mem:/{printf "%.0f", $3/$2*100}')
    if [ $mem_percent -gt 85 ]; then
        tips+=("${YELLOW}⚠${NC} 内存使用率高 (${mem_percent}%)，考虑减小batch size")
    fi

    # 检查Swap
    local swap_used=$(free -m | awk '/^Swap:/{print $3}')
    if [ $swap_used -gt 1000 ]; then
        tips+=("${RED}⚠${NC} Swap使用量大 (${swap_used}MB)，性能会受影响")
    fi

    # 检查GPU
    if command -v nvidia-smi &> /dev/null; then
        local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$gpu_util" ] && [ $gpu_util -lt 50 ]; then
            tips+=("${YELLOW}⚠${NC} GPU利用率低 (${gpu_util}%)，检查数据加载是否为瓶颈")
        fi
    fi

    # 检查磁盘
    local disk_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ $disk_usage -gt 90 ]; then
        tips+=("${RED}⚠${NC} 磁盘空间不足 (${disk_usage}%)，建议清理")
    fi

    # 显示提示
    if [ ${#tips[@]} -eq 0 ]; then
        echo -e "  ${GREEN}✓ 系统运行良好，无需调整${NC}"
    else
        for tip in "${tips[@]}"; do
            echo -e "  $tip"
        done
    fi
}

# 主监控循环
main_loop() {
    local refresh_interval=2  # 刷新间隔（秒）

    while true; do
        clear_screen

        echo -e "${CYAN}$(date '+%Y-%m-%d %H:%M:%S')${NC}"
        echo ""

        get_memory_info
        get_cpu_info
        get_gpu_info
        get_disk_info
        get_network_info
        get_training_processes
        get_performance_tips

        echo -e "\n${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${CYAN}刷新间隔: ${refresh_interval}秒 | 按 Ctrl+C 退出${NC}"

        sleep $refresh_interval
    done
}

# 单次检查模式
single_check() {
    clear_screen
    echo -e "${CYAN}$(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo ""

    get_memory_info
    get_cpu_info
    get_gpu_info
    get_disk_info
    get_network_info
    get_training_processes
    get_performance_tips
}

# 帮助信息
show_help() {
    echo "WSL2 性能监控脚本 - 大模型训练专用"
    echo ""
    echo "用法:"
    echo "  $0          # 持续监控模式（默认）"
    echo "  $0 -o       # 单次检查模式"
    echo "  $0 -h       # 显示帮助"
    echo ""
    echo "监控内容:"
    echo "  - 内存和Swap使用情况"
    echo "  - CPU使用率和负载"
    echo "  - GPU利用率和显存"
    echo "  - 磁盘空间"
    echo "  - 网络流量"
    echo "  - 训练进程状态"
    echo "  - 性能优化建议"
}

# 主程序入口
case "${1:-}" in
    -o|--once)
        single_check
        ;;
    -h|--help)
        show_help
        ;;
    *)
        main_loop
        ;;
esac
