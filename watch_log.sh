#!/bin/bash
# GraphRAG 实时日志监控 (带颜色高亮)

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}GraphRAG 实时日志监控${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""

# 检查日志文件
if [ ! -f "logs/indexing-engine.log" ]; then
    echo -e "${RED}❌ 日志文件不存在: logs/indexing-engine.log${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}📋 实时日志 (Ctrl+C 退出):${NC}"
echo -e "${CYAN}----------------------------------------${NC}"

# 实时监控日志，带颜色高亮
tail -f logs/indexing-engine.log | while read line; do
    # 高亮进度信息
    if echo "$line" | grep -qi "progress:"; then
        echo -e "${GREEN}✓ $line${NC}"

    # 高亮错误
    elif echo "$line" | grep -qi "error\|exception\|failed"; then
        echo -e "${RED}✗ $line${NC}"

    # 高亮警告
    elif echo "$line" | grep -qi "warning\|warn"; then
        echo -e "${YELLOW}⚠ $line${NC}"

    # 高亮完成信息
    elif echo "$line" | grep -qi "complete\|finished\|done\|success"; then
        echo -e "${GREEN}✓ $line${NC}"

    # 高亮开始信息
    elif echo "$line" | grep -qi "starting\|begin\|creating"; then
        echo -e "${CYAN}▶ $line${NC}"

    # 普通日志
    else
        echo "$line"
    fi
done
