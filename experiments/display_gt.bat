@echo off
chcp 65001 > nul
cd /d c:\Users\kevin\Desktop\llm_debate\experiments
python show_gt_simple.py > gt_display.txt 2>&1
type gt_display.txt
