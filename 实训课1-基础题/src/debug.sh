#!/bin/bash

echo "启动DCU版本调试..."

# 使用hipgdb进行调试
hipgdb ./outputfile_dcu

echo "调试会话结束" 