#!/bin/bash

# 检查是否传递了文件夹路径
if [ -z "$1" ]; then
  echo "请提供文件夹路径作为参数"
  exit 1
fi

# 获取传递的文件夹路径
TARGET_DIR="$1"

# 判断文件夹是否存在
if [ ! -d "$TARGET_DIR" ]; then
  echo "文件夹不存在：$TARGET_DIR"
  exit 1
fi

# 遍历文件夹中的所有文件
for file in "$TARGET_DIR"/*; do
  # 判断是否是文件
  if [ -f "$file" ]; then
    ./../build/BNN -I -m ../build/resnet18.ml -f $file >> ./test_on_dataset.txt
  fi
done