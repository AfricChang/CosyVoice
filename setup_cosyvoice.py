#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path


def run_command(cmd):
    """运行命令并实时显示输出"""
    print(f"执行命令: {cmd}")
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    return process.returncode


def setup_environment():
    """设置conda环境"""
    print("正在设置conda环境...")
    
    # 检查conda环境是否存在
    result = subprocess.run("conda env list | grep cosyvoice", shell=True, capture_output=True, text=True)
    if "cosyvoice" in result.stdout:
        print("cosyvoice环境已存在，跳过创建步骤")
    else:
        # 创建conda环境
        run_command("conda create -n cosyvoice -y python=3.10")
        
    # 激活环境并安装依赖
    activate_cmd = "conda activate cosyvoice && "
    
    # 安装pynini
    run_command(activate_cmd + "conda install -y -c conda-forge pynini==2.1.5")
    
    # 安装依赖
    run_command(activate_cmd + "pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com")
    
    # 确保git子模块已更新
    run_command("git submodule update --init --recursive")
    
    print("环境设置完成")


def link_model_files(model_path):
    """链接或复制外部模型文件到项目目录"""
    print(f"正在链接模型文件从 {model_path}...")
    
    # 创建目标目录
    target_dir = Path("pretrained_models/CosyVoice2-0.5B")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 寻找模型路径下的所有文件
    model_files = list(Path(model_path).glob("*"))
    
    if not model_files:
        print(f"警告: 在 {model_path} 中没有找到模型文件")
        return
    
    # 链接或复制每个文件
    for file_path in model_files:
        target_path = target_dir / file_path.name
        
        # 如果目标已存在，先删除
        if target_path.exists():
            if target_path.is_symlink() or target_path.is_file():
                target_path.unlink()
            elif target_path.is_dir():
                shutil.rmtree(target_path)
        
        # 尝试创建符号链接，如果失败则复制文件
        try:
            target_path.symlink_to(file_path)
            print(f"已创建符号链接: {target_path} -> {file_path}")
        except (OSError, NotImplementedError):
            if file_path.is_dir():
                shutil.copytree(file_path, target_path)
            else:
                shutil.copy2(file_path, target_path)
            print(f"已复制: {file_path} -> {target_path}")
    
    print("模型文件链接完成")


def check_installation():
    """检查安装是否成功"""
    print("正在检查安装...")
    
    # 确认模型文件存在
    model_dir = Path("pretrained_models/CosyVoice2-0.5B")
    if not model_dir.exists() or not any(model_dir.iterdir()):
        print("错误: 模型文件不存在或为空")
        return False
    
    # 尝试导入必要的包
    try:
        sys.path.append('./third_party/Matcha-TTS')
        from cosyvoice.cli.cosyvoice import CosyVoice2
        print("导入CosyVoice模块成功")
        return True
    except ImportError as e:
        print(f"错误: 导入失败 - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="设置CosyVoice环境并链接模型文件")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="外部模型文件的路径")
    
    args = parser.parse_args()
    
    setup_environment()
    link_model_files(args.model_path)
    
    if check_installation():
        print("CosyVoice设置成功!")
        print("请使用 'conda activate cosyvoice' 激活环境后运行测试脚本")
    else:
        print("CosyVoice设置过程中遇到问题，请检查以上错误信息")


if __name__ == "__main__":
    main() 