#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
from pathlib import Path
import torchaudio
import torch
import numpy as np
import soundfile as sf

# 添加Matcha-TTS依赖
sys.path.append('./third_party/Matcha-TTS')

# 导入CosyVoice相关模块
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav


def save_audio(speech, sample_rate, output_path):
    """保存音频文件"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存音频
    if isinstance(speech, torch.Tensor):
        speech = speech.squeeze().cpu().numpy()
    
    if len(speech.shape) > 1 and speech.shape[0] == 1:
        speech = speech.squeeze(0)
    
    sf.write(output_path, speech, sample_rate)
    print(f"已保存音频文件到: {output_path}")


def test_zero_shot(model, text, prompt_file, prompt_text, output_dir):
    """测试零样本语音克隆"""
    print(f"\n===== 测试零样本语音克隆 =====")
    print(f"合成文本: {text}")
    print(f"参考音频: {prompt_file}")
    print(f"参考文本: {prompt_text}")
    
    # 加载参考音频
    prompt_speech_16k = load_wav(prompt_file, 16000)
    
    # 生成输出路径
    output_path = os.path.join(output_dir, "zero_shot_output.wav")
    
    # 执行推理
    start_time = time.time()
    for i, result in enumerate(model.inference_zero_shot(text, prompt_text, prompt_speech_16k, stream=False)):
        # 仅保存第一个输出（多个batch场景）
        if i == 0:
            save_audio(result['tts_speech'], model.sample_rate, output_path)
    
    # 计算耗时
    elapsed = time.time() - start_time
    print(f"零样本语音克隆完成，耗时: {elapsed:.2f}秒")
    

def test_cross_lingual(model, text, prompt_file, output_dir):
    """测试跨语言语音克隆"""
    print(f"\n===== 测试跨语言语音克隆 =====")
    print(f"合成文本: {text}")
    print(f"参考音频: {prompt_file}")
    
    # 加载参考音频
    prompt_speech_16k = load_wav(prompt_file, 16000)
    
    # 生成输出路径
    output_path = os.path.join(output_dir, "cross_lingual_output.wav")
    
    # 执行推理
    start_time = time.time()
    for i, result in enumerate(model.inference_cross_lingual(text, prompt_speech_16k, stream=False)):
        # 仅保存第一个输出（多个batch场景）
        if i == 0:
            save_audio(result['tts_speech'], model.sample_rate, output_path)
    
    # 计算耗时
    elapsed = time.time() - start_time
    print(f"跨语言语音克隆完成，耗时: {elapsed:.2f}秒")


def test_fine_grained_control(model, text, prompt_file, output_dir):
    """测试精细控制合成"""
    print(f"\n===== 测试精细控制合成 =====")
    print(f"合成文本: {text}")
    print(f"参考音频: {prompt_file}")
    
    # 加载参考音频
    prompt_speech_16k = load_wav(prompt_file, 16000)
    
    # 生成输出路径
    output_path = os.path.join(output_dir, "fine_grained_control_output.wav")
    
    # 执行推理
    start_time = time.time()
    for i, result in enumerate(model.inference_cross_lingual(text, prompt_speech_16k, stream=False)):
        # 仅保存第一个输出（多个batch场景）
        if i == 0:
            save_audio(result['tts_speech'], model.sample_rate, output_path)
    
    # 计算耗时
    elapsed = time.time() - start_time
    print(f"精细控制合成完成，耗时: {elapsed:.2f}秒")


def test_instruct(model, text, instruct, prompt_file, output_dir):
    """测试指令控制语音合成"""
    print(f"\n===== 测试指令控制语音合成 =====")
    print(f"合成文本: {text}")
    print(f"指令: {instruct}")
    print(f"参考音频: {prompt_file}")
    
    # 加载参考音频
    prompt_speech_16k = load_wav(prompt_file, 16000)
    
    # 生成输出路径
    output_path = os.path.join(output_dir, "instruct_output.wav")
    
    # 执行推理
    start_time = time.time()
    for i, result in enumerate(model.inference_instruct2(text, instruct, prompt_speech_16k, stream=False)):
        # 仅保存第一个输出（多个batch场景）
        if i == 0:
            save_audio(result['tts_speech'], model.sample_rate, output_path)
    
    # 计算耗时
    elapsed = time.time() - start_time
    print(f"指令控制语音合成完成，耗时: {elapsed:.2f}秒")


def test_streaming(model, text, prompt_file, prompt_text, output_dir):
    """测试流式语音合成"""
    print(f"\n===== 测试流式语音合成 =====")
    print(f"合成文本: {text}")
    print(f"参考音频: {prompt_file}")
    print(f"参考文本: {prompt_text}")
    
    # 加载参考音频
    prompt_speech_16k = load_wav(prompt_file, 16000)
    
    # 生成输出路径
    output_dir = os.path.join(output_dir, "streaming")
    os.makedirs(output_dir, exist_ok=True)
    
    # 执行推理
    start_time = time.time()
    chunks = []
    for i, result in enumerate(model.inference_zero_shot(text, prompt_text, prompt_speech_16k, stream=True)):
        chunk_path = os.path.join(output_dir, f"chunk_{i}.wav")
        save_audio(result['tts_speech'], model.sample_rate, chunk_path)
        chunks.append(result['tts_speech'])
    
    # 合并所有chunk
    if chunks:
        combined = torch.cat(chunks, dim=1)
        combined_path = os.path.join(output_dir, "combined_streaming.wav")
        save_audio(combined, model.sample_rate, combined_path)
    
    # 计算耗时
    elapsed = time.time() - start_time
    print(f"流式语音合成完成，耗时: {elapsed:.2f}秒")


def run_all_tests(model_dir, output_dir, prompt_file=None):
    """运行所有测试"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查参考音频文件
    if prompt_file is None or not os.path.exists(prompt_file):
        # 使用默认的参考音频
        prompt_file = "./asset/zero_shot_prompt.wav"
        if not os.path.exists(prompt_file):
            print(f"错误: 未找到参考音频文件 {prompt_file}")
            return
    
    print(f"加载CosyVoice2模型: {model_dir}")
    try:
        model = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)
        print("模型加载成功")
    except Exception as e:
        print(f"错误: 模型加载失败 - {e}")
        return
    
    # 测试文本
    chinese_text = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    english_text = "The sunset painted the sky with brilliant hues of orange and purple, casting a warm glow over the peaceful countryside."
    emotion_text = "在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。"
    
    # 参考文本
    prompt_text = "希望你以后能够做的比我还好呦。"
    
    # 指令文本
    instruct_text = "用四川话说这句话"
    
    # 运行各种测试
    test_zero_shot(model, chinese_text, prompt_file, prompt_text, output_dir)
    test_cross_lingual(model, english_text, prompt_file, output_dir)
    test_fine_grained_control(model, emotion_text, prompt_file, output_dir)
    test_instruct(model, chinese_text, instruct_text, prompt_file, output_dir)
    test_streaming(model, chinese_text, prompt_file, prompt_text, output_dir)
    
    print(f"\n所有测试完成! 输出文件保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="测试CosyVoice2的语音合成和声音克隆功能")
    parser.add_argument("--model_dir", type=str, default="pretrained_models/CosyVoice2-0.5B",
                        help="CosyVoice2模型目录路径")
    parser.add_argument("--output_dir", type=str, default="test_outputs",
                        help="测试输出保存目录")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="参考音频文件路径(如果不提供，将使用默认音频)")
    
    args = parser.parse_args()
    
    run_all_tests(args.model_dir, args.output_dir, args.prompt_file)


if __name__ == "__main__":
    main() 