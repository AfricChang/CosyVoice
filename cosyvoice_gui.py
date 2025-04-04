#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QTextEdit, QPushButton, 
                            QComboBox, QFileDialog, QGroupBox, QRadioButton,
                            QSlider, QMessageBox, QLineEdit, QButtonGroup, QStatusBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QFileInfo
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

# 确保脚本的目录存在于Python路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'third_party/Matcha-TTS'))

# 导入CosyVoice相关模块
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
except ImportError as e:
    print(f"导入CosyVoice模块失败: {e}")
    print("请确保已安装所有依赖项，包括modelscope, hyperpyyaml等")
    sys.exit(1)

def process_text_by_lines(text):
    """将文本按行分割，并组合成不超过150字的片段"""
    segments = []
    current_segment = ""
    max_length = 150
    
    # 按行分割文本
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:  # 跳过空行
            continue
            
        # 如果添加当前行会超出长度限制
        if len(current_segment) + len(line) + 1 > max_length:  # +1 是为了可能添加的换行符
            if current_segment:
                segments.append(current_segment)
            current_segment = line
        else:
            # 如果不是第一行，添加换行符
            if current_segment:
                current_segment += "\n" + line
            else:
                current_segment = line
    
    # 添加最后一个段落
    if current_segment:
        segments.append(current_segment)
        
    return segments


class SynthesisThread(QThread):
    """语音合成线程，防止界面卡死"""
    finished = pyqtSignal(str)  # 完成信号，携带生成的音频文件路径
    progress = pyqtSignal(str)  # 进度更新信号
    error = pyqtSignal(str)     # 错误信号

    def __init__(self, model, mode, text, prompt_file, prompt_text, instruct_text, output_dir):
        super().__init__()
        self.model = model
        self.mode = mode
        self.text = text
        self.prompt_file = prompt_file
        self.prompt_text = prompt_text
        self.instruct_text = instruct_text
        self.output_dir = output_dir
        self.output_path = None

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 为输出文件生成唯一文件名
            timestamp = int(time.time())
            self.output_path = os.path.join(self.output_dir, f"{self.mode}_{timestamp}.wav")
            
            # 加载参考音频
            if self.prompt_file:
                self.progress.emit("正在加载参考音频...")
                prompt_speech_16k = load_wav(self.prompt_file, 16000)
            else:
                prompt_speech_16k = None
                
            # 将长文本分割成多个片段
            text_segments = process_text_by_lines(self.text)
            
            # 显示更详细的分割信息
            segment_info = "\n".join([f"片段{i+1}: 字符数{len(s)}" for i, s in enumerate(text_segments)])
            self.progress.emit(f"文本已分割为{len(text_segments)}个片段:\n{segment_info}")
            
            # 存储每个片段合成的音频
            audio_segments = []
            
            # 根据模式执行不同的合成方法
            for i, segment in enumerate(text_segments):
                # 显示片段内容的前30个字符，但如果内容包含换行符，只显示第一行
                display_text = segment.split('\n')[0] if '\n' in segment else segment[:30]
                self.progress.emit(f"开始合成第{i+1}/{len(text_segments)}个片段: {display_text}...")
                
                if self.mode == "zero_shot" and prompt_speech_16k is not None:
                    for j, result in enumerate(self.model.inference_zero_shot(
                            segment, self.prompt_text, prompt_speech_16k, stream=False)):
                        if j == 0:
                            audio_segments.append(result['tts_speech'])
                            break
                            
                elif self.mode == "cross_lingual" and prompt_speech_16k is not None:
                    for j, result in enumerate(self.model.inference_cross_lingual(
                            segment, prompt_speech_16k, stream=False)):
                        if j == 0:
                            audio_segments.append(result['tts_speech'])
                            break
                            
                elif self.mode == "instruct" and prompt_speech_16k is not None:
                    for j, result in enumerate(self.model.inference_instruct2(
                            segment, self.instruct_text, prompt_speech_16k, stream=False)):
                        if j == 0:
                            audio_segments.append(result['tts_speech'])
                            break
                else:
                    self.error.emit(f"无法执行{self.mode}模式，请检查参数设置")
                    return
            # 合并所有音频片段
            if audio_segments:
                self.progress.emit("正在合并所有音频片段...")
                
                if len(audio_segments) == 1:
                    combined_speech = audio_segments[0]
                else:
                    # 确保所有片段都是张量或都是numpy数组
                    if isinstance(audio_segments[0], torch.Tensor):
                        # 如果是张量，在时间维度上拼接
                        combined_speech = torch.cat(audio_segments, dim=1 if len(audio_segments[0].shape) > 1 else 0)
                    else:
                        # 如果是numpy数组，在时间维度上拼接
                        combined_speech = np.concatenate(audio_segments, axis=0)
                
                # 记录合并后的总长度
                total_len = combined_speech.shape[-1] if isinstance(combined_speech, torch.Tensor) else len(combined_speech)
                
                # 检查合并后的音频数据是否有效
                if total_len == 0:
                    self.error.emit("错误: 合并后的音频长度为0，请检查音频片段")
                    return
                
                # 保存合并后的音频
                self._save_audio(combined_speech, self.output_path)
                # 显示合并后的音频详细信息
                if isinstance(combined_speech, torch.Tensor):
                    samples = combined_speech.shape[-1]
                else:
                    samples = len(combined_speech)
                duration = samples / self.model.sample_rate
                
                # 验证保存的音频文件
                if not os.path.exists(self.output_path) or os.path.getsize(self.output_path) == 0:
                    self.error.emit("错误: 保存的音频文件无效，请检查磁盘空间和权限")
                    return
            else:
                self.error.emit("没有生成任何音频片段")
                return
                
            self.progress.emit("语音合成完成!")
            self.finished.emit(self.output_path)
            
        except Exception as e:
            self.error.emit(f"合成过程中出错: {str(e)}")
            
    def _save_audio(self, speech, output_path):
        """保存音频文件"""
        if isinstance(speech, torch.Tensor):
            speech = speech.squeeze().cpu().numpy()
        
        if len(speech.shape) > 1 and speech.shape[0] == 1:
            speech = speech.squeeze(0)
        
        sf.write(output_path, speech, self.model.sample_rate)
        self.progress.emit(f"已保存音频到: {output_path}")


class CosyVoiceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_loaded = False
        self.current_audio_path = None
        self.player = QMediaPlayer()
        self.initUI()
        self.load_model()
        
    def initUI(self):
        """初始化界面"""
        self.setWindowTitle('CosyVoice语音合成')
        self.setGeometry(100, 100, 800, 600)
        
        # 主部件和布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)  # 增加组件间距
        
        # 模型加载部分
        model_group = QGroupBox("模型设置")
        model_layout = QHBoxLayout()
        
        self.model_path_edit = QLineEdit("pretrained_models/CosyVoice2-0.5B")
        model_layout.addWidget(QLabel("模型路径:"))
        model_layout.addWidget(self.model_path_edit)
        
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # 合成模式选择
        mode_group = QGroupBox("合成模式")
        mode_layout = QVBoxLayout()
        mode_layout.setSpacing(10)  # 增加按钮间距
        
        self.mode_group = QButtonGroup()
        
        # 合成文本输入
        text_group = QGroupBox("合成文本")
        text_layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("请输入要合成的文本")
        self.text_edit.setText("收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。")
        text_layout.addWidget(self.text_edit)
        text_group.setLayout(text_layout)
        main_layout.addWidget(text_group)
        
        # 创建模式选择按钮并设置样式
        self.zero_shot_radio = QRadioButton("零样本声音克隆")
        self.cross_lingual_radio = QRadioButton("跨语言声音克隆")
        self.instruct_radio = QRadioButton("指令控制语音合成")
        
        # 设置按钮样式
        for radio in [self.zero_shot_radio, self.cross_lingual_radio, self.instruct_radio]:
            radio.setStyleSheet("""
                QRadioButton {
                    font-size: 14px;
                    padding: 8px;
                    border: 2px solid #ddd;
                    border-radius: 5px;
                    background-color: #f5f5f5;
                }
                QRadioButton:checked {
                    border-color: #4a90e2;
                    background-color: #e3f2fd;
                }
                QRadioButton:hover {
                    background-color: #e6e6e6;
                }
            """)
        
        self.cross_lingual_radio.setChecked(True)
        
        self.mode_group.addButton(self.zero_shot_radio, 0)
        self.mode_group.addButton(self.cross_lingual_radio, 1)
        self.mode_group.addButton(self.instruct_radio, 2)
        
        # 添加模式切换信号处理
        self.mode_group.buttonClicked.connect(self.on_mode_changed)
        
        mode_layout.addWidget(self.zero_shot_radio)
        mode_layout.addWidget(self.cross_lingual_radio)
        mode_layout.addWidget(self.instruct_radio)
        
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)
        
        # 参考音频设置
        prompt_group = QGroupBox("参考音频设置")
        prompt_layout = QVBoxLayout()
        prompt_layout.setSpacing(10)  # 增加控件间距
        
        # 音频来源选择
        source_layout = QHBoxLayout()
        self.source_group = QButtonGroup()
        
        self.builtin_radio = QRadioButton("内置音色")
        self.custom_radio = QRadioButton("自定义音频")
        self.custom_radio.setChecked(True)
        
        self.source_group.addButton(self.builtin_radio, 0)
        self.source_group.addButton(self.custom_radio, 1)
        self.source_group.buttonClicked.connect(self.on_source_changed)
        
        source_layout.addWidget(QLabel("音频来源:"))
        source_layout.addWidget(self.builtin_radio)
        source_layout.addWidget(self.custom_radio)
        prompt_layout.addLayout(source_layout)
        
        # 内置音色选择
        self.builtin_container = QWidget()
        builtin_layout = QHBoxLayout()
        self.builtin_combo = QComboBox()
        
        # 扫描AudioSamples目录获取所有wav文件和对应文本
        audio_samples_dir = os.path.join(os.path.dirname(__file__), "AudioSamples")
        self.wav_files = [f for f in os.listdir(audio_samples_dir) if f.endswith('.wav')]
        self.txt_files = [f for f in os.listdir(audio_samples_dir) if f.endswith('.txt')]
        
        # 创建音色名称到文件的映射
        self.voice_mapping = {}
        voice_names = []
        
        for wav_file in self.wav_files:
            try:
                # 提取名字部分（去掉数字和扩展名）
                base_name = wav_file.split('.')[0]
                name_part = wav_file.split('_')[1].split('.')[0]
                
                # 查找对应的txt文件
                txt_file = f"{base_name}.txt"
                if txt_file in self.txt_files:
                    with open(os.path.join(audio_samples_dir, txt_file), 'r', encoding='utf-8') as f:
                        prompt_text = f.read().strip()
                else:
                    prompt_text = ""
                
                self.voice_mapping[name_part] = {
                    'wav': wav_file,
                    'txt': prompt_text
                }
                voice_names.append(name_part)
            except IndexError:
                continue
        
        # 去重并排序
        voice_names = sorted(list(set(voice_names)))
        
        # 添加到下拉框
        self.builtin_combo.addItems(voice_names)
        builtin_layout.addWidget(QLabel("内置音色:"))
        builtin_layout.addWidget(self.builtin_combo)
        
        # 添加试听按钮
        self.preview_btn = QPushButton("试听")
        self.preview_btn.clicked.connect(self.preview_builtin_voice)
        builtin_layout.addWidget(self.preview_btn)
        
        self.builtin_container.setLayout(builtin_layout)
        self.builtin_container.hide()
        prompt_layout.addWidget(self.builtin_container)
        
        # 自定义音频文件选择
        self.custom_container = QWidget()
        custom_layout = QVBoxLayout()
        
        prompt_file_layout = QHBoxLayout()
        self.prompt_file_edit = QLineEdit()
        self.prompt_file_edit.setPlaceholderText("参考音频文件路径")
        self.prompt_file_btn = QPushButton("选择文件")
        self.prompt_file_btn.clicked.connect(self.select_prompt_file)
        prompt_file_layout.addWidget(QLabel("参考音频:"))
        prompt_file_layout.addWidget(self.prompt_file_edit)
        prompt_file_layout.addWidget(self.prompt_file_btn)
        custom_layout.addLayout(prompt_file_layout)
        
        # 检查voice_mapping中是否存在"飞镜"
        if "飞镜" in self.voice_mapping:
            self.builtin_combo.setCurrentText("飞镜")
        # 参考文本设置（零样本克隆模式下需要）
        self.prompt_text_container = QWidget()
        prompt_text_layout = QVBoxLayout()
        self.prompt_text_edit = QLineEdit()
        self.prompt_text_edit.setPlaceholderText("输入与参考音频对应的文本内容")
        self.prompt_text_edit.setText("希望你以后能够做的比我还好呦。")
        prompt_text_layout.addWidget(QLabel("参考文本:"))
        prompt_text_layout.addWidget(self.prompt_text_edit)
        self.prompt_text_container.setLayout(prompt_text_layout)
        custom_layout.addWidget(self.prompt_text_container)
        
        self.custom_container.setLayout(custom_layout)
        prompt_layout.addWidget(self.custom_container)
        
        # 指令文本（指令模式下需要）
        self.instruct_text_container = QWidget()
        instruct_text_layout = QVBoxLayout()
        self.instruct_text_edit = QLineEdit()
        self.instruct_text_edit.setPlaceholderText("输入控制指令，如'用四川话说这句话'")
        self.instruct_text_edit.setText("用四川话说这句话")
        instruct_text_layout.addWidget(QLabel("指令文本:"))
        instruct_text_layout.addWidget(self.instruct_text_edit)
        self.instruct_text_container.setLayout(instruct_text_layout)
        prompt_layout.addWidget(self.instruct_text_container)
        
        # 初始化显示状态
        self.instruct_text_container.hide()
        
        prompt_group.setLayout(prompt_layout)
        main_layout.addWidget(prompt_group)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)  # 增加按钮间距
        
        self.synthesize_btn = QPushButton("开始合成")
        self.synthesize_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 20px;
                font-size: 14px;
                background-color: #4a90e2;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.synthesize_btn.clicked.connect(self.start_synthesis)
        self.synthesize_btn.setEnabled(False)  # 初始禁用，等模型加载成功后启用
        
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)  # 初始禁用，等合成完成后启用
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)  # 初始禁用
        
        self.open_output_dir_btn = QPushButton("打开输出目录")
        self.open_output_dir_btn.clicked.connect(self.open_output_directory)
        
        control_layout.addWidget(self.synthesize_btn)
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.open_output_dir_btn)
        main_layout.addLayout(control_layout)
        
        # 合成日志输出
        log_group = QGroupBox("合成日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("合成过程信息将在这里显示...")
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("欢迎使用CosyVoice语音合成器")
        
        # 设置主布局
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
    def load_model(self):
        """加载模型"""
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            QMessageBox.warning(self, "警告", "请输入有效的模型路径")
            return
            
        self.statusBar.showMessage("正在加载模型，请稍候...")
        self.load_model_btn.setEnabled(False)
        
        try:
            # 尝试加载模型
            self.model = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False)
            self.model_loaded = True
            self.synthesize_btn.setEnabled(True)
            self.statusBar.showMessage(f"模型加载成功: {model_path}")
            QMessageBox.information(self, "成功", "CosyVoice模型加载成功!")
        except Exception as e:
            self.statusBar.showMessage(f"模型加载失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
        finally:
            self.load_model_btn.setEnabled(True)
            
    def select_prompt_file(self):
        """选择参考音频文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择参考音频文件", "", 
            "音频文件 (*.wav *.mp3 *.flac);;所有文件 (*)", 
            options=options
        )
        
        if file_path:
            self.prompt_file_edit.setText(file_path)
            
    def on_source_changed(self):
        """处理音频来源切换"""
        if self.builtin_radio.isChecked():
            self.builtin_container.show()
            self.custom_container.hide()
            
            # 自动设置参考文本
            selected_voice = self.builtin_combo.currentText()
            if selected_voice:
                voice_info = self.voice_mapping.get(selected_voice)
                if voice_info and voice_info['txt']:
                    self.prompt_text_edit.setText(voice_info['txt'])
                
                # 设置内置音色文件路径
                if voice_info:
                    audio_file = os.path.join("AudioSamples", voice_info['wav'])
                    self.prompt_file_edit.setText(audio_file)
        else:
            self.builtin_container.hide()
            self.custom_container.show()
            
            # 清空自定义音频路径
            self.prompt_file_edit.clear()
            
    def start_synthesis(self):
        """开始语音合成"""
        if not self.model_loaded:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
            
        # 获取合成参数
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "警告", "请输入要合成的文本")
            return
        if self.builtin_radio.isChecked:
            selected_voice = self.builtin_combo.currentText()
            voice_info = self.voice_mapping.get(selected_voice)
            if voice_info:
                prompt_file = os.path.join(os.path.dirname(__file__), "AudioSamples", voice_info['wav'])
            prompt_text = self.prompt_text_edit.text().strip()
        else:
            prompt_file = self.prompt_file_edit.text().strip()

        if not prompt_file:
            QMessageBox.warning(self, "警告", "请选择参考音频文件")
            return
            
        if not os.path.exists(prompt_file):
            QMessageBox.warning(self, "警告", f"参考音频文件不存在: {prompt_file}")
            return
            
        # 确定合成模式
        if self.zero_shot_radio.isChecked():
            mode = "zero_shot"
            prompt_text = self.prompt_text_edit.text().strip()
            if not prompt_text:
                QMessageBox.warning(self, "警告", "零样本克隆模式需要输入参考文本")
                return
        elif self.cross_lingual_radio.isChecked():
            mode = "cross_lingual"
            prompt_text = ""  # 跨语言模式不需要参考文本
        elif self.instruct_radio.isChecked():
            mode = "instruct"
            prompt_text = ""  # 指令模式不需要参考文本
            instruct_text = self.instruct_text_edit.text().strip()
            if not instruct_text:
                QMessageBox.warning(self, "警告", "指令模式需要输入指令文本")
                return
        else:
            QMessageBox.warning(self, "警告", "请选择合成模式")
            return
            
        # 创建输出目录
        output_dir = os.path.join(current_dir, "synthesis_outputs")
        
        # 禁用UI控件，防止重复操作
        self.synthesize_btn.setEnabled(False)
        self.statusBar.showMessage("正在合成语音，请稍候...")
        
        # 创建并启动合成线程
        self.synthesis_thread = SynthesisThread(
            self.model,
            mode,
            text,
            prompt_file,
            self.prompt_text_edit.text().strip(),
            self.instruct_text_edit.text().strip(),
            output_dir
        )
        
        self.synthesis_thread.progress.connect(self.update_status)
        self.synthesis_thread.error.connect(self.show_error)
        self.synthesis_thread.finished.connect(self.synthesis_finished)
        
        self.synthesis_thread.start()
        
    def update_status(self, message):
        """更新状态信息到日志输出框"""
        self.log_text.append(message)
        
    def show_error(self, error_message):
        """显示错误消息"""
        error_msg = f"错误: {error_message}"
        self.log_text.append(error_msg)
        QMessageBox.critical(self, "错误", error_message)
        self.synthesize_btn.setEnabled(True)
        
    def synthesis_finished(self, output_path):
        """合成完成处理"""
        self.current_audio_path = output_path
        self.synthesize_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        
        # 自动播放合成的语音
        self.play_audio()
        
    def play_audio(self):
        """播放音频"""
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            QMessageBox.warning(self, "警告", "没有可播放的音频文件")
            return
            
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.current_audio_path)))
        self.player.play()
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar.showMessage(f"正在播放: {self.current_audio_path}")
        
        # 播放结束后恢复按钮状态
        self.player.stateChanged.connect(self.player_state_changed)
        
    def preview_builtin_voice(self):
        """试听内置音色"""
        selected_voice = self.builtin_combo.currentText()
        if not selected_voice:
            return
            
        # 获取对应的音频文件
        voice_info = self.voice_mapping.get(selected_voice)
        if not voice_info:
            QMessageBox.warning(self, "警告", f"未找到音色 {selected_voice} 的音频文件")
            return
            
        audio_file = os.path.join(os.path.dirname(__file__), "AudioSamples", voice_info['wav'])
        
        # 播放音频
        if os.path.exists(audio_file):
            media = QMediaContent(QUrl.fromLocalFile(audio_file))
            self.player.setMedia(media)
            self.player.play()
        else:
            QMessageBox.warning(self, "警告", f"音频文件 {audio_file} 不存在")
        
    def stop_audio(self):
        """停止播放音频"""
        self.player.stop()
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar.showMessage("播放已停止")
        
    def player_state_changed(self, state):
        """媒体播放器状态变化处理"""
        if state == QMediaPlayer.StoppedState:
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.statusBar.showMessage("播放完成")
    
    def open_output_directory(self):
        """打开合成输出目录"""
        output_dir = os.path.join(current_dir, "synthesis_outputs")
        os.makedirs(output_dir, exist_ok=True)
        os.startfile(output_dir)
            
    def closeEvent(self, event):
        """应用关闭时的处理"""
        self.player.stop()
        event.accept()
        
    def on_mode_changed(self, button):
        """处理合成模式切换"""
        # 根据选择的模式显示/隐藏相关控件
        if button == self.zero_shot_radio:
            self.prompt_text_container.show()
            self.instruct_text_container.hide()
        elif button == self.cross_lingual_radio:
            self.prompt_text_container.hide()
            self.instruct_text_container.hide()
        elif button == self.instruct_radio:
            self.prompt_text_container.hide()
            self.instruct_text_container.show()


def main():
    app = QApplication(sys.argv)
    window = CosyVoiceGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    print("loading model...")
    main()
