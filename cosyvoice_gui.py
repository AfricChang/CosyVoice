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
                
            # 根据模式执行不同的合成方法
            self.progress.emit("开始语音合成...")
            
            if self.mode == "zero_shot" and prompt_speech_16k is not None:
                for i, result in enumerate(self.model.inference_zero_shot(
                        self.text, self.prompt_text, prompt_speech_16k, stream=False)):
                    if i == 0:
                        self._save_audio(result['tts_speech'], self.output_path)
                        break
                        
            elif self.mode == "cross_lingual" and prompt_speech_16k is not None:
                for i, result in enumerate(self.model.inference_cross_lingual(
                        self.text, prompt_speech_16k, stream=False)):
                    if i == 0:
                        self._save_audio(result['tts_speech'], self.output_path)
                        break
                        
            elif self.mode == "instruct" and prompt_speech_16k is not None:
                for i, result in enumerate(self.model.inference_instruct2(
                        self.text, self.instruct_text, prompt_speech_16k, stream=False)):
                    if i == 0:
                        self._save_audio(result['tts_speech'], self.output_path)
                        break
            else:
                self.error.emit(f"无法执行{self.mode}模式，请检查参数设置")
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
        
        # 合成文本输入
        text_group = QGroupBox("合成文本")
        text_layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("请输入要合成的文本")
        self.text_edit.setText("收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。")
        text_layout.addWidget(self.text_edit)
        text_group.setLayout(text_layout)
        main_layout.addWidget(text_group)
        
        # 合成模式选择
        mode_group = QGroupBox("合成模式")
        mode_layout = QHBoxLayout()
        
        self.mode_group = QButtonGroup()
        self.zero_shot_radio = QRadioButton("零样本声音克隆")
        self.cross_lingual_radio = QRadioButton("跨语言声音克隆")
        self.instruct_radio = QRadioButton("指令控制语音合成")
        
        self.zero_shot_radio.setChecked(True)
        
        self.mode_group.addButton(self.zero_shot_radio, 0)
        self.mode_group.addButton(self.cross_lingual_radio, 1)
        self.mode_group.addButton(self.instruct_radio, 2)
        
        mode_layout.addWidget(self.zero_shot_radio)
        mode_layout.addWidget(self.cross_lingual_radio)
        mode_layout.addWidget(self.instruct_radio)
        
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)
        
        # 参考音频设置
        prompt_group = QGroupBox("参考音频设置")
        prompt_layout = QVBoxLayout()
        
        prompt_file_layout = QHBoxLayout()
        self.prompt_file_edit = QLineEdit()
        self.prompt_file_edit.setPlaceholderText("参考音频文件路径")
        self.prompt_file_btn = QPushButton("选择文件")
        self.prompt_file_btn.clicked.connect(self.select_prompt_file)
        prompt_file_layout.addWidget(QLabel("参考音频:"))
        prompt_file_layout.addWidget(self.prompt_file_edit)
        prompt_file_layout.addWidget(self.prompt_file_btn)
        prompt_layout.addLayout(prompt_file_layout)
        
        # 参考文本设置（零样本克隆模式下需要）
        self.prompt_text_edit = QLineEdit()
        self.prompt_text_edit.setPlaceholderText("输入与参考音频对应的文本内容")
        self.prompt_text_edit.setText("希望你以后能够做的比我还好呦。")
        prompt_layout.addWidget(QLabel("参考文本:"))
        prompt_layout.addWidget(self.prompt_text_edit)
        
        # 指令文本（指令模式下需要）
        self.instruct_text_edit = QLineEdit()
        self.instruct_text_edit.setPlaceholderText("输入控制指令，如'用四川话说这句话'")
        self.instruct_text_edit.setText("用四川话说这句话")
        prompt_layout.addWidget(QLabel("指令文本:"))
        prompt_layout.addWidget(self.instruct_text_edit)
        
        prompt_group.setLayout(prompt_layout)
        main_layout.addWidget(prompt_group)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        self.synthesize_btn = QPushButton("开始合成")
        self.synthesize_btn.clicked.connect(self.start_synthesis)
        self.synthesize_btn.setEnabled(False)  # 初始禁用，等模型加载成功后启用
        
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)  # 初始禁用，等合成完成后启用
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)  # 初始禁用
        
        control_layout.addWidget(self.synthesize_btn)
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.stop_btn)
        main_layout.addLayout(control_layout)
        
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
        """更新状态栏信息"""
        self.statusBar.showMessage(message)
        
    def show_error(self, error_message):
        """显示错误消息"""
        self.statusBar.showMessage(f"错误: {error_message}")
        QMessageBox.critical(self, "错误", error_message)
        self.synthesize_btn.setEnabled(True)
        
    def synthesis_finished(self, output_path):
        """合成完成处理"""
        self.current_audio_path = output_path
        self.synthesize_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.statusBar.showMessage(f"合成完成! 输出文件: {output_path}")
        
        # 自动播放合成的语音
        self.play_audio()
        
    def play_audio(self):
        """播放合成的音频"""
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            QMessageBox.warning(self, "警告", "没有可播放的音频文件")
            return
            
        # 设置媒体播放器
        url = QUrl.fromLocalFile(QFileInfo(self.current_audio_path).absoluteFilePath())
        self.player.setMedia(QMediaContent(url))
        self.player.play()
        
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar.showMessage(f"正在播放: {self.current_audio_path}")
        
        # 播放结束后恢复按钮状态
        self.player.stateChanged.connect(self.player_state_changed)
        
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
            
    def closeEvent(self, event):
        """应用关闭时的处理"""
        self.player.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = CosyVoiceGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
