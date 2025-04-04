# CosyVoice本地部署指南

本项目提供了在本地部署CosyVoice语音合成模型的脚本和工具，让您能够轻松使用CosyVoice的语音合成和声音克隆功能。

## 目录

- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [详细说明](#详细说明)
  - [环境部署](#环境部署)
  - [测试运行](#测试运行)
  - [Web界面](#web界面)
- [常见问题](#常见问题)

## 系统要求

- Windows 10或更高版本
- Python 3.10
- CUDA支持的GPU (推荐NVIDIA GTX 1060 6GB或更高)
- Miniconda或Anaconda
- 至少10GB的磁盘空间
- 至少8GB内存 (推荐16GB)

## 快速开始

1. 确保您已安装[Miniconda](https://docs.conda.io/en/latest/miniconda.html)或Anaconda
2. 克隆或下载本仓库
3. 确保模型文件路径正确(`E:\AI\AI_Models\aigiPanel\aigcpanel-server-cosyvoice2-0.5b-win-x86-v0.1.0\aigcpanelmodels`)
4. 双击运行`start_cosyvoice.bat`脚本
5. 选择所需操作:
   - 选项1: 部署CosyVoice环境
   - 选项2: 运行测试脚本
   - 选项3: 启动Web界面

## 详细说明

### 环境部署

环境部署脚本(`setup_cosyvoice.py`)将执行以下操作:

1. 创建名为`cosyvoice`的conda环境
2. 安装必要的依赖项，包括pynini、pytorch等
3. 将外部模型文件链接到项目目录中
4. 验证安装是否成功

要手动部署环境，请执行:

```bash
# 创建conda环境
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice

# 安装pynini
conda install -y -c conda-forge pynini==2.1.5

# 安装依赖
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# 更新git子模块
git submodule update --init --recursive

# 链接模型文件
python setup_cosyvoice.py --model_path "E:\AI\AI_Models\aigiPanel\aigcpanel-server-cosyvoice2-0.5b-win-x86-v0.1.0\aigcpanelmodels"
```

### 测试运行

测试脚本(`test_cosyvoice.py`)将运行以下测试:

1. 零样本语音克隆: 使用参考音频和文本合成新语音
2. 跨语言语音克隆: 使用中文参考音频合成英文语音(或反之)
3. 精细控制合成: 合成含有情感标记的语音
4. 指令控制语音合成: 根据指令(如方言、风格)合成语音
5. 流式语音合成: 测试流式合成模式

要手动运行测试，请执行:

```bash
conda activate cosyvoice
python test_cosyvoice.py --model_dir pretrained_models/CosyVoice2-0.5B --output_dir test_outputs
```

您也可以指定自己的参考音频文件:

```bash
python test_cosyvoice.py --prompt_file path/to/your/audio.wav
```

### Web界面

CosyVoice提供了一个易于使用的网页界面，您可以通过以下方式启动:

```bash
conda activate cosyvoice
python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
```

启动后，在浏览器中访问 http://localhost:50000 即可使用Web界面。

Web界面提供以下功能:
- 预训练音色合成
- 3秒极速声音克隆
- 跨语种声音克隆
- 自然语言控制

## 常见问题

**问题: 运行时出现"ImportError: No module named 'xxx'"**

解决方案: 确保您已激活cosyvoice环境并安装了所有依赖:
```bash
conda activate cosyvoice
pip install -r requirements.txt
```

**问题: 模型加载失败**

解决方案: 确保模型文件路径正确，并且已成功链接到pretrained_models目录:
```bash
python setup_cosyvoice.py --model_path "您的模型路径"
```

**问题: CUDA相关错误**

解决方案: 确保您已安装正确版本的CUDA和相应的PyTorch版本。如果在Windows上没有CUDA支持，可以使用CPU模式，但性能会降低:
```python
model = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False, device="cpu")
```

**问题: 合成的语音质量不佳**

解决方案: 尝试使用更高质量的参考音频，确保参考音频和参考文本内容一致，同时确保参考音频的采样率至少为16kHz。

---

如有任何问题或建议，请随时提问。祝您使用愉快! 