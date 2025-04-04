@echo off
setlocal enabledelayedexpansion

REM 设置编码为UTF-8
chcp 65001 > nul

echo ====================================
echo CosyVoice 部署和测试脚本
echo ====================================

REM 检查Python是否安装
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请先安装Python 3.10
    goto :end
)

REM 检查conda是否安装
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误: 未找到conda，请先安装Miniconda或Anaconda
    goto :end
)

REM 设置模型路径
set MODEL_PATH=E:\AI\AI_Models\aigiPanel\aigcpanel-server-cosyvoice2-0.5b-win-x86-v0.1.0\aigcpanelmodels

REM 检查模型路径是否存在
if not exist "%MODEL_PATH%" (
    echo 错误: 未找到模型路径 %MODEL_PATH%
    echo 请检查模型路径是否正确
    goto :end
)

echo 模型路径: %MODEL_PATH%

echo.
echo 请选择要执行的操作:
echo 1. 部署CosyVoice环境
echo 2. 运行测试脚本
echo 3. 启动Web界面
echo 4. 退出
echo.

set /p choice=请输入选项 [1-4]: 

if "%choice%"=="1" (
    call :deploy_env
) else if "%choice%"=="2" (
    call :run_test
) else if "%choice%"=="3" (
    call :start_web
) else if "%choice%"=="4" (
    goto :end
) else (
    echo 无效的选项，请重新运行脚本并选择有效选项
    goto :end
)

goto :end

:deploy_env
echo.
echo ===== 开始部署CosyVoice环境 =====
echo.

REM 激活基础conda环境，然后运行setup脚本
call conda activate base
python setup_cosyvoice.py --model_path "%MODEL_PATH%"

if %errorlevel% neq 0 (
    echo.
    echo 环境部署过程中出现错误，请检查上面的错误信息
) else (
    echo.
    echo 环境部署完成!
)
goto :eof

:run_test
echo.
echo ===== 开始运行CosyVoice测试 =====
echo.

REM 激活cosyvoice环境，然后运行测试脚本
call conda activate cosyvoice
python test_cosyvoice.py --model_dir pretrained_models/CosyVoice2-0.5B --output_dir test_outputs

if %errorlevel% neq 0 (
    echo.
    echo 测试过程中出现错误，请检查上面的错误信息
) else (
    echo.
    echo 测试完成!
    echo 输出文件保存在 test_outputs 目录中
)
goto :eof

:start_web
echo.
echo ===== 启动CosyVoice Web界面 =====
echo.

REM 激活cosyvoice环境，然后启动web界面
call conda activate cosyvoice
python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B

if %errorlevel% neq 0 (
    echo.
    echo Web界面启动过程中出现错误，请检查上面的错误信息
)
goto :eof

:end
echo.
echo 脚本执行完毕
pause 