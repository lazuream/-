# 项目简介

本项目是基于 PyQt5 开发的数字图像处理功能界面。

- `.ui` 文件需要在 Qt Creator 中运行。
- 没有使用外部 icon 文件。

## 主要环境依赖

- scipy 1.13.1
- pyinstaller 6.11.1
- pyinstaller-hooks-contrib 2024.10
- PyQt5 5.15.9
- pyqt5-plugins 5.15.9.2.3
- PyQt5-Qt5 5.15.2
- PyQt5-sip 12.16.1
- pyqt5-tools 5.15.9.3.3
- PySocks 1.7.1
- python-dateutil 2.9.0.post0
- python-dotenv 1.0.1
- python-json-logger 3.2.0

## 打包 exe 文件

1. 进入对应 conda 环境。
2. 进入项目所在文件夹。
3. 输入以下命令进行打包：
   ```bash
   pyinstaller -F -w main.pyz(注意：使用 -D 打包全部文件可能会导致 python39.dll 文件缺失的问题。)
4. 如果希望添加新东西，比如一些 icon 文件，则输入以下命令：
   ```bash
   pyinstaller -F -w -i "文件地址" main.py（注意：文件地址不需要引号）
  
