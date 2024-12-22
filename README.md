本项目是基于pyqt5开发的数字图像处理功能界面
.ui文件需要在qtcreater中运行
没有使用外部icon文件
主要环境依赖为
scipy                               1.13.1
pyinstaller                         6.11.1
pyinstaller-hooks-contrib           2024.10
PyQt5                               5.15.9
pyqt5-plugins                       5.15.9.2.3
PyQt5-Qt5                           5.15.2
PyQt5-sip                           12.16.1
pyqt5-tools                         5.15.9.3.3
PySocks                             1.7.1
python-dateutil                     2.9.0.post0
python-dotenv                       1.0.1
python-json-logger                  3.2.0

打包exe文件：
  1）进入对应conda环境
  2）进入项目所在文件夹
  3）输入：pyinstaller -F -w main.py(-D打包全部文件会显示python39.dll文件缺失)
  4）如果希望添加新东西，比如一些icon文件，则输入：pyinstaller -F -w -i"文件地址(不需要引号)" main.py
