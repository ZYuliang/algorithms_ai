
python setup.py check

pip install twine
# 1. 编译
python setup.py build
# 2. 生成发布压缩包：
python setup.py sdist
# 3. 生成网络发布包wheel文件：
python setup.py bdist_wheel

twine upload dist/SICA-2.1.4.tar.gz上传对应版本的包
需要数据密码和账号
twine upload dist/*


1.如果报403错误，是有人使用的包与你的包保持同名

2.如果报400错误，是之前生成了一个对应的dist文件，你需要将之前的dist文件删除之后，再进行重新的下载安装