# BESIII_HEPML
  这个包是对项目
  [https://github.com/arogozhnikov/hep_ml](https://github.com/arogozhnikov/hep_ml)
  的改进版本，添加了xgboost模型和bin-reweight模型的支持，并支持自动优化超参数。仅供科大BESIII合作组内部交流使用。

## 1. 环境配置
### Python环境直接调用法（建议）
如果服务器上已经有配置好的软件环境，则直接使用该环境的可执行文件即可。例如存在地址为 XXX/python 的可执行文件，则使用
```
alias python "XXX/python"
```
将命令定向到该可执行文件，然后使用
```
python YYY.py
```
执行脚本文件即可
本人在高能所配置可直接使用的环境地址为
```
/besfs5/users/leizh/software/anaconda/envs/py38/bin/python
```
### Python重新安装法（不建议）
如果服务器上没有配置好的软件环境，则需自行安装。步骤如下：

安装[anaconda](https://www.anaconda.com/)

建立名为XXX的python环境，建议版本>=3.8
```
conda create -n XXX python=3.8
```
进入XXX环境
```
conda activate XXX
```
创建名为pip_requirement.txt的临时文件，内容如下
```
numpy
pandas
matplotlib
lmfit
configparser
pyyaml
sympy
uproot
xgboost
sklearn
```
并使用以下命令自动安装依赖
```
pip install -r pip_requirement.txt
```
其他环境配置使用与方法参照anaconda教程
### 程序包配置
使用以下命令拉去该项目到本地
```
git clone https://github.com/Lei-Zuohong/BESIII_HEPML.git
```
将其中hreweight程序包文件夹放入任意位置XXX/hreweight，并使用以下命令将其所在路径加入环境变量PYTHONPATH
```
setenv PYTHONPATH XXX
```
终端运行以下命令不报错即配置成功
```
python
>>> import hrweight as hreweight
```

## 2. 程序包使用








