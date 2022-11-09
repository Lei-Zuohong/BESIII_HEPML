# BESIII_HEPML
  这个包是对项目[https://github.com/arogozhnikov/hep_ml](https://github.com/arogozhnikov/hep_ml)的改进版本，添加了xgboost模型和bin-reweight模型的支持，并支持自动优化超参数。

## 环境配置
### 1.Python环境直接调用
如果服务器上已经有配置好的软件环境，则直接使用该环境的可执行文件即可。例如存在地址为 P/python 的可执行文件，则使用
```
alias python "P/python"
```
然后使用
```
python XXX.py
```
执行脚本文件即可

### 2.Python安装
如果服务器上没有配置好的软件环境，则需自行安装。步骤如下：
安装[anaconda](https://www.anaconda.com/)
建立独立python环境