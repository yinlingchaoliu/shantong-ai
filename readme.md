
# 制作exe文件
```shell
pip install pyinstaller 
pyinstaller -F main.py 
```

# 制作环境
```shell
 python -m venv myenv 
 source myenv/bin/activate
```


# conda创建环境
```shell
conda create -n myenv3.12  python=3.12
conda activate myenv3.12
conda env export > environment.yml
```


# 安装依赖
```shell
sh install.sh
```
