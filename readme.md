
# 制作exe文件
```shell
pip install pyinstaller 
pyinstaller -F main.py 
```

# 制作环境
```shell
python -m venv myenv 
source myenv/bin/activate
 
pyenv install 3.8.5  # 安装特定版本的Python
pyenv global 3.8.5  # 设置全局Python版本为3.8.5
pyenv local 3.8.5  # 在当前目录设置Python版本为3.8.5
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

# 添加大文件
```shell
# 初始化 Git LFS
git lfs install
# 告诉 Git LFS 跟踪 .pth 文件（根据你的实际文件类型调整）
git lfs track "*.pth"
# 或者直接跟踪特定文件
git lfs track "notebook/data/checkpoint.pth"
git add .gitattributes
# 将所有历史记录中的 .pth 文件迁移到 LFS
git lfs migrate import --include="*.pth" --everything

git lfs migrate import --include="notebook/data/checkpoint.pth" --everything
git push --set-upstream origin main --force-with-lease
```