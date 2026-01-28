# SSL 模块导入错误修复指南

## 问题描述
```
ImportError: DLL load failed while importing _ssl: 找不到指定的模块。
```

这是 Windows 上 conda 环境的常见 SSL 问题。

## 解决方案（按顺序尝试）

### 方案 1：重新安装 Python（最有效，推荐）

如果 `_ssl` DLL 加载失败，通常是 Python 的 SSL 模块损坏。重新安装 Python 可以修复：

```powershell
conda activate odt
# 先查看当前 Python 版本
python --version

# 重新安装相同版本的 Python（例如 3.13.5）
conda install python=3.13.5 -y
# 或者使用你当前的版本号
```

### 方案 2：重新安装 openssl 和 certifi

在 PowerShell 或 CMD 中执行：

```powershell
conda activate odt
conda install -c conda-forge openssl certifi -y
conda update openssl certifi -y
```

### 方案 3：修复 conda 环境

```powershell
conda activate odt
conda update --all -y
conda install openssl -y
```

### 方案 4：重新创建环境（最后手段）

```powershell
# 备份当前环境
conda list --export > packages.txt

# 创建新环境
conda create -n odt_new python=3.11 -y
conda activate odt_new

# 安装必要的包
conda install openssl certifi -y
pip install openai scikit-learn numpy tqdm

# 如果需要其他包，从 packages.txt 中安装
```

## 验证修复

运行以下 Python 代码验证 SSL 是否正常：

```python
import ssl
import socket
print("SSL 模块导入成功！")
print(f"SSL 版本: {ssl.OPENSSL_VERSION}")
```

## 如果仍然失败

1. 检查是否有多个 Python 安装冲突
2. 尝试使用 `pip install --upgrade certifi` 而不是 conda
3. 检查系统环境变量 PATH 是否正确

