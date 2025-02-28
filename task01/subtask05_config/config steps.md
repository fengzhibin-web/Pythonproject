# 环境配置步骤说明文档

## 一、引言

本说明文档旨在指导用户使用 Anaconda 进行虚拟环境的创建，并在该虚拟环境中安装 深度学习库(以 PyTorch为例)。通过详细的步骤和指令，帮助用户快速搭建所需的开发环境。

## 二、环境配置步骤

### 1. 创建新的虚拟环境

首先，我们需要打开 Anaconda Prompt，它是 Anaconda 提供的命令行工具，可用于执行各种与 Anaconda 相关的操作。在 Anaconda Prompt 中，执行以下指令来创建一个名为 `occ` 的新虚拟环境，并指定 Python 版本为 3.9.21：

```bash
conda create -n occ python=3.9.21
```
### 2. 激活虚拟环境
创建好虚拟环境后，需要将其激活，以便后续在该环境中进行操作。在 Anaconda Prompt 中执行以下指令来激活 occ 虚拟环境：
```bash
conda activate occ
```
### 3. 添加清华镜像源
为了加快包的下载速度，我们可以添加清华镜像源。在 Anaconda Prompt 中执行以下指令添加 PyTorch 的清华镜像源：
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
```
### 3. 添加清华镜像源
为了加快包的下载速度，我们可以添加清华镜像源。在 Anaconda Prompt 中执行以下指令添加 PyTorch 的清华镜像源：
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
```

### 4. 检查镜像源是否添加成功
添加镜像源后，我们可以通过以下指令查看 channels 处是否成功添加了该镜像源：
```bash
conda config --show
```

### 5. 安装 PyTorch 及其相关库
在激活的 occ 虚拟环境中，执行以下指令来安装 PyTorch、torchvision 和 torchaudio 库，并且指定仅使用 CPU 版本：
```bash
conda install pytorch torchvision torchaudio cpuonly -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
```

### 6. 检查库是否安装成功
安装完成后，我们可以通过以下指令查看当前虚拟环境中已安装的所有库，以此来检查 PyTorch 及其相关库是否安装成功，其余库的安装方法同上
```bash
conda list
```
***
### 7.在pycharm中使用配置好的虚拟环境
1. 打开Pycharm项目的`Settings` -> `Project Interpreter` -> `Show All`
2. 点击`+`添加虚拟环境（默认为`base`）
3. 选择`Existing environment`（现有环境）
4. 选择conda环境目录（在annaconda安装目录的envs下面），进入环境文件夹后选择`python.exe`即可
5. 选择好后，点击`ok`，直到回到`setting`界面，选择好刚载入的环境解释器，确认即可


## 三、总结
通过以上步骤，你已经成功创建了一个名为 occ 的虚拟环境，并在其中安装了 PyTorch、torchvision 和 torchaudio 库。现在你可以在该环境中进行相关的深度学习开发工作了。