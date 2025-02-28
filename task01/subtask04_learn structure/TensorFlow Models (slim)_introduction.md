# TensorFlow Models (slim) 项目介绍
# TF-slim的维护者：
Sergio Guadarrama, github: sguada
Sergio Guadarrama，github：sguada
# 代码风格规范
## Python 样式
遵循 PEP 8 Python 样式指南，但 TensorFlow 使用 2 个空格而不是 4 个空格
## Python 运算
`TensorFlow` 运算是一种给定输入张量、返回输出张量（或在构建计算图时向计算图添加运算）的函数。

1. 第一个参数应当是张量，然后是基本的 `Python` 参数。最后一个参数是默认值为 `None` 的 `name`。
2. 张量参数应当是单个张量或者多个张量的可迭代对象。也就是说，“张量或张量列表”过于宽泛。请参见 `assert_proper_iterable`。
3. 如果使用张量作为参数的运算正在使用 `C++` 运算，则应调用` convert_to_tensor` 将非张量输入转换为张量。请注意，参数在文档中仍被描述为特定 `dtype` 的 `Tensor `对象。
4. 每个 `Python` 运算都应具有一个 `name_scope`。如下所示，以字符串形式传递运算的名称。
运算应包含带参数和返回声明的大量 `Python` 注释，这些注释说明了每个值的类型和含义。应在说明中指定可能的形状、`dtype` 或秩。请参阅文档详细信息。
5. 为提高可用性，“示例”部分中包括带运算的输入/输出的用法示例。
6. 避免显式使用 `tf.Tensor.eval` 或 `tf.Session.run`。例如，要编写依赖于张量值的逻辑，请使用 `TensorFlow` 控制流。或者，将运算限制为仅在启用 `Eager Execution` 时 `(tf.executing_eagerly())` 才运行。
## 目录结构
### - datasets：提供五个常用的图像数据集
### - deployment：提供模型部署脚本
### - nets：提供一些模型的原始文件
### - preprocessing：提供一些预训练模型
### - scripts：
##### 1. 提供转换数据集格式的脚本
##### 2. 提供评估模型性能的脚本
##### 3. 提供导出推理图的脚本
##### 3. 提供自动化预处理ImageNet数据的脚本

### - BULID：通过 Bazel 构建系统，将项目中的 Python 代码、Shell 脚本、数据文件等组织成多个可构建的目标，明确了各目标之间的依赖关系，方便进行项目的构建、测试和部署。同时，通过注释部分依赖，可以灵活调整构建过程中使用的依赖项。
### - READEME:提供项目描述 ，配置方法，使用方法，作者信息等项目相关信息
---
## 核心模块
## 1.datasets （数据集）
**主要提供了
-- Flowers
-- Cifar
-- MNIST
-- ImageNet
-- VisualWakeWords
五个常用图像数据集用于模型训练
使用数据集前，需要使用如下脚本将原始数据集转化为TensorFlow 的原生 TFRecord 格式：**
```
$ DATA_DIR=/tmp/data/flowers
$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"
```
**执行脚本后，将创建`TFRecord `文件，代表训练和验证数据，每个数据被分为五个文件，此外，还将找到 `$DATA_DIR/labels.txt`文件，其中包含从整数标签到类名的映射**
**可以使用相同的脚本来创建 mnist、cifar10 和 visualwakewords 数据。但是，对于 ImageNet，您必须按照说明进行作 这里。请注意，您首先必须在 image-net.org 注册一个帐户。此外，下载可能需要几个小时，并且可能最多使用 500GB。**
**此外，还提供了专门用于将 ImageNet 数据集下载并处理为原生 TFRecord 格式的自动化脚本，使用`script`文件夹中的`download_and_convert`脚本即可实现**

## 2.script（脚本）
**该模块为模型训练提供了不同的脚本，分别有数据集格式转换脚本，模型性能评估脚本，导出推理图脚本**
## 3.preprocessing(预处理)
**提供了五种模型的预训练版本，相较于原始版本，这些模型以及经过了大体量的数据集训练，神经网络参数比原始模型要高，功能更强大**


