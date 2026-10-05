# 小波驱动的深度伪造开放集检测

本项目采用“已知域真/假学习 + 三维证据空间中的球形区域拒识”输出三种结果：

- `real`：真图；
- `fake`：伪造图；
- `unknown`：位于真球和假球之外、球体归属冲突，或输入质量不合格。

`unknown` 不是已知域训练数据中的第三类标签。教师和学生的主任务仍学习
`real=0`、`fake=1`；校准阶段再把真假概率、类别原型相似度、原型距离、
RGB—高频分歧和低置信度组合为三维证据点，但只用真/假校准数据拟合真球和假球。
能量默认不参与核心证据，只能通过配置作为消融项启用。
只有独占真球或假球的样本会被接纳；两个球都不进入时直接输出 `unknown=2`。
数据集中没有 `unknown` 类，也不使用 OE/OOD 样本训练或校准。

## 代码入口

每个操作现在都有独立脚本，不再使用统一的模式切换参数：

| 操作 | 独立脚本 | 主要输出 |
|---|---|---|
| 检查数据并训练完整频带教师 | `train_teacher.py` | `data_audit.json`、`best_teacher.pth` |
| 测试教师 | `test_teacher.py` | 教师已知集指标 |
| 训练 RGB—高频学生 | `train_student.py` | `outputs/stage2_student/best_student.pth` |
| 测试学生已知集 | `test_student.py` | 学生真/假指标 |
| 校准真/假可信球 | `calibrate_trusted_spheres.py` | `prototypes.pth`、`calibration.json` |
| 检测真/假/未知 | `detect_images.py` | 每张图片记录、三类数量和开放集指标 |
| 退化鲁棒性测试 | `test_robustness.py` | `robustness_metrics.json` |
| 跨数据集/生成器测试 | `test_generalization.py` | `generalization_metrics.json` |

具体业务逻辑位于 `pipelines/stage1.py` 和 `pipelines/stage2.py`，命令行参数位于 `pipelines/cli.py`。主函数中不再混放训练、校准、测试等实现。

## 数据目录

在当前 `code` 文件夹下使用以下结构：

```text
dataset/
├── id/
│   ├── train/
│   │   ├── real/FFPP/
│   │   └── fake/
│   │       ├── face_swap/FFPP/
│   │       └── attribute_edit/
│   ├── val/                       # 与 train 使用相同目录层级，但数据不重复
│   └── test/                      # 与 train 使用相同目录层级，但数据不重复
├── calibration/
│   └── id/                        # 与 train 使用相同目录层级，仅用于可信球校准
└── generalization/
    ├── CelebDF/{real,fake/face_swap}/
    └── DFDC/{real,fake/face_swap}/
```

不要把图片直接放在 `fake` 容器目录。每张假图必须进入
`face_swap`（换脸）或`attribute_edit`（面部/属性编辑）之一。本任务不再细分
具体的换脸或编辑算法。数据集中的每一级目录都带有 `README.md`，其中说明了
该目录允许存放的图片、标签、来源和禁止混入的内容；完整规范见
[`dataset/README.md`](dataset/README.md)。

数据加载时会先对所有数据源执行统一的图像处理：使用OpenCV检测最大人脸，按配置
扩大边界并裁成正方形；检测失败时使用确定性的中心正方形裁剪。随后训练阶段再执行
随机裁剪和增强，验证、测试、校准与推理阶段执行相同的确定性缩放与中心裁剪。
该流程由`data/preprocessing.py`统一实现，避免模型直接利用FF++完整画面与其他数据集
人脸小图之间的尺寸和背景差异。

训练加载器默认使用分层平衡采样，不删除任何原图。采样目标为
`real:face_swap:attribute_edit = 2:1:1`，即主类别`real:fake = 1:1`，假类内部
`face_swap:attribute_edit = 1:1`。

## WSL 环境

代码和数据可以放在 Windows 的 C 盘，同时从 WSL 运行。Windows 中的
`C:\Users\Lenovo\Desktop\被动防御\code` 对应 WSL 中的：

```bash
/mnt/c/Users/Lenovo/Desktop/被动防御/code
```

进入项目并安装环境：

```bash
cd "/mnt/c/Users/Lenovo/Desktop/被动防御/code"
python -m pip install -r requirements.txt
```

在 WSL 命令中，Windows 文件夹同样要写成 `/mnt/c/...`，不能写 `C:\...`。

## 推荐运行顺序

### 1. 检查数据并训练完整频带教师

```bash
python train_teacher.py
```

程序首先检查`train`、`val`、`test`和`calibration`的目录、类别、损坏图片与跨划分
内容重复，并保存`outputs/stage1_teacher/data_audit.json`。目录缺失、类别为空或图片损坏
时停止训练；重复图片只记录警告，不阻止训练。随后教师使用
`RGB + LL2 + H1 + H2`开始训练。

如需单独测试已训练教师：

```bash
python test_teacher.py
```

### 2. 训练 RGB—高频学生

```bash
python train_student.py
```

训练时教师被冻结。教师仍使用完整频带；学生只接收 `RGB + H1 + H2`，不接收 `LL2`。
默认配置中，局部遮挡和高频方向随机丢弃只用于阶段一；阶段二不再丢弃
RGB 或高频分支，主要进行知识蒸馏和退化一致性学习。

学生在门控融合后使用两个独立投影：`z_bin` 只服务于真/假分类、原型和可信球，
`z_sub` 只服务于`face_swap`与`attribute_edit`两个伪造子类。两个伪造子类
共同形成唯一的假原型和假球。

如需单独检查学生在已知真/假测试集上的表现：

```bash
python test_student.py
```

### 3. 校准三维证据和真/假可信球

```bash
python circle.py
```

该步骤使用训练集特征拟合已知原型、证据空间和真/假马氏椭球，并使用独立的
`calibration/id/real` 与 `calibration/id/fake` 完成以下操作：

1. 用真/假数据标准化原型距离、分支分歧和低置信度；
2. 构造 `[真证据, 假证据, 新颖度证据]` 三维坐标；
3. 使用训练集分别估计真球和假球的中心及协方差；
4. 在校准集上计算本类马氏距离，并以
   `ceil((n + 1) × 目标覆盖率)` 对应的顺序统计量确定共形半径。

默认目标覆盖率为 90%。例如每类有 188 个校准样本时，半径取排序后第 171 个
本类距离，而不是普通经验 90% 分位数对应的第 170 个距离。

当前模型结构和校准格式均已升级。旧学生检查点与旧版 `calibration.json` 不能继续
使用；需要重新训练学生并重新运行本步骤。若做能量消融，可将配置中的
`calibration.include_energy` 改为 `true` 后单独校准和评价。

### 4. 检测 inference 图片并生成 Excel 报告

```bash
python detect_images.py
```

程序默认递归检测 `dataset/inference`。该目录不需要建立类别子文件夹，也不需要
提供真实标签。终端依次输出：

1. 真实球和伪造球的半径；
2. 两个球的参考 AUC、AUC 来源和模型决策一致性 AUC；
3. 真、换脸、属性编辑、未知四类图片数量；
4. 真实图片被判断为真的比例、假图片被判断为假的比例；
5. 每张图片的真实标签、模型判断、是否正确、置信度、球心距离和判断区域。

检测结果保存为：

```text
outputs/stage2_student/predictions.json
outputs/stage2_student/open_set_test_metrics.json
outputs/stage2_student/detection_results.xlsx
```

Excel 报告包含 `检测汇总` 和 `逐图结果` 两个工作表。`检测汇总` 记录四类数量、
球半径、球级 AUC、AUC 来源以及各判断区域数量；`逐图结果` 每行对应一张图片。

模型决策一致性 AUC 使用模型最终判断作为伪标签，衡量判断结果和球距离分数是否
一致，不等同于真实性能 AUC。参考 AUC 来自校准集；检测有标签测试集时优先显示
当前测试集重新计算的球级 AUC。

当前 `dataset/inference` 的真实标签在模型完成判断后按以下文件名规则读取：

```text
*_source.*   -> 真
*_swapped.*  -> 假（换脸）
F_STGN_*.*   -> 假（属性编辑）
```

这些标签不进入神经网络，也不参与真/假/未知决策，只用于检测后的正确识别率统计。
真实图片正确识别率为“真实图片中最终判断为真的比例”；假图片正确识别率为“假图片
中最终判断为假的比例”。被拒识为未知的已知图片按识别失败计算。

也可以检测任意 Windows 文件夹：

```bash
python detect_images.py --input-dir "C:\Users\Lenovo\Desktop\待检测图片"
```

项目内相对路径既可以写成 `inference`，也可以写成 `dataset/inference`。

`predictions.json` 中每张图片都有路径、四分类判断、真/假概率、置信度、新颖度连续分数、
能量（诊断项）、原型距离、分支分歧、三维证据点、到真/假球心的距离、球体归属、
最终判定区域和质量拒识原因。只有假球单独接纳的图片才会包含有效的
`fake_subtype` 与 `fake_subtype_confidence`；真和未知统一记录为 `N/A`。

## 论文评价实验

### 测试图像退化鲁棒性

```bash
python test_robustness.py
```

脚本会在运行时自动对 `dataset/id/test` 生成 JPEG、二次 JPEG 编码、模糊、
噪声、缩放和裁图版本，不会修改或保存原图。每个退化条件分别输出真球和假球的
one-vs-rest AUC、球内纯度、球内条件错误率、本类覆盖率和误接纳率，并额外报告
未知输出率和质量拒识率，不再以整体 accuracy 作为核心指标。

### 测试跨数据集或跨生成器泛化性

按 `dataset/generalization/<域名>/real` 与 `fake` 放置带标签测试数据，然后执行：

```bash
python test_generalization.py
```

也可以指定其他目录：

```bash
python test_generalization.py --data-root "/mnt/c/path/to/generalization"
```

输出每个域的真球/假球 one-vs-rest AUC、球内纯度、球内条件错误率、
本类覆盖率、误接纳率，以及“假进入真球”的高风险错误率。

## 自定义参数

每个脚本都可单独指定配置、设备或检查点，例如：

```bash
python detect_images.py \
  --config configs/stage2_student.yaml \
  --checkpoint outputs/stage2_student/best_student.pth \
  --device cuda:0 \
  --input-dir "/mnt/c/Users/Lenovo/Desktop/待检测图片"
```

使用 `python 脚本名.py --help` 可查看该步骤支持的参数。

## 自动化测试

```bash
python -m pytest -q
```

测试覆盖二层 Haar 小波尺寸、真/假数据语义、教师和学生接口、学生不使用 LL2、
教师冻结、统一人脸裁剪、分层平衡采样、原型距离、真/假可信球校准、球外拒识、
三种输出数量以及检查点恢复。
