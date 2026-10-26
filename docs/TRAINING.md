# 采集、打标与训练

## 1. 标签与左右手

词表固定为 `1、2、3、4、5、6、7、8、9、I Love U、ok`。9表示握拳，0/10不存在。左右手分别采样、保留实际手别标记，当前共享同一个MLP分类模型；两手不需要各训练一个模型。

当前任务面向手背俯拍。每类每只手每个独立轮次建议采30–50个有变化的样本，至少3轮，并覆盖多名操作员；5–6个只适合验证采集流程。相邻连拍高度相似，不能当作独立人员/场景覆盖。对手套、遮挡、远近和光照，应以现场条件补采和验证。

## 2. 按钮采集

```powershell
python collect_digits.py --camera 1 --operator p01
```

点击采集冻结当前画面，选择标签及实际左/右手，保存后继续下一轮。再次启动会追加已有数据，默认位置为 `data/current/digits.jsonl`；“开始新的独立采集轮次”用于建立会话标记。`--demo`写入单独演示目录，不写正式数据。

## 3. 已有照片制作数据集

```text
hand_images/
  1/                 左手1
  1 - 副本/          右手1
  ...
  9/                 左手握拳
  9 - 副本/          右手握拳
  I Love U/
  I Love U - 副本/
  ok/
  ok - 副本/
```

```powershell
python import_hand_images.py --source "D:/datasets/hand_images" --output data/my_dorsal
```

类别由文件夹人工指定，MediaPipe生成关键点和边界框伪标注；会保存去重清单、有效JSONL、无手/歧义记录、逐图标注和质检图。抽查关键点是否落在手上、左右手与类别是否正确，确认后使用该JSONL训练。

JSONL每行包含标签、实际手别、operator、session、points、63维features及来源信息。导入自同一照片批次的数据不能通过随意修改session名称伪装为独立采集轮次。

## 4. 划分与增强

多人员/多会话优先按组划分；每组应采齐所有类别。现有训练器的group模式保留验证组，但用该组作早停，仍需另建独立验收集。

```powershell
# 数据具有至少3个真实独立会话组时
python train_digits.py --data data/current/digits.jsonl --group-by session --epochs 400 --augment-copies 4
# 单批次照片的初步实验：各类别/实际手别按时间分段，并保留间隔
python train_digits.py --data data/my_dorsal/digits.jsonl --split temporal --epochs 400 --augment-copies 4
```

先划分，再只扩增训练集合：以腕点为原点、掌距归一化，随机镜像，Z轴±12°、X/Y轴±5°小旋转，标准差0.004的有限关键点噪声。每原样本增加4个变体即总共5份。验证/测试集合不扩增。镜像只改变训练特征，不将记录中的实际手别冒充真实采集结果。

## 5. 权重接入

默认训练完成后写入 `models/digits.pt` 与 `outputs/training_report.json`，同时输出训练样本增强记录。主程序重启直接加载权重及随模型保存的标签、置信度阈值。可用 `--output-dir outputs/experiments/my_run` 先做隔离实验，通过独立测试后再替换默认权重。替换前保留上一版本。

原始训练数据不随公开仓库分发；仓库提供当前已训练权重和聚合评估摘要，因此可以直接推理，但要复现或继续训练需按上述步骤提供自己的数据。

## 6. 动作序列与专项目标检测

动作识别模块可采集2秒单手完整序列，经质检后重采样为20帧，追加到 `data/actions.jsonl`。必须包含展开→捏合/握拳→展开的时间变化，静态照片不能代替动作序列。现版本动作识别是规则状态机，LSTM/TCN是后续方向。

若要自动识别包装破损，需另外采集有代表性的正常/异常照片，人工标注目标框或分割区域，划分独立测试集，并通过 `train_detector.py --help` 使用自有数据训练。当前通用YOLO12输出物体类别，不承担自动判损。
