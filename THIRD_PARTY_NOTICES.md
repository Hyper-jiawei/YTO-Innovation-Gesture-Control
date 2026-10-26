# 第三方来源与素材说明

本仓库包含项目自编的手势工作站代码、训练脚本、已训练分类器及演示资料，并依赖或参考下列项目。公开展示不改变第三方代码、模型或商标原有权利；不为未明确授权的材料额外授予许可。

| 项目/资产 | 使用方式 | 来源与许可 |
|---|---|---|
| MediaPipe | 手部定位与21点关键点；models/gesture_recognizer.task任务文件 | [Google AI Edge](https://github.com/google-ai-edge/mediapipe)，[模型文档](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer)，Apache-2.0；许可副本见licenses/mediapipe.txt |
| Ultralytics / YOLO12n | 目标检测接口、models/yolo12n.pt通用模型 | [Ultralytics](https://github.com/ultralytics/ultralytics)，[YOLO12文档](https://docs.ultralytics.com/models/yolo12/)，AGPL-3.0/官方企业授权方案；随附AGPL许可文本见licenses/ultralytics.txt |
| PyTorch / torchvision | MLP训练、GPU推理 | [PyTorch](https://github.com/pytorch/pytorch)，[torchvision](https://github.com/pytorch/vision)，各依赖遵循其上游许可 |
| ZXing-C++ | 一维与二维条码解码 | [zxing-cpp](https://github.com/zxing-cpp/zxing-cpp)，Apache-2.0 |
| handpose_x | 手姿与数据处理思路参考 | [Hyper-jiawei/handpose_x](https://github.com/Hyper-jiawei/handpose_x)，MIT，副本见licenses/handpose_x.txt |
| gestures-recog | 关键点采集、训练及动态序列思路参考 | [fukuuy/gestures-recog](https://github.com/fukuuy/gestures-recog)；发布检查时GitHub未返回明确仓库许可证，本仓库未整体复制其源文件或数据，不对其授权状态作扩大解释 |
| YTO标识与五个GIF | 项目主页、品牌展示及演示 | 由项目方提供，权利归相应权利人；不授予商标使用权 |

其他Python依赖以各上游项目和安装包的许可为准。本项目没有把所有第三方依赖重新声明为MIT。集成、修改或再分发时请保留来源与适用许可，并按实际使用方式核对Ultralytics许可要求。

原始人员照片、训练样本、现场事件日志与本机依赖目录不在公开发布范围；当前digits.pt为项目自采手背样本训练出的分类器，聚合指标与校验值见docs/validation_metrics.json。
