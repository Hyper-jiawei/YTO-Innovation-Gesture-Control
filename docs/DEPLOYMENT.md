# 部署与运行

## 推荐环境

开发验证环境为 Windows、Python 3.10.21、NVIDIA RTX5060，PyTorch 2.7.1 / torchvision 0.22.1 / CUDA12.8。MediaPipe关键点走CPU，MLP和YOLO在CUDA可用时使用GPU；没有CUDA时可退回CPU，但速度需要实测。

## 新电脑安装

```powershell
conda create -n mediapipe python=3.10 -y
conda activate mediapipe
python -m pip install -r requirements-base.txt
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-extras.txt
python check_environment.py
python main.py --camera 1
```

`check_environment.py`会检查CUDA前向与反向计算，结果写入 `outputs/environment.json`。CPU环境应安装对应CPU版PyTorch，且不使用上面的CUDA安装地址。依赖版本是当前开发快照；其他平台或Python版本需重新验证。

仓库提供以下模型：

| 文件 | 用途 |
|---|---|
| models/digits.pt | 自采手背数据训练的11类MLP |
| models/gesture_recognizer.task | MediaPipe手部定位与关键点任务包；项目不采用其内置手势类别作为自定义分类结果 |
| models/yolo12n.pt | 通用YOLO12n目标检测 |
| models/gesture_labels.json | 当前标签含义及控制映射 |

模型SHA256与评估摘要见 `docs/validation_metrics.json`。模型文件加载仅使用可信来源；外部权重应自行核验来源与格式。

## 启动参数

```powershell
# 默认相机为1，可改为0或auto；界面显示实际分辨率
python main.py --camera 1
# 无相机的明确标注合成演示，不产生真实业务操作
python main.py --demo
# 自定义框选稳定时长与容差
python main.py --camera 1 --dwell 1.1 --hold-tolerance 0.06 --hold-iou 0.35
# 禁用目标检测
python main.py --camera 1 --no-yolo
# 使用自己的检测权重
python main.py --camera 1 --yolo models/my_detector.pt
```

原开发机仍可运行 `./start.ps1`，默认定位 `C:/ProgramData/Anaconda3/envs/mediapipe/python.exe`。`start.ps1`、`collect.ps1`、`train.ps1`、`setup_gpu.ps1`中的解释器路径可按本机Conda路径调整。使用Python命令则不依赖该固定路径。

`bootstrap.py`会优先使用可选的 `.deps/`、`.extras/` 项目依赖目录；公开仓库不提供这两个体积较大的目录。新环境按上述pip命令安装即可，无需复制原开发环境。

## 操作

- 扫码、拍照、破损取证：两食指指向目标对角，稳定1.1秒；进度圈闭合即选择。拍照缩略图持续保留；破损识别临时显示在手指离开后隐藏。
- 停止：双手握拳（9）保持1秒。
- 开启：双手同时OK至少0.2秒→双手保持入镜并松开至少0.15秒→3秒内再次同时OK至少0.2秒。持续单次OK不启动。
- 格口绑定：仅格口模式双手同数2–5可绑定，顶部大字确认。
- 人工急停后必须复位；尚未完成的取证任务会拦截启动。历史待确认报告继续保留，但不拦截本地模拟启停验证。无归档按钮。

当前设备控制为本地模拟，未接PLC；异常报告为本地保存，未发往企业系统。部署联调需另行实现协议、权限、反馈及安全互锁。

## 输出与隐私

`outputs/`保存照片、报告、扫码诊断及事件日志；`data/`保存采集与导入样本。二者默认不提交Git。五个公开演示GIF由项目方提供，存放在 `docs/assets/`。首次使用会自动生成必要目录。

## 验证

```powershell
python -B -s -m unittest discover -s tests -v
python main.py --demo --smoke 3
```

UI测试需要可用桌面/Tk环境。全部测试会涉及PyTorch、MediaPipe及图形界面；结果不能替代现场验收。不同开发环境先核对 `requirements-*.txt` 与GPU检查结果。
