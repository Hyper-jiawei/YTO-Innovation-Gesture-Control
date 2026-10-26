import bootstrap
import argparse
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Train parcel / exposed-content detector on GPU')
    p.add_argument('--data',required=True,type=Path,help='YOLO data.yaml，按操作员/日期分割数据')
    p.add_argument('--weights',required=True,type=Path,help='本地 YOLO 检测权重')
    p.add_argument('--epochs',type=int,default=100)
    a=p.parse_args()
    from ultralytics import YOLO
    from lg.vision import cuda_device
    device=cuda_device()
    if device=='cpu': raise RuntimeError('GPU 尚未验证通过；先运行 check_environment.py')
    YOLO(str(a.weights)).train(data=str(a.data.resolve()),epochs=a.epochs,imgsz=640,batch=4,
        device=0,workers=0,project=str(bootstrap.ROOT/'outputs/detector_training'),name='parcel')
