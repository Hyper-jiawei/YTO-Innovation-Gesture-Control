"""Independently inspect YOLO12 on a selected image region, without camera/UI."""
import bootstrap
import argparse,json
from pathlib import Path
import cv2,numpy as np
from lg.vision import Detector,draw_detections

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--roi',type=float,nargs=4,default=[0,0,1,1],metavar=('X1','Y1','X2','Y2'))
    parser.add_argument('--model',type=Path,default=bootstrap.ROOT/'models/yolo12n.pt')
    args=parser.parse_args()
    a,b,c,d=args.roi
    if not (0<=a<c<=1 and 0<=b<d<=1): parser.error('ROI must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1')
    frame=cv2.imdecode(np.fromfile(args.image,np.uint8),cv2.IMREAD_COLOR)
    if frame is None: parser.error('Cannot read image')
    detector=Detector(args.model); detections=detector.detect_region(frame,args.roi)
    folder=bootstrap.ROOT/'outputs/detection_demo';folder.mkdir(parents=True,exist_ok=True)
    ok,encoded=cv2.imencode('.jpg',draw_detections(frame,detections))
    if not ok: raise RuntimeError('Cannot encode detection result')
    encoded.tofile(str(folder/'annotated.jpg'))
    report=dict(model=detector.model_name,device=detector.device,roi=args.roi,detections=detections,
                bbox_coordinates='normalized_full_frame')
    (folder/'detections.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
