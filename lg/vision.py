"""Camera, MediaPipe landmarks, optional trained classifier and YOLO refinement."""
import time
from pathlib import Path
import cv2
import numpy as np
from .core import Hand, Tracker, features
from .labels import LOVE_LABEL,LABELS,FIST_DIGIT

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
               (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
               (15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]


def cuda_device():
    try:
        import torch
        if torch.cuda.is_available():
            x = torch.ones((32,32), device='cuda')
            _ = (x@x).sum().item()  # actual kernel, not just driver discovery
            return 'cuda:0'
    except (ImportError, RuntimeError, AssertionError, OSError):
        pass
    return 'cpu'


def network(torch, classes):
    return torch.nn.Sequential(torch.nn.Linear(63,128), torch.nn.ReLU(),
                               torch.nn.Dropout(.15), torch.nn.Linear(128,64),
                               torch.nn.ReLU(), torch.nn.Linear(64,classes))


class DigitClassifier:
    def __init__(self, root):
        self.model = None
        path = Path(root)/'models/digits.pt'
        self.device = 'cpu'
        if path.exists():
            import torch
            self.torch = torch
            self.device = cuda_device()
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            self.labels = ckpt['labels']
            if not set(self.labels).issubset(LABELS):
                raise ValueError('模型词表已过期：请使用包含 1–9、I Love U、ok 的最新手背模型')
            self.threshold = ckpt.get('confidence_threshold',.85)
            self.model = network(torch,len(self.labels)).to(self.device)
            self.model.load_state_dict(ckpt['state_dict'])
            self.model.eval()

    def predict(self, points):
        if self.model is None:
            # Old hardcoded finger patterns conflict with the user's folder labels.
            return None
        t = self.torch
        with t.inference_mode():
            x = t.tensor(features(points), device=self.device).unsqueeze(0)
            confidence, idx = self.model(x).softmax(-1).max(-1)
        label = self.labels[idx.item()]
        if confidence.item() < self.threshold:
            return None
        if label.isdigit(): return int(label)
        return label if label in (LOVE_LABEL,'ok') else None


class Hands:
    def __init__(self, root, classify=True):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        # Existing task bundle includes a hand landmarker. Ignore canned gestures.
        model = Path(root)/'models/gesture_recognizer.task'
        if not model.exists():
            model = Path(root).parent/'gesture_recognizer.task'
        if not model.exists():
            raise FileNotFoundError(f'缺少关键点模型: {model}')
        self.recognizer = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(base_options=BaseOptions(model_asset_path=str(model)),
                running_mode=vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=.6, min_tracking_confidence=.6))
        self.tracker = Tracker()
        self.classifier = DigitClassifier(root) if classify else None
        self.ts = 0

    def process(self, frame):
        # Retain the original high-resolution frame for barcode/photo crops.
        # Landmark inference can use a smaller copy with identical normalized axes.
        if frame.shape[1]>960:
            frame=cv2.resize(frame,(960,max(1,round(frame.shape[0]*960/frame.shape[1]))),interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        self.ts = max(self.ts+1, int(time.monotonic()*1000))
        result = self.recognizer.recognize_for_video(image,self.ts)
        hands = []
        for landmarks in result.hand_landmarks:
            p = np.array([(l.x,l.y,l.z) for l in landmarks],dtype=np.float32)
            box = (*p[:,:2].min(0), *p[:,:2].max(0))
            hands.append(Hand(p,self.classifier.predict(p) if self.classifier else None,box))
        return self.tracker.update(hands)

    def close(self):
        self.recognizer.close()


def open_camera(source,width=1920,height=1080):
    if str(source).isdigit():
        candidates = [int(source)]
    elif source == 'auto':
        candidates = [0,1,2,3]
    else:
        candidates = [str(source)]
    for index in candidates:
        for configure in ([True,False] if isinstance(index,int) else [False]):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if isinstance(index,int) else cv2.VideoCapture(index)
            if configure:
                cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
                cap.set(cv2.CAP_PROP_FPS,30)
            ok, frame = cap.read()
            if ok:
                cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
                return cap, frame, index
            cap.release()
    raise RuntimeError(f'无法读取摄像头/视频 {source}；关闭占用摄像头的程序，或用 --camera 0/1 指定设备。')


def crop(frame, roi):
    h,w = frame.shape[:2]
    a,b,c,d = roi
    x1,y1 = max(0,min(w-1,int(a*w))),max(0,min(h-1,int(b*h)))
    x2,y2 = max(x1+1,min(w,int(c*w))),max(y1+1,min(h,int(d*h)))
    return frame[y1:y2,x1:x2].copy(), (x1,y1,x2,y2)


def draw(frame, hands, roi=None, progress=0, dwell_seconds=1.1):
    out = frame.copy()
    h,w = out.shape[:2]
    for index,hand in enumerate(hands):
        pts = (hand.points[:,:2]*[w,h]).astype(int)
        color = (220,190,30) if index == 0 else (50,170,255)
        for a,b in CONNECTIONS:
            cv2.line(out,tuple(pts[a]),tuple(pts[b]),color,2)
        for i,p in enumerate(pts):
            cv2.circle(out,tuple(p),6 if i==8 else 3,color,-1)
        label='9 FIST' if hand.digit==FIST_DIGIT else (hand.digit if hand.digit is not None else '?')
        cv2.putText(out,f'ID {hand.track_id} / {label}',
                    tuple(pts[0]),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
    if roi:
        _,(a,b,c,d) = crop(out,roi)
        cv2.rectangle(out,(a,b),(c,d),(80,230,80),2)
        cv2.putText(out,f'HOLD {progress:.0%}',(a,max(25,b-12)),cv2.FONT_HERSHEY_SIMPLEX,.7,(80,230,80),2)
        # Both rings use the joint dwell timer: selection needs BOTH tips stable.
        fraction = float(np.clip(progress,0,1))
        radius = max(16, min(30,int(min(w,h)*.04)))
        for hand in hands:
            center = tuple((hand.points[8,:2]*[w,h]).astype(int))
            cv2.circle(out,center,radius,(50,60,70),5,cv2.LINE_AA)
            if fraction > 0:
                cv2.ellipse(out,center,(radius,radius),-90,0,360*fraction,
                            (80,230,80) if fraction >= 1 else (255,220,40),4,cv2.LINE_AA)
            remaining = max(0,dwell_seconds*(1-fraction))
            cv2.putText(out,'OK' if fraction >= 1 else f'{remaining:.1f}s',
                        (center[0]+radius+5,center[1]+5),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,250,250),1,cv2.LINE_AA)
    return out


class Detector:
    def __init__(self, model_path=None):
        self.model = None
        self.model_name = Path(model_path).name if model_path else 'disabled'
        self.device = cuda_device() if model_path else 'disabled'
        if model_path:
            if not Path(model_path).is_file():
                raise FileNotFoundError(f'YOLO 权重不存在: {model_path}')
            from ultralytics import YOLO
            self.model = YOLO(str(model_path))

    def detect_region(self, frame, roi):
        """Infer only the selected crop; return full-frame normalized boxes."""
        if self.model is None: return []
        image,(x1,y1,x2,y2)=crop(frame,roi)
        height,width=frame.shape[:2]
        result=self.model.predict(image,device=self.device,verbose=False,conf=.25,imgsz=640)[0]
        detections=[]
        for box in result.boxes:
            a,b,c,d=box.xyxy[0].cpu().tolist()
            cls=int(box.cls.item())
            bbox=[(x1+max(0,min(x2-x1,a)))/width,(y1+max(0,min(y2-y1,b)))/height,
                  (x1+max(0,min(x2-x1,c)))/width,(y1+max(0,min(y2-y1,d)))/height]
            detections.append(dict(label=str(result.names[cls]),class_id=cls,
                                   confidence=float(box.conf.item()),bbox=bbox))
        return detections

    def refine(self, frame, roi):
        """Select one fully enclosed detection; keep gesture ROI if ambiguous.

        General pretrained detectors cannot promise parcel or damage recognition.
        """
        if self.model is None:
            return roi, '手动精确框选'
        from .core import iou
        h,w = frame.shape[:2]
        results = self.model.predict(frame,device=self.device,verbose=False,conf=.5)[0]
        candidates = []
        for box in results.boxes:
            b = (box.xyxy[0].cpu().numpy()/[w,h,w,h]).tolist()
            if b[0]>=roi[0] and b[1]>=roi[1] and b[2]<=roi[2] and b[3]<=roi[3] and iou(b,roi)>.2:
                candidates.append((b, results.names[int(box.cls.item())]))
        if len(candidates)==1:
            return tuple(candidates[0][0]), f'YOLO: {candidates[0][1]} / {self.device}'
        return roi, f'候选 {len(candidates)} 个，保留食指框选'


def draw_detections(frame,detections):
    out=frame.copy(); h,w=out.shape[:2]
    for detection in detections:
        a,b,c,d=[round(float(v)*s) for v,s in zip(detection['bbox'],[w,h,w,h])]
        color=(214,68,112)
        cv2.rectangle(out,(a,b),(c,d),color,2,cv2.LINE_AA)
        label=f"{detection['label']} {detection['confidence']:.2f}"
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.6,2)
        baseline=min(h-5,max(th+5,b+th+5))
        cv2.rectangle(out,(a,baseline-th-5),(min(w,a+tw+8),baseline+3),color,-1)
        cv2.putText(out,label,(a+4,baseline-1),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2,cv2.LINE_AA)
    return out
