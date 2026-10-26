"""ROI-only, multi-pass barcode decoding. Never infer content from printed text."""
import cv2
import numpy as np


def scan_crop(frame,roi):
    """Add a small real-image margin for quiet zones; never search the full frame."""
    h,w=frame.shape[:2]
    x1,y1,x2,y2=[float(v) for v in roi]
    dx=max(6/w,(x2-x1)*.10)
    dy=max(6/h,(y2-y1)*.12)
    from .vision import crop
    image,bounds=crop(frame,(max(0,x1-dx),max(0,y1-dy),min(1,x2+dx),min(1,y2+dy)))
    a,b,c,d=bounds
    # Only accept codes whose centres remain inside the user-selected rectangle.
    region=(x1*w-a,y1*h-b,x2*w-a,y2*h-b)
    return image,region


def decode_barcodes(image,content_region=None):
    if image is None or image.size==0:
        return [],dict(decoder='none',reason='empty_roi',width=0,height=0,attempts=0)
    h,w=image.shape[:2]
    info=dict(decoder='ZXing 1D/2D',width=w,height=h,attempts=0)
    try:
        import zxingcpp
    except ImportError:
        text,points,_=cv2.QRCodeDetector().detectAndDecode(image)
        if text and content_region is not None and points is not None:
            cx,cy=points.reshape(-1,2).mean(0)
            x1,y1,x2,y2=content_region
            if not (x1<=cx<=x2 and y1<=cy<=y2): text=''
        info['decoder']='OpenCV QR only'
        return ([dict(text=text,format='QRCode',method='opencv')] if text else []),info
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    # ZXing already rotates/inverts. Additional passes address contrast/sampling.
    def variants():
        yield gray,'original',1.,0,0,zxingcpp.Binarizer.LocalAverage
        # Some linear readers in the installed wheel do not honour try_invert.
        yield 255-gray,'explicit_invert',1.,0,0,zxingcpp.Binarizer.LocalAverage
        yield gray,'global_histogram',1.,0,0,zxingcpp.Binarizer.GlobalHistogram
        scale=min(3.,2400/max(h,w))
        if scale>1.05:
            large=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            yield large,'upscale',scale,0,0,zxingcpp.Binarizer.LocalAverage
        enhanced=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(gray)
        yield enhanced,'contrast',1.,0,0,zxingcpp.Binarizer.LocalAverage
        _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        margin=max(16,int(min(h,w)*.1))
        padded=cv2.copyMakeBorder(binary,margin,margin,margin,margin,cv2.BORDER_CONSTANT,value=255)
        yield padded,'otsu_quiet_zone',1.,margin,margin,zxingcpp.Binarizer.BoolCast
        # Screen moire can vary between rows; a low-cost smooth pass is bounded.
        yield cv2.GaussianBlur(gray,(3,3),.5),'light_smoothing',1.,0,0,zxingcpp.Binarizer.LocalAverage
    for view,method,scale,ox,oy,binarizer in variants():
        info['attempts']+=1
        codes=zxingcpp.read_barcodes(np.ascontiguousarray(view),try_rotate=True,try_downscale=True,
                                    try_invert=True,binarizer=binarizer,return_errors=False)
        found=[]
        for code in codes:
            if not code.text: continue
            if content_region is not None:
                corners=[code.position.top_left,code.position.top_right,code.position.bottom_left,code.position.bottom_right]
                cx=(sum(p.x for p in corners)/4-ox)/scale
                cy=(sum(p.y for p in corners)/4-oy)/scale
                x1,y1,x2,y2=content_region
                if not (x1<=cx<=x2 and y1<=cy<=y2): continue
            found.append(dict(text=code.text,format=str(code.format),method=method))
        if found:
            info['method']=method
            return list({c['text']:c for c in found}.values()),info
    return [],info


class ScanWindow:
    """Fixed selection, multiple fresh frames, one final result; does not repeat HOLD."""
    def __init__(self,seconds=2.5,max_frames=8,interval=.18):
        self.seconds,self.max_frames,self.interval=seconds,max_frames,interval
        self.roi=None

    def start(self,roi,now):
        self.roi=tuple(float(v) for v in roi)
        self.deadline=now+self.seconds
        self.next_attempt=now
        self.attempts=0

    def cancel(self): self.roi=None

    def step(self,frame,now,station):
        if self.roi is None: return
        if now<self.next_attempt and now<self.deadline: return
        roi=self.roi
        image,region=scan_crop(frame,roi)
        result=station.scan(image,final=False,content_region=region)
        station.last_scan_info.update(frame_width=int(frame.shape[1]),frame_height=int(frame.shape[0]))
        self.attempts+=1
        self.next_attempt=now+self.interval
        if result is not None:
            self.cancel()
        elif now>=self.deadline or self.attempts>=self.max_frames:
            station.scan_failed(image,roi,frames=self.attempts)
            self.cancel()
