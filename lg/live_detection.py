"""Mode-gated YOLO worker: one latest frame, bounded evidence jobs, fresh cache."""
from collections import deque
from dataclasses import dataclass
import threading
import time
from .core import iou


@dataclass
class DetectionSnapshot:
    frame: object
    roi: tuple
    detections: list
    captured_at: float
    completed_at: float
    inference_ms: float


def inside_region(detections,roi):
    selected=[]
    for item in detections:
        a,b,c,d=item['bbox'];x,y=(a+c)/2,(b+d)/2
        intersection=max(0,min(c,roi[2])-max(a,roi[0]))*max(0,min(d,roi[3])-max(b,roi[1]))
        if roi[0]<=x<=roi[2] and roi[1]<=y<=roi[3] and intersection/max((c-a)*(d-b),1e-9)>=.5:
            selected.append(item)
    return selected


class LiveDetector:
    def __init__(self,detector,interval=.05,prewarm=False):
        self.detector=detector;self.interval=interval
        self.condition=threading.Condition();self.inference_lock=threading.Lock()
        self.active=False;self.closed=False;self.generation=0
        self.pending=None;self.jobs=deque();self.completed={};self.latest=None
        self.error=None;self.counter=0
        self.prewarm=prewarm and detector.model is not None
        self.warmup_done=threading.Event();self.warmup_ms=None
        self.thread=threading.Thread(target=self._run,daemon=True,name='yolo-latest-frame')
        self.thread.start()

    def activate(self,enabled):
        with self.condition:
            if self.active!=enabled:
                self.generation+=1;self.pending=None;self.jobs.clear();self.completed.clear()
                self.latest=None;self.error=None;self.active=enabled
                self.condition.notify_all()

    def submit(self,frame,roi=None,now=None):
        with self.condition:
            if self.active and not self.closed and not self.error:
                self.pending=(self.generation,frame.copy(),tuple(roi or (0,0,1,1)),time.monotonic() if now is None else now,None)
                self.condition.notify()

    def request(self,frame,roi,now):
        with self.condition:
            if not self.active or self.closed or self.error or len(self.jobs)>=4: return None
            self.counter+=1;token=self.counter
            self.jobs.append((self.generation,frame.copy(),tuple(roi),now,token))
            self.condition.notify()
            return token

    def take(self,token):
        with self.condition: return self.completed.pop(token,None)

    def snapshot(self,roi,now,max_age=.75):
        with self.condition:
            cached=self.latest
            if not self.active or self.error or cached is None: return None
            if now-cached.captured_at>max_age: return None
            if cached.roi!=(0,0,1,1) and iou(cached.roi,roi)<.7: return None
            return DetectionSnapshot(cached.frame,tuple(roi),inside_region(cached.detections,roi),
                                     cached.captured_at,cached.completed_at,cached.inference_ms)

    def refine(self,frame,roi):
        # The model has one inference owner at a time, including photo mode.
        with self.inference_lock: return self.detector.refine(frame,roi)

    def _run(self):
        # One synthetic warmup overlaps camera/MediaPipe initialization. Its result
        # is never published as a real camera detection or used as evidence.
        if self.prewarm:
            import numpy as np
            started=time.perf_counter()
            try:
                with self.inference_lock:
                    self.detector.detect_region(np.zeros((640,640,3),np.uint8),(0,0,1,1))
            except Exception as exc:
                with self.condition:
                    self.error=f'{type(exc).__name__}: {exc}'
                    for queued in self.jobs:
                        self.completed[queued[-1]]=self.error
                    self.jobs.clear();self.pending=None
            self.warmup_ms=(time.perf_counter()-started)*1000
        self.warmup_done.set()
        next_live=0.
        while True:
            with self.condition:
                while True:
                    if self.closed: return
                    if self.active and not self.error and self.jobs:
                        job=self.jobs.popleft();break
                    if self.active and not self.error and self.pending is not None:
                        remaining=next_live-time.monotonic()
                        if remaining<=0:
                            job=self.pending;self.pending=None;break
                        self.condition.wait(remaining)
                    else: self.condition.wait()
            generation,frame,roi,captured_at,token=job
            started=time.monotonic();started_precise=time.perf_counter()
            try:
                with self.inference_lock: detections=self.detector.detect_region(frame,roi)
                completed=time.monotonic()
                result=DetectionSnapshot(frame,roi,detections,captured_at,completed,(time.perf_counter()-started_precise)*1000)
                error=None
            except Exception as exc:
                result=None;error=f'{type(exc).__name__}: {exc}'
            next_live=started+self.interval
            with self.condition:
                if not self.active or generation!=self.generation: continue
                if error:
                    self.error=error
                    if token is not None: self.completed[token]=error
                    for queued in self.jobs:
                        if queued[-1] is not None: self.completed[queued[-1]]=error
                    self.jobs.clear();self.pending=None
                else:
                    if token is not None: self.completed[token]=result
                    self.latest=result

    def close(self):
        with self.condition:
            self.closed=True;self.pending=None;self.jobs.clear();self.condition.notify_all()
        self.thread.join(timeout=5)


class DamageDisplay:
    def __init__(self,tolerance=.08):
        self.tolerance=tolerance;self.clear()

    def clear(self):
        self.roi=None;self.pair=None

    def select(self,roi,pair):
        self.roi=tuple(roi);self.pair=tuple(pair)

    def visible(self,current_roi,pair):
        valid=(self.roi is not None and current_roi is not None and tuple(pair)==self.pair
               and max(abs(a-b) for a,b in zip(current_roi,self.roi))<=self.tolerance)
        if not valid: self.clear()
        return valid
