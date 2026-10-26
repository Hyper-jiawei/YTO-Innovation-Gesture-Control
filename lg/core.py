"""Geometry and temporal state machines; independent of camera/UI/MediaPipe."""
from dataclasses import dataclass
import numpy as np
from .labels import FIST_DIGIT
from .controls import HoldTrigger,DoubleOKTrigger,STOP_SECONDS,GATE_IDS


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area = lambda r: max(0, r[2]-r[0]) * max(0, r[3]-r[1])
    return inter / max(area(a) + area(b) - inter, 1e-9)


def features(points):
    p = np.asarray(points, dtype=np.float32).reshape(21, 3).copy()
    p -= p[0].copy()
    scale = np.linalg.norm(p[9])
    if scale < 1e-6 or not np.isfinite(p).all():
        raise ValueError('Invalid landmarks')
    return (p / scale).flatten()


def index_pointing(points):
    """ROI pointing ignores thumb pose; digit classification is a separate task."""
    p = np.asarray(points)
    u, v = p[5]-p[6], p[8]-p[6]
    cosine = np.dot(u,v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-9)
    angle = np.degrees(np.arccos(np.clip(cosine,-1,1)))
    if not (angle >= 145 and np.linalg.norm(p[8]-p[0]) > 1.05*np.linalg.norm(p[6]-p[0])):
        return False
    for base in (9,13,17):
        u,v = p[base]-p[base+1],p[base+3]-p[base+1]
        cosine = np.dot(u,v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-9)
        if (np.degrees(np.arccos(np.clip(cosine,-1,1))) > 155
                and np.linalg.norm(p[base+3]-p[0]) > 1.12*np.linalg.norm(p[base+1]-p[0])):
            return False
    return True


@dataclass
class Hand:
    points: np.ndarray
    digit: object
    box: tuple
    track_id: int = -1


class Tracker:
    """Greedy IoU association, stable numeric identity, resets on loss."""
    def __init__(self):
        self.previous = []
        self.next_id = 0

    def update(self, hands):
        for h in hands:
            h.track_id = -1
        pairs = sorted(((iou(a.box,b.box), ai, bi) for ai,a in enumerate(hands)
                        for bi,b in enumerate(self.previous)), reverse=True)
        used_a, used_b = set(), set()
        for score, ai, bi in pairs:
            if score >= .25 and ai not in used_a and bi not in used_b:
                hands[ai].track_id = self.previous[bi].track_id
                used_a.add(ai); used_b.add(bi)
        for h in hands:
            if h.track_id < 0:
                h.track_id = self.next_id
                self.next_id += 1
        self.previous = hands
        return sorted(hands, key=lambda h: h.points[0,0])


class Dwell:
    def __init__(self, seconds=1.1, tolerance=.06, min_iou=.35):
        self.seconds, self.tolerance, self.min_iou = seconds, tolerance, min_iou
        self.reset()

    def reset(self):
        self.key = None
        self.anchor = None
        self.start = 0
        self.fired = False
        self.progress = 0.

    def update(self, key, positions, boxes, now):
        if key is None:
            self.reset()
            return False
        if self.fired and key == self.key:
            self.progress = 1.
            return False
        positions = np.asarray(positions)
        stable = (key == self.key and self.anchor is not None
                  and np.max(np.linalg.norm(positions-self.anchor,axis=-1)) <= self.tolerance
                  and all(iou(a,b) >= self.min_iou for a,b in zip(boxes,self.boxes)))
        if not stable:
            # A fired pose is handled above and stays closed until release.
            self.key, self.anchor, self.boxes = key, positions.copy(), list(boxes)
            self.start, self.fired = now, False
        self.progress = min(1., max(0., (now-self.start)/self.seconds))
        if self.progress >= 1 and not self.fired:
            self.fired = True
            return True
        return False


class Engine:
    def __init__(self, dwell=1.1, tolerance=.06, min_iou=.35, min_roi_pixels=4):
        self.dwell = Dwell(dwell,tolerance,min_iou)
        self.stop_hold=HoldTrigger(STOP_SECONDS)
        self.start_hold=DoubleOKTrigger()
        self.mode = None
        self.pair = None
        self.roi = None
        self.code = None
        self.last_time = None
        self.min_roi_pixels = min_roi_pixels
        self.control_name=None
        self.control_progress=0.

    def update(self, hands, mode, now, frame_size=None):
        events = []
        self.roi = self.code = None
        pair = tuple(h.track_id for h in hands)
        gap = self.last_time is not None and now-self.last_time > .5
        self.last_time = now
        changed=mode != self.mode or pair != self.pair
        if changed or (gap and not self.dwell.fired):
            self.dwell.reset()
        if changed or gap:
            self.stop_hold.reset();self.start_hold.reset()
        self.mode, self.pair = mode, pair
        digits = [h.digit for h in hands]
        stopping=digits==[FIST_DIGIT,FIST_DIGIT]
        starting=digits==['ok','ok']
        stop=self.stop_hold.update(pair if stopping else None,now)
        if stopping: self.start_hold.reset()
        start=self.start_hold.update(starting,now)
        self.control_name=None
        if stopping: self.control_name='停止传送带'
        elif starting: self.control_name='开启传送带 · OK 2/2' if self.start_hold.phase>=2 else '开启传送带 · OK 1/2'
        elif self.start_hold.phase==1: self.control_name='请松开双手'
        elif self.start_hold.phase==2: self.control_name='请再次同时 OK'
        self.control_progress=self.stop_hold.progress if stopping else self.start_hold.progress
        if stopping or starting:
            self.dwell.reset()
            return [('stop' if stop else 'start',None)] if stop or start else []
        key = None
        positions, boxes = [], []
        if len(hands) == 2:
            positions = [h.points[8,:2] for h in hands]
            boxes = [h.box for h in hands]
            if mode in ('scan','photo','damage') and all(h.digit == 1 or index_pointing(h.points) for h in hands):
                a,b = positions
                # Public event payloads must use Python scalars, not NumPy float32.
                roi = tuple(max(0.,min(1.,float(v))) for v in (min(a[0],b[0]), min(a[1],b[1]),
                                                             max(a[0],b[0]), max(a[1],b[1])))
                # Four pixels, not 8% of the frame: accept small/thin labels.
                min_width = self.min_roi_pixels/frame_size[0] if frame_size else 1e-6
                min_height = self.min_roi_pixels/frame_size[1] if frame_size else 1e-6
                if roi[2]-roi[0] >= min_width and roi[3]-roi[1] >= min_height:
                    self.roi = roi
                    key = (mode, pair)
            elif mode == 'command' and all(isinstance(d,(int,np.integer)) and d in GATE_IDS for d in digits) and digits[0]==digits[1]:
                self.code = f'{digits[0]}{digits[1]}'
                key = ('command', pair, self.code)
        if self.dwell.update(key, positions, boxes, now):
            events.append(('bind_gate', int(digits[0])) if mode == 'command' else (mode,self.roi))
        return events
