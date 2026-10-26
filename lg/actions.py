"""Fifth module: normalized landmark time series -> explicit action cycles.

Like the source repo's sequence pipeline, observations retain timestamps and
normalized hand geometry. No LSTM/GRU model is claimed without motion samples.
"""
from collections import deque
import numpy as np
from .labels import FIST_DIGIT


def hand_metrics(points):
    p=np.asarray(points,dtype=np.float32)
    palm=max(float(np.linalg.norm(p[9]-p[0])),1e-6)
    pinch=float(np.linalg.norm(p[4]-p[8])/palm)
    extended=closed=0
    for base in (5,9,13,17):
        u,v=p[base]-p[base+1],p[base+3]-p[base+1]
        angle=np.degrees(np.arccos(np.clip(np.dot(u,v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-9),-1,1)))
        ratio=np.linalg.norm(p[base+3]-p[0])/max(np.linalg.norm(p[base+1]-p[0]),1e-6)
        extended+=int(angle>150 and ratio>1.08)
        closed+=int(angle<125 or ratio<.9)
    return dict(pinch=pinch,extended=extended,closed=closed,
                thumb_closed=bool(np.linalg.norm(p[4]-p[5])/palm<.55))


class ActionCycle:
    """Require open -> closed -> open; hysteresis, debounce, timeout, no repeats."""
    def __init__(self,debounce=.12,timeout=2.5):
        self.debounce,self.timeout=debounce,timeout
        self.reset()

    def reset(self):
        self.phase='waiting_open'; self.observation=None; self.since=0.; self.started=None

    def update(self,state,now):
        if self.started is not None and now-self.started>self.timeout:
            self.reset()
        if state is None:
            self.observation=None; return False
        if state!=self.observation:
            self.observation=state; self.since=now
        if now-self.since<self.debounce: return False
        if self.phase=='waiting_open' and state=='open': self.phase='ready'
        elif self.phase=='ready' and state=='closed':
            self.phase='closed'; self.started=now
        elif self.phase=='closed' and state=='open':
            self.phase='ready'; self.started=None
            return True
        return False


class ActionRecognizer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tracks={}; self.last_time=None
        self.status='等待展开手，再捏合或握拳并释放'

    def update(self,hands,now):
        if self.last_time is not None and now-self.last_time>.35:
            self.reset()
        self.last_time=now
        ids={h.track_id for h in hands}
        self.tracks={k:v for k,v in self.tracks.items() if k in ids}
        events=[]; statuses=[]
        for hand in hands:
            track=self.tracks.setdefault(hand.track_id,dict(pinch=ActionCycle(),fist=ActionCycle(),history=deque(maxlen=90)))
            m=hand_metrics(hand.points)
            track['history'].append(dict(time=float(now),**m))
            # In live mode use the trained vocabulary: 9 is the operator's fist.
            # Geometry remains available for explicit synthetic/unclassified inputs.
            fist=hand.digit==FIST_DIGIT if hand.digit is not None else m['closed']==4 and m['thumb_closed']
            fist_state='closed' if fist else ('open' if m['extended']>=3 else None)
            if fist or m['closed']==4: track['pinch'].reset()
            # A fist can also bring thumb/index close; do not emit both actions.
            pinch_state=None if fist or m['closed']==4 else ('closed' if m['pinch']<.32 else ('open' if m['pinch']>.62 else None))
            for name,state in [('pinch',pinch_state),('fist',fist_state)]:
                if track[name].update(state,now):
                    events.append(dict(action=name,track_id=int(hand.track_id),monotonic_time=float(now),samples=len(track['history'])))
            phases={'waiting_open':'先展开','ready':'已就绪','closed':'等待释放'}
            statuses.append(f"ID {hand.track_id} 捏合:{phases[track['pinch'].phase]} / 握拳:{phases[track['fist'].phase]}")
        self.status='；'.join(statuses) if statuses else '未检测到手；动作计时已清空'
        return events


class SequenceRecorder:
    """User-labelled two-second motion samples, independent from static photos."""
    def __init__(self):
        import uuid
        self.active=None; self.session=uuid.uuid4().hex[:12]

    def start(self,label,now):
        if label not in ('pinch','fist','idle'): raise ValueError('Invalid action label')
        if self.active is not None: raise ValueError('当前动作仍在录制')
        self.active=dict(label=label,start=now,track=None,frames=[],times=[])

    def step(self,hands,now,path):
        if self.active is None: return None
        from pathlib import Path
        from .core import features
        import json,uuid
        from datetime import datetime,timezone
        row=self.active
        if len(hands)!=1 or (row['track'] is not None and hands[0].track_id!=row['track']):
            self.active=None
            return '动作样本已取消：请只保留一只手，并全程保持可见'
        row['track']=hands[0].track_id
        row['frames'].append(features(hands[0].points).tolist())
        row['times'].append(float(now-row['start']))
        if now-row['start']<2: return None
        self.active=None
        if len(row['frames'])<10 or max(np.diff(row['times']),default=0)>.35:
            return '动作样本已取消：有效帧不足或视频中断'
        # Reference repo resamples each gesture sequence to 20 frames.
        times=np.array(row['times']); vectors=np.array(row['frames'])
        grid=np.linspace(times[0],times[-1],20)
        sequence=np.stack([np.interp(grid,times,vectors[:,i]) for i in range(63)],axis=-1)
        record=dict(id=uuid.uuid4().hex,label=row['label'],time=datetime.now(timezone.utc).isoformat(),
                    session=self.session,physical_hand='unspecified',operator='unspecified',
                    features=sequence.tolist(),relative_times=grid.tolist(),original_frame_count=len(times),
                    duration=float(times[-1]-times[0]),source='live_action_recording',track_id=int(row['track']))
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a',encoding='utf-8') as stream: stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        return f"动作样本已保存：{row['label']}，{len(times)} 帧 → 20 帧；请改变姿态再录下一轮"
