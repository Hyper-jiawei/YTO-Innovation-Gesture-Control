"""Shared operator controls and timing, independent of camera and UI."""
STOP_SECONDS=1.0
START_SECONDS=.2
GATE_IDS=(2,3,4,5)
OPERATOR_GUIDE=(
    ('停止传送带','双手握拳保持 1 秒'),
    ('开启传送带','双手连续两次 OK（中间松开，3 秒内）'),
    ('绑定摆臂格口','在格口模式，双手相同数字 2–5'),
)


class HoldTrigger:
    def __init__(self,seconds):
        self.seconds=seconds;self.reset()

    def reset(self):
        self.key=None;self.since=None;self.fired=False;self.progress=0.

    def update(self,key,now):
        if key is None:
            self.reset();return False
        if key!=self.key:
            self.reset();self.key=key;self.since=now
        self.progress=min(1.,max(0.,(now-self.since)/self.seconds))
        if self.progress>=1 and not self.fired:
            self.fired=True;return True
        return False


class DoubleOKTrigger:
    """Two debounced OK poses, separated by a visible release, within 3 seconds."""
    def __init__(self):
        self.hold=HoldTrigger(START_SECONDS)
        self.release=HoldTrigger(.15)
        self.reset()

    def reset(self):
        self.hold.reset();self.release.reset()
        self.phase=0;self.first_at=None;self.progress=0.

    def update(self,active,now):
        if self.phase in (1,2) and now-self.first_at>3:
            self.reset()
        if self.phase==3:
            if not active: self.reset()
            return False
        if self.phase==1:
            if self.release.update(True if not active else None,now):
                self.phase=2;self.hold.reset()
            self.progress=.5
            return False
        fired=self.hold.update(True if active else None,now)
        self.progress=(.5 if self.phase==2 else 0)+self.hold.progress*.5
        if fired:
            if self.phase==0:
                self.phase=1;self.first_at=now;self.progress=.5
            else:
                self.phase=3;self.progress=1.
                return True
        return False
