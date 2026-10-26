"""MediaPipe single-sample collection: select label, pose, press Space/S."""
import bootstrap
import argparse
import time
import uuid
import cv2
import numpy as np
from lg.vision import Hands, open_camera, draw
from lg.data import sample_counts, save_snapshot
from lg.labels import LABELS,ACTIVE_DATASET


def main():
    p=argparse.ArgumentParser(description='One key press saves one MediaPipe hand sample')
    p.add_argument('--operator',default='p01',help='匿名操作员编号；用于分组验证')
    p.add_argument('--session',default=None,help='每次独立采集使用新编号')
    p.add_argument('--camera',default='1')
    p.add_argument('--label',default='1',choices=LABELS)
    p.add_argument('--hand',choices=['left','right'],help='人的实际左/右手，不是画面位置；也可用 L/R 键选择')
    p.add_argument('--side',choices=['left','right','both'],default='both',help='仅筛选画面半区；通常无需设置')
    a=p.parse_args()
    session=a.session or uuid.uuid4().hex[:10]
    path=bootstrap.ROOT/ACTIVE_DATASET
    counts=sample_counts(path)
    detector=Hands(bootstrap.ROOT,classify=False)
    cap=None
    window='MediaPipe hand-back collection / SPACE or S = save ONE'
    message='Select label 1-9/U/O and physical hand L/R; pose then SPACE/S'
    message_until=0
    saved=0
    success=False
    try:
        cap,frame,_=open_camera(a.camera)
        cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window,1100,850)
        while True:
            hands=detector.process(frame)
            view=draw(frame,hands)
            hand=a.hand or 'CHOOSE L/R'
            color=(80,230,80) if success else (80,140,255)
            if time.monotonic() < message_until:
                cv2.rectangle(view,(1,1),(view.shape[1]-2,view.shape[0]-2),color,6)
            panel=np.full((205,max(900,view.shape[1]),3),(27,23,18),dtype=np.uint8)
            labels=LABELS
            lines=[f'LABEL: {a.label} | PHYSICAL HAND: {hand} | saved this run: {saved}',
                   '1-9: label (9=fist) | U: I Love U | O: ok | L/R: hand | SPACE/S: save | Q: exit',
                   f'Session: {session} | Operator: {a.operator} | Vary pose between samples',
                   'LEFT  '+ ' '.join(f'{label}:{counts[("left",label)]}' for label in labels),
                   'RIGHT '+ ' '.join(f'{label}:{counts[("right",label)]}' for label in labels),
                   message]
            for i,line in enumerate(lines):
                cv2.putText(panel,line,(12,26+i*31),cv2.FONT_HERSHEY_SIMPLEX,.47,color if i==5 else (230,235,240),1,cv2.LINE_AA)
            display=np.vstack([cv2.copyMakeBorder(view,0,0,0,panel.shape[1]-view.shape[1],cv2.BORDER_CONSTANT),panel])
            cv2.imshow(window,display)
            key=cv2.waitKey(1)&255
            if key==ord('q') or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1: break
            if ord('1')<=key<=ord('9'): a.label=chr(key)
            elif key==ord('u'): a.label='I Love U'
            elif key==ord('o'): a.label='ok'
            elif key==ord('l'): a.hand='left'
            elif key==ord('r'): a.hand='right'
            elif key in (32,ord('s')):
                try:
                    row=save_snapshot(path,hands,a.label,a.operator,session,a.hand,a.side)
                    counts[(row['physical_hand'],row['label'])]+=1
                    saved+=1
                    success=True
                    message=f'SAVED +1 | {a.hand} / {a.label} | total {counts[(a.hand,a.label)]}'
                except (ValueError,OSError) as e:
                    success=False
                    message=f'NOT SAVED: {e}'
                message_until=time.monotonic()+1.2
                print(message,flush=True)
            ok,frame=cap.read()
            if not ok: raise RuntimeError('Camera/video input ended')
    finally:
        detector.close()
        if cap: cap.release()
        cv2.destroyAllWindows()
    print(f'Saved {saved} samples to {path}; session={session}')

if __name__=='__main__': main()
