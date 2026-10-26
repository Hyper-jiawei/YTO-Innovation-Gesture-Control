import argparse
import queue
import threading
import time
from pathlib import Path
import tkinter as tk
from .core import Engine
from .vision import Hands, Detector, open_camera, crop, draw, draw_detections
from .actions import ActionRecognizer, SequenceRecorder
from .business import Station
from .demo import scene, observations, action_observations
from .barcode import ScanWindow
from .live_detection import LiveDetector,DamageDisplay

MODES = {'扫码框选':'scan','目标拍照':'photo','破损件上报':'damage','格口绑定':'command','动作识别':'action'}


class App:
    def __init__(self, args):
        self.args = args
        self.root_path = Path(__file__).resolve().parents[1]
        self.root = tk.Tk()
        self.root.title('YTO手势隔空操作系统' + (' | 离线模拟' if args.demo else ' | 实时识别'))
        width=min(1480,self.root.winfo_screenwidth()-80)
        height=min(980 if args.demo else 920,self.root.winfo_screenheight()-100)
        self.root.geometry(f'{width}x{height}')
        self.root.minsize(1080,740)
        self.commands, self.frames = queue.Queue(), queue.Queue(maxsize=2)
        self.stop = threading.Event()
        self.last_event = None
        self.shown = 0
        self.make_ui()
        self.worker = threading.Thread(target=self.run_worker,daemon=True)
        self.worker.start()
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.poll_timer=self.root.after(30,self.poll)
        self.smoke_timer=None
        if args.smoke:
            self.smoke_timer=self.root.after(int(args.smoke*1000),self.close)

    def make_ui(self):
        from .ui import build_ui
        build_ui(self, MODES)

    def publish(self, packet):
        if self.frames.full():
            try: self.frames.get_nowait()
            except queue.Empty: pass
        self.frames.put_nowait(packet)

    def run_worker(self):
        cap = detector = hands_model = live = None
        try:
            station = Station(self.root_path, self.root_path/'outputs/demo' if self.args.demo else None)
            detector = Detector(self.args.yolo)
            live=LiveDetector(detector,prewarm=not self.args.demo)
            engine = Engine(self.args.dwell,self.args.hold_tolerance,self.args.hold_iou,self.args.min_roi_pixels)
            mode = self.args.mode
            demo_digits = (1,1)
            demo_action='open'
            scanner=ScanWindow()
            actions=ActionRecognizer(); recorder=SequenceRecorder()
            display=DamageDisplay(self.args.hold_tolerance+.02)
            pending_evidence={}
            action_count=0
            source = '离线模拟（视觉输入为合成数据）'

            def save_damage(roi,evidence_frame,detections,note=None):
                if note is None:
                    labels=', '.join(f"{d['label']} {d['confidence']:.0%}" for d in detections)
                    note=f'{detector.model_name} / {detector.device}: '+(labels or '未检测到已知目标，保留人工取证')
                annotated=draw_detections(evidence_frame,detections)
                image,_=crop(annotated,roi)
                station.evidence(image,'damage',roi,note,detections)

            def start_belt():
                if pending_evidence: station.emit('blocked','破损件检测尚未完成，请等待取证完成后重做两次 OK')
                else: station.start()

            def cancel_pending(note):
                for record in pending_evidence.values():
                    save_damage(record['roi'],record['frame'],[],note)
                pending_evidence.clear()

            if self.args.demo:
                base = scene()
            else:
                hands_model = Hands(self.root_path)
                cap,base,index = open_camera(self.args.camera,self.args.camera_width,self.args.camera_height)
                source = f'摄像头/视频 {index} · {base.shape[1]}×{base.shape[0]} | 关键点 CPU | 数字 {hands_model.classifier.device}'
            previous = time.monotonic()
            while not self.stop.is_set():
                while not self.commands.empty():
                    action,value = self.commands.get_nowait()
                    if action=='mode':
                        if mode!=value:
                            scanner.cancel(); actions.reset(); display.clear()
                            recorder.active=None
                            cancel_pending('切换模式取消检测，保留人工框选证据')
                        mode=value
                    elif action=='demo':
                        demo_digits=value
                        if value is None: demo_action=None
                        elif value==(5,5): demo_action='open'
                        elif value==(9,9): demo_action='fist'
                    elif action=='demo_action': demo_action=value
                    elif action=='reset': station.reset()
                    elif action=='ack': station.ack()
                    elif action=='estop': station.stop()
                    elif action=='start': start_belt()
                    elif action=='bind_gate': station.bind_gate(value)
                    elif action=='record_action' and mode=='action':
                        if self.args.demo:
                            station.emit('info','离线模拟不保存真实动作训练样本；请使用摄像头模式')
                        elif recorder.active is None:
                            recorder.start(value,time.monotonic())
                            station.emit('info','开始录制 2 秒：仅一只手入镜，完成完整动作后释放')
                live.activate(mode=='damage')
                if self.args.demo:
                    frame = base.copy()
                    hands = action_observations(demo_action) if mode=='action' else (observations(demo_digits) if demo_digits is not None else [])
                else:
                    frame=base
                    hands=hands_model.process(frame)
                now=time.monotonic()
                if mode=='action':
                    for action_event in actions.update(hands,now):
                        action_name={'pinch':'捏合','fist':'握拳'}[action_event['action']]
                        action_count+=1
                        station.emit('action',f"动作轨迹已记录：{action_name} · 累计 {action_count}",**action_event)
                    recording_result=recorder.step(hands,now,self.root_path/'data/actions.jsonl')
                    if recording_result: station.emit('action_sample',recording_result)
                events=engine.update(hands,mode,now,frame_size=(frame.shape[1],frame.shape[0]))
                # Start inference on mode entry, before the dwell selection fires.
                if mode=='damage': live.submit(frame,engine.roi,now)
                for kind,value in events:
                    if kind=='stop': station.halt('已执行：双手握拳保持 1 秒，传送带停止（模拟）')
                    elif kind=='start': start_belt()
                    elif kind=='bind_gate': station.bind_gate(value)
                    elif kind=='damage':
                        station.halt('传送带停止：破损件取证中（模拟）')
                        display.select(value,engine.pair)
                        snapshot=live.snapshot(value,now)
                        if snapshot is not None:
                            save_damage(value,snapshot.frame,snapshot.detections)
                        else:
                            token=live.request(frame,value,now)
                            if token is None:
                                save_damage(value,frame,[],live.error or '检测不可用，保留人工取证')
                            else:
                                pending_evidence[token]=dict(roi=value,frame=frame.copy())
                    elif kind=='scan':
                        scanner.start(value,now)
                        station.emit('scan_pending','区域已锁定，正在连续扫码；可移开手指，保持标签清晰')
                    else:
                        roi,note=live.refine(frame,value)
                        image,_=crop(frame,roi)
                        station.evidence(image,kind,roi,note)
                for token,record in list(pending_evidence.items()):
                    result=live.take(token)
                    if result is None: continue
                    if isinstance(result,str):
                        save_damage(record['roi'],record['frame'],[],result)
                    else:
                        save_damage(record['roi'],result.frame,result.detections)
                    pending_evidence.pop(token)
                scanner.step(frame,now,station)
                preview_image=station.last_scan_image
                preview_revision=('scan',station.scan_revision)
                if mode=='photo':
                    preview_image=station.last_photo_image
                    preview_revision=('photo',station.photo_revision)
                clear_preview=False;damage_message=None;visible=False;selected_detections=[]
                rendered=draw(frame,hands,scanner.roi or engine.roi,engine.dwell.progress,self.args.dwell)
                if mode=='damage':
                    visible=display.visible(engine.roi,engine.pair)
                    preview_image=None;clear_preview=True
                    damage_message='等待双指框选；移开手指隐藏识别结果'
                    if visible:
                        snapshot=live.snapshot(display.roi,now)
                        if snapshot is not None:
                            selected_detections=snapshot.detections
                            rendered=draw_detections(rendered,selected_detections)
                            preview_image,_=crop(draw_detections(snapshot.frame,selected_detections),display.roi)
                            preview_revision=snapshot.completed_at
                            clear_preview=False
                            labels=' / '.join(f"{d['label']} {d['confidence']:.0%}" for d in selected_detections)
                            damage_message=labels or '当前区域未检测到已知目标'
                        else: damage_message='YOLO 正在预热/更新，请保持框选'
                    if live.error: damage_message=f'YOLO 检测异常：{live.error}'
                fps=1/max(now-previous,1e-3); previous=now
                timing=live.latest.inference_ms if mode=='damage' and live.latest is not None else None
                yolo_info=f'{detector.model_name} {detector.device}'
                if mode=='damage': yolo_info+=f' / {timing:.0f}ms' if timing is not None else ' / 预热中'
                packet=dict(frame=rendered,info=f'{source} | {fps:.1f} FPS | {yolo_info}',
                            digits=' / '.join(str(h.digit) if h.digit is not None else '?' for h in hands) or '等待双手',
                            bound_gate=station.bound_gate,last_command=station.last_command,
                            control_name=engine.control_name,control_progress=engine.control_progress,gate_mode=mode=='command',
                            action_mode=mode=='action',action_status=actions.status if mode=='action' else '',
                            damage_mode=mode=='damage',damage_visible=visible,damage_message=damage_message,
                            detections=selected_detections,clear_preview=clear_preview,
                            recording=recorder.active is not None,
                            progress=engine.dwell.progress,running=station.running,estop=station.estop,
                            count=len(station.seen),pending=len(station.pending),events=station.events[:],
                            scan_image=preview_image,scan_revision=preview_revision,preview_mode=mode,scanning=scanner.roi is not None)
                self.publish(packet)
                if cap:
                    ok,base=cap.read()
                    if not ok: raise RuntimeError('视频结束或摄像头读取中断；模拟传送带已停止')
                else:
                    self.stop.wait(.03)
        except Exception as e:
            print(f'WORKER_ERROR {type(e).__name__}: {e}',flush=True)
            self.publish(dict(error=f'{type(e).__name__}: {e}'))
        finally:
            if live: live.close()
            if 'cancel_pending' in locals(): cancel_pending('程序关闭，保留人工框选证据')
            if cap: cap.release()
            if hands_model: hands_model.close()
            if 'station' in locals() and station.running:
                station.halt('传送带停止：程序关闭（模拟）')

    def poll(self):
        from .ui import render_packet
        self.poll_timer=None
        packet=None
        while not self.frames.empty(): packet=self.frames.get_nowait()
        if packet: render_packet(self,packet)
        if not self.stop.is_set(): self.poll_timer=self.root.after(40,self.poll)

    def close(self):
        if self.stop.is_set(): return
        self.stop.set()
        for timer in (self.poll_timer,self.smoke_timer):
            if timer is not None: self.root.after_cancel(timer)
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        self.worker.join(timeout=5)
        if self.args.smoke:
            print(f'UI_SMOKE rendered_frames={self.shown}')
            if not self.shown: raise RuntimeError('UI smoke failed: no preview rendered')


def main(default_mode='scan'):
    parser=argparse.ArgumentParser(description='YTO手势隔空操作系统：统一启动，无后台服务')
    parser.add_argument('--camera',default='1',help='设备编号或本地视频路径（默认 1）；auto 自动尝试 0–3')
    parser.add_argument('--camera-width',type=int,default=1920,help='请求摄像头宽度；实际值显示于界面')
    parser.add_argument('--camera-height',type=int,default=1080,help='请求摄像头高度；不支持时使用相机实际能力')
    parser.add_argument('--demo',action='store_true',help='明确标识的离线合成输入')
    parser.add_argument('--mode',choices=list(MODES.values()),default=default_mode)
    parser.add_argument('--dwell',type=float,default=1.1,help='停留触发秒数，默认 1.1')
    parser.add_argument('--min-roi-pixels',type=int,default=4,help='框选宽高的最小像素，默认 4')
    parser.add_argument('--hold-tolerance',type=float,default=.06,help='指尖相对起点允许波动，默认 0.06')
    parser.add_argument('--hold-iou',type=float,default=.35,help='停留时手框与起始框的最小 IoU，默认 0.35')
    parser.add_argument('--yolo',help='可选本地检测权重；不自动下载')
    parser.add_argument('--no-yolo',action='store_true',help='禁用自动加载的通用检测模型')
    parser.add_argument('--smoke',type=float,default=0,help='UI 冒烟验证秒数后关闭')
    args=parser.parse_args()
    bundled = Path(__file__).resolve().parents[1]/'models/yolo12n.pt'
    if args.no_yolo:
        args.yolo = None
    elif not args.yolo and not args.demo and bundled.exists():
        args.yolo = str(bundled)
    if args.dwell < .3: parser.error('--dwell 应至少为 0.3 秒')
    if not 0 < args.hold_tolerance <= .2: parser.error('--hold-tolerance 应在 (0, 0.2] 内')
    if not 0 <= args.hold_iou <= 1: parser.error('--hold-iou 应在 [0, 1] 内')
    if args.min_roi_pixels < 1: parser.error('--min-roi-pixels 应至少为 1')
    if min(args.camera_width,args.camera_height)<1: parser.error('摄像头分辨率必须大于零')
    App(args).run()
