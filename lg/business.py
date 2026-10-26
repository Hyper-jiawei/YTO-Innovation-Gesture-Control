"""Local simulator and durable evidence/outbox. No PLC/network writes."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import cv2
from .barcode import decode_barcodes
from .controls import GATE_IDS


class Station:
    def __init__(self, root, output_dir=None):
        self.folder = Path(output_dir) if output_dir else Path(root)/'outputs'
        self.folder.mkdir(exist_ok=True,parents=True)
        self.running = False
        self.estop = False
        self.seen = set()
        self.pending = {}
        self.events = []
        self.last_barcode = ''
        self.bound_gate=None
        self.last_command='等待操作指令'
        self.last_photo_image=None
        self.photo_revision=0
        self.last_scan_image=None
        self.last_scan_info={}
        self.scan_revision=0
        # Reload audit state: duplicate scans stay duplicates after restart.
        log = self.folder/'events.jsonl'
        if log.exists():
            for line in log.read_text(encoding='utf-8').splitlines():
                try:
                    event = json.loads(line)
                    if event['kind']=='scan':
                        self.seen.add(event['text'])
                    elif event['kind']=='damage':
                        self.pending[event['id']] = event
                    elif event['kind']=='bind_gate' and event.get('gate') in GATE_IDS:
                        self.bound_gate=event['gate']
                    elif event['kind']=='ack':
                        self.pending.pop(event['report_id'],None)
                except (ValueError,KeyError):
                    continue

    def emit(self, kind, message, **fields):
        event = dict(time=datetime.now(timezone.utc).isoformat(),kind=kind,message=message,**fields)
        with (self.folder/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(event,ensure_ascii=False)+'\n')
        self.events.append(event)
        if kind in ('start','stop','estop','reset','blocked','bind_gate','invalid_code'):
            self.last_command=message
        return message

    def stop(self):
        self.running = False
        self.estop = True
        return self.emit('estop','人工急停 / 急停已锁定（模拟）')

    def halt(self,message='传送带停止（模拟）'):
        self.running=False
        return self.emit('stop',message)

    def reset(self):
        self.estop = False
        self.running = False
        return self.emit('reset','急停已人工复位，传送带保持停止')

    def start(self):
        if self.estop:
            return self.emit('blocked','启动被急停锁定拦截')
        self.running=True
        return self.emit('start','传送带启动（模拟）')

    def bind_gate(self,gate):
        if type(gate) is not int or gate not in GATE_IDS:
            return self.emit('invalid_code','格口未绑定：仅支持 2–5 号摆臂格口')
        self.bound_gate=gate
        return self.emit('bind_gate',f'已绑定 {gate} 号摆臂格口（模拟）',gate=gate)

    def command(self,code):
        if isinstance(code,str) and len(code)==2 and code[0]==code[1] and code[0] in '2345':
            return self.bind_gate(int(code[0]))
        return self.emit('invalid_code','格口未绑定：请双手比出相同数字 2–5')

    def scan(self, image, final=True, content_region=None):
        codes,info=decode_barcodes(image,content_region)
        texts=sorted({code['text'] for code in codes})
        decoder=info['decoder']
        self.last_scan_image=image.copy()
        self.last_scan_info=info
        self.scan_revision+=1
        if not texts:
            return self.scan_failed(image) if final else None
        for text in texts:
            self.last_barcode = text
            if text in self.seen:
                self.emit('duplicate',f'重复扫码，不重复计数: {text}',text=text)
            else:
                code=next(c for c in codes if c['text']==text)
                self.emit('scan',f'扫码成功: {text}',text=text,decoder=decoder,format=code['format'],method=code['method'])
                self.seen.add(text)
        return self.events[-1]['message']

    def scan_failed(self,image,roi=None,frames=1):
        folder=self.folder/'scans'
        folder.mkdir(parents=True,exist_ok=True)
        ok,encoded=cv2.imencode('.png',image)
        if ok: encoded.tofile(str(folder/'last_failed_roi.png'))
        info=dict(self.last_scan_info,frames=frames,roi=[float(v) for v in roi] if roi else None,
                  diagnostic_image='scans/last_failed_roi.png')
        (folder/'last_failed_roi.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        size=f"{info.get('width',0)}×{info.get('height',0)}"
        return self.emit('scan_miss',f'未解码（区域 {size}，已重试 {frames} 帧）。请框全条码并留白，移近/对焦后重框。',**info)

    def evidence(self, image, kind, roi, detector_note='', detections=None):
        # Also accept NumPy coordinates from detectors/direct module callers.
        roi = [float(value) for value in roi]
        identity = uuid.uuid4().hex[:12]
        path = self.folder/f'{kind}_{identity}.jpg'
        ok, encoded = cv2.imencode('.jpg',image)
        if not ok:
            raise RuntimeError('取证图片编码失败')
        encoded.tofile(str(path))  # supports Chinese Windows paths
        if kind=='photo':
            self.last_photo_image=image.copy()
            self.photo_revision+=1
        if kind=='damage' and self.running:
            self.halt('传送带停止：破损件取证（模拟）')
        message = '异常件已取证并隔离，等待本地确认' if kind=='damage' else '目标区域拍照已保存'
        event = self.emit(kind,message + (f'；{detector_note}' if detector_note else ''),
                          id=identity,image=path.name,roi=roi,barcode=self.last_barcode,
                          status='pending_local' if kind=='damage' else 'saved',detector=detector_note,
                          detections=detections or [],bbox_coordinates='normalized_full_frame')
        if kind=='damage':
            report = self.events[-1].copy()
            (self.folder/f'report_{identity}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            self.pending[identity]=report
        return event

    def ack(self):
        if not self.pending:
            return self.emit('info','没有待确认的异常件')
        identity = next(iter(self.pending))
        self.emit('ack','异常件已本地确认归档（未上传企业系统）',report_id=identity)
        path = self.folder/f'report_{identity}.json'
        report = json.loads(path.read_text(encoding='utf-8'))
        report['status']='acknowledged_local'
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        self.pending.pop(identity)
        return self.events[-1]['message']
