import sys,time,json,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import numpy as np
from lg.live_detection import LiveDetector,DetectionSnapshot,DamageDisplay,inside_region
from lg.core import Engine
from lg.demo import observations
from lg.business import Station


def wait_for(predicate,timeout=2):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=predicate()
        if value: return value
        time.sleep(.005)
    raise AssertionError('Background result timeout')


class FakeDetector:
    model_name='yolo12n.pt';device='cpu';model=object()
    def __init__(self): self.calls=[]
    def detect_region(self,frame,roi):
        self.calls.append((int(frame[0,0,0]),tuple(roi),time.monotonic()))
        return [dict(label='book',class_id=73,confidence=.91,bbox=[.35,.3,.6,.6])]


class LiveDetectionTests(unittest.TestCase):
    def test_warmup_result_is_never_published_as_camera_evidence(self):
        detector=FakeDetector();live=LiveDetector(detector,prewarm=True)
        try:
            self.assertTrue(live.warmup_done.wait(1))
            self.assertEqual(len(detector.calls),1)
            self.assertIsNone(live.latest)
            live.activate(True)
            self.assertIsNone(live.snapshot((0,0,1,1),time.monotonic()))
            live.submit(np.full((10,10,3),3,np.uint8))
            result=wait_for(lambda:live.snapshot((0,0,1,1),time.monotonic()))
            self.assertEqual(result.frame[0,0,0],3)
        finally: live.close()

    def test_mode_entry_preheats_and_switch_off_invalidates(self):
        detector=FakeDetector();live=LiveDetector(detector,interval=.001)
        try:
            frame=np.zeros((100,100,3),np.uint8)
            live.submit(frame);time.sleep(.02)
            self.assertEqual(detector.calls,[])
            live.activate(True);live.submit(frame)
            snapshot=wait_for(lambda:live.snapshot((.2,.2,.8,.8),time.monotonic()))
            self.assertEqual(snapshot.detections[0]['label'],'book')
            self.assertIsNone(live.snapshot((.2,.2,.8,.8),snapshot.captured_at+1))
            live.activate(False)
            self.assertIsNone(live.snapshot((.2,.2,.8,.8),time.monotonic()))
            count=len(detector.calls);live.submit(frame);time.sleep(.02)
            self.assertEqual(len(detector.calls),count)
        finally: live.close()

    def test_latest_frame_replaces_backlog(self):
        started=threading.Event();release=threading.Event()
        class Slow(FakeDetector):
            def detect_region(self,frame,roi):
                if not self.calls: started.set();release.wait(2)
                return super().detect_region(frame,roi)
        detector=Slow();live=LiveDetector(detector,interval=.001)
        try:
            live.activate(True);live.submit(np.zeros((10,10,3),np.uint8))
            self.assertTrue(started.wait(1))
            for n in range(1,100): live.submit(np.full((10,10,3),n,np.uint8))
            release.set();wait_for(lambda:len(detector.calls)==2)
            self.assertEqual([row[0] for row in detector.calls],[0,99])
        finally: release.set();live.close()

    def test_old_inflight_result_cannot_cross_mode_generation(self):
        first=threading.Event();second=threading.Event();release1=threading.Event();release2=threading.Event()
        class Slow(FakeDetector):
            def detect_region(self,frame,roi):
                if frame[0,0,0]==1: first.set();release1.wait(2)
                else: second.set();release2.wait(2)
                return super().detect_region(frame,roi)
        live=LiveDetector(Slow(),interval=.001)
        try:
            live.activate(True);live.submit(np.full((10,10,3),1,np.uint8));self.assertTrue(first.wait(1))
            live.activate(False);live.activate(True);live.submit(np.full((10,10,3),2,np.uint8))
            release1.set();self.assertTrue(second.wait(1))
            self.assertIsNone(live.latest)
            release2.set();wait_for(lambda:live.latest)
            self.assertEqual(live.latest.frame[0,0,0],2)
        finally: release1.set();release2.set();live.close()

    def test_evidence_job_retains_original_frame_and_roi(self):
        live=LiveDetector(FakeDetector())
        try:
            live.activate(True);frame=np.full((100,100,3),4,np.uint8);roi=(.2,.2,.8,.8)
            token=live.request(frame,roi,time.monotonic());frame[:]=9
            result=wait_for(lambda:live.take(token))
            self.assertEqual(result.frame[0,0,0],4);self.assertEqual(result.roi,roi)
            self.assertIsNone(live.take(token))
        finally: live.close()

    def test_detection_error_does_not_publish_stale_result(self):
        class Broken(FakeDetector):
            def detect_region(self,frame,roi): raise RuntimeError('test model failure')
        live=LiveDetector(Broken())
        try:
            live.activate(True);token=live.request(np.zeros((10,10,3),np.uint8),(0,0,1,1),time.monotonic())
            message=wait_for(lambda:live.take(token))
            self.assertIn('test model failure',message)
            self.assertIsNone(live.snapshot((0,0,1,1),time.monotonic()))
        finally: live.close()

    def test_selection_clear_on_loss_move_or_new_identity(self):
        for roi,pair in [(None,(1,2)),((.6,.6,.9,.9),(1,2)),((.2,.2,.8,.8),(8,9))]:
            display=DamageDisplay();display.select((.2,.2,.8,.8),(1,2))
            self.assertTrue(display.visible((.21,.2,.79,.81),(1,2)))
            self.assertFalse(display.visible(roi,pair));self.assertIsNone(display.roi)

    def test_roi_filter_rejects_background_and_unrelated_crop(self):
        boxes=[dict(label='book',bbox=[.3,.3,.5,.5]),dict(label='person',bbox=[.8,.1,.9,.2])]
        self.assertEqual([r['label'] for r in inside_region(boxes,(.2,.2,.6,.6))],['book'])
        live=LiveDetector(FakeDetector())
        try:
            live.activate(True);now=time.monotonic()
            live.latest=DetectionSnapshot(None,(.1,.1,.3,.3),boxes,now,now,10)
            self.assertIsNone(live.snapshot((.6,.6,.8,.8),now))
        finally: live.close()


class ConveyorGestureTests(unittest.TestCase):
    def test_double_ok_all_modes_once_until_release(self):
        for mode in ('scan','photo','damage','command','action'):
            engine=Engine();events=[]
            sequence=[('ok','ok')]*5+[(5,5)]*3+[('ok','ok')]*12
            for i,digits in enumerate(sequence): events+=engine.update(observations(digits),mode,i*.1)
            self.assertEqual(events,[('start',None)])
            engine.update(observations((5,5)),mode,2.)
            for i,digits in enumerate(sequence): events+=engine.update(observations(digits),mode,2.1+i*.1)
            self.assertEqual(events,[('start',None),('start',None)])

    def test_single_ok_unknown_or_hand_loss_cannot_start(self):
        for digits in [('ok',5),('ok',None),(None,None)]:
            engine=Engine()
            for i in range(10): self.assertFalse(engine.update(observations(digits),'damage',i*.1))
        engine=Engine();engine.update(observations(('ok','ok')),'scan',0)
        engine.update([],'scan',.2)
        self.assertFalse(engine.update(observations(('ok','ok')),'scan',.4))
        self.assertFalse(engine.update(observations(('ok','ok')),'scan',1.2))

    def test_one_fist_cannot_stop(self):
        engine=Engine()
        for i in range(20):
            self.assertFalse(engine.update(observations((9,5)),'scan',i*.1))

    def test_gesture_stop_does_not_lock_and_ok_restarts(self):
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            station=Station(bootstrap.ROOT,folder);engine=Engine();station.start()
            for i,digits in enumerate([(9,9)]*12+[(5,5)]*3+[('ok','ok')]*4+[(5,5)]*3+[('ok','ok')]*4):
                for kind,_ in engine.update(observations(digits),'scan',i*.1):
                    if kind=='stop': station.halt('双手握拳1秒：停止')
                    elif kind=='start': station.start()
            self.assertFalse(station.estop);self.assertTrue(station.running)
            self.assertEqual([e['kind'] for e in station.events],['start','stop','start'])
            station.stop();station.start();self.assertFalse(station.running)


class AppLiveFlowTests(unittest.TestCase):
    def args(self,mode):
        return SimpleNamespace(demo=True,mode=mode,yolo=None,dwell=.3,hold_tolerance=.06,hold_iou=.35,
                               min_roi_pixels=4,smoke=0)

    def test_detection_before_selection_and_hide_after_hand_leaves(self):
        from lg.app import App
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            detector=FakeDetector();station=Station(bootstrap.ROOT,folder);packets=[];packet_times=[]
            with patch('lg.app.Detector',return_value=detector),patch('lg.app.Station',return_value=station):
                app=App(self.args('damage'));original_publish=app.publish
                def publish(packet): packets.append(packet);packet_times.append(time.monotonic());original_publish(packet)
                app.publish=publish
                app.root.after(750,lambda:app.commands.put(('demo',None)))
                app.root.after(1150,app.close);app.run()
            self.assertGreater(len(detector.calls),2)
            self.assertTrue(any(p.get('damage_visible') and p.get('detections') for p in packets))
            self.assertTrue(any(p.get('clear_preview') and not p.get('damage_visible') and p.get('scan_image') is None for p in packets[-5:]))
            reports=[e for e in station.events if e['kind']=='damage']
            self.assertEqual(len(reports),1);self.assertEqual(reports[0]['detections'][0]['label'],'book')
            self.assertLess(detector.calls[0][2],packet_times[next(i for i,p in enumerate(packets) if p.get('damage_visible'))])

    def test_window_controls_start_stop_start_without_manual_reset(self):
        from lg.app import App
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            station=Station(bootstrap.ROOT,folder)
            with patch('lg.app.Station',return_value=station):
                app=App(self.args('command'))
                latest=[None];published=[];original=app.publish
                def publish(packet):
                    latest[0]=packet;published.append(time.monotonic());original(packet)
                app.publish=publish
                app.commands.put(('demo',('ok','ok')))
                app.root.after(400,lambda:app.commands.put(('demo',(5,5))))
                app.root.after(700,lambda:app.commands.put(('demo',('ok','ok'))))
                state=dict(phase=0,frames=0,at=time.monotonic())
                deadline=time.monotonic()+8
                def drive():
                    controls=[e['kind'] for e in station.events if e['kind'] in ('start','stop','estop')]
                    phase=state['phase']
                    if phase==0 and controls==['start']:
                        app.commands.put(('demo',(9,9)));state.update(phase=1,frames=len(published),at=time.monotonic())
                    elif phase==1 and controls==['start','stop']:
                        app.commands.put(('demo',('ok','ok')));state['phase']=4
                        app.root.after(400,lambda:app.commands.put(('demo',(5,5))))
                        app.root.after(700,lambda:app.commands.put(('demo',('ok','ok'))))
                    elif phase==4 and controls[:3]==['start','stop','start']:
                        app.close();return
                    if time.monotonic()>deadline: app.close();return
                    app.root.after(15,drive)
                app.root.after(20,drive);app.run()
            controls=[e['kind'] for e in station.events if e['kind'] in ('start','stop','estop')]
            self.assertEqual(controls[:3],['start','stop','start']);self.assertNotIn('estop',controls)


if __name__=='__main__': unittest.main()
