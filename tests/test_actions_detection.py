import sys,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import numpy as np
from lg.actions import ActionCycle,ActionRecognizer,SequenceRecorder,hand_metrics
from lg.core import Hand,Engine
from lg.vision import Detector,draw_detections,crop
from lg.business import Station


def pose(kind):
    p=np.zeros((21,3),np.float32); p[0]=[.5,.85,0]
    for base,x in zip((5,9,13,17),(.38,.47,.56,.65)):
        p[base:base+4]=[[x,.62,0],[x,.42,0],[x,.55 if kind=='fist' else .3,0],[x,.68 if kind=='fist' else .2,0]]
    p[1:5]=[[.4,.75,0],[.3,.68,0],[.22,.6,0],[.18,.52,0]]
    if kind=='fist': p[4]=[.43,.64,0]
    if kind=='pinch': p[4]=p[8]+[.02,0,0]
    return Hand(p,None,(*p[:,:2].min(0),*p[:,:2].max(0)),7)


class ActionTests(unittest.TestCase):
    def test_geometry_and_scale_invariance(self):
        self.assertEqual(hand_metrics(pose('open').points)['extended'],4)
        self.assertEqual(hand_metrics(pose('fist').points)['closed'],4)
        self.assertLess(hand_metrics(pose('pinch').points)['pinch'],.32)
        p=pose('pinch').points
        self.assertAlmostEqual(hand_metrics(p)['pinch'],hand_metrics(p*2+[.2,.3,.1])['pinch'],places=5)

    def test_complete_actions_once_and_distinct(self):
        for kind in ('pinch','fist'):
            with self.subTest(kind=kind):
                recognizer=ActionRecognizer(); events=[]
                for i,phase in enumerate(['open']*4+[kind]*4+['open']*12):
                    events.extend(recognizer.update([pose(phase)],i*.1))
                self.assertEqual([e['action'] for e in events],[kind])
                json.dumps(events)

    def test_held_pose_never_counts_as_motion(self):
        recognizer=ActionRecognizer()
        for i in range(20): self.assertFalse(recognizer.update([pose('fist')],i*.1))
        for i in range(20,24): self.assertFalse(recognizer.update([pose('open')],i*.1))

    def test_loss_identity_change_and_frame_gap_cancel(self):
        for interruption in ('loss','identity','gap'):
            recognizer=ActionRecognizer()
            for i,kind in enumerate(['open']*4+['fist']*4): recognizer.update([pose(kind)],i*.1)
            if interruption=='loss': recognizer.update([],.8)
            for i in range(9,13):
                hand=pose('open')
                if interruption=='identity': hand.track_id=8
                now=i*.1+(1 if interruption=='gap' else 0)
                self.assertFalse(recognizer.update([hand],now))

    def test_debounce_and_timeout(self):
        c=ActionCycle()
        for state,t in [('open',0),('open',.2),('closed',.3),('open',.35),('open',.5)]:
            self.assertFalse(c.update(state,t))
        c.update('closed',.6);c.update('closed',.8)
        self.assertFalse(c.update('open',3.5));self.assertFalse(c.update('open',3.7))

    def test_only_label_nine_held_one_second_stops(self):
        for digit in (7,9):
            engine=Engine();output=[]
            for i,kind in enumerate(['fist']*12):
                hands=[pose(kind),pose(kind)];hands[1].track_id=8
                for hand in hands: hand.digit=5 if kind=='open' else digit
                output+=engine.update(hands,'command',i*.1)
            self.assertEqual(sum(kind=='stop' for kind,_ in output),1 if digit==9 else 0)

    def test_fist_action_uses_trained_label_nine(self):
        for digit in (7,9):
            recognizer=ActionRecognizer();events=[]
            for i,kind in enumerate(['open']*4+['fist']*4+['open']*4):
                hand=pose(kind);hand.digit=5 if kind=='open' else digit
                hand.points[4]=[.18,.52,0]  # Numeric label determines learned fist pose.
                events+=recognizer.update([hand],i*.1)
            self.assertEqual([e['action'] for e in events],['fist'] if digit==9 else [])

    def test_thumb_extended_does_not_count_as_fist(self):
        recognizer=ActionRecognizer();events=[]
        for i,kind in enumerate(['open']*4+['fist']*4+['open']*4):
            hand=pose(kind)
            if kind=='fist': hand.points[4]=[.18,.52,0]
            events+=recognizer.update([hand],i*.1)
        self.assertEqual(events,[])

    def test_recorder_resamples_and_appends(self):
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as temp:
            target=Path(temp)/'actions.jsonl'; recorder=SequenceRecorder()
            for run in range(2):
                recorder.start('pinch',run*3)
                for i in range(21): result=recorder.step([pose('open' if i<5 or i>15 else 'pinch')],run*3+i*.1,target)
                self.assertIn('已保存',result)
            saved=[json.loads(line) for line in target.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(saved),2);self.assertEqual(np.array(saved[0]['features']).shape,(20,63))
            recorder.start('fist',7); self.assertIn('取消',recorder.step([],7,target))
            self.assertEqual(len(target.read_text(encoding='utf-8').splitlines()),2)


class DatasetDetectionTests(unittest.TestCase):
    def test_slow_inference_does_not_retrigger_latched_selection(self):
        from lg.demo import observations
        engine=Engine(dwell=.3);hands=observations()
        for now in (0,.1,.2): self.assertFalse(engine.update(hands,'damage',now))
        self.assertEqual(engine.update(hands,'damage',.31)[0][0],'damage')
        for now in (2,2.1,2.2,2.5): self.assertFalse(engine.update(hands,'damage',now))
        self.assertTrue(engine.dwell.fired)
        engine.update([],'damage',2.6)
        self.assertFalse(engine.dwell.fired)

    def test_folder_labels(self):
        from import_hand_images import folder_label
        self.assertEqual(folder_label('ok- 副本'),('ok','right'))
        self.assertEqual(folder_label('I Love U'),('I Love U','left'))
        self.assertEqual(folder_label('3 - 副本'),('3','right'))
        with self.assertRaises(ValueError): folder_label('unlabelled')
        for removed in ('0','10','0 - 副本','10 - 副本'):
            with self.assertRaises(ValueError): folder_label(removed)

    def test_removed_labels_rejected_by_collection_and_commands(self):
        from lg.data import save_snapshot
        from lg.demo import observations
        from lg.labels import LABELS
        self.assertEqual(LABELS,[str(i) for i in range(1,10)]+['I Love U','ok'])
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            for label in ('0','10'):
                with self.assertRaises(ValueError):
                    save_snapshot(Path(folder)/'digits.jsonl',observations()[:1],label,'p01','s1','left')
            station=Station(bootstrap.ROOT,folder)
            station.command('10')
            self.assertFalse(station.running)
            self.assertEqual(station.events[-1]['kind'],'invalid_code')
            station.start();self.assertTrue(station.running)
            station.command('11');self.assertTrue(station.running)
            self.assertIsNone(station.bound_gate)
            station.halt();self.assertFalse(station.running)
            engine=Engine(dwell=.3)
            for now in (0,.1,.2,.31):
                self.assertEqual(engine.update(observations((1,0)),'command',now),[])

    def test_detector_scopes_roi_and_maps_coordinates(self):
        detector=Detector();detector.device='cpu';detector.model=MagicMock()
        box=MagicMock();box.xyxy[0].cpu.return_value.tolist.return_value=[10,5,60,40]
        box.cls.item.return_value=73;box.conf.item.return_value=.91
        detector.model.predict.return_value=[SimpleNamespace(boxes=[box],names={73:'book'})]
        frame=np.zeros((200,300,3),np.uint8)
        result=detector.detect_region(frame,(.2,.25,.8,.75))
        self.assertEqual(detector.model.predict.call_args.args[0].shape,(100,180,3))
        np.testing.assert_allclose(result[0]['bbox'],[70/300,55/200,120/300,90/200])
        self.assertEqual(result[0]['label'],'book');json.dumps(result)
        annotated=draw_detections(frame,result)
        self.assertGreater(np.count_nonzero(annotated),0)
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            station=Station(folder)
            image,_=crop(annotated,(.2,.25,.8,.75))
            station.evidence(image,'damage',(.2,.25,.8,.75),'yolo12n.pt',result)
            report=json.loads(next(station.folder.glob('report_*.json')).read_text(encoding='utf-8'))
            self.assertEqual(report['detections'],result)
            self.assertEqual(report['bbox_coordinates'],'normalized_full_frame')

    def test_disabled_detector_does_not_fabricate_objects(self):
        self.assertEqual(Detector().detect_region(np.zeros((10,10,3),np.uint8),(0,0,1,1)),[])


class AppFlowTests(unittest.TestCase):
    def args(self,mode):
        return SimpleNamespace(demo=True,mode=mode,yolo=None,dwell=.3,hold_tolerance=.06,hold_iou=.35,
                               min_roi_pixels=4,smoke=0)

    def test_fifth_module_emits_completed_pinch_in_window(self):
        from lg.app import App
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            station=Station(bootstrap.ROOT,folder)
            with patch('lg.app.Station',return_value=station):
                app=App(self.args('action'))
                app.root.after(350,lambda:app.commands.put(('demo_action','pinch')))
                app.root.after(800,lambda:app.commands.put(('demo_action','open')))
                app.root.after(1400,app.close)
                app.run()
            self.assertEqual([e['action'] for e in station.events if e['kind']=='action'],['pinch'])
            self.assertGreater(app.shown,0)

    def test_damage_flow_uses_scoped_detection_and_persists_label(self):
        from lg.app import App
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            station=Station(bootstrap.ROOT,folder)
            detector=MagicMock(device='cpu',model_name='yolo12n.pt')
            detector.detect_region.return_value=[dict(label='book',class_id=73,confidence=.91,bbox=[.35,.3,.6,.6])]
            with patch('lg.app.Station',return_value=station),patch('lg.app.Detector',return_value=detector):
                app=App(self.args('damage'))
                app.root.after(1100,app.close);app.run()
            self.assertGreater(detector.detect_region.call_count,1)
            self.assertFalse(detector.refine.called)
            report=json.loads(next(Path(folder).glob('report_*.json')).read_text(encoding='utf-8'))
            self.assertEqual(report['detections'][0]['label'],'book')
            self.assertTrue((Path(folder)/report['image']).exists())


if __name__=='__main__': unittest.main()
