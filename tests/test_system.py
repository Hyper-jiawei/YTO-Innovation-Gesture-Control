import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import json
import tempfile
import unittest
from unittest.mock import patch,MagicMock
import numpy as np
from lg.core import Dwell,Engine,Tracker,iou,features,index_pointing
from lg.demo import observations,scene
from lg.business import Station
from lg.vision import crop,Hands
from lg.data import save_snapshot,sample_counts
from lg.labels import LOVE_LABEL,LABELS,canonical_label,ACTIVE_DATASET


class TemporalTests(unittest.TestCase):
    def test_augmentation_reproducible_and_normalized(self):
        from lg.augmentation import augment_landmarks
        vector=features(observations()[0].points)
        before=vector.copy()
        a,meta=augment_landmarks(vector,np.random.default_rng(42))
        b,other=augment_landmarks(vector,np.random.default_rng(42))
        np.testing.assert_array_equal(a,b)
        self.assertEqual(meta,other)
        np.testing.assert_array_equal(vector,before)
        p=a.reshape(21,3)
        np.testing.assert_allclose(p[0],0,atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.norm(p[9])),1,places=5)
        self.assertTrue(np.isfinite(a).all())
        self.assertLessEqual(abs(meta['z_degrees']),12)
        self.assertLessEqual(abs(meta['y_degrees']),5)

    def setUp(self):
        self.boxes=[(.1,.1,.3,.3),(.6,.6,.9,.9)]
        self.positions=np.array([[.2,.2],[.8,.8]])

    def test_iou(self):
        self.assertEqual(iou((0,0,1,1),(0,0,1,1)),1)
        self.assertEqual(iou((0,0,1,1),(2,2,3,3)),0)

    def test_dwell_once_until_release(self):
        d=Dwell(seconds=1.2)
        self.assertFalse(d.update('a',self.positions,self.boxes,0))
        self.assertTrue(d.update('a',self.positions,self.boxes,1.3))
        self.assertFalse(d.update('a',self.positions+.1,self.boxes,2))
        self.assertFalse(d.update('a',self.positions+.1,self.boxes,4))
        d.update(None,[],[],4.1)
        d.update('a',self.positions,self.boxes,5)
        self.assertTrue(d.update('a',self.positions,self.boxes,6.3))

    def test_anchor_prevents_slow_drift(self):
        d=Dwell()
        for i in range(60):
            self.assertFalse(d.update('a',self.positions+i*.005,self.boxes,i*.1))

    def test_box_motion_resets(self):
        d=Dwell()
        d.update('a',self.positions,self.boxes,0)
        self.assertFalse(d.update('a',self.positions,[(0,0,.9,.9)]*2,1.3))

    def test_one_second_fist_hold_once_until_release(self):
        engine=Engine();hands=observations((9,9))
        for now in np.arange(0,1,.1): self.assertFalse(engine.update(hands,'scan',float(now)))
        self.assertEqual(engine.update(hands,'scan',1.001),[('stop',None)])
        self.assertFalse(engine.update(hands,'scan',1.2))
        engine.update([], 'scan',1.3)
        self.assertFalse(engine.update(hands,'scan',1.4))

    def test_loss_resets_stop_hold(self):
        engine=Engine();hands=observations((9,9))
        engine.update(hands,'scan',0);engine.update(hands,'scan',.3)
        engine.update([],'scan',.4)
        for now in (.6,.9,1.2): self.assertFalse(engine.update(hands,'scan',now))
        self.assertEqual(engine.update(hands,'scan',1.61),[('stop',None)])

    def test_stalled_input_cannot_finish_stop_hold(self):
        engine=Engine();hands=observations((9,9))
        engine.update(hands,'scan',0)
        self.assertFalse(engine.update(hands,'scan',2))
        self.assertEqual(engine.control_progress,0)

    def test_roi_and_mode_exclusion(self):
        e=Engine(dwell=1.2); h=observations()
        for t in np.arange(0,1.11,.1): e.update(h,'scan',float(t))
        events=e.update(h,'scan',1.3)
        self.assertEqual(events[0][0],'scan')
        self.assertEqual(len(events[0][1]),4)
        self.assertTrue(all(type(value) is float for value in events[0][1]))
        self.assertFalse(e.update(h,'command',1.4))
        for t in np.arange(1.5,2.51,.1): e.update(h,'command',float(t))
        self.assertEqual(e.update(h,'command',2.7),[])  # Label 1 never binds a gate.

    def test_lost_hand_and_identity_reset(self):
        e=Engine(); h=observations()
        e.update(h,'scan',0); e.update([],'scan',1)
        self.assertFalse(e.update(h,'scan',1.3))
        h[0].track_id=123
        self.assertFalse(e.update(h,'scan',3))

    def test_stalled_frames_cancel_dwell(self):
        e=Engine(); h=observations()
        e.update(h,'scan',0)
        self.assertFalse(e.update(h,'scan',2))
        self.assertEqual(e.dwell.progress,0)

    def test_tracker(self):
        t=Tracker(); h=t.update(observations()); ids=[v.track_id for v in h]
        self.assertEqual([v.track_id for v in t.update(observations())],ids)
        t.update([])
        self.assertNotEqual([v.track_id for v in t.update(observations())],ids)

    def test_features_translation_scale(self):
        p=observations()[0].points
        np.testing.assert_allclose(features(p),features(p*2+[.2,.1,0]),atol=1e-5)
        with self.assertRaises(ValueError): features(np.zeros((21,3)))

    def test_relaxed_hold_two_seconds_and_release(self):
        d=Dwell(seconds=2.)
        self.assertFalse(d.update('roi',self.positions,self.boxes,0))
        jitter=self.positions+np.array([[.035,.02],[-.03,.02]])
        smaller=[(.12,.12,.28,.28),(.63,.63,.87,.87)]
        self.assertFalse(d.update('roi',jitter,smaller,1.9))
        self.assertTrue(d.update('roi',jitter,smaller,2.01))
        self.assertFalse(d.update('roi',jitter+.05,smaller,3))
        self.assertEqual(d.progress,1)
        d.update(None,[],[],3.1)
        self.assertEqual(d.progress,0)

    def test_new_default_dwell_is_1_1_seconds(self):
        e=Engine(); h=observations()
        events=[]
        for t in np.arange(0,1.1,.1):
            events+=e.update(h,'scan',float(t))
        self.assertFalse(events)
        self.assertEqual(e.update(h,'scan',1.101)[0][0],'scan')

    def test_small_and_thin_roi_can_trigger(self):
        for width,height in ((.02,.02),(.06,.01),(.01,.06)):
            with self.subTest(width=width,height=height):
                e=Engine(); hands=observations()
                hands[0].points[8,:2]=[.5,.5]
                hands[1].points[8,:2]=[.5+width,.5+height]
                events=[]
                for now in (0,.3,.6,.9,1.101):
                    events+=e.update(hands,'photo',now,frame_size=(640,480))
                self.assertEqual(len(events),1)
                self.assertEqual(events[0][0],'photo')

    def test_degenerate_roi_rejected(self):
        for width,height in ((0,.1),(.1,0),(.001,.1),(.1,.001)):
            with self.subTest(width=width,height=height):
                e=Engine(); hands=observations()
                hands[0].points[8,:2]=[.5,.5]
                hands[1].points[8,:2]=[.5+width,.5+height]
                for now in (0,.3,.6,.9,1.101):
                    self.assertFalse(e.update(hands,'photo',now,frame_size=(640,480)))
                    self.assertIsNone(e.roi)

    def test_roi_ignores_thumb_but_rejects_open_hand(self):
        p=np.zeros((21,3),np.float32)
        p[0]=[.2,.7,0]
        p[5:9]=[[.2,.5,0],[.2,.4,0],[.2,.3,0],[.2,.2,0]]
        for base in (9,13,17):
            p[base:base+4]=[[.25,.5,0],[.25,.4,0],[.25,.5,0],[.25,.6,0]]
        self.assertTrue(index_pointing(p))
        p[4]=[.7,.1,0]
        self.assertTrue(index_pointing(p))
        p[9:13]=[[.25,.5,0],[.25,.4,0],[.25,.3,0],[.25,.2,0]]
        self.assertFalse(index_pointing(p))

    def test_love_gesture_not_a_numeric_control(self):
        engine=Engine(); hands=observations((LOVE_LABEL,1))
        for now in (0,.3,.6,.9,1.2):
            self.assertFalse(engine.update(hands,'command',now))
            self.assertIsNone(engine.code)

    def test_temporal_split_disjoint_and_complete_classes(self):
        from lg.training import temporal_split
        rows=[dict(label=label,physical_hand=hand,time=i) for label in LABELS
              for hand in ('left','right') for i in range(40)]
        train,val,test,gap=temporal_split(rows)
        parts=[set(v.tolist()) for v in (train,val,test,gap)]
        self.assertEqual(len(set.union(*parts)),len(rows))
        self.assertEqual(sum(map(len,parts)),len(rows))
        for part in (train,val,test):
            self.assertEqual({rows[i]['label'] for i in part},set(LABELS))
        self.assertEqual(len(gap),8*len(LABELS))


class IntegrationTests(unittest.TestCase):
    def test_exported_augmentation_has_only_training_parents(self):
        from lg.augmentation import expand_training
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            rows=[dict(label='1',physical_hand='left',features=features(observations()[0].points).tolist()) for _ in range(8)]
            path=Path(folder)/'augmented.jsonl'
            vectors,parents=expand_training(rows,[0,2,4],4,42,'testhash',path)
            self.assertEqual(vectors.shape,(15,63))
            self.assertEqual(set(parents.tolist()),{0,2,4})
            saved=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(saved),15)
            self.assertTrue(all(r['source_index'] in {0,2,4} and r['split']=='train' and r['label']=='1' for r in saved))
            self.assertTrue(all('points' not in r for r in saved))
            self.assertEqual(sum(r['variant']>0 for r in saved),12)

    def test_gui_capture_freezes_then_saves_once(self):
        from lg.collector import CaptureSession
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            path=Path(folder)/'digits.jsonl'
            session=CaptureSession(path)
            hands=observations()[:1]
            original=hands[0].points.copy()
            session.freeze(hands,'p01')
            hands[0].points[:]=0
            with self.assertRaises(ValueError): session.new_session('p02')
            with self.assertRaises(ValueError): session.save('3',None)
            self.assertFalse(path.exists())
            row=session.save('3','right')
            np.testing.assert_allclose(row['points'],original)
            self.assertEqual(row['capture'],'gui_snapshot')
            self.assertEqual(row['physical_hand'],'right')
            with self.assertRaises(ValueError): session.save('3','right')
            self.assertEqual(len(path.read_text().splitlines()),1)
            session.freeze(observations()[:1],'p01'); session.cancel()
            self.assertEqual(len(path.read_text().splitlines()),1)
            previous=session.session
            session.new_session('p02')
            self.assertNotEqual(previous,session.session)
            self.assertEqual(session.counts[('right','3')],1)

    def test_gui_label_buttons_and_next_round(self):
        from lg.collector import CollectorApp
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            args=SimpleNamespace(demo=True,operator='test',session=None,label='1',hand=None,side='both',camera='1',smoke=0)
            app=CollectorApp(args,folder)
            try:
                app.root.update()
                app.latest=dict(frame=scene(),hands=observations()[:1])
                app.open_capture()
                self.assertIsNotNone(app.dialog)
                app.label_buttons['7'].invoke(); app.hand_buttons['right'].invoke()
                app.save_capture()
                self.assertIsNone(app.dialog)
                self.assertEqual(app.capture.saved,1)
                self.assertIn('保存成功',app.feedback.cget('text'))
                app.latest=dict(frame=scene(),hands=observations()[:1])
                app.open_capture()
                self.assertEqual(app.label_var.get(),'7')
                app.cancel_capture()
                self.assertEqual(app.capture.saved,1)
            finally:
                app.close(); app.worker.join(timeout=3)

    def test_numpy_roi_photo_and_damage_persistence(self):
        for kind in ('photo','damage'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
                station=Station(folder)
                roi=np.array([.28,.23,.72,.78],dtype=np.float32)
                image,_=crop(scene(),roi)
                station.evidence(image,kind,roi)
                event=station.events[-1]
                self.assertTrue(all(type(value) is float for value in event['roi']))
                self.assertTrue((station.folder/event['image']).is_file())
                persisted=json.loads((station.folder/'events.jsonl').read_text(encoding='utf-8'))
                self.assertEqual(persisted['roi'],event['roi'])
                np.testing.assert_allclose(persisted['roi'],roi)
                if kind=='damage':
                    report=station.folder/f"report_{event['id']}.json"
                    self.assertEqual(json.loads(report.read_text(encoding='utf-8'))['roi'],persisted['roi'])
                    restored=Station(folder)
                    self.assertIn(event['id'],restored.pending)
                    restored.ack()
                    self.assertEqual(json.loads(report.read_text(encoding='utf-8'))['status'],'acknowledged_local')

    def test_hold_to_photo_with_float32_landmarks(self):
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            engine=Engine(dwell=.3)
            hands=observations()
            self.assertEqual(hands[0].points.dtype,np.float32)
            events=[]
            for now in (0,.1,.2,.31):
                events+=engine.update(hands,'photo',now)
            self.assertEqual(len(events),1)
            kind,roi=events[0]
            image,_=crop(scene(),roi)
            station=Station(folder)
            station.evidence(image,kind,roi)
            self.assertEqual(station.events[-1]['kind'],'photo')
            self.assertEqual(engine.update(hands,'photo',.4),[])

    def test_single_sample_collection_success_and_rejection(self):
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            path=Path(folder)/'digits.jsonl'
            for h in ([],observations()):
                with self.assertRaises(ValueError): save_snapshot(path,h,'1','p01','s1','left')
            self.assertFalse(path.exists())
            with self.assertRaises(ValueError): save_snapshot(path,observations()[:1],'1','p01','s1',None)
            row=save_snapshot(path,observations()[:1],'1','p01','s1','right')
            self.assertEqual(row['side'],'left')
            self.assertEqual(row['physical_hand'],'right')
            self.assertEqual(len(path.read_text().splitlines()),1)
            self.assertEqual(sample_counts(path)[('right','1')],1)

    def test_collector_keys_and_feedback(self):
        import collect_digits_keyboard as collect_digits
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            capture=MagicMock()
            frame=np.zeros((480,640,3),np.uint8)
            capture.read.return_value=(True,frame)
            detector=MagicMock()
            detector.process.side_effect=lambda f: observations()[:1]
            shown=[]
            with patch.object(bootstrap,'ROOT',Path(folder)), \
                 patch.object(collect_digits,'Hands',return_value=detector) as factory, \
                 patch.object(collect_digits,'open_camera',return_value=(capture,frame,1)), \
                 patch.object(sys,'argv',['collect_digits.py','--operator','test']), \
                 patch.object(collect_digits.cv2,'namedWindow'),patch.object(collect_digits.cv2,'resizeWindow'), \
                 patch.object(collect_digits.cv2,'destroyAllWindows'), \
                 patch.object(collect_digits.cv2,'imshow',side_effect=lambda n,i:shown.append(i.copy())), \
                 patch.object(collect_digits.cv2,'getWindowProperty',return_value=1), \
                 patch.object(collect_digits.cv2,'waitKey',side_effect=[ord('l'),ord('1'),32,ord('r'),ord('u'),ord('s'),ord('q')]):
                collect_digits.main()
                factory.assert_called_once_with(Path(folder),classify=False)
            rows=[json.loads(line) for line in (Path(folder)/ACTIVE_DATASET).read_text().splitlines()]
            self.assertEqual([(r['physical_hand'],r['label']) for r in rows],[('left','1'),('right',LOVE_LABEL)])
            self.assertTrue(any(tuple(i[1,1])==(80,230,80) for i in shown))

    def test_two_tip_rings_render(self):
        from lg.vision import draw
        hands=observations()
        image=np.zeros((720,960,3),np.uint8)
        base=draw(image,hands,(.28,.23,.72,.78),0)
        full=draw(image,hands,(.28,.23,.72,.78),1)
        for hand in hands:
            x,y=(hand.points[8,:2]*[960,720]).astype(int)
            self.assertTrue(np.any(base[max(0,y-30):y+30,x-30:x+30]!=full[max(0,y-30):y+30,x-30:x+30]))

    def test_scan_duplicate_restart_damage_ack_estop(self):
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            s=Station(folder)
            frame=scene()
            image,_=crop(frame,(.28,.23,.72,.78))
            s.scan(image); self.assertEqual(len(s.seen),1)
            s.scan(image); self.assertEqual(s.events[-1]['kind'],'duplicate')
            s=Station(folder); self.assertEqual(len(s.seen),1)
            s.start(); self.assertTrue(s.running)
            s.evidence(image,'damage',(.28,.23,.72,.78))
            self.assertFalse(s.running); self.assertEqual(len(s.pending),1)
            s=Station(folder); self.assertEqual(len(s.pending),1)
            s.ack(); self.assertFalse(s.pending)
            report=next((Path(folder)/'outputs').glob('report_*.json'))
            self.assertEqual(json.loads(report.read_text(encoding='utf-8'))['status'],'acknowledged_local')
            s.stop(); s.start(); self.assertFalse(s.running)
            s.reset(); self.assertFalse(s.running)
            s.start(); self.assertTrue(s.running)

    def test_real_mediapipe_blank_frame(self):
        detector=Hands(bootstrap.ROOT)
        try: self.assertEqual(detector.process(np.zeros((480,640,3),np.uint8)),[])
        finally: detector.close()

if __name__=='__main__': unittest.main()
