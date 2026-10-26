import sys,json,tempfile,unittest,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
from lg.core import Engine
from lg.demo import observations
from lg.business import Station


class GateControlTests(unittest.TestCase):
    def test_same_numbers_bind_all_four_gates_once(self):
        for gate in range(2,6):
            engine=Engine(dwell=.3);events=[]
            for now in (0,.1,.2,.31,.4):
                events+=engine.update(observations((gate,gate)),'command',now)
            self.assertEqual(events,[('bind_gate',gate)])

    def test_mixed_numbers_and_six_to_eight_never_bind(self):
        for digits in ((1,1),(1,2),(3,5),(6,6),(7,7),(8,8)):
            engine=Engine(dwell=.3)
            for i in range(16): self.assertFalse(engine.update(observations(digits),'command',i*.1))

    def test_scanning_with_two_index_fingers_does_not_bind_gate(self):
        engine=Engine(dwell=.3);events=[]
        for now in (0,.1,.2,.31): events+=engine.update(observations((1,1)),'scan',now)
        self.assertEqual([event[0] for event in events],['scan'])

    def test_binding_persists_and_no_longer_stops_or_acknowledges(self):
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            station=Station(bootstrap.ROOT,folder);station.start()
            station.command('11');self.assertTrue(station.running);self.assertIsNone(station.bound_gate)
            station.command('22');self.assertTrue(station.running);self.assertEqual(station.bound_gate,2)
            restored=Station(bootstrap.ROOT,folder);self.assertEqual(restored.bound_gate,2)
            self.assertFalse(restored.running)
            station.bind_gate(5);self.assertEqual(station.bound_gate,5)
            for invalid in (0,1,6,9,'3',True):
                station.bind_gate(invalid);self.assertEqual(station.bound_gate,5)

    def test_action_diagnostics_do_not_replace_executed_command(self):
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            station=Station(bootstrap.ROOT,folder);station.start();message=station.last_command
            station.emit('action','动作轨迹已记录：握拳')
            self.assertEqual(station.last_command,message)
            station.stop();station.start()
            self.assertIn('拦截',station.last_command);self.assertFalse(station.running)


class ConsoleWindowTests(unittest.TestCase):
    def test_fixed_status_shows_command_and_gate_not_completed_action(self):
        from lg.app import App
        args=SimpleNamespace(demo=True,mode='command',yolo=None,dwell=.3,hold_tolerance=.06,hold_iou=.35,min_roi_pixels=4,smoke=0)
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            station=Station(bootstrap.ROOT,folder);snapshots=[]
            with patch('lg.app.Station',return_value=station):
                app=App(args);app.commands.put(('demo',('ok','ok')))
                app.root.after(400,lambda:app.commands.put(('demo',(5,5))))
                app.root.after(700,lambda:app.commands.put(('demo',('ok','ok'))))
                deadline=time.monotonic()+8;phase=[0]
                def drive():
                    if phase[0]==0 and station.running:
                        app.commands.put(('demo',(3,3)));phase[0]=1
                    elif phase[0]==1 and station.bound_gate==3 and '3 号' in app.gate_status.cget('text'):
                        snapshots.append((app.belt.cget('text'),app.command_status.cget('text'),app.gate_status.cget('text')))
                        app.commands.put(('demo',(9,9)));phase[0]=2
                    elif phase[0]==2 and not station.running and '已停止' in app.belt.cget('text'):
                        snapshots.append((app.belt.cget('text'),app.command_status.cget('text'),app.gate_status.cget('text')))
                        app.close();return
                    if time.monotonic()>deadline: app.close();return
                    app.root.after(20,drive)
                app.root.after(20,drive);app.run()
            self.assertEqual(len(snapshots),2)
            self.assertIn('运行中',snapshots[0][0]);self.assertIn('已绑定',snapshots[0][1])
            self.assertIn('1 秒',snapshots[1][1]);self.assertIn('3 号',snapshots[1][2])
            self.assertFalse(any('捏合完成' in text or '握拳完成' in text for row in snapshots for text in row))


if __name__=='__main__': unittest.main()
