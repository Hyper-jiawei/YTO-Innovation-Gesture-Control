import unittest,tempfile,time
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import bootstrap
from lg.app import App
from lg.business import Station
from lg.core import Engine
from lg.demo import observations

class PhotoControlTests(unittest.TestCase):
    def test_one_ok_or_timeout_never_starts(self):
        e=Engine()
        for i in range(60): self.assertFalse(e.update(observations(('ok','ok')),'photo',i*.1))
        for i in range(60,64): e.update(observations((5,5)),'photo',i*.1)
        for i in range(64,70): self.assertFalse(e.update(observations(('ok','ok')),'photo',i*.1))

    def test_photo_preview_persists_and_archive_button_removed(self):
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            station=Station(bootstrap.ROOT,folder);packets=[];shown=[]
            args=SimpleNamespace(demo=True,mode='photo',yolo=None,dwell=.2,hold_tolerance=.06,hold_iou=.35,min_roi_pixels=4,smoke=0)
            with patch('lg.app.Station',return_value=station):
                app=App(args);original=app.publish
                def publish(packet): packets.append(packet);original(packet)
                app.publish=publish
                app.root.after(600,lambda:app.commands.put(('demo',None)))
                def finish():
                    shown.append(app.scan_preview.cget('image'))
                    def labels(w):
                        result=[]
                        try: result.append(w.cget('text'))
                        except Exception: pass
                        for child in w.winfo_children(): result+=labels(child)
                        return result
                    self.assertFalse(any('归档' in t for t in labels(app.root)))
                    app.close()
                app.root.after(1000,finish);app.run()
            self.assertTrue(shown[0]);self.assertIsNotNone(station.last_photo_image)
            self.assertTrue(all(p['scan_image'] is not None for p in packets[-4:]))
            self.assertEqual(station.photo_revision,1)
            station.pending['test']={};station.start();self.assertTrue(station.running)
            station.stop();station.start();self.assertFalse(station.running)

if __name__=='__main__': unittest.main()
