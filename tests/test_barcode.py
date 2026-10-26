import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import unittest
from unittest.mock import patch,MagicMock
import tempfile
import json
import cv2
import numpy as np
import zxingcpp
from lg.barcode import decode_barcodes,scan_crop,ScanWindow
from lg.business import Station


def make_code(text='SAMPLE-BARCODE-123',format=zxingcpp.BarcodeFormat.Code128):
    return np.asarray(zxingcpp.create_barcode(text,format).to_image(scale=2,add_quiet_zones=True))


class BarcodeTests(unittest.TestCase):
    def test_real_linear_formats_orientation_and_contrast(self):
        for format,text in [(zxingcpp.BarcodeFormat.Code128,'SAMPLE-BARCODE-123'),
                            (zxingcpp.BarcodeFormat.EAN13,'5901234123457')]:
            code=make_code(text,format)
            for variant in [code,np.rot90(code,2).copy(),np.rot90(code).copy(),255-code,
                            (code.astype(float)*.25+140).astype(np.uint8)]:
                with self.subTest(format=str(format),shape=variant.shape):
                    codes,_=decode_barcodes(variant)
                    self.assertIn(text,[c['text'] for c in codes])

    def test_no_fabricated_result_and_multi_pass(self):
        codes,info=decode_barcodes(np.full((80,180,3),180,np.uint8))
        self.assertEqual(codes,[])
        self.assertEqual(info['attempts'],7)

    def test_outside_selection_not_accepted(self):
        code=make_code('OUTSIDE')
        frame=np.full((200,800),255,np.uint8)
        frame[50:50+code.shape[0],300:300+code.shape[1]]=code
        codes,_=decode_barcodes(frame,(0,0,200,200))
        self.assertFalse(codes)

    def test_margin_clipped_to_frame(self):
        frame=np.zeros((480,640,3),np.uint8)
        image,region=scan_crop(frame,(0,0,.1,.1))
        self.assertGreater(image.shape[1],64)
        self.assertGreater(image.shape[0],48)
        self.assertEqual(region[:2],(0,0))

    def test_retry_uses_new_frame_and_logs_only_final_result(self):
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            station=Station(folder)
            scan=ScanWindow()
            frame=np.full((480,800,3),255,np.uint8)
            roi=(.1,.2,.9,.7)
            scan.start(roi,0)
            scan.step(frame,0,station)
            self.assertEqual(station.events,[])
            self.assertIsNotNone(scan.roi)
            code=make_code()
            frame[170:170+code.shape[0],140:140+code.shape[1]]=cv2.cvtColor(code,cv2.COLOR_GRAY2BGR)
            scan.step(frame,.3,station)
            self.assertIsNone(scan.roi)
            self.assertEqual(station.events[-1]['text'],'SAMPLE-BARCODE-123')
            self.assertEqual(len(station.seen),1)

    def test_failed_retry_one_log_and_diagnostic(self):
        with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'outputs') as folder:
            station=Station(folder)
            scan=ScanWindow(max_frames=2)
            scan.start((.1,.1,.9,.9),0)
            frame=np.full((120,200,3),255,np.uint8)
            scan.step(frame,0,station); scan.step(frame,.3,station)
            self.assertIsNone(scan.roi)
            self.assertEqual(len(station.events),1)
            self.assertEqual(station.events[-1]['kind'],'scan_miss')
            self.assertTrue((station.folder/'scans/last_failed_roi.png').exists())
            self.assertEqual(json.loads((station.folder/'scans/last_failed_roi.json').read_text(encoding='utf-8'))['frames'],2)

    def test_camera_requests_hd_before_first_read_and_native_fallback(self):
        from lg.vision import open_camera
        cap=MagicMock(); cap.read.return_value=(True,np.zeros((1080,1920,3),np.uint8))
        with patch('lg.vision.cv2.VideoCapture',return_value=cap):
            _,frame,index=open_camera('1')
        self.assertEqual(index,1)
        calls=cap.mock_calls
        read_index=next(i for i,c in enumerate(calls) if c[0]=='read')
        self.assertIn(unittest.mock.call.set(cv2.CAP_PROP_FRAME_WIDTH,1920),calls[:read_index])
        self.assertEqual(frame.shape[:2],(1080,1920))
        bad=MagicMock(); bad.read.return_value=(False,None)
        native=MagicMock(); native.read.return_value=(True,np.zeros((480,640,3),np.uint8))
        with patch('lg.vision.cv2.VideoCapture',side_effect=[bad,native]):
            result,_,_=open_camera('1')
        self.assertIs(result,native); bad.release.assert_called_once()


if __name__=='__main__': unittest.main()
