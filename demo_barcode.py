"""Decode a local image, or generate real checksum-valid 1D regression fixtures."""
import bootstrap
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from lg.barcode import decode_barcodes,scan_crop


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',type=Path,help='本地图像；不提供时生成真实 Code128/EAN13 测试图')
    p.add_argument('--roi',type=float,nargs=4,help='归一化 x1 y1 x2 y2')
    a=p.parse_args()
    output=bootstrap.ROOT/'outputs/barcode_demo'
    output.mkdir(parents=True,exist_ok=True)
    if a.image:
        frame=cv2.imdecode(np.fromfile(str(a.image),np.uint8),cv2.IMREAD_COLOR)
        if frame is None: raise ValueError('无法读取图像')
        image,region=scan_crop(frame,a.roi) if a.roi else (frame,None)
        cases=[(a.image.name,image,region)]
    else:
        import zxingcpp
        cases=[]
        for name,format,text in [('code128',zxingcpp.BarcodeFormat.Code128,'SAMPLE-BARCODE-123'),('ean13',zxingcpp.BarcodeFormat.EAN13,'5901234123457')]:
            image=np.asarray(zxingcpp.create_barcode(text,format).to_image(scale=3,add_quiet_zones=True))
            cases.extend([(name,image,None),(name+'_upside_down',np.rot90(image,2).copy(),None),
                          (name+'_low_contrast',(image.astype(np.float32)*.25+140).astype(np.uint8),None)])
    report=[]
    for name,image,region in cases:
        codes,info=decode_barcodes(image,region)
        ok,encoded=cv2.imencode('.png',image)
        if ok: encoded.tofile(str(output/(Path(name).stem+'.png')))
        report.append(dict(case=name,codes=codes,diagnostics=info))
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
