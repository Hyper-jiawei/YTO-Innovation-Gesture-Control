"""Folder-labelled dorsal images -> auditable MediaPipe pseudo annotations."""
import bootstrap
import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime,timezone,timedelta
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from lg.core import features
from lg.labels import LABELS


def folder_label(name):
    right='副本' in name
    label=re.sub(r'\s*-?\s*副本.*$','',name).strip(' -')
    if label.lower()=='ok': label='ok'
    if label not in LABELS:
        raise ValueError(f'Unrecognized folder label: {name}')
    return label,'right' if right else 'left'


def capture_time(path,index):
    match=re.search(r'(\d{8})_(\d{2})_(\d{2})_(\d{2})',path.stem)
    if match:
        stamp=datetime.strptime(''.join(match.groups()),'%Y%m%d%H%M%S').replace(tzinfo=timezone(timedelta(hours=8)))
        return stamp.timestamp()+index*.0001
    return float(index)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path(r'C:\Users\Hyper\Pictures\hand_images1023'))
    p.add_argument('--output',type=Path,default=bootstrap.ROOT/'data/hand_images1023')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions,vision
    options=vision.GestureRecognizerOptions(base_options=BaseOptions(model_asset_path=str(bootstrap.ROOT/'models/gesture_recognizer.task')),
        running_mode=vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5)
    recognizer=vision.GestureRecognizer.create_from_options(options)
    rows,manifest,seen,previews=[],[],{},{}
    files=[]
    for folder in sorted(v for v in a.source.iterdir() if v.is_dir()):
        label,hand=folder_label(folder.name)
        files.extend((path,label,hand) for path in sorted(folder.rglob('*')) if path.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp'))
    try:
        for index,(path,label,hand) in enumerate(files):
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            entry=dict(source=str(path),label=label,physical_hand=hand,view='dorsal',sha256=digest)
            if digest in seen:
                entry.update(status='duplicate',duplicate_of=seen[digest]); manifest.append(entry); continue
            seen[digest]=str(path)
            try:
                with Image.open(path) as opened: image=ImageOps.exif_transpose(opened).convert('RGB')
                width,height=image.size
                thumb=image.copy(); thumb.thumbnail((960,960))
                rgb=np.asarray(thumb).copy()
                result=recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
                candidates=[]
                for landmarks in result.hand_landmarks:
                    points=np.array([(l.x,l.y,l.z) for l in landmarks],np.float32)
                    lo,hi=points[:,:2].min(0),points[:,:2].max(0)
                    candidates.append((float(np.prod(hi-lo)),points,(*lo,*hi)))
                candidates.sort(key=lambda v:v[0],reverse=True)
                if not candidates:
                    entry['status']='no_hand'
                elif len(candidates)>1 and candidates[0][0]<1.8*candidates[1][0]:
                    entry['status']='ambiguous_multiple_hands'
                else:
                    _,points,box=candidates[0]
                    row=dict(label=label,physical_hand=hand,view='dorsal',operator='images1023',session='images1023_'+hand,
                        time=capture_time(path,index),source=str(path),source_sha256=digest,image_width=width,image_height=height,
                        points=points.tolist(),features=features(points).tolist(),bbox=[float(v) for v in box],
                        capture='folder_images_mediapipe_pseudo_annotation')
                    rows.append(row); entry.update(status='accepted',row_index=len(rows)-1,detected_hands=len(candidates))
                    annotation=dict(row,landmark_pixels=(points*[width,height,width]).tolist())
                    (a.output/'annotations').mkdir(exist_ok=True)
                    (a.output/'annotations'/f'{digest[:16]}.json').write_text(json.dumps(annotation,ensure_ascii=False,indent=2),encoding='utf-8')
                    key=(label,hand)
                    if key not in previews:
                        from lg.vision import draw
                        from lg.core import Hand
                        preview=cv2.cvtColor(draw(np.asarray(image)[:,:,::-1].copy(),[Hand(points,label,box,0)]),cv2.COLOR_BGR2RGB)
                        pic=Image.fromarray(preview); pic.thumbnail((320,180))
                        previews[key]=pic
            except Exception as e: entry.update(status='error',error=repr(e))
            manifest.append(entry)
            if (index+1)%50==0: print(f'processed={index+1}/{len(files)} accepted={len(rows)}',flush=True)
    finally: recognizer.close()
    (a.output/'digits.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (a.output/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest),encoding='utf-8')
    (a.output/'rejected_images.json').write_text(json.dumps([r for r in manifest if r['status']!='accepted'],ensure_ascii=False,indent=2),encoding='utf-8')
    report=dict(source=str(a.source),total_images=len(files),accepted=len(rows),status_counts=dict(Counter(r['status'] for r in manifest)),
                class_hand_counts={label:{hand:sum(r['label']==label and r['physical_hand']==hand for r in rows) for hand in ('left','right')} for label in sorted({r['label'] for r in manifest})},
                missing_labels=sorted(set(LABELS)-{r['label'] for r in rows}),
                note='Folder labels are human-provided; landmarks/bboxes are MediaPipe pseudo annotations, not manually verified ground truth.')
    (a.output/'dataset_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    sheet=Image.new('RGB',(4*340,((len(previews)+3)//4)*215),'#f3f4f9'); pen=ImageDraw.Draw(sheet)
    for i,((label,hand),pic) in enumerate(sorted(previews.items())):
        x,y=(i%4)*340,(i//4)*215
        sheet.paste(pic,(x+10,y+25)); pen.text((x+10,y+5),f'{label} / {hand}',fill='black')
    sheet.save(a.output/'annotation_contact_sheet.jpg')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
