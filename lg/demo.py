"""Explicit synthetic observations for reproducible offline demonstrations."""
import cv2
import numpy as np
from .core import Hand


def scene():
    image = np.full((720,960,3),(40,33,24),np.uint8)
    for x in range(0,960,80):
        cv2.line(image,(x,0),(x,720),(65,55,43),2)
    cv2.rectangle(image,(300,190),(660,540),(120,170,195),-1)
    qr = cv2.QRCodeEncoder_create().encode('DEMO-PARCEL-20261020-001')
    qr = cv2.copyMakeBorder(qr,4,4,4,4,cv2.BORDER_CONSTANT,value=255)
    qr = cv2.resize(qr,(210,210),interpolation=cv2.INTER_NEAREST)
    image[255:465,375:585] = cv2.cvtColor(qr,cv2.COLOR_GRAY2BGR)
    cv2.putText(image,'OFFLINE SIMULATION / NO CAMERA',(25,45),cv2.FONT_HERSHEY_SIMPLEX,.8,(50,210,255),2)
    return image


def observations(digits=(1,1)):
    hands = []
    for i,(x,y) in enumerate(((.28,.23),(.72,.78))):
        p = np.zeros((21,3),dtype=np.float32)
        for j in range(21):
            p[j,:2] = [x+(j%4-1)*.012,y+.06+(j//4)*.015]
        p[8,:2] = [x,y]
        p[0,:2] = [x,y+.15]
        box = (*p[:,:2].min(0),*p[:,:2].max(0))
        hands.append(Hand(p,digits[i],box,i))
    return hands


def action_observations(kind='open'):
    """Explicit synthetic joints for testing temporal action cycles without camera."""
    if kind is None: return []
    p=np.zeros((21,3),np.float32); p[0]=[.5,.85,0]
    for base,x in zip((5,9,13,17),(.38,.47,.56,.65)):
        p[base:base+4]=[[x,.62,0],[x,.42,0],
                      [x,.55 if kind=='fist' else .3,0],[x,.68 if kind=='fist' else .2,0]]
    p[1:5]=[[.4,.75,0],[.3,.68,0],[.22,.6,0],[.18,.52,0]]
    if kind=='fist': p[4]=[.43,.64,0]
    if kind=='pinch': p[4]=p[8]+[.02,0,0]
    return [Hand(p,None,(*p[:,:2].min(0),*p[:,:2].max(0)),0)]
