import cv2
import numpy as np
from collections import namedtuple

Section = namedtuple("Section", "up, down, high")

def split2sec(split, I):
    sec = []
    sec_s = 0
    for interval in split:
        sec_e = sec_s + interval
        print(sec_s, sec_e)
        mask = np.where((sec_s<I) & (I<=sec_e), 1, 0)
        print(I[0,0], (I-sec_e)[0,0], (sec_s-I)[0,0])
        sec.append(Section._make([
            mask*(I-sec_s)*255/interval, mask*(sec_e-I)*255/interval, mask*255]))
        sec_s = sec_e
    return sec


# split = [255,255,255]
split = [255,255,255]
# split = [255,255,255,255,255,255,765]   #,255,255


max_val = sum(split)+1
I = np.zeros((100, max_val), np.float)
for i in range(max_val): I[:, i] = i
# I = np.ones((400,400), np.float)*510
sec = split2sec(split, I)
print(sec)

        
# blue  = sec[0].up + sec[1].high + sec[2].high
# green = sec[1].up + sec[2].high
# red   = sec[2].up

blue  = sec[2].up
green = sec[1].up + sec[2].high
red   = sec[0].up + sec[1].high + sec[2].high
        
# blue  = sec[0].up   + sec[1].high + sec[2].high + sec[3].down               + sec[5].up +\
        # sec[6].high #+ sec[7].high + sec[8].high
# green = sec[1].up   + sec[2].high + sec[3].down + sec[4].up   + sec[5].high +\
        # sec[6].high #+ sec[7].up   + sec[8].high
# red   = sec[2].up   + sec[3].high + sec[4].high + sec[5].high +\
        # sec[6].down #              + sec[8].up

img = np.stack([ blue.astype(np.uint8),
                 green.astype(np.uint8),
                 red.astype(np.uint8) ], axis=-1)
print(img[0,0,:])
cv2.imwrite('test.png', img)



    
