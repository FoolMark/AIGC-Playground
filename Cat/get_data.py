import os
import cv2
import numpy as np
from PIL import Image
if __name__ == '__main__':
    SRC_ROOT = 'I:\\Dataset\\Cat\\archive'
    dirs = os.listdir(SRC_ROOT)
    try:
        for dir in dirs:
            os.mkdir(os.path.join('./images/',dir))
    except:
        pass
    DIRS = os.listdir(SRC_ROOT)
    ct = 0
    dst_lmk = np.float32([[96.,64.],[160.,64.],[128.,112.]])
    for DIR in DIRS:
        SRC_PATH = os.path.join(SRC_ROOT,DIR)
        IMG_DIRS = os.listdir(SRC_PATH)
        for IMG in IMG_DIRS:
            if IMG.split('.')[-1] != 'jpg':
                continue
            else:
                img = np.array(Image.open(os.path.join(SRC_PATH,IMG)))
                with open(os.path.join(SRC_PATH,IMG+'.cat'),'r') as f:
                    lmks = f.readline().split()[1:7]
                src_lmk = np.zeros(6).astype(np.float32)
                for i in range(6):
                    src_lmk[i] = np.float32(lmks[i])
                src_lmk = src_lmk.reshape((3,2))
                M = cv2.getAffineTransform(src_lmk,dst_lmk)
                dst_img = cv2.warpAffine(img,M,(256,256))[:,:,::-1]
                DST_IMG = os.path.join('images',DIR,IMG)
                cv2.imwrite(DST_IMG,dst_img)
            ct += 1    
            if ct % 500 == 0:
                print(ct)    
