import cv2
import numpy as np
def calculate_CCs(image_fps,ROIs):
    GCC=[]
    RCC=[]
    BCC=[]
    for img_path in image_fps:
        img = cv2.imread(img_path)
        for roi in ROIs:
            B=img[roi[:,0],roi[:,1],0]
            G=img[roi[:,0],roi[:,1],1]
            R=img[roi[:,0],roi[:,1],2]
            Total_sum=img[roi[:,0],roi[:,1]].sum()
            GCC.append(G.sum()/Total_sum)
            RCC.append(R.sum()/Total_sum)
            BCC.append(B.sum()/Total_sum)
    GCC=np.array(GCC)
    BCC=np.array(BCC)
    RCC=np.array(RCC)
    return RCC, GCC, BCC