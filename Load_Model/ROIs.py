from keras.models import load_model
import segmentation_models
from sklearn.cluster import KMeans
import cv2
import numpy as np
import random
from queue import Queue
def DFS(ROI,ROI_IMG,vis,a,b):
    
    q = Queue()
    q.put((a,b))
    row=[-1,0,0,1]
    col=[0,-1,1,0]
    ht,wd=ROI_IMG.shape
    vis[a,b] = 1
    while(q.qsize()):
        m=q.qsize()
        i,j=q.get()
        ROI.append([i,j])
        for k in range(4):
            x=i+row[k]
            y=j+col[k]
            if(x>=0 and x<ht and y>=0 and y<wd):
                if(vis[x,y]!=1 and ROI_IMG[x,y]==255):
                    q.put((x,y))
                    vis[x,y] = 1

def get_ROI(ROI_IMG,min_area):
    ROI=[]
    vis=ROI_IMG*0
    hi,wd=ROI_IMG.shape
    for i in  range(hi):
        for j in range(wd):
            if(vis[i,j]==0 and ROI_IMG[i,j]==255):
                DFS(ROI,ROI_IMG,vis,i,j)
                if(len(ROI)>=min_area):
                    return ROI
                ROI=[]
    return ROI 
def get_Resized_ROI(ROI_IMG,scale_h,scale_w):
    h,w=ROI_IMG.shape
    h,w=int(scale_h*h),int(scale_w*w)
    resized_ROI_image=cv2.resize(ROI_IMG,(w, h))
    resized_ROI_image=255*(resized_ROI_image==255)
    
    ROI=[]
    for i in range(h):
        for j in range(w):
            if(resized_ROI_image[i,j]==255):
                ROI.append([i,j])
    ROI=np.array(ROI)
    return ROI

def get_ROIs(points,labels,min_area,scale_h,scale_w):
    N=len(labels)
    ROIs=[]
    for i in range(N):
        if(len(points[labels==i])==0):
            continue
        x1=min(points[labels==i][:,1])
        y1=min(points[labels==i][:,0])
        x2=max(points[labels==i][:,1])
        y2=max(points[labels==i][:,0])
        
        ROI_IMG = np.zeros((y2-y1+1,x2-x1+1))
        ROI_IMG[points[labels==i][:,0]-y1,points[labels==i][:,1]-x1] = 255
        
        ROI = get_ROI(ROI_IMG,min_area)
        if(len(ROI) == 0):
            continue
        ROI = np.array(ROI)
        
        ROI_IMG = ROI_IMG*0
        ROI_IMG[ROI[:,0],ROI[:,1]] = 255
        
        ROI = get_Resized_ROI(ROI_IMG,scale_h,scale_w)
        ROI[:,0]+=int(y1*scale_h)
        ROI[:,1]+=int(x1*scale_w)
        
        ROIs.append(ROI)
    return ROIs

def get_required_ROIs(model,class_number, img_path, required_rois):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H,W,_=image.shape
    h,w=256,256
    if(H<h):
        h=H
    if(W<w):
        w=W

    mask=cv2.resize(image, (w, h))
    mask = np.expand_dims(mask, axis=0)
    mask = model.predict(mask,verbose=0)
    mask=mask[..., class_number].squeeze()

    points=[]
    for i in range(h):
        for j in range(w):
            if(mask[i,j]>0.9):
                points.append([i,j])
    points=np.array(points)

    scale_h=H/h
    scale_w=W/w
    thresold_area=20000*(H*W/1200000)
    ROI_area=int(thresold_area/(scale_h*scale_w))
    n_ROIs = points.shape[0]//ROI_area

    k_means = KMeans(n_clusters=n_ROIs,max_iter=300)
    k_means.fit(points)
    labels= k_means.labels_

    min_roi_area=int(15000*(H*W/1200000)/(scale_h*scale_w))
    ROIs=get_ROIs(points,labels,min_roi_area,scale_h,scale_w)
    
    # check required_rois
    if required_rois>0 and required_rois<len(ROIs):
        rois=random.sample(ROIs, required_rois)
    else:
        rois=ROIs

    for r in rois:
        image[(r[:,0]),(r[:,1]),0]=0
        image[(r[:,0]),(r[:,1]),1]=130
        image[(r[:,0]),(r[:,1]),2]=0

    return rois,image