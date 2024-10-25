import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import os
import keras
from keras.models import load_model
import segmentation_models

from phenoAI.Load_Model.ROIs import get_required_ROIs
from phenoAI.Load_Model.image_fps import get_image_fps
from phenoAI.Load_Model.calculate_CCs import calculate_CCs
from phenoAI.Load_Model.pheno_Parameters import extract_phenology_Parameters,plot_CC
from phenoAI.Load_Model.save_CCs_TimeSeries import save_CCs_TimeSeries
from phenoAI.Load_Model.extract_model import get_classes_model


class loadModel:

    def __init__(self,model,dataset_path,class_name,date_pattern,required_rois=5):
        # check required_rois
        assert type(required_rois)==int, "Type error: Data type of required number of ROIs is not an integer" 
        
        print('Loading Vegetation segmentation Model...',end='\r')
        model,classes=get_classes_model(model)
        print('Loading Vegetation segmentation Model...|done')
        
        classes=classes.split('$')
        class_number=-1
        for i in range(len(classes)):
            if classes[i]==class_name:
                class_number=i
                break
        assert class_number!=-1, class_name+" is not present in category list "+str(classes)
        # check dataset path
        assert os.path.isdir(dataset_path), dataset_path+", path does not exist or is not valid" 
        
        print('Importing Dataset...',end='\r')
        self.DoY,self.image_fps=get_image_fps(dataset_path,date_pattern,required_rois)
        print('Importing Dataset...|done')

        assert (len(self.image_fps)!=0), dataset_path+", does not contain images" 
        
        img_path=random.choice(self.image_fps)
        
        print('Creating ROIs in the vegetation segment...',end='\r')
        self.ROIs, self.roi_img = get_required_ROIs(model,class_number,img_path,required_rois)
        print('Creating ROIs in the vegetation segment...|done')
        
        print('Extracting Chromatic chromatic coordinate from Images...',end='\r')
        self.RCC, self.GCC, self.BCC = calculate_CCs(self.image_fps,self.ROIs)
        print('Extracting Chromatic chromatic coordinate from Images...|done')
        
        print('Calculating phenological Parameters...',end='\r')
        initial_parameters = np.array([0.3, 0.5, 0.04, 129, 0.03, 265])
        self.pRCC = extract_phenology_Parameters(self.DoY,self.RCC,initial_parameters)
        self.pGCC = extract_phenology_Parameters(self.DoY,self.GCC,initial_parameters)
        self.pBCC = extract_phenology_Parameters(self.DoY,self.BCC,initial_parameters)
        print('Calculating phenological Parameters...|done')
        

    def showROIs(self):
        h,w,_=self.roi_img.shape
        plt.figure(figsize=(10, 10*w/h))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.roi_img)
        plt.show()
        plt.close()

    def saveROIsImage(self,path_location):
        # check path 

        img = cv2.cvtColor(self.roi_img, cv2.COLOR_BGR2RGB)

        if not (path_location[-4:]=='.JPG' or path_location[-4:]=='.jpg' or path_location[-4:]=='.png'
        or path_location[-4:]=='.PNG' or path_location[-4:]=='JPEG' or path_location[-4:]=='jpeg'):

            assert os.path.exists(path_location), path_location+", does not exist or is not valid"
            path_location+="/ROIs_image.jpg"

        assert cv2.imwrite(path_location,img), path_location+", does not exist or is not valid"
        print('Saved')
    
    def getDoY(self):
        return self.DoY
    def getGCC(self):
        return self.GCC
    def getBCC(self):
        return self.BCC
    def getRCC(self):
        return self.RCC

    def setInitialParameters(self,min, max, slope1, SoS, slope2 , EoS):
        p=np.array([min, max, slope1, SoS, slope2 , EoS])
        self.pRCC = extract_phenology_Parameters(self.DoY,self.RCC,p)
        self.pGCC = extract_phenology_Parameters(self.DoY,self.GCC,p)
        self.pBCC = extract_phenology_Parameters(self.DoY,self.BCC,p)
        
    def setInitialGCCParameters(self,min, max, slope1, SoS, slope2 , EoS):
        p=np.array([min, max, slope1, SoS, slope2 , EoS])
        self.pGCC = extract_phenology_Parameters(self.DoY,self.GCC,p)

    def setInitialRCCParameters(self,min, max, slope1, SoS, slope2 , EoS):
        p=np.array([min, max, slope1, SoS, slope2 , EoS])
        self.pRCC = extract_phenology_Parameters(self.DoY,self.RCC,p)

    def setInitialBCCParameters(self,min, max, slope1, SoS, slope2 , EoS):
        p=np.array([p1,p2,p3,p4,p5,p6])
        self.pBCC = extract_phenology_Parameters(self.DoY,self.BCC,p)

    def extractGCCParameters(self):
        return self.pGCC
    def extractBCCParameters(self):
        return self.pBCC
    def extractRCCParameters(self):
        return self.pRCC
        
    def plotGCC(self):
        plot_CC(self.DoY,self.GCC,self.pGCC,"GCC")
    def plotRCC(self):
        plot_CC(self.DoY,self.RCC,self.pRCC,"RCC")
    def plotBCC(self):
        plot_CC(self.DoY,self.BCC,self.pBCC,"BCC")

    def saveBCCPlot(self,path_location):
        plot_CC(self.DoY,self.BCC,self.pBCC,"BCC",True,path_location)
        print('Saved')
    def saveGCCPlot(self,path_location):
        plot_CC(self.DoY,self.GCC,self.pGCC,"GCC",True,path_location)
        print('Saved')
    def saveRCCPlot(self,path_location):
        plot_CC(self.DoY,self.RCC,self.pRCC,"RCC",True,path_location)
        print('Saved')

    def saveCCsTimeSeries(self, path_location):
        save_CCs_TimeSeries(path_location,self.DoY,self.RCC, self.GCC, self.BCC,self.pRCC,self.pGCC,self.pBCC)
