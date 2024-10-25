import os
import tensorflow as tf
import keras
import segmentation_models as sm
from zipfile import ZipFile
import datetime
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from phenoAI.Train_Pheno_AI.Json_to_mask import get_classes
from phenoAI.Train_Pheno_AI.data_preparation import Dataset,Dataloder
from phenoAI.Train_Pheno_AI.Augmentation import augmentation



class trainPhenoAI:
    def __init__(self,dataset_path, epochs = 15,learning_rate = 0.001, batch_size = 0,is_augmentation = True):

        assert os.path.isdir(dataset_path), dataset_path+", path does not exist or is not valid" 

        print('Saving Image masks of Json labels...',end='\r')
        self.classes = get_classes(dataset_path)
        print('Saving Image masks of Json labels...|done')

        classes=self.classes
        print('Orderwise Classes names: ',classes.split('$'))

        print('Loading Dataset images and masks for training...', end='\r')
        if is_augmentation:
            augmentation = augmentation()
        else:
            augmentation = None
        dataset=Dataset(dataset_path,classes,augmentation)
        print('Loading Dataset images and masks for training...|done')

        BATCH_SIZE = max(1,int(0.03*len(dataset)))
        if batch_size!=0:
            BATCH_SIZE=batch_size

        dataset_indexes = list(range(0,len(dataset)))

        train_dataset, test_dataset = train_test_split(dataset_indexes, test_size = 0.15, random_state = 42)

        self.train_dataloader = Dataloder(train_dataset,dataset, batch_size=BATCH_SIZE)
        self.test_dataloader = Dataloder(test_dataset,dataset, batch_size=1)

        BACKBONE = 'efficientnetb3'
        LR = learning_rate
        EPOCHS = epochs
        preprocess_input = sm.get_preprocessing(BACKBONE)
        n_classes = len(self.classes.split('$'))+1
        activation = 'softmax'
        sm.set_framework('tf.keras')
        self.model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
        
        optim = keras.optimizers.Adam(LR)
        dice_loss = sm.losses.JaccardLoss()
        focal_loss = sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)

        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
        self.model.compile(optim, total_loss, metrics)

        print("Training the model...")

        self.history = self.model.fit(
            self.train_dataloader, 
            steps_per_epoch=len(self.train_dataloader), 
            epochs=EPOCHS, 
            callbacks = [keras.callbacks.ReduceLROnPlateau(), tf.keras.callbacks.EarlyStopping('val_loss', patience=5)],  
            validation_data=self.test_dataloader, 
            validation_steps=len(self.test_dataloader),     
        )

    def reTrain(self,epochs):

        print('Training for',epochs,'more epochs.')
        history = self.model.fit(
            self.train_dataloader, 
            steps_per_epoch=len(self.train_dataloader), 
            epochs=epochs, 
            callbacks = [keras.callbacks.ReduceLROnPlateau(), tf.keras.callbacks.EarlyStopping('val_loss', patience=5)], 
            validation_data=self.test_dataloader, 
            validation_steps=len(self.test_dataloader),     
        )
        self.history.history['loss'] = self.history.history['loss'] + history.history['loss']
        self.history.history['val_loss'] = self.history.history['val_loss'] + history.history['val_loss']
        self.history.history['iou_score'] = self.history.history['iou_score'] + history.history['iou_score']
        self.history.history['val_iou_score'] = self.history.history['val_iou_score'] + history.history['val_iou_score']
        self.history.history['f1-score'] = self.history.history['f1-score'] + history.history['f1-score']
        self.history.history['val_f1-score'] = self.history.history['val_f1-score'] + history.history['val_f1-score']

    def performanceReport(self):

        plt.figure(figsize=(10, 3))
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
        plt.figure(figsize=(10, 3))
        plt.plot(self.history.history['iou_score'])
        plt.plot(self.history.history['val_iou_score'])
        plt.title('Model iou_score')
        plt.ylabel('iou_score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()      

        scores = self.model.evaluate(self.test_dataloader,verbose=0)
        print('Test performance:')  
        print("Loss: ",scores[0])
        print("iou :",scores[1])
        print("f1-score :",scores[2])

    def saveModel(self,path_location):
        current_time = datetime.datetime.now()
        current_time=current_time.strftime("%m-%d-%Y_%H.%M")
        curr_name=current_time+"_model"
        zip_name="/"+curr_name+".zip"
        if (path_location[-4:]=='.zip' or path_location[-4:]=='.ZIP'):
            i=-1
            l=len(path_location)
            while(i>-l and (path_location[i]!='/' and path_location[i]!='\\')):
                i-=1
            if(i==-l):
                zip_name=path_location
                path_location=""
            else:
                zip_name=path_location[i:]
                path_location=path_location[0:i]

        assert os.path.exists(path_location), path_location+ " , path does not exist or is not valid"

        zip_name=path_location+zip_name
        model_path=path_location+curr_name+".h5"

        self.model.save(model_path)
        with ZipFile(zip_name, 'w') as zip:
            zip.write(model_path,curr_name+'_Model.h5')
            zip.writestr('classes.txt',self.classes)
        os.remove(model_path)
        print("Model saved to ",zip_name)
