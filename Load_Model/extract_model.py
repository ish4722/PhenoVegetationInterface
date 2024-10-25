from zipfile import ZipFile
import os
import keras
from keras.models import load_model
import segmentation_models
import shutil
def get_classes_model(model):

    if(type(model)==str):
        assert model.endswith('.zip') or model.endswith('.ZIP') , model+", should be a zip file or getPhenoModel object's model"
        assert os.path.isfile(model), model+", path does not exist or is not valid"

        with ZipFile(model) as zip:
            try:
                classes=str(zip.read('classes.txt'))[2:-1]
            except:
                assert 0,model+", is not valid model zip file"
                
            listOfFileNames = zip.namelist()

            assert len(listOfFileNames)==2, model+", is not valid model zip file"

            for fileName in listOfFileNames:
                if fileName.endswith('.h5'):
                    folder=model+fileName
                    zip.extract(fileName,folder)
                    model=load_model(folder+"/"+fileName,compile=False)
                    shutil.rmtree(folder)
                    return model,classes

            assert 0,model+", is not valid model zip file"
    
    return model
