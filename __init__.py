from phenoAI.Load_Model.main import loadModel
from phenoAI.Train_Pheno_AI.main import trainPhenoAI

print(
'''Documentation:
This Package contains two module-

1.trainPhenoAI(dataset_path, epochs,learning_rate, batch_size,is_augmentation): To make phenological Model by training through Image Dataset.
dataset_path: dataset location(Containing images and labels).
Optional Parameters:
epochs,learning_rate, batch_size,is_augmentation(True if we want to add more data using Augmentation)

It has below functions:
reTrain(epochs): For Training the model for more epochs
performanceReport(): Gives us loss(or metrics) vs epochs graphs, scores of test data prediction
saveModel(path_location): Where phenological Model to be saved. You can provide folder location(model will be saved by default name in that folder) or folder location along with model name in zip format, for example- 'C:/your_model_name.zip'

2. loadModel(model,dataset_path,class_name,date_pattern,required_rois): To make analysis object. It provides phenological parameters, chronological coefficients, plots etc.
Here, model: it can be model zip file location obtained by trainPhenoAI module
dataset_path: Images location
class_name: name of category on which you want to analyse. Name should be matching with one of the names used while labelling the dataset
date_pattern: This is used for extracting date from image name. This should match with file name. It should contains 'yyyy'(for year), 'dd'(for day) and 'mm'(for month), and '*'. '*' used for covering zero aur more characters in the file name. Example: For 'Sd_20221204-15_sdg.jpg','pd_20191024-1ald.jpg' etc, pattern can be:'*yyyymmdd-*' or '*_yyyymmdd-*' or '*yyyymmdd*'.
required_rois:number of ROIs

This module object associates below functions:
showROIs(): shows images with ROIs selected
saveROIsImage(path_location): save this ROIs Image in the provided path_location.
setInitialParameters(min, max, slope1, SoS, slope2 , EoS): set initial parameters of plotting function for all GCC,BCC,RCC
setInitialGCCParameters(min, max, slope1, SoS, slope2 , EoS): same used for individual GCC
setInitialBCCParameters(min, max, slope1, SoS, slope2 , EoS): for BCC
setInitialECCParameters(min, max, slope1, SoS, slope2 , EoS): for RCC
extractGCCParameters(): Gives 6 phenological parameters
extractBCCParameters(): for BCC
extractRCCParameters(): for RCC
plotGCC(): plots GCC verses Day of Year(DoY) graph
plotGCC(),plotRCC(): for BCC and RCC
saveGCCPlot(path_location):save GCC vs DoY plot images in given location
saveBCCPlot(path_location),saveRCCPlot(path_location) for BCC and RCC
saveCCsTimeSeries(path_location): save all record GCC,BCC,RCC, 6 parameters in Excel file in the given location

'''
)