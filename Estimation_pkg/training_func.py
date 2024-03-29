import numpy as np
import gc
import os
import torch
import torch.nn as nn
from Training_pkg.iostream import load_TrainingVariables, load_geophysical_biases_data, load_geophysical_species_data, load_monthly_obs_data, Learning_Object_Datasets
from Training_pkg.utils import *
from Training_pkg.Model_Func import train, predict
from Training_pkg.data_func import normalize_Func, get_trainingdata_within_sart_end_YEAR
from Training_pkg.Statistic_Func import regress2, linear_regression, Cal_RMSE
from Training_pkg.Net_Construction import *


from Evaluation_pkg.utils import *
from Evaluation_pkg.data_func import GetXIndex,GetYIndex

from Estimation_pkg.iostream import save_trained_model_forEstimation

def Train_Model_forEstimation(train_beginyears, train_endyears, width, height, sitesnumber,start_YYYY, TrainingDatasets):
    true_input, mean, std = Learning_Object_Datasets(bias=bias,Normalized_bias=normalize_bias,Normlized_Speices=normalize_species,Absolute_Species=absolute_species,Log_PM25=log_species,species=species)
    Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets)
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    nchannel   = len(channel_names)
    seed       = 19980130
    typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
    site_index = np.array(range(sitesnumber))
    
    for imodel in range(len(train_beginyears)):
        Normalized_TrainingData    = get_trainingdata_within_sart_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=train_beginyears[imodel],training_end_YYYY=train_endyears[imodel],start_YYYY=start_YYYY,sitesnumber=sitesnumber)
        training_array_index       = GetXIndex(index=site_index,beginyear=train_beginyears[imodel],endyear=train_endyears[imodel],sitenumber=sitesnumber)
        learning_objective_index   = GetYIndex(index=site_index,beginyear=train_beginyears[imodel],endyear=train_endyears[imodel],sitenumber=sitesnumber)
        testing_array_index        = GetXIndex(index=np.array(range(100)),beginyear=train_beginyears[imodel],endyear=train_endyears[imodel],sitenumber=sitesnumber) # These two testing arrrays are meaningless here
        teating_objective_index    = GetYIndex(index=np.array(range(100)),beginyear=train_beginyears[imodel],endyear=train_endyears[imodel],sitenumber=sitesnumber) #
        cnn_model = initial_network(width=width)
        X_train = Normalized_TrainingData[training_array_index,:,:,:]
        y_train = true_input[learning_objective_index]
        X_test  = Normalized_TrainingData[testing_array_index,:,:,:] 
        y_test  = true_input[teating_objective_index]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn_model.to(device)
        torch.manual_seed(21)
        train_loss, train_acc, valid_losses, test_acc  = train(model=cnn_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, BATCH_SIZE=batchsize, learning_rate=lr0, TOTAL_EPOCHS=epoch)
        save_trained_model_forEstimation(cnn_model=cnn_model, model_outdir=model_outdir, typeName=typeName, version=version, species=species, nchannel=nchannel, special_name=special_name,beginyear=train_beginyears[imodel], endyear=train_endyears[imodel], width=width, height=height)
        del X_train, y_train
        gc.collect()
    del true_input, Initial_Normalized_TrainingData
    gc.collect()
    return