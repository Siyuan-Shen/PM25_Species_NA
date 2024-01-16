import torch
import numpy as np
import torch
import torch.nn as nn
import os
import gc
from sklearn.model_selection import RepeatedKFold
import random
import csv

from Training_pkg.iostream import load_TrainingVariables, load_geophysical_biases_data, load_geophysical_species_data, load_monthly_obs_data, Learning_Object_Datasets
from Training_pkg.utils import *
from Training_pkg.Model_Func import train, predict
from Training_pkg.data_func import normalize_Func, get_trainingdata_within_sart_end_YEAR
from Training_pkg.Statistic_Func import regress2, linear_regression, Cal_RMSE
from Training_pkg.Net_Construction import *

from Evaluation_pkg.utils import *
from Evaluation_pkg.data_func import GetXIndex,GetYIndex,Get_XY_indices, Get_XY_arraies, Get_final_output, ForcedSlopeUnity_Func, CalculateAnnualR2, CalculateMonthR2, calculate_Statistics_results
from Evaluation_pkg.iostream import save_trained_model, output_text, save_loss_accuracy, save_data_recording, load_data_recording, AVD_output_text, Output_Text_Sites_Number, save_BLOO_data_recording, save_BLOO_loss_accuracy, load_BLOO_data_recording, load_BLOO_loss_accuracy

from visualization_pkg.Assemble_Func import plot_save_loss_accuracy_figure

def BLOO_AVD_Spatial_CrossValidation(width, height, sitesnumber,start_YYYY, TrainingDatasets):
    # *------------------------------------------------------------------------------*#
    ##   Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    ### Get training data, label data, initial observation data and geophysical species
    
    SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
    geophysical_species, lat, lon = load_geophysical_species_data(species=species)
    true_input, mean, std = Learning_Object_Datasets(bias=bias,Normalized_bias=normalize_bias,Normlized_Speices=normalize_species,Absolute_Species=absolute_species,Log_PM25=log_species,species=species)
    Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets)
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    nchannel   = len(channel_names)
    seed       = 19980130
    typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
    site_index = np.array(range(sitesnumber))
    
    rkf = RepeatedKFold(n_splits=kfold, n_repeats=repeats, random_state=seed)
    # *------------------------------------------------------------------------------*#
    ## Begining the Cross-Validation.
    ## Multiple Models will be trained in each fold.
    # *------------------------------------------------------------------------------*#
    final_data_recording, obs_data_recording, geo_data_recording, testing_population_data_recording, training_final_data_recording, training_obs_data_recording, training_dataForSlope_recording = initialize_AVD_DataRecording(beginyear=beginyears[0],endyear=endyears[-1])
    count = 0
    test_index_number = np.array([],dtype = int)
    train_index_number = np.array([],dtype=int)
    for init_train_index, test_index in rkf.split(site_index):
        print('Initial Train index: ',len(init_train_index))
        train_index = GetBufferTrainingIndex(test_index=test_index,train_index=init_train_index,buffer=Buffer_size,sitelat=lat,sitelon=lon)
        test_index_number = np.append(test_index_number,len(test_index))
        train_index_number = np.append(train_index_number,len(train_index))
        for imodel in range(len(beginyears)):
            Normalized_TrainingData = get_trainingdata_within_sart_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=beginyears[imodel],training_end_YYYY=endyears[imodel],start_YYYY=start_YYYY,sitesnumber=sitesnumber)
            X_Training_index, X_Testing_index, Y_Training_index, Y_Testing_index = Get_XY_indices(train_index=train_index,test_index=test_index,beginyear=beginyears[imodel],endyear=endyears[imodel], sitesnumber=sitesnumber)
            X_train, X_test, y_train, y_test = Get_XY_arraies(Normalized_TrainingData=Normalized_TrainingData,true_input=true_input,X_Training_index=X_Training_index,X_Testing_index=X_Testing_index,Y_Training_index=Y_Training_index,Y_Testing_index=Y_Testing_index)

            # *------------------------------------------------------------------------------*#
            ## Training Process.
            # *------------------------------------------------------------------------------*#

            if ResNet_setting:
                block = resnet_block_lookup_table(ResNet_Blocks)
                cnn_model = ResNet(nchannel=nchannel,block=block,blocks_num=ResNet_blocks_num,num_classes=1,include_top=True,groups=1,width_per_group=width)#cnn_model = Net(nchannel=nchannel)
    
 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cnn_model.to(device)
            torch.manual_seed(21)
            train_loss, train_acc, valid_losses, test_acc  = train(model=cnn_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, BATCH_SIZE=batchsize, learning_rate=lr0, TOTAL_EPOCHS=epoch)
            save_trained_model(cnn_model=cnn_model, model_outdir=model_outdir, typeName=typeName, version=version, species=species, nchannel=nchannel, special_name=special_name, count=count, width=width, height=height)
            for iyear in range((endyears[imodel]-beginyears[imodel]+1)):
                yearly_test_index   = GetXIndex(index=test_index, beginyear=(beginyears[imodel]+iyear),endyear=(beginyears[imodel]+iyear), sitenumber=sitesnumber)
                yearly_train_index  = GetXIndex(index=train_index, beginyear=(beginyears[imodel]+iyear),endyear=(beginyears[imodel]+iyear), sitenumber=sitesnumber)
                yearly_test_input  = Normalized_TrainingData[yearly_test_index,:,:,:]
                yearly_train_input = Normalized_TrainingData[yearly_train_index,:,:,:]
                yearly_test_Yindex  = GetYIndex(index=test_index,beginyear=(beginyears[imodel]+iyear), endyear=(beginyears[imodel]+iyear), sitenumber=sitesnumber)
                yearly_train_Yindex = GetYIndex(index=train_index,beginyear=(beginyears[imodel]+iyear), endyear=(beginyears[imodel]+iyear), sitenumber=sitesnumber)
                
                Validation_Prediction = predict(yearly_test_input, cnn_model, 3000)
                Training_Prediction   = predict(yearly_train_input, cnn_model, 3000)
                final_data = Get_final_output(Validation_Prediction, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean,std,yearly_test_Yindex)
                train_final_data = Get_final_output(Training_Prediction, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean, std,yearly_train_Yindex)
                if ForcedSlopeUnity:
                    final_data = ForcedSlopeUnity_Func(train_final_data=train_final_data,train_obs_data=SPECIES_OBS[yearly_train_Yindex]
                                                   ,test_final_data=Validation_Prediction,train_area_index=train_index,test_area_index=test_index,
                                                   endyear=beginyears[imodel]+iyear,beginyear=beginyears[imodel]+iyear,EachMonth=EachMonthForcedSlopeUnity)

                # *------------------------------------------------------------------------------*#
                ## Recording observation and prediction for this model this fold.
                # *------------------------------------------------------------------------------*#

                Validation_obs_data   = SPECIES_OBS[yearly_test_Yindex]
                Training_obs_data     = SPECIES_OBS[yearly_train_Yindex]
                Geophysical_test_data = geophysical_species[yearly_test_Yindex]

                for imonth in range(len(MONTH)):
                    final_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]]          = np.append(final_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], final_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                    obs_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]]            = np.append(obs_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], Validation_obs_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                    geo_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]]            = np.append(geo_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], Geophysical_test_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                    training_final_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]] = np.append(training_final_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], train_final_data[imonth*len(train_index):(imonth+1)*len(train_index)])
                    training_obs_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]]   = np.append(training_obs_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], Training_obs_data[imonth*len(train_index):(imonth+1)*len(train_index)])
                 
        count += 1

    test_CV_R2, train_CV_R2, geo_CV_R2, RMSE_CV_R2, NRMSE_CV_R2, slope_CV_R2 = calculate_Statistics_results(test_beginyear=test_beginyear, test_endyear=test_endyear,
                                                                                                                final_data_recording=final_data_recording, obs_data_recording=obs_data_recording,
                                                                                                                geo_data_recording=geo_data_recording, training_final_data_recording=training_final_data_recording,
                                                                                                                training_obs_data_recording=training_obs_data_recording)
        
    txtfile_outdir = txt_outdir + '{}/{}/Results/results-BLOOCV/'.format(species, version)
    if not os.path.isdir(txtfile_outdir):
        os.makedirs(txtfile_outdir)
    
    txt_outfile =  txtfile_outdir + 'BLOO-{}km-SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.csv'.format(Buffer_size,typeName,species,version,nchannel,width,height,special_name)

    Output_Text_Sites_Number(outfile=txt_outfile, status='w', train_index_number=train_index_number, test_index_number=test_index_number, buffer=Buffer_size)
    AVD_output_text(outfile=txt_outfile,status='a', test_CV_R2=test_CV_R2, train_CV_R2=train_CV_R2, geo_CV_R2=geo_CV_R2, RMSE_CV_R2=RMSE_CV_R2, NRMSE_CV_R2=NRMSE_CV_R2,
                        slope_CV_R2=slope_CV_R2)
    save_BLOO_loss_accuracy(model_outdir=model_outdir,loss=train_loss, accuracy=train_acc,valid_loss=valid_losses, valid_accuracy=test_acc,typeName=typeName,
                       version=version,species=species, nchannel=nchannel,special_name=special_name, width=width, height=height,buffer_radius=Buffer_size)
    final_longterm_data, obs_longterm_data = get_annual_longterm_array(beginyear=test_beginyear, endyear=test_endyear, final_data_recording=final_data_recording,obs_data_recording=obs_data_recording)
    save_BLOO_data_recording(obs_data=obs_longterm_data,final_data=final_longterm_data,
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH='Annual',nchannel=nchannel,special_name=special_name,width=width,height=height,buffer_radius=Buffer_size)
           
    for imonth in range(len(MONTH)):
        final_longterm_data, obs_longterm_data = get_monthly_longterm_array(beginyear=test_beginyear, imonth=imonth,endyear=test_endyear, final_data_recording=final_data_recording,obs_data_recording=obs_data_recording)
        save_BLOO_data_recording(obs_data=obs_longterm_data,final_data=final_longterm_data,
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH=MONTH[imonth],nchannel=nchannel,special_name=special_name,width=width,height=height,buffer_radius=Buffer_size)
      

    return