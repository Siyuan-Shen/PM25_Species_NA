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
from Evaluation_pkg.data_func import GetXIndex,Get_SecondStage_XIndex,GetYIndex,Get_XY_indices, Get_XY_arraies, Get_final_output, ForcedSlopeUnity_Func, CalculateAnnualR2, CalculateMonthR2, calculate_Statistics_results
from Evaluation_pkg.iostream import load_coMonitor_Population,save_SecondStage_trained_model,save_trained_model, output_text, save_loss_accuracy, save_data_recording, load_data_recording, AVD_output_text, Output_Text_Sites_Number, save_BLOO_data_recording, save_BLOO_loss_accuracy, load_BLOO_data_recording, load_BLOO_loss_accuracy
from Evaluation_pkg.Evaluation_func import Get_SecondStage_GeoSpecies
from visualization_pkg.Assemble_Func import plot_save_loss_accuracy_figure

def BLOO_AVD_Spatial_CrossValidation(buffer_radius,width, height, sitesnumber,start_YYYY, TrainingDatasets):
    # *------------------------------------------------------------------------------*#
    ##   Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    ### Get training data, label data, initial observation data and geophysical species
    beginyears = BLOO_beginyears
    endyears   = BLOO_endyears
    SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
    geophysical_species, lat, lon = load_geophysical_species_data(species=species)
    true_input, mean, std = Learning_Object_Datasets(bias=bias,Normalized_bias=normalize_bias,Normlized_Speices=normalize_species,Absolute_Species=absolute_species,Log_PM25=log_species,species=species)
    Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets)
    population_data = load_coMonitor_Population()
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    nchannel   = Get_MainStream_nchannels()
    seed       = 19980130
    typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
    site_index = np.array(range(sitesnumber))
    
    rkf = RepeatedKFold(n_splits=BLOO_kfold, n_repeats=BLOO_repeats, random_state=seed)
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
        train_index = GetBufferTrainingIndex(test_index=test_index,train_index=init_train_index,buffer=buffer_radius,sitelat=lat,sitelon=lon)
        test_index_number = np.append(test_index_number,len(test_index))
        train_index_number = np.append(train_index_number,len(train_index))
        for imodel in range(len(beginyears)):
            Normalized_TrainingData = get_trainingdata_within_sart_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=beginyears[imodel],training_end_YYYY=endyears[imodel],start_YYYY=start_YYYY,sitesnumber=sitesnumber)
            X_Training_index, X_Testing_index, Y_Training_index, Y_Testing_index = Get_XY_indices(train_index=train_index,test_index=test_index,beginyear=beginyears[imodel],endyear=endyears[imodel], sitesnumber=sitesnumber)
            X_train, X_test, y_train, y_test = Get_XY_arraies(Normalized_TrainingData=Normalized_TrainingData,true_input=true_input,X_Training_index=X_Training_index,X_Testing_index=X_Testing_index,Y_Training_index=Y_Training_index,Y_Testing_index=Y_Testing_index)

            # *------------------------------------------------------------------------------*#
            ## Training Process.
            # *------------------------------------------------------------------------------*#

            cnn_model = initial_network(width=width,main_stream_nchannels=nchannel)
 
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
                if combine_with_GeophysicalSpeceis_Switch:
                    nearest_distance = get_nearest_test_distance(area_test_index=test_index,area_train_index=train_index,site_lat=lat,site_lon=lon)
                    coeficient = get_coefficients(nearest_site_distance=nearest_distance,cutoff_size=cutoff_size,beginyear=beginyears[imodel],
                                              endyear = endyears[imodel])
                    final_data = (1.0-coeficient)*final_data + coeficient * geophysical_species[yearly_test_Yindex]
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
                population_test_data  = population_data[yearly_test_index]
                
                if ~TwoStage_Settings:
                    for imonth in range(len(MONTH)):
                        final_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]]              = np.append(final_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], final_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                        obs_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]]                = np.append(obs_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], Validation_obs_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                        geo_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]]                = np.append(geo_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], Geophysical_test_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                        training_final_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]]     = np.append(training_final_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], train_final_data[imonth*len(train_index):(imonth+1)*len(train_index)])
                        training_obs_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]]       = np.append(training_obs_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], Training_obs_data[imonth*len(train_index):(imonth+1)*len(train_index)])
                        testing_population_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]] = np.append(testing_population_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], population_test_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                if iyear == 0:
                    trainset_Train_Array_recording = yearly_train_input
                    trainset_Final_Array_recording = train_final_data
                    testset_Train_Array_recording = yearly_test_input
                    testset_Final_Array_recording = final_data
                else:
                    trainset_Train_Array_recording = np.append(trainset_Train_Array_recording,yearly_train_input)
                    trainset_Final_Array_recording = np.append(trainset_Final_Array_recording,train_final_data)
                    testset_Train_Array_recording = np.append(testset_Train_Array_recording,yearly_test_input)
                    testset_Final_Array_recording = np.append(testset_Final_Array_recording,final_data)
            if TwoStage_Settings:

                SecondStage_trainset_training_data = Get_SecondStage_GeoSpecies(first_stage_final_data=trainset_Final_Array_recording,first_stage_training_data=trainset_Train_Array_recording,width=width)
                SecondStage_testset_training_data = Get_SecondStage_GeoSpecies(first_stage_final_data=testset_Final_Array_recording,first_stage_training_data=testset_Train_Array_recording,width=width)
                Normalized_SecondStage_trainset_training_data, input_mean, input_std = normalize_Func(inputarray=SecondStage_trainset_training_data)
                Normalized_SecondStage_testset_training_data, input_mean, input_std = normalize_Func(inputarray=SecondStage_testset_training_data)
                Normalized_TrainingData = get_trainingdata_within_sart_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=beginyears[imodel],training_end_YYYY=endyears[imodel],start_YYYY=start_YYYY,sitesnumber=sitesnumber)
                X_Training_index, X_Testing_index, Y_Training_index, Y_Testing_index = Get_XY_indices(train_index=train_index,test_index=test_index,beginyear=beginyears[imodel],endyear=endyears[imodel], sitesnumber=sitesnumber)
                X_train, X_test, y_train, y_test = Get_XY_arraies(Normalized_TrainingData=Normalized_TrainingData,true_input=true_input,X_Training_index=X_Training_index,X_Testing_index=X_Testing_index,Y_Training_index=Y_Training_index,Y_Testing_index=Y_Testing_index)
                
                X_train = np.append(X_train,Normalized_SecondStage_trainset_training_data,axis=1)
                X_test  = np.append(X_test,Normalized_SecondStage_testset_training_data,axis=1)

                # *------------------------------------------------------------------------------*#
                ## Training Process.
                # *------------------------------------------------------------------------------*#

                cnn_model_secondStage = initial_network(width=width,main_stream_nchannels=(nchannel+len(TwoStage_Variables)))
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                cnn_model_secondStage.to(device)
                torch.manual_seed(21)
                train_loss, train_acc, valid_losses, test_acc  = train(model=cnn_model_secondStage, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, BATCH_SIZE=batchsize, learning_rate=lr0, TOTAL_EPOCHS=epoch)
                save_SecondStage_trained_model(cnn_model=cnn_model_secondStage, model_outdir=model_outdir, typeName=typeName, version=version, species=species, nchannel=nchannel, special_name=special_name, count=count, width=width, height=height)
            
                for iyear in range((endyears[imodel]-beginyears[imodel]+1)):
                    yearly_test_index   = GetXIndex(index=test_index, beginyear=(beginyears[imodel]+iyear),endyear=(beginyears[imodel]+iyear), sitenumber=sitesnumber)
                    yearly_train_index  = GetXIndex(index=train_index, beginyear=(beginyears[imodel]+iyear),endyear=(beginyears[imodel]+iyear), sitenumber=sitesnumber)
                    yearly_test_Yindex  = GetYIndex(index=test_index,beginyear=(beginyears[imodel]+iyear), endyear=(beginyears[imodel]+iyear), sitenumber=sitesnumber)
                    yearly_train_Yindex = GetYIndex(index=train_index,beginyear=(beginyears[imodel]+iyear), endyear=(beginyears[imodel]+iyear), sitenumber=sitesnumber)
                    
                    SecondStage_yearly_test_index = Get_SecondStage_XIndex(beginyear=(beginyears[imodel]+iyear),endyear=(beginyears[imodel]+iyear), sitenumber=len(test_index))
                    SecondStage_yearly_train_index = Get_SecondStage_XIndex(beginyear=(beginyears[imodel]+iyear),endyear=(beginyears[imodel]+iyear), sitenumber=len(train_index))

                    yearly_test_input  = Normalized_TrainingData[yearly_test_index,:,:,:]
                    yearly_train_input = Normalized_TrainingData[yearly_train_index,:,:,:]

                    yearly_test_input  = np.append(yearly_test_input, Normalized_SecondStage_testset_training_data[SecondStage_yearly_test_index,:,:,:],axis=1)
                    yearly_train_input = np.append(yearly_train_input,Normalized_SecondStage_trainset_training_data[SecondStage_yearly_train_index,:,:,:],axis=1)

                    Validation_Prediction = predict(yearly_test_input, cnn_model_secondStage, 3000)
                    Training_Prediction   = predict(yearly_train_input, cnn_model_secondStage, 3000)
                    final_data = Get_final_output(Validation_Prediction, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean,std,yearly_test_Yindex)
                    train_final_data = Get_final_output(Training_Prediction, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean, std,yearly_train_Yindex)
                
                    if combine_with_GeophysicalSpeceis_Switch:
                        nearest_distance = get_nearest_test_distance(area_test_index=test_index,area_train_index=train_index,site_lat=lat,site_lon=lon)
                        coeficient = get_coefficients(nearest_site_distance=nearest_distance,cutoff_size=cutoff_size,beginyear=beginyears[imodel],
                                              endyear = endyears[imodel])
                        final_data = (1.0-coeficient)*final_data + coeficient * geophysical_species[yearly_test_Yindex]
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
                    population_test_data  = population_data[yearly_test_index]

                    for imonth in range(len(MONTH)):
                        final_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]]              = np.append(final_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], final_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                        obs_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]]                = np.append(obs_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], Validation_obs_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                        geo_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]]                = np.append(geo_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], Geophysical_test_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                        training_final_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]]     = np.append(training_final_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], train_final_data[imonth*len(train_index):(imonth+1)*len(train_index)])
                        training_obs_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]]       = np.append(training_obs_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], Training_obs_data[imonth*len(train_index):(imonth+1)*len(train_index)])
                        testing_population_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]] = np.append(testing_population_data_recording[str(beginyears[imodel]+iyear)][MONTH[imonth]], population_test_data[imonth*len(test_index):(imonth+1)*len(test_index)])
             
        count += 1

    test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE, PWM_NRMSE, slope, PWAModel, PWAMonitors = calculate_Statistics_results(test_beginyear=BLOO_test_beginyear, test_endyear=BLOO_test_endyear,
                                                                                                                final_data_recording=final_data_recording, obs_data_recording=obs_data_recording,
                                                                                                                geo_data_recording=geo_data_recording, training_final_data_recording=training_final_data_recording,
                                                                                                                training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording)
         
    txtfile_outdir = txt_outdir + '{}/{}/Results/results-BLOOCV/'.format(species, version)
    if not os.path.isdir(txtfile_outdir):
        os.makedirs(txtfile_outdir)
    
    txt_outfile =  txtfile_outdir + 'Buffered-{}km-{}fold-SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.csv'.format(buffer_radius,BLOO_kfold,typeName,species,version,nchannel,width,height,special_name)

    Output_Text_Sites_Number(outfile=txt_outfile, status='w', train_index_number=train_index_number, test_index_number=test_index_number, buffer=buffer_radius)
    AVD_output_text(outfile=txt_outfile,status='a', test_CV_R2=test_CV_R2, train_CV_R2=train_CV_R2, geo_CV_R2=geo_CV_R2, RMSE=RMSE, NRMSE=NRMSE,PMW_NRMSE=PWM_NRMSE,
                        slope=slope,PWM_Model=PWAModel,PWM_Monitors=PWAMonitors)
    save_BLOO_loss_accuracy(model_outdir=model_outdir,loss=train_loss, accuracy=train_acc,valid_loss=valid_losses, valid_accuracy=test_acc,typeName=typeName,
                       version=version,species=species, nchannel=nchannel,special_name=special_name, width=width, height=height,buffer_radius=buffer_radius)
    final_longterm_data, obs_longterm_data = get_annual_longterm_array(beginyear=BLOO_test_beginyear, endyear=BLOO_test_endyear, final_data_recording=final_data_recording,obs_data_recording=obs_data_recording)
    save_BLOO_data_recording(obs_data=obs_longterm_data,final_data=final_longterm_data,
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH='Annual',nchannel=nchannel,special_name=special_name,width=width,height=height,buffer_radius=buffer_radius)
           
    for imonth in range(len(MONTH)):
        final_longterm_data, obs_longterm_data = get_monthly_longterm_array(beginyear=BLOO_test_beginyear, imonth=imonth,endyear=BLOO_test_endyear, final_data_recording=final_data_recording,obs_data_recording=obs_data_recording)
        save_BLOO_data_recording(obs_data=obs_longterm_data,final_data=final_longterm_data,
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH=MONTH[imonth],nchannel=nchannel,special_name=special_name,width=width,height=height,buffer_radius=buffer_radius)
      

    return

def Get_Buffer_sites_number(buffer_radius,width, height, sitesnumber,start_YYYY, TrainingDatasets):
    # *------------------------------------------------------------------------------*#
    ##   Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    ### Get training data, label data, initial observation data and geophysical species
    
    SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
    geophysical_species, lat, lon = load_geophysical_species_data(species=species)
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    seed       = 19980130
    site_index = np.array(range(sitesnumber))
    
    rkf = RepeatedKFold(n_splits=BLOO_kfold, n_repeats=repeats, random_state=seed)
    # *------------------------------------------------------------------------------*#
    ## Begining the Cross-Validation.
    ## Multiple Models will be trained in each fold.
    # *------------------------------------------------------------------------------*#
    count = 0
    
    test_index_number = np.array([],dtype = int)
    train_index_number = np.array([],dtype=int)
    for init_train_index, test_index in rkf.split(site_index):
        
        train_index = GetBufferTrainingIndex(test_index=test_index,train_index=init_train_index,buffer=buffer_radius,sitelat=lat,sitelon=lon)
        test_index_number = np.append(test_index_number,len(test_index))
        train_index_number = np.append(train_index_number,len(train_index))
    print('Fold:', BLOO_kfold, 'Number of training sites:',np.mean(train_index_number), ' buffer radius: ', buffer_radius)
    return