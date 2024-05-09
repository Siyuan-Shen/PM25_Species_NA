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
from Evaluation_pkg.iostream import save_sensitivity_test_trained_model,save_sensitivity_test_data_recording,save_sensitivity_test_loss_accuracy,load_sensitivity_test_data_recording,load_sensitivity_test_loss_accuracy,load_coMonitor_Population,save_trained_model, output_text, save_loss_accuracy, save_data_recording, load_data_recording, SensitivityTests_output_text

from visualization_pkg.Assemble_Func import plot_save_loss_accuracy_figure

def Sensitivity_Test_AVD_CrossValidation(width, height, sitesnumber,start_YYYY, TrainingDatasets,total_channel_names,main_stream_channel_names, side_stream_channel_names,exclude_channel_names):
    # *------------------------------------------------------------------------------*#
    ##   Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    ### Get training data, label data, initial observation data and geophysical species
    exclude_names_suffix = ''
    for iname in exclude_channel_names:
        exclude_names_suffix += '-'+iname

    SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
    geophysical_species, lat, lon = load_geophysical_species_data(species=species)
    true_input, mean, std = Learning_Object_Datasets(bias=bias,Normalized_bias=normalize_bias,Normlized_Speices=normalize_species,Absolute_Species=absolute_species,Log_PM25=log_species,species=species)
    Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets)
    population_data = load_coMonitor_Population()
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    nchannel   = len(total_channel_names)
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
    for train_index, test_index in rkf.split(site_index):
        for imodel in range(len(beginyears)):
            Normalized_TrainingData = get_trainingdata_within_sart_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=beginyears[imodel],training_end_YYYY=endyears[imodel],start_YYYY=start_YYYY,sitesnumber=sitesnumber)
            X_Training_index, X_Testing_index, Y_Training_index, Y_Testing_index = Get_XY_indices(train_index=train_index,test_index=test_index,beginyear=beginyears[imodel],endyear=endyears[imodel], sitesnumber=sitesnumber)
            X_train, X_test, y_train, y_test = Get_XY_arraies(Normalized_TrainingData=Normalized_TrainingData,true_input=true_input,X_Training_index=X_Training_index,X_Testing_index=X_Testing_index,Y_Training_index=Y_Training_index,Y_Testing_index=Y_Testing_index)

            # *------------------------------------------------------------------------------*#
            ## Training Process.
            # *------------------------------------------------------------------------------*#

            cnn_model = initial_network(width=width,main_stream_nchannel=len(main_stream_channel_names),side_stream_nchannel=len(side_stream_channel_names))

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cnn_model.to(device)
            torch.manual_seed(21)
            train_loss, train_acc, valid_losses, test_acc  = train(model=cnn_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, BATCH_SIZE=batchsize, learning_rate=lr0, TOTAL_EPOCHS=epoch,
                                                                   initial_channel_names=total_channel_names,main_stream_channels=main_stream_channel_names,side_stream_channels=side_stream_channel_names)
            save_sensitivity_test_trained_model(cnn_model=cnn_model, model_outdir=model_outdir, typeName=typeName, version=version, species=species, nchannel=nchannel, special_name=special_name, count=count, width=width, height=height,exclude_names_suffix=exclude_names_suffix)
            for iyear in range((endyears[imodel]-beginyears[imodel]+1)):
                yearly_test_index   = GetXIndex(index=test_index, beginyear=(beginyears[imodel]+iyear),endyear=(beginyears[imodel]+iyear), sitenumber=sitesnumber)
                yearly_train_index  = GetXIndex(index=train_index, beginyear=(beginyears[imodel]+iyear),endyear=(beginyears[imodel]+iyear), sitenumber=sitesnumber)
                yearly_test_Yindex  = GetYIndex(index=test_index,beginyear=(beginyears[imodel]+iyear), endyear=(beginyears[imodel]+iyear), sitenumber=sitesnumber)
                yearly_train_Yindex = GetYIndex(index=train_index,beginyear=(beginyears[imodel]+iyear), endyear=(beginyears[imodel]+iyear), sitenumber=sitesnumber)
                yearly_test_input  = Normalized_TrainingData[yearly_test_index,:,:,:]
                yearly_train_input = Normalized_TrainingData[yearly_train_index,:,:,:]

                Validation_Prediction = predict(inputarray=yearly_test_input, model=cnn_model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_channel_names)
                Training_Prediction   = predict(inputarray=yearly_train_input,  model=cnn_model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_channel_names)
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

    test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE, PWM_NRMSE, slope, PWAModel, PWAMonitors = calculate_Statistics_results(test_beginyear=Sensitivity_Test_test_beginyear, test_endyear=Sensitivity_Test_test_endyear,
                                                                                                                final_data_recording=final_data_recording, obs_data_recording=obs_data_recording,
                                                                                                                geo_data_recording=geo_data_recording, training_final_data_recording=training_final_data_recording,
                                                                                                                training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording)
        
    txtfile_outdir = txt_outdir + '{}/{}/Results/results-Sensitivity_Tests/statistical_indicators/'.format(species, version)
    if not os.path.isdir(txtfile_outdir):
        os.makedirs(txtfile_outdir)
    

    txt_outfile =  txtfile_outdir + 'Sensitivity_Tests_{}_{}_{}_{}Channel_{}x{}{}_Exclude{}.csv'.format(typeName,species,version,nchannel,width,height,special_name,exclude_names_suffix)
    SensitivityTests_output_text(outfile=txt_outfile,status='w', test_beginyears=Sensitivity_Test_test_beginyear,test_endyears=Sensitivity_Test_test_endyear,test_CV_R2=test_CV_R2, train_CV_R2=train_CV_R2, geo_CV_R2=geo_CV_R2, RMSE=RMSE, NRMSE=NRMSE,PMW_NRMSE=PWM_NRMSE,
                        slope=slope,PWM_Model=PWAModel,PWM_Monitors=PWAMonitors,exclude_channels_names=exclude_channel_names)
    save_sensitivity_test_loss_accuracy(model_outdir=model_outdir,loss=train_loss, accuracy=train_acc,valid_loss=valid_losses, valid_accuracy=test_acc,typeName=typeName,
                       version=version,species=species, nchannel=nchannel,special_name=special_name, width=width, height=height, exclude_names_suffix=exclude_names_suffix)
    final_longterm_data, obs_longterm_data = get_annual_longterm_array(beginyear=Sensitivity_Test_test_beginyear, endyear=Sensitivity_Test_test_endyear, final_data_recording=final_data_recording,obs_data_recording=obs_data_recording)
    save_sensitivity_test_data_recording(obs_data=obs_longterm_data,final_data=final_longterm_data,
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH='Annual',nchannel=nchannel,special_name=special_name,width=width,height=height,exclude_names_suffix=exclude_names_suffix)
           
    for imonth in range(len(MONTH)):
        final_longterm_data, obs_longterm_data = get_monthly_longterm_array(beginyear=Sensitivity_Test_test_beginyear, imonth=imonth,endyear=Sensitivity_Test_test_endyear, final_data_recording=final_data_recording,obs_data_recording=obs_data_recording)
        save_sensitivity_test_data_recording(obs_data=obs_longterm_data,final_data=final_longterm_data,
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH=MONTH[imonth],nchannel=nchannel,special_name=special_name,width=width,height=height,exclude_names_suffix=exclude_names_suffix)
      
    return