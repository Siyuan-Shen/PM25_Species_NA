import torch
import numpy as np
import os
import csv
from Evaluation_pkg.utils import *

def save_trained_model(cnn_model, model_outdir, typeName, version, species, nchannel, special_name, count, width, height):
    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    model_outfile = outdir +  'SpatialCV_{}_{}_{}x{}_{}Channel{}_No{}.pt'.format(typeName, species, width,height, nchannel,special_name, count)
    torch.save(cnn_model, model_outfile)

def save_loss_accuracy(model_outdir, loss, accuracy, typeName, version, species, nchannel, special_name, width, height):

    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    loss_outfile = outdir + 'SpatialCV_loss_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, width, height, nchannel,special_name)
    accuracy_outfile = outdir + 'SpatialCV_accuracy_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, width, height, nchannel,special_name)
    np.save(loss_outfile, loss)
    np.save(accuracy_outfile, accuracy)
    return

def save_data_recording(obs_data, final_data,species, version, typeName, beginyear, MONTH, nchannel, special_name, width, height):
    outdir = txt_outdir + '{}/{}/Results/results-DataRecording/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    obs_data_outfile   = outdir + '{}-{}-Obs-DataRecording_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    final_data_outfile = outdir + '{}-{}-Final-DataRecording_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    
    np.save(obs_data_outfile, obs_data)
    np.save(final_data_outfile, final_data)
    return

def load_data_recording(species, version, typeName, beginyear, MONTH, nchannel, special_name, width, height):
    indir = txt_outdir + '{}/{}/Results/results-DataRecording/'.format(species, version)
    obs_data_infile   = indir + '{}-{}-Obs-DataRecording_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    final_data_infile = indir + '{}-{}-Final-DataRecording_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    
    obs_data = np.load(obs_data_infile)
    final_data = np.load(final_data_infile)

    return obs_data, final_data

def load_loss_accuracy(model_outdir, typeName, version, species, nchannel, special_name, width, height):

    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    loss_outfile = outdir +'SpatialCV_loss_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, width, height, nchannel,special_name)
    accuracy_outfile = outdir + 'SpatialCV_accuracy_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, width, height, nchannel,special_name)
    loss = np.load(loss_outfile)
    accuracy = np.load(accuracy_outfile)
    return loss, accuracy

def output_text(outfile:str,status:str,CV_R2,annual_CV_R2,month_CV_R2,training_annual_CV_R2,training_month_CV_R2,
                geo_annual_CV_R2, geo_month_CV_R2,
                CV_slope,annual_CV_slope,month_CV_slope,
                CV_RMSE,annual_CV_RMSE,month_CV_RMSE,
                beginyear:str,endyear:str,species:str,kfold:int,repeats:int):
    
    MONTH = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    CV_R2[-1] = np.mean(CV_R2[0:kfold * repeats])
    annual_CV_R2[-1] = np.mean(annual_CV_R2[0:kfold * repeats])
    CV_slope[-1] = np.mean(CV_slope[0:kfold * repeats])
    annual_CV_slope[-1] = np.mean(annual_CV_slope[0:kfold * repeats])
    CV_RMSE[-1] = np.mean(CV_RMSE[0:kfold * repeats])
    annual_CV_RMSE[-1] = np.mean(annual_CV_RMSE[0:kfold * repeats])
    training_annual_CV_R2[-1] = np.mean(training_annual_CV_R2[0:kfold * repeats])
    geo_annual_CV_R2[-1] = np.mean(geo_annual_CV_R2[0:kfold * repeats])

    with open(outfile,status) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Species: {} ; Time Period: {} - {}'.format(species, beginyear, endyear)])
        writer.writerow(['R2 for monthly validation', '\nMax: ',str(np.round(np.max(CV_R2),4)),'Min: ',str(np.round(np.min(CV_R2),4)),
                         'Avg: ',str(np.round(CV_R2[-1],4)),'\nSlope for monthly validation Max: ',str(np.round(np.max(CV_slope),4)),'Min: ',str(np.round(np.min(CV_slope),4)),
                         'Avg: ',str(np.round(CV_slope[-1],4)),'\nRMSE for monthly validation Max: ',str(np.round(np.max(CV_RMSE),4)),'Min: ',str(np.round(np.min(CV_RMSE),4)),
                         'Avg: ',str(np.round(CV_RMSE[-1],4))])
        writer.writerow(['#####################   Annual average validation ####################', '\n R2 Max: ', str(np.round(np.max(annual_CV_R2), 4)), 'Min: ',
                         str(np.round(np.min(annual_CV_R2), 4)),
                         'Avg: ', str(np.round(annual_CV_R2[-1], 4)),'\nSlope for Annual average validation Max: ', str(np.round(np.max(annual_CV_slope), 4)), 'Min: ',
                         str(np.round(np.min(annual_CV_slope), 4)),
                         'Avg: ', str(np.round(annual_CV_slope[-1], 4)),'\nRMSE for Annual average validation Max: ', str(np.round(np.max(annual_CV_RMSE), 4)), 'Min: ',
                         str(np.round(np.min(annual_CV_RMSE), 4)),
                         'Avg: ', str(np.round(annual_CV_RMSE[-1], 4))])
        writer.writerow(['###################### Annual Training ####################', '\n Training R2 - Max: ', str(np.round(np.max(training_annual_CV_R2), 4)), 'Min: ',
                         str(np.round(np.min(training_annual_CV_R2), 4)),
                         'Avg: ', str(np.round(training_annual_CV_R2[-1], 4))])
        writer.writerow(['###################### Annual Geophysical ####################', '\n  R2 - Max: ', str(np.round(np.max(geo_annual_CV_R2), 4)), 'Min: ',
                         str(np.round(np.min(geo_annual_CV_R2), 4)),
                         'Avg: ', str(np.round(geo_annual_CV_R2[-1], 4))])
        for imonth in range(len(MONTH)):
            month_CV_R2[imonth,-1] = np.mean(month_CV_R2[imonth,0:kfold * repeats])
            month_CV_slope[imonth,-1] = np.mean(month_CV_slope[imonth,0:kfold * repeats])
            month_CV_RMSE[imonth,-1] = np.mean(month_CV_RMSE[imonth,0:kfold * repeats])
            training_month_CV_R2[imonth,-1] = np.mean(training_month_CV_R2[imonth,0:kfold * repeats])
            geo_month_CV_R2[imonth,-1] = np.mean(geo_month_CV_R2[imonth,0:kfold * repeats])
            writer.writerow([' -------------------------- {} ------------------------'.format(MONTH[imonth]), '\n R2 - Max: ', str(np.round(np.max(month_CV_R2[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_R2[imonth,:]), 4)), 'Avg: ',str(np.round(month_CV_R2[imonth,-1],4)),
                             '\nSlope - Max: ', str(np.round(np.max(month_CV_slope[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_slope[imonth,:]), 4)), 'Avg: ',str(np.round(month_CV_slope[imonth,-1],4)),
                             '\nRMSE -  Max: ', str(np.round(np.max(month_CV_RMSE[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_RMSE[imonth,:]), 4)), 'Avg: ',str(np.round(month_CV_RMSE[imonth,-1],4)),
                             '\nTraining R2 - Max: ',str(np.round(np.max(training_month_CV_R2[imonth,:]), 4)), 'Min: ',str(np.round(np.min(training_month_CV_R2[imonth,:]), 4)), 'Avg: ',
                             str(np.round(training_month_CV_R2[imonth,-1],4)),
                             '\nGeophysical R2 - Max: ',str(np.round(np.max(geo_month_CV_R2[imonth,:]), 4)), 'Min: ',str(np.round(np.min(geo_month_CV_R2[imonth,:]), 4)), 'Avg: ',
                             str(np.round(geo_month_CV_R2[imonth,-1],4))])

    return