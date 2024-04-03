import toml
import os
from Training_pkg.utils import *
from Training_pkg.iostream import load_TrainingVariables
from visualization_pkg.Assemble_Func import plot_save_loss_accuracy_figure, plot_save_estimation_map_figure
from visualization_pkg.Evaluation_plot import regression_plot
from Evaluation_pkg.Spatial_CrossValidation import Normal_Spatial_CrossValidation, AVD_Spatial_CrossValidation, FixedNumber_AVD_Spatial_CrossValidation
from Evaluation_pkg.BLOO_CrossValidation import BLOO_AVD_Spatial_CrossValidation, Get_Buffer_sites_number
from Evaluation_pkg.iostream import load_loss_accuracy, load_data_recording
from Evaluation_pkg.utils import *
from Estimation_pkg.Estimation import Estimation_Func
from Estimation_pkg.utils import *
from Uncertainty_pkg.uncertainty_estimation import Derive_Estimation_Uncertainty
from Uncertainty_pkg.utils import Uncertainty_Switch

cfg = toml.load('./config.toml')

if __name__ == '__main__':
    typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
    nchannel   = Get_MainStream_nchannels()

    if Spatial_CrossValidation_Switch:
        cfg_outdir = Config_outdir + '{}/{}/Results/results-SpatialCV/'.format(species, version)
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        if not os.path.isdir(cfg_outdir):
            os.makedirs(cfg_outdir)
        cfg_outfile = cfg_outdir + 'config_SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.toml'.format(typeName,species,version,nchannel,width,height,special_name)
        AVD_Spatial_CrossValidation(width=width,height=height,sitesnumber=sitesnumber,start_YYYY=start_YYYY,TrainingDatasets=TrainingDatasets)
        f = open(cfg_outfile,'w')
        toml.dump(cfg, f)
        f.close()

    if Spatial_CV_LossAccuracy_plot_Switch:
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        loss, accuracy, valid_loss, valid_accuracy = load_loss_accuracy(model_outdir=model_outdir,typeName=typeName, version=version, species=species,nchannel=nchannel,special_name=special_name, width=width, height=height)
        plot_save_loss_accuracy_figure(loss=loss,accuracy=accuracy, valid_loss=valid_loss, valid_accuracy=valid_accuracy,typeName=typeName,species=species,version=version,nchannel=nchannel,width=width,height=height,special_name=special_name)
    
    if regression_plot_switch:
        #MONTH = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        annual_obs_data, annual_final_data = load_data_recording(species=species,version=version,typeName=typeName, beginyear='Alltime', MONTH='Annual',
                                                                 nchannel=nchannel,special_name=special_name,width=width,height=height)
        regression_plot(plot_obs_pm25=annual_final_data,plot_pre_pm25=annual_obs_data,species=species, version=version, typeName=typeName, beginyear='Alltime',
                        MONTH='Annual', nchannel=nchannel,special_name=special_name,width=width,height=height)
        for imonth in range(len(MONTH)):
            monthly_obs_data, monthly_final_data = load_data_recording(species=species,version=version,typeName=typeName, beginyear='Alltime', MONTH=MONTH[imonth],
                                                                 nchannel=nchannel,special_name=special_name,width=width,height=height)
            regression_plot(plot_obs_pm25=monthly_final_data,plot_pre_pm25=monthly_obs_data,species=species, version=version, typeName=typeName, beginyear='Alltime',
                        MONTH=MONTH[imonth], nchannel=nchannel,special_name=special_name,width=width,height=height)

    if BLOO_CrossValidation_Switch:
        cfg_outdir = Config_outdir + '{}/{}/Results/results-BLOOCV/'.format(species, version)
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        for buffer_radius in Buffer_size:
            if not os.path.isdir(cfg_outdir):
                os.makedirs(cfg_outdir)
            cfg_outfile = cfg_outdir + 'config_BLOO_SpatialCV_{}km-buffer_{}_{}_{}_{}Channel_{}x{}{}.toml'.format(buffer_radius,typeName,species,version,nchannel,width,height,special_name)
            BLOO_AVD_Spatial_CrossValidation(buffer_radius=buffer_radius,width=width,height=height,sitesnumber=sitesnumber,start_YYYY=start_YYYY,TrainingDatasets=TrainingDatasets)
            #Get_Buffer_sites_number(buffer_radius=buffer_radius,width=width,height=height,sitesnumber=sitesnumber,start_YYYY=start_YYYY,TrainingDatasets=TrainingDatasets)
            f = open(cfg_outfile,'w')
            toml.dump(cfg, f)
            f.close()

    if FixNumber_Spatial_CrossValidation_Switch:
        cfg_outdir = Config_outdir + '{}/{}/Results/results-FixNumberCV/'.format(species, version)
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        for i in range(len(Fixednumber_test_sites)):
            if not os.path.isdir(cfg_outdir):
                os.makedirs(cfg_outdir)
            cfg_outfile = cfg_outdir + 'config_FixNumber_SpatialCV_{}-test-sites_{}-train-sites_{}_{}_{}_{}Channel_{}x{}{}.toml'.format(Fixednumber_test_sites[i],Fixednumber_train_sites[i],typeName,species,version,nchannel,width,height,special_name)
            FixedNumber_AVD_Spatial_CrossValidation(Fixednumber_train_site=Fixednumber_train_sites[i],Fixednumber_test_site=Fixednumber_test_sites[i],width=width,height=height,sitesnumber=sitesnumber,start_YYYY=start_YYYY,TrainingDatasets=TrainingDatasets)
            f = open(cfg_outfile,'w')
            toml.dump(cfg, f)
            f.close()
            
    if Estimation_Switch:
        Estimation_Func()

        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        if Estimation_visualization_Switch:
            plot_save_estimation_map_figure(Estimation_Map_Plot=Map_Plot_Switch,typeName=typeName,width=
                                            width,height=height,species=species,version=version,Area=Map_Plot_Area,PLOT_YEARS=Map_Plot_YEARS, PLOT_MONTHS=Map_Plot_MONTHS)
        cfg_outdir = Config_outdir + '{}/{}/Estimation/configuration-files/'.format(species, version)
        if not os.path.isdir(cfg_outdir):
            os.makedirs(cfg_outdir)
        cfg_outfile = cfg_outdir + 'config_Estimation_{}_{}_{}_{}Channel_{}x{}{}.toml'.format(typeName,species,version,nchannel,width,height,special_name)
        f = open(cfg_outfile,'w')
        toml.dump(cfg, f)
        f.close()


    if Uncertainty_Switch:
        Derive_Estimation_Uncertainty()
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        cfg_outdir = Config_outdir + '{}/{}/Uncertainty_Results/configuration-files/'.format(species, version)
        if not os.path.isdir(cfg_outdir):
            os.makedirs(cfg_outdir)
        cfg_outfile = cfg_outdir + 'config_Uncertainty_{}_{}_{}_{}Channel_{}x{}{}.toml'.format(typeName,species,version,nchannel,width,height,special_name)
        f = open(cfg_outfile,'w')
        toml.dump(cfg, f)
        f.close()


