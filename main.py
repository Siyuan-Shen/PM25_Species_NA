import toml
import os
from Training_pkg.utils import *
from Training_pkg.iostream import load_TrainingVariables
from visualization_pkg.Assemble_Func import plot_save_loss_accuracy_figure
from visualization_pkg.Evaluation_plot import regression_plot
from Evaluation_pkg.Spatial_CrossValidation import Normal_Spatial_CrossValidation
from Evaluation_pkg.iostream import load_loss_accuracy, load_data_recording
from Evaluation_pkg.utils import *


cfg = toml.load('./config.toml')

if __name__ == '__main__':
    typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
    nchannel   = len(channel_names)

    if Spatial_CrossValidation_Switch:
        cfg_outdir = Config_outdir + '{}/{}/Results/results-SpatialCV/'.format(species, version)
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        if not os.path.isdir(cfg_outdir):
            os.makedirs(cfg_outdir)
        cfg_outfile = cfg_outdir + 'config_SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.csv'.format(typeName,species,version,nchannel,width,height,special_name)
        Normal_Spatial_CrossValidation(width=width,height=height,sitesnumber=sitesnumber,start_YYYY=start_YYYY,TrainingDatasets=TrainingDatasets)
        f = open(cfg_outfile,'w')
        toml.dump(cfg, f)
        f.close()

    if Spatial_CV_LossAccuracy_plot_Switch:
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        loss, accuracy = load_loss_accuracy(model_outdir=model_outdir,typeName=typeName, version=version, species=species,nchannel=nchannel,special_name=special_name, width=width, height=height)
        plot_save_loss_accuracy_figure(loss=loss,accuracy=accuracy,typeName=typeName,species=species,version=version,nchannel=nchannel,width=width,height=height,special_name=special_name)
    
    if regression_plot_switch:
        #MONTH = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        MONTH = ['01', '07']
        annual_obs_data, annual_final_data = load_data_recording(species=species,version=version,typeName=typeName, beginyear='Alltime', MONTH='Annual',
                                                                 nchannel=nchannel,special_name=special_name,width=width,height=height)
        regression_plot(plot_obs_pm25=annual_final_data,plot_pre_pm25=annual_obs_data,species=species, version=version, typeName=typeName, beginyear='Alltime',
                        MONTH='Annual', nchannel=nchannel,special_name=special_name,width=width,height=height)
        for imonth in range(len(MONTH)):
            monthly_obs_data, monthly_final_data = load_data_recording(species=species,version=version,typeName=typeName, beginyear='Alltime', MONTH=MONTH[imonth],
                                                                 nchannel=nchannel,special_name=special_name,width=width,height=height)
            regression_plot(plot_obs_pm25=monthly_final_data,plot_pre_pm25=monthly_obs_data,species=species, version=version, typeName=typeName, beginyear='Alltime',
                        MONTH=MONTH[imonth], nchannel=nchannel,special_name=special_name,width=width,height=height)

