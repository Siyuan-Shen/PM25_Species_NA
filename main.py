import toml
import os
import pprint
from Training_pkg.utils import *
from Training_pkg.iostream import load_TrainingVariables
from visualization_pkg.Assemble_Func import plot_save_loss_accuracy_figure, plot_save_estimation_map_figure
from visualization_pkg.Evaluation_plot import regression_plot,every_point_regression_plot
from Evaluation_pkg.Spatial_CrossValidation import Normal_Spatial_CrossValidation, AVD_Spatial_CrossValidation, FixedNumber_AVD_Spatial_CrossValidation
from Evaluation_pkg.Sensitivity_Spatial_CrossValidation import Sensitivity_Test_AVD_CrossValidation
from Evaluation_pkg.BLOO_CrossValidation import BLOO_AVD_Spatial_CrossValidation, Get_Buffer_sites_number
from Evaluation_pkg.BLCO_CrossValidation import BLCO_AVD_Spatial_CrossValidation
from Evaluation_pkg.SHAPvalue_analysis import Spatial_CV_SHAP_Analysis
from Evaluation_pkg.iostream import load_loss_accuracy, load_data_recording, load_month_based_data_recording
from Evaluation_pkg.utils import *
from Estimation_pkg.Estimation import Estimation_Func
from Estimation_pkg.Quality_Control import Calculate_Regional_PWM_PM_Components
from Estimation_pkg.utils import *
from Uncertainty_pkg.uncertainty_estimation import Derive_Estimation_Uncertainty
from Uncertainty_pkg.utils import Uncertainty_Switch

cfg = toml.load('./config.toml')
pprint.pprint(cfg)

if __name__ == '__main__':
    typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
    total_channel_names, main_stream_channel_names, side_channel_names = Get_channel_names(channels_to_exclude=[])
    nchannel   = len(total_channel_names)

    if Spatial_CrossValidation_Switch:
        cfg_outdir = Config_outdir + '{}/{}/Results/results-SpatialCV/configuration-files/'.format(species, version)
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        if not os.path.isdir(cfg_outdir):
            os.makedirs(cfg_outdir)
        cfg_outfile = cfg_outdir + 'config_SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.toml'.format(typeName,species,version,nchannel,width,height,special_name)
        f = open(cfg_outfile,'w')
        toml.dump(cfg, f)
        f.close()
        AVD_Spatial_CrossValidation(width=width,height=height,sitesnumber=sitesnumber,start_YYYY=start_YYYY,TrainingDatasets=TrainingDatasets,
                                    total_channel_names=total_channel_names,main_stream_channel_names=main_stream_channel_names,side_stream_nchannel_names=side_channel_names)

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
        obs_data, final_data,geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording, lat_recording, lon_recording = load_month_based_data_recording(species=species,version=version,typeName=typeName, beginyear=beginyears[0],endyear=endyears[-1],
                                                                 nchannel=nchannel,special_name=special_name,width=width,height=height)
        regression_plot(plot_obs_pm25=annual_obs_data,plot_pre_pm25=annual_final_data,species=species, version=version, typeName=typeName, beginyear='Alltime',
                        MONTH='Annual', nchannel=nchannel,special_name=special_name,width=width,height=height)
        every_point_regression_plot(plot_obs_pm25=obs_data,plot_pre_pm25=final_data,species=species, version=version, typeName=typeName,plot_beginyear=every_point_begin_years,plot_endyear=every_point_end_years,
                        MONTH='Annual', nchannel=nchannel,special_name=special_name,width=width,height=height)
        for imonth in range(len(MONTH)):
            monthly_obs_data, monthly_final_data = load_data_recording(species=species,version=version,typeName=typeName, beginyear='Alltime', MONTH=MONTH[imonth],
                                                                 nchannel=nchannel,special_name=special_name,width=width,height=height)
            regression_plot(plot_obs_pm25=monthly_obs_data,plot_pre_pm25=monthly_final_data,species=species, version=version, typeName=typeName, beginyear='Alltime',
                        MONTH=MONTH[imonth], nchannel=nchannel,special_name=special_name,width=width,height=height)
            every_point_regression_plot(plot_obs_pm25=obs_data,plot_pre_pm25=final_data,species=species, version=version, typeName=typeName,plot_beginyear=every_point_begin_years,plot_endyear=every_point_end_years,
                        MONTH=MONTH[imonth], nchannel=nchannel,special_name=special_name,width=width,height=height)

    if SHAP_Analysis_switch:
        cfg_outdir = Config_outdir + '{}/{}/Results/results-SpatialCV/configuration-files/'.format(species, version)
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        if not os.path.isdir(cfg_outdir):
            os.makedirs(cfg_outdir)
        cfg_outfile = cfg_outdir + 'config_SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.toml'.format(typeName,species,version,nchannel,width,height,special_name)
        f = open(cfg_outfile,'w')
        toml.dump(cfg, f)
        f.close()
        Spatial_CV_SHAP_Analysis(width=width,height=height,sitesnumber=sitesnumber
                                 ,start_YYYY=start_YYYY,TrainingDatasets=TrainingDatasets,total_channel_names=total_channel_names,main_stream_channel_names=main_stream_channel_names,
                                 side_stream_nchannel_names=side_channel_names)
    if BLOO_CrossValidation_Switch:
        cfg_outdir = Config_outdir + '{}/{}/Results/results-BLOOCV/configuration-files/'.format(species, version)
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        for buffer_radius in BLOO_Buffer_size:
            if not os.path.isdir(cfg_outdir):
                os.makedirs(cfg_outdir)
            cfg_outfile = cfg_outdir + 'config_BLOO_SpatialCV_{}km-buffer_{}_{}_{}_{}Channel_{}x{}{}.toml'.format(buffer_radius,typeName,species,version,nchannel,width,height,special_name)
            f = open(cfg_outfile,'w')
            toml.dump(cfg, f)
            f.close()
            BLOO_AVD_Spatial_CrossValidation(buffer_radius=buffer_radius,width=width,height=height,sitesnumber=sitesnumber,start_YYYY=start_YYYY,TrainingDatasets=TrainingDatasets,
                                             total_channel_names=total_channel_names,main_stream_channel_names=main_stream_channel_names,side_stream_channel_names=side_channel_names)
            #Get_Buffer_sites_number(buffer_radius=buffer_radius,width=width,height=height,sitesnumber=sitesnumber,start_YYYY=start_YYYY,TrainingDatasets=TrainingDatasets)
    
    if BLCO_CrossValidation_Switch:
        if utilize_self_isolated_sites:
            cfg_outdir = Config_outdir + '{}/{}/Results/results-SelfIsolated_BLCOCV/configuration-files/'.format(species, version)
        else:
            cfg_outdir = Config_outdir + '{}/{}/Results/results-BLCOCV/configuration-files/'.format(species, version)
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        for buffer_radius in BLCO_Buffer_size:
            if not os.path.isdir(cfg_outdir):
                os.makedirs(cfg_outdir)
            if utilize_self_isolated_sites:
                cfg_outfile = cfg_outdir + 'config_SelfIsolated_BLCO_SpatialCV_{}km-buffer_{}_{}_{}_{}Channel_{}x{}{}.toml'.format(buffer_radius,typeName,species,version,nchannel,width,height,special_name)
            else:
                cfg_outfile = cfg_outdir + 'config_BLCO_SpatialCV_{}km-buffer_{}_{}_{}_{}Channel_{}x{}{}.toml'.format(buffer_radius,typeName,species,version,nchannel,width,height,special_name)
            f = open(cfg_outfile,'w')
            toml.dump(cfg, f)
            f.close()
            BLCO_AVD_Spatial_CrossValidation(buffer_radius=buffer_radius,BLCO_kfold=BLCO_kfold,width=width,height=height,sitesnumber=sitesnumber,start_YYYY=start_YYYY,TrainingDatasets=TrainingDatasets,
                                             total_channel_names=total_channel_names,main_stream_channel_names=main_stream_channel_names,side_stream_channel_names=side_channel_names)
            #Get_Buffer_sites_number(buffer_radius=buffer_radius,width=width,height=height,sitesnumber=sitesnumber,start_YYYY=start_YYYY,TrainingDatasets=TrainingDatasets)

    if FixNumber_Spatial_CrossValidation_Switch:
        cfg_outdir = Config_outdir + '{}/{}/Results/results-FixNumberCV/configuration-files/'.format(species, version)
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        for i in range(len(Fixednumber_test_sites)):
            if not os.path.isdir(cfg_outdir):
                os.makedirs(cfg_outdir)
            cfg_outfile = cfg_outdir + 'config_FixNumber_SpatialCV_{}-test-sites_{}-train-sites_{}_{}_{}_{}Channel_{}x{}{}.toml'.format(Fixednumber_test_sites[i],Fixednumber_train_sites[i],typeName,species,version,nchannel,width,height,special_name)
            f = open(cfg_outfile,'w')
            toml.dump(cfg, f)
            f.close()
            FixedNumber_AVD_Spatial_CrossValidation(Fixednumber_train_site=Fixednumber_train_sites[i],Fixednumber_test_site=Fixednumber_test_sites[i],width=width,height=height,sitesnumber=sitesnumber,start_YYYY=start_YYYY,TrainingDatasets=TrainingDatasets,
                                                    total_channel_names=total_channel_names, main_stream_channel_names=main_stream_channel_names, side_stream_nchannel_names=side_channel_names)
            
    if Estimation_Switch:
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        cfg_outdir = Config_outdir + '{}/{}/Estimation/configuration-files/'.format(species, version)
        if not os.path.isdir(cfg_outdir):
            os.makedirs(cfg_outdir)
        cfg_outfile = cfg_outdir + 'config_Estimation_{}_{}_{}_{}Channel_{}x{}{}.toml'.format(typeName,species,version,nchannel,width,height,special_name)
        f = open(cfg_outfile,'w')
        toml.dump(cfg, f)
        f.close()
        Estimation_Func(total_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,side_channel_names=side_channel_names)

        if Estimation_PWMPM_Cal_Switch:
            Calculate_Regional_PWM_PM_Components()
        
        if Estimation_visualization_Switch:
            plot_save_estimation_map_figure(Estimation_Map_Plot=Map_Plot_Switch,ForcedSlopeUnity_Map_Plot_Switch=ForcedSlopeUnity_Map_Plot_Switch,typeName=typeName,width=
                                            width,height=height,species=species,version=version,Area=Map_Plot_Area,PLOT_YEARS=Map_Plot_YEARS, PLOT_MONTHS=Map_Plot_MONTHS)
        


    if Uncertainty_Switch:
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        cfg_outdir = Config_outdir + '{}/{}/Uncertainty_Results/configuration-files/'.format(species, version)
        if not os.path.isdir(cfg_outdir):
            os.makedirs(cfg_outdir)
        cfg_outfile = cfg_outdir + 'config_Uncertainty_{}_{}_{}_{}Channel_{}x{}{}.toml'.format(typeName,species,version,nchannel,width,height,special_name)
        f = open(cfg_outfile,'w')
        toml.dump(cfg, f)
        f.close()
        Derive_Estimation_Uncertainty(total_channel_names=total_channel_names,width=width,height=height)
        
        

    if Sensitivity_Test_Switch:
        for igroup in range(len(Sensitivity_Test_Sensitivity_Test_Variables)):
            total_channel_names, main_stream_channel_names, side_channel_names = Get_channel_names(channels_to_exclude=Sensitivity_Test_Sensitivity_Test_Variables[igroup])
            print('Exclude Variables: {} \nTotal Channel Names: {}'.format(Sensitivity_Test_Sensitivity_Test_Variables[igroup],total_channel_names))
            width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=total_channel_names)
            Sensitivity_Test_AVD_CrossValidation(width=width,height=height,sitesnumber=sitesnumber,start_YYYY=start_YYYY,TrainingDatasets=TrainingDatasets,
                                                 total_channel_names=total_channel_names,main_stream_channel_names=main_stream_channel_names,side_stream_channel_names=side_channel_names,
                                                 exclude_channel_names=Sensitivity_Test_Sensitivity_Test_Variables[igroup])
        
        cfg_outdir = Config_outdir + '{}/{}/Results/results-Sensitivity_Tests/configuration-files/'.format(species, version)
        if not os.path.isdir(cfg_outdir):
            os.makedirs(cfg_outdir)
        cfg_outfile = cfg_outdir + 'config_Sensitivity-Tests_{}_{}_{}_{}Channel_{}x{}{}.toml'.format(typeName,species,version,nchannel,width,height,special_name)
        f = open(cfg_outfile,'w')
        toml.dump(cfg, f)
        f.close()