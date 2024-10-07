import torch
import numpy as np
import os
import csv
import netCDF4 as nc
import time
from Estimation_pkg.utils import *


def load_trained_month_based_model_forEstimation(model_outdir, typeName, version, species, nchannel, special_name,beginyear, endyear,month_index, width, height):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)
    outdir = model_outdir + '{}/{}/Results/Estimation-Trained_Models/'.format(species, version)
    PATH = outdir +  'Estimation_{}_{}_{}x{}_{}-{}_{}_{}Channel{}.pt'.format(typeName, species, width,height, beginyear, endyear,Selected_MONTHS_str, nchannel,special_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = torch.load(PATH, map_location=torch.device(device)).eval()
    model.to(device)
    return model


def load_trained_model_forEstimation(model_outdir, typeName, version, species, nchannel, special_name,beginyear, endyear, width, height):
    outdir = model_outdir + '{}/{}/Results/Estimation-Trained_Models/'.format(species, version)
    PATH = outdir +  'Estimation_{}_{}_{}x{}_{}-{}_{}Channel{}.pt'.format(typeName, species, width,height, beginyear, endyear, nchannel,special_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = torch.load(PATH, map_location=torch.device(device)).eval()
    model.to(device)
    return model

def load_map_data(channel_names, YYYY, MM):
    inputfiles = inputfiles_table(YYYY=YYYY,MM=MM)
    indir = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/'
    lat_infile = indir + 'tSATLAT_NA.npy'
    lon_infile = indir + 'tSATLON_NA.npy'
    SATLAT = np.load(lat_infile)
    SATLON = np.load(lon_infile)
    output = np.zeros((len(channel_names), len(SATLAT), len(SATLON)))
    loading_time_start = time.time()
    for i in range(len(channel_names)):
        infile = inputfiles[channel_names[i]]
        tempdata = np.load(infile)
        print('{} has been loaded!'.format(channel_names[i]))
        output[i,:,:] = tempdata
    loading_time_end = time.time()
    print('Loading time cost: ', loading_time_end - loading_time_start, 's')
    return output

def load_estimation_map_data(YYYY:str, MM:str,SPECIES:str, version:str, special_name):
    indir = Estimation_outdir + '{}/{}/Map_Estimation/{}/'.format(SPECIES,version,YYYY)
    infile = indir + '{}_{}_{}{}{}.nc'.format(SPECIES,version,YYYY,MM,special_name)
    MapData = nc.Dataset(infile)
    lat = MapData.variables['lat'][:]
    lon = MapData.variables['lon'][:]
    SPECIES_Map = MapData.variables[SPECIES][:]
    SPECIES_Map = np.array(SPECIES_Map)
    return SPECIES_Map, lat, lon

def load_ForcedSlopeUnity_estimation_map_data(YYYY:str, MM:str,SPECIES:str, version:str, special_name):
    indir = Estimation_outdir + '{}/{}/ForcedSlopeUnity_Map_Estimation/{}/'.format(SPECIES,version,YYYY)
    infile = indir + '{}_{}_{}{}{}_ForcedSlopeUnity.nc'.format(SPECIES,version,YYYY,MM,special_name)
    MapData = nc.Dataset(infile)
    lat = MapData.variables['lat'][:]
    lon = MapData.variables['lon'][:]
    SPECIES_Map = MapData.variables[SPECIES][:]
    SPECIES_Map = np.array(SPECIES_Map)
    return SPECIES_Map, lat, lon


def load_Annual_estimation_map_data(YYYY:str,SPECIES:str, version:str, special_name):
    indir = Estimation_outdir + '{}/{}/Map_Estimation/{}/'.format(SPECIES,version,YYYY)
    infile = indir + 'Annual_{}_{}_{}{}.nc'.format(SPECIES,version,YYYY,special_name)
    MapData = nc.Dataset(infile)
    lat = MapData.variables['lat'][:]
    lon = MapData.variables['lon'][:]
    SPECIES_Map = MapData.variables[SPECIES][:]
    SPECIES_Map = np.array(SPECIES_Map)
    return SPECIES_Map, lat, lon

def load_ForcedSlope_forEstimation(model_indir, typeName, version, species, nchannel, special_name,beginyear, endyear, month_index,width, height):
    indir = model_indir + '{}/{}/Results/Estimation-ForcedSlopeUnity_Dicts/'.format(species, version)
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)
    dic_infile = indir + 'Estimation-ForcedSlopeUnity_Dicts_{}_{}_{}x{}_{}-{}_{}_{}Channel{}.npy'.format(typeName, species, width,height, beginyear, endyear,Selected_MONTHS_str, nchannel,special_name)
    ForcedSlopeUnity_Dictionary_forEstimation = np.load(dic_infile,allow_pickle=True).item()
    return ForcedSlopeUnity_Dictionary_forEstimation
def save_ForcedSlope_forEstimation(ForcedSlopeUnity_Dictionary_forEstimation, model_outdir, typeName, version, species, nchannel, special_name,beginyear, endyear, month_index,width, height):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)

    outdir = model_outdir + '{}/{}/Results/Estimation-ForcedSlopeUnity_Dicts/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    dic_outfile = outdir + 'Estimation-ForcedSlopeUnity_Dicts_{}_{}_{}x{}_{}-{}_{}_{}Channel{}.npy'.format(typeName, species, width,height, beginyear, endyear,Selected_MONTHS_str, nchannel,special_name)
    np.save(dic_outfile,ForcedSlopeUnity_Dictionary_forEstimation)
    return

def save_trained_model_forEstimation(cnn_model, model_outdir, typeName, version, species, nchannel, special_name,beginyear, endyear, width, height):
    outdir = model_outdir + '{}/{}/Results/Estimation-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    model_outfile = outdir +  'Estimation_{}_{}_{}x{}_{}-{}_{}Channel{}.pt'.format(typeName, species, width,height, beginyear, endyear, nchannel,special_name)
    torch.save(cnn_model, model_outfile)
    return 

def save_trained_month_based_model_forEstimation(cnn_model, model_outdir, typeName, version, species, nchannel, special_name,beginyear, endyear, month_index,width, height):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)

    outdir = model_outdir + '{}/{}/Results/Estimation-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    model_outfile = outdir +  'Estimation_{}_{}_{}x{}_{}-{}_{}_{}Channel{}.pt'.format(typeName, species, width,height, beginyear, endyear,Selected_MONTHS_str, nchannel,special_name)
    torch.save(cnn_model, model_outfile)
    return 


def save_annual_final_map_data(final_data:np.array, YYYY:str, extent:list, SPECIES:str, version:str, special_name):
    outdir = Estimation_outdir + '{}/{}/Map_Estimation/{}/'.format(SPECIES,version,YYYY)
    
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    outfile = outdir + 'Annual_{}_{}_{}{}.nc'.format(SPECIES,version,YYYY,special_name)
    lat_size = final_data.shape[0]
    lon_size = final_data.shape[1]
    lat_delta = 0.01 #(extent[1]-extent[0])/(lat_size-1)
    lon_delta = 0.01 #(extent[3]-extent[2])/(lon_size-1)

    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    MapData.TITLE = 'Convolutional Neural Network Annual {} Estimation over North America Area.'.format(SPECIES)
    MapData.CONTACT = 'SIYUAN SHEN <s.siyuan@wustl.edu>'
    MapData.LAT_DELTA = lat_delta
    MapData.LON_DELTA = lon_delta
    MapData.TIMECOVERAGE    = '{}'.format(YYYY)

    lat = MapData.createDimension("lat",lat_size)
    lon = MapData.createDimension("lon",lon_size)
    PM25 = MapData.createVariable(SPECIES,'f4',('lat','lon',))
    latitudes = MapData.createVariable("lat","f4",("lat",))
    longitudes = MapData.createVariable("lon","f4",("lon",))
    latitudes[:] = np.arange(extent[0],np.round(extent[1]+lat_delta,decimals=5),lat_delta)
    longitudes[:] = np.arange(extent[2],np.round(extent[3]+lon_delta,decimals=5),lon_delta) 
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    latitudes.standard_name = 'latitude'
    latitudes.long_name = 'latitude'
    longitudes.standard_name = 'longitude'
    longitudes.long_name = 'longitude'
    PM25.units = 'ug/m3'
    PM25.long_name = 'Convolutional Neural Network derived Annual {} [ug/m^3]'.format(SPECIES)
    PM25[:] = final_data
    return
    
def save_final_map_data(final_data:np.array, YYYY:str, MM:str, extent:list, SPECIES:str, version:str, special_name):
    outdir = Estimation_outdir + '{}/{}/Map_Estimation/{}/'.format(SPECIES,version,YYYY)
    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + '{}_{}_{}{}{}.nc'.format(SPECIES,version,YYYY,MM,special_name)
    lat_size = final_data.shape[0]
    lon_size = final_data.shape[1]
    lat_delta = 0.01
    lon_delta = 0.01

    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    MapData.TITLE = 'Convolutional Neural Network Monthly {} Estimation over North America Area.'.format(SPECIES)
    MapData.CONTACT = 'SIYUAN SHEN <s.siyuan@wustl.edu>'
    MapData.LAT_DELTA = lat_delta
    MapData.LON_DELTA = lon_delta
    MapData.TIMECOVERAGE    = '{}/{}'.format(MM,YYYY)

    lat = MapData.createDimension("lat",lat_size)
    lon = MapData.createDimension("lon",lon_size)
    PM25 = MapData.createVariable(SPECIES,'f4',('lat','lon',))
    latitudes = MapData.createVariable("lat","f4",("lat",))
    longitudes = MapData.createVariable("lon","f4",("lon",))
    latitudes[:] = np.arange(extent[0],np.round(extent[1]+lat_delta,decimals=5),lat_delta)
    longitudes[:] = np.arange(extent[2],np.round(extent[3]+lon_delta,decimals=5),lon_delta) 
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    latitudes.standard_name = 'latitude'
    latitudes.long_name = 'latitude'
    longitudes.standard_name = 'longitude'
    longitudes.long_name = 'longitude'
    PM25.units = 'ug/m3'
    PM25.long_name = 'Convolutional Neural Network derived Monthly {} [ug/m^3]'.format(SPECIES)
    PM25[:] = final_data
    return

def save_ForcedSlopeUnity_final_map_data(final_data:np.array, YYYY:str, MM:str, extent:list, SPECIES:str, version:str, special_name):
    outdir = Estimation_outdir + '{}/{}/ForcedSlopeUnity_Map_Estimation/{}/'.format(SPECIES,version,YYYY)
    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + '{}_{}_{}{}{}_ForcedSlopeUnity.nc'.format(SPECIES,version,YYYY,MM,special_name)
    lat_size = final_data.shape[0]
    lon_size = final_data.shape[1]
    lat_delta = 0.01
    lon_delta = 0.01

    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    MapData.TITLE = 'Convolutional Neural Network Monthly {} Estimation over North America Area.'.format(SPECIES)
    MapData.CONTACT = 'SIYUAN SHEN <s.siyuan@wustl.edu>'
    MapData.LAT_DELTA = lat_delta
    MapData.LON_DELTA = lon_delta
    MapData.TIMECOVERAGE    = '{}/{}'.format(MM,YYYY)

    lat = MapData.createDimension("lat",lat_size)
    lon = MapData.createDimension("lon",lon_size)
    PM25 = MapData.createVariable(SPECIES,'f4',('lat','lon',))
    latitudes = MapData.createVariable("lat","f4",("lat",))
    longitudes = MapData.createVariable("lon","f4",("lon",))
    latitudes[:] = np.arange(extent[0],np.round(extent[1]+lat_delta,decimals=5),lat_delta)
    longitudes[:] = np.arange(extent[2],np.round(extent[3]+lon_delta,decimals=5),lon_delta) 
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    latitudes.standard_name = 'latitude'
    latitudes.long_name = 'latitude'
    longitudes.standard_name = 'longitude'
    longitudes.long_name = 'longitude'
    PM25.units = 'ug/m3'
    PM25.long_name = 'Convolutional Neural Network derived Monthly {} [ug/m^3]'.format(SPECIES)
    PM25[:] = final_data
    return

def save_combinedGeo_map_data(final_data:np.array, YYYY:str, MM:str, extent:list, SPECIES:str, version:str, special_name):
    outdir = Estimation_outdir + '{}/{}/Map_Estimation/Combined_withGeo/{}/'.format(SPECIES,version,YYYY)
    
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    outfile = outdir + 'Combined-{}km-Geo{}_{}_{}_{}{}{}.nc'.format(Coefficient_start_distance,SPECIES,SPECIES,version,YYYY,MM,special_name)
    lat_size = final_data.shape[0]
    lon_size = final_data.shape[1]
    lat_delta = 0.01
    lon_delta = 0.01

    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    MapData.TITLE = 'Convolutional Neural Network Monthly {} Estimation combined with Geophysical data over North America Area.'.format(SPECIES)
    MapData.CONTACT = 'SIYUAN SHEN <s.siyuan@wustl.edu>'
    MapData.LAT_DELTA = lat_delta
    MapData.LON_DELTA = lon_delta
    MapData.TIMECOVERAGE    = '{}/{}'.format(MM,YYYY)

    lat = MapData.createDimension("lat",lat_size)
    lon = MapData.createDimension("lon",lon_size)
    PM25 = MapData.createVariable(SPECIES,'f4',('lat','lon',))
    latitudes = MapData.createVariable("lat","f4",("lat",))
    longitudes = MapData.createVariable("lon","f4",("lon",))
    latitudes[:] = np.arange(extent[0],np.round(extent[1]+lat_delta,decimals=5),lat_delta)
    longitudes[:] = np.arange(extent[2],np.round(extent[3]+lon_delta,decimals=5),lon_delta) 
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    latitudes.standard_name = 'latitude'
    latitudes.long_name = 'latitude'
    longitudes.standard_name = 'longitude'
    longitudes.long_name = 'longitude'
    PM25.units = 'ug/m3'
    PM25.long_name = 'Convolutional Neural Network combined with Geophysicl data Monthly {} [ug/m^3]'.format(SPECIES)
    PM25[:] = final_data
    return

def Monthly_PWM_PM_output_text(PWM_PM_dic,species,YYYY,MM,outfile,areas_list):
    with open(outfile,'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Area','Time Period','PWM {} [ug/m^3]'.format(species)])
        for iarea in areas_list:
            for iyear in range(len(YYYY)):
                for imonth in range(len(MM)):
                    writer.writerow([iarea,'{}-{}'.format(YYYY[iyear],MM[imonth]),str(np.round(PWM_PM_dic[iarea][iyear*12+imonth],4))])
        
    return

def Annual_PWM_PM_output_text(PWM_PM_dic,species,YYYY,outfile,areas_list):
    with open(outfile,'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Area','Time Period','PWM {} [ug/m^3]'.format(species)])
        for iarea in areas_list:
            for iyear in range(len(YYYY)):
                writer.writerow([iarea,'{}'.format(YYYY[iyear]),str(np.round(PWM_PM_dic[iarea][iyear],4))])  
    return
