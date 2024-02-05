import torch
import numpy as np
import os
import csv
import netCDF4 as nc
import time
from Estimation_pkg.utils import *


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
    lat = MapData.variables['latitude'][:]
    lon = MapData.variables['longitude'][:]
    SPECIES_Map = MapData.variables[SPECIES][:]
    return SPECIES_Map, lat, lon

def save_trained_model_forEstimation(cnn_model, model_outdir, typeName, version, species, nchannel, special_name,beginyear, endyear, width, height):
    outdir = model_outdir + '{}/{}/Results/Estimation-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    model_outfile = outdir +  'Estimation_{}_{}_{}x{}_{}-{}_{}Channel{}.pt'.format(typeName, species, width,height, beginyear, endyear, nchannel,special_name)
    torch.save(cnn_model, model_outfile)
    return 

def save_final_map_data(final_data:np.array, YYYY:str, MM:str, extent:list, SPECIES:str, version:str, special_name):
    outdir = Estimation_outdir + '{}/{}/Map_Estimation/{}/'.format(SPECIES,version,YYYY)
    
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    outfile = outdir + '{}_{}_{}{}{}.nc'.format(SPECIES,version,YYYY,MM,special_name)
    lat_size = final_data.shape[0]
    lon_size = final_data.shape[1]
    lat_delta = (extent[1]-extent[0])/lat_size
    lon_delta = (extent[3]-extent[2])/lon_size

    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    MapData.TITLE = 'Convolutional Neural Network Monthly {} Estimation over North America Area.'.format(SPECIES)
    MapData.CONTACT = 'SIYUAN SHEN <s.siyuan@wustl.edu>'
    MapData.LAT_DELTA = lat_delta
    MapData.LON_DELTA = lon_delta
    MapData.TIMECOVERAGE    = '{}/{}'.format(MM,YYYY)

    lat = MapData.createDimension("lat",lat_size)
    lon = MapData.createDimension("lon",lon_size)
    PM25 = MapData.createVariable(SPECIES,'f4',('lat','lon',))
    latitudes = MapData.createVariable("latitude","f4",("lat",))
    longitudes = MapData.createVariable("longitude","f4",("lon",))
    latitudes[:] = np.arange(extent[0],extent[1],lat_delta)
    longitudes[:] = np.arange(extent[2],extent[3],lon_delta) 
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
    outdir = Estimation_outdir + '{}/{}/Map_Estimation/{}/'.format(SPECIES,version,YYYY)
    
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    outfile = outdir + 'Combined-Geo{}_{}_{}_{}{}{}.nc'.format(SPECIES,SPECIES,version,YYYY,MM,special_name)
    lat_size = final_data.shape[0]
    lon_size = final_data.shape[1]
    lat_delta = (extent[1]-extent[0])/lat_size
    lon_delta = (extent[3]-extent[2])/lon_size

    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    MapData.TITLE = 'Convolutional Neural Network Monthly {} Estimation combined with Geophysical data over North America Area.'.format(SPECIES)
    MapData.CONTACT = 'SIYUAN SHEN <s.siyuan@wustl.edu>'
    MapData.LAT_DELTA = lat_delta
    MapData.LON_DELTA = lon_delta
    MapData.TIMECOVERAGE    = '{}/{}'.format(MM,YYYY)

    lat = MapData.createDimension("lat",lat_size)
    lon = MapData.createDimension("lon",lon_size)
    PM25 = MapData.createVariable(SPECIES,'f4',('lat','lon',))
    latitudes = MapData.createVariable("latitude","f4",("lat",))
    longitudes = MapData.createVariable("longitude","f4",("lon",))
    latitudes[:] = np.arange(extent[0],extent[1],lat_delta)
    longitudes[:] = np.arange(extent[2],extent[3],lon_delta) 
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
