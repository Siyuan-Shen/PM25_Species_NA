import csv 
import numpy as np
import netCDF4 as nc
import os
from Training_pkg.utils import *
from Estimation_pkg.utils import *
from Uncertainty_pkg.utils import *


def load_BLOO_rRMSE():
    indir = '/my-projects/Projects/PM25_Speices_DL_2023/code/Training_Evaluation_Estimation/{}/{}/Results/results-BLOOCV/'.format(species, version)
    rRMSE = np.zeros((13,len(Buffer_radii_forUncertainty)))
    for iradius in range(len(Buffer_radii_forUncertainty)):
        infile = indir + 'BLOO-{}km-SpatialCV_PM25-bias_PM25_v1.0.0_29Channel_11x11_ResNet_Basic1111_1.csv'.format(Buffer_radii_forUncertainty[iradius])
        with open(infile, newline='') as f:
            reader = csv.reader(f)
            count = 0
            for row in reader:
                if count > 1:
                    for i in range(len(row)):
                        print(row[i])
                        if row[i] == '\n NRMSE -  Avg: ':
                            
                            rRMSE[count-2,iradius] = row[i+1]
            
                count += 1
    return rRMSE

def load_rRMSE_map_data( MM:str, version:str, special_name):
    indir = Uncertainty_outdir + '{}/{}/Uncertainty_Results/rRMSE_Map/'.format(species,version)
    infile = indir + 'rRMSE_Map_{}_{}_{}{}.nc'.format(species,version,MM,special_name)
    MapData = nc.Dataset(infile)
    SPECIES_Map = MapData.variables[species][:]
    lat = MapData.variables['latitude'][:]
    lon = MapData.variables['longitude'][:]
    SPECIES_Map = np.array(SPECIES_Map)
    print('Type of SPECIES_MAP: {}'.format(type(SPECIES_Map)))
    return SPECIES_Map, lat, lon

def load_pixels_nearest_sites_distances_map():
    indir = '/my-projects/Projects/PM25_Speices_DL_2023/data/Pixels2sites_distances/'
   
    infile = indir + '{}_nearest_site_distances_forEachPixel.nc'.format(species)
    
    MapData = nc.Dataset(infile)
    Distance_Map = MapData.variables['Distance'][:]
    Distance_Map = np.array(Distance_Map)
    return Distance_Map
def load_absolute_uncertainty_map_data(YYYY:str, MM:str, version:str, special_name):
    indir = Estimation_outdir + '{}/{}/Uncertainty_Results/Absolute-Uncertainty_Map/{}/'.format(species,version,YYYY)
    infile = indir + 'AbsoluteUncertainty_{}_{}_{}{}{}.nc'.format(species,version,YYYY,MM,special_name)
    MapData = nc.Dataset(infile)
    lat = MapData.variables['latitude'][:]
    lon = MapData.variables['longitude'][:]
    SPECIES_Map = MapData.variables[species][:]
    SPECIES_Map = np.array(SPECIES_Map)
    return SPECIES_Map, lat, lon

def load_NA_GeoLatLon():
    indir = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/'
    lat_infile = indir + 'tSATLAT_NA.npy'
    lon_infile = indir + 'tSATLON_NA.npy'
    NA_GeoLAT = np.load(lat_infile)
    NA_GeoLON = np.load(lon_infile)
    return NA_GeoLAT, NA_GeoLON

def load_NA_GeoLatLon_Map():
    indir = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/'
    lat_infile = indir + 'tSATLAT_NA_MAP.npy'
    lon_infile = indir + 'tSATLON_NA_MAP.npy'
    NA_GeoLAT_MAP = np.load(lat_infile)
    NA_GeoLON_MAP = np.load(lon_infile)
    return NA_GeoLAT_MAP, NA_GeoLON_MAP

def save_nearest_site_distances_forEachPixel(nearest_distance_map,extent_lat,extent_lon):
    outdir = '/my-projects/Projects/PM25_Speices_DL_2023/data/Pixels2sites_distances/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + '{}_nearest_site_distances_forEachPixel.nc'.format(species)
    
    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    MapData.TITLE = 'Nearset distance for each pixel from {} sites'.format(species)
    MapData.CONTACT = 'SIYUAN SHEN <s.siyuan@wustl.edu>'

    lat = MapData.createDimension("lat",len(extent_lat))
    lon = MapData.createDimension("lon",len(extent_lon))
    Distance = MapData.createVariable('Distance','f4',('lat','lon',))
    latitudes = MapData.createVariable("latitude","f4",("lat",))
    longitudes = MapData.createVariable("longitude","f4",("lon",))
    latitudes[:] = extent_lat
    longitudes[:] = extent_lon
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    latitudes.standard_name = 'latitude'
    latitudes.long_name = 'latitude'
    longitudes.standard_name = 'longitude'
    longitudes.long_name = 'longitude'
    Distance.units = 'kilometer'
    Distance[:] = nearest_distance_map
    return
def save_rRMSE_uncertainty_Map(Map_rRMSE:np.array,MM:str,):

    outdir = Uncertainty_outdir + '{}/{}/Uncertainty_Results/rRMSE_Map/'.format(species,version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + 'rRMSE_Map_{}_{}_{}{}.nc'.format(species,version,MM,special_name)
    lat_size = Map_rRMSE.shape[0]
    lon_size = Map_rRMSE.shape[1]
    lat_delta = (Extent[1]-Extent[0])/lat_size
    lon_delta = (Extent[3]-Extent[2])/lon_size

    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    MapData.TITLE = 'Convolutional Neural Network Monthly {} rRMSE Map over North America Area.'.format(species)
    MapData.CONTACT = 'SIYUAN SHEN <s.siyuan@wustl.edu>'
    MapData.LAT_DELTA = lat_delta
    MapData.LON_DELTA = lon_delta
    MapData.TIMECOVERAGE    = '{}'.format(MM)

    lat = MapData.createDimension("lat",lat_size)
    lon = MapData.createDimension("lon",lon_size)
    rRMSE = MapData.createVariable(species,'f4',('lat','lon',))
    latitudes = MapData.createVariable("latitude","f4",("lat",))
    longitudes = MapData.createVariable("longitude","f4",("lon",))
    latitudes[:] = np.arange(Extent[0],Extent[1],lat_delta)
    longitudes[:] = np.arange(Extent[2],Extent[3],lon_delta) 
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    latitudes.standard_name = 'latitude'
    latitudes.long_name = 'latitude'
    longitudes.standard_name = 'longitude'
    longitudes.long_name = 'longitude'
    rRMSE.units = 'unitless'
    rRMSE.long_name = 'Convolutional Neural Network derived Monthly {} rRMSE'.format(species)
    rRMSE[:] = Map_rRMSE
    return

def save_absolute_uncertainty_data(final_data:np.array, YYYY:str, MM:str):
    outdir = Estimation_outdir + '{}/{}/Uncertainty_Results/Absolute-Uncertainty_Map/{}/'.format(species,version,YYYY)
    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + 'AbsoluteUncertainty_{}_{}_{}{}{}.nc'.format(species,version,YYYY,MM,special_name)
    lat_size = final_data.shape[0]
    lon_size = final_data.shape[1]
    lat_delta = (Extent[1]-Extent[0])/lat_size
    lon_delta = (Extent[3]-Extent[2])/lon_size

    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    MapData.TITLE = 'Convolutional Neural Network Monthly {} Absolute Uncertainty Estimation over North America Area.'.format(species)
    MapData.CONTACT = 'SIYUAN SHEN <s.siyuan@wustl.edu>'
    MapData.LAT_DELTA = lat_delta
    MapData.LON_DELTA = lon_delta
    MapData.TIMECOVERAGE    = '{}/{}'.format(MM,YYYY)

    lat = MapData.createDimension("lat",lat_size)
    lon = MapData.createDimension("lon",lon_size)
    PM25 = MapData.createVariable(species,'f4',('lat','lon',))
    latitudes = MapData.createVariable("latitude","f4",("lat",))
    longitudes = MapData.createVariable("longitude","f4",("lon",))
    latitudes[:] = np.arange(Extent[0],Extent[1],lat_delta)
    longitudes[:] = np.arange(Extent[2],Extent[3],lon_delta) 
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    latitudes.standard_name = 'latitude'
    latitudes.long_name = 'latitude'
    longitudes.standard_name = 'longitude'
    longitudes.long_name = 'longitude'
    PM25.units = 'ug/m3'
    PM25.long_name = 'Convolutional Neural Network derived Monthly {} absolute Uncertainty [ug/m^3]'.format(species)
    PM25[:] = final_data
    return
