import numpy as np 
import os
from scipy.interpolate import NearestNDInterpolator
import time

from Training_pkg.iostream import load_monthly_obs_data
from Training_pkg.utils import *

from Estimation_pkg.data_func import get_landtype,get_extent_index
from Estimation_pkg.utils import *

from Evaluation_pkg.utils import *

from Uncertainty_pkg.iostream import load_NA_GeoLatLon,load_NA_GeoLatLon_Map,save_nearest_site_distances_forEachPixel
from Uncertainty_pkg.utils import get_extent_lat_lon_map


def convert_distance_to_rRMSE_uncertainty(buffer_radii, BLOO_rRMSE, map_distances,):
    map_uncertainty = np.zeros(map_distances.shape,dtype=np.float64)
    for iradius in range(len(buffer_radii)-1):
        d_left  = buffer_radii[iradius]
        d_right = buffer_radii[iradius+1]
        rRMSE_left  = BLOO_rRMSE[iradius]
        rRMSE_right = BLOO_rRMSE[iradius+1]
        pixels_index = np.where((map_distances >= d_left) & (map_distances < d_right))
        
        print('d_left: {}, d_right: {}, rRMSE_left: {}, rRMSE_right: {}'.format(d_left,d_right,rRMSE_left,rRMSE_right))
        map_uncertainty[pixels_index] = (map_distances[pixels_index]-d_left)/(d_right-d_left) * (rRMSE_right - rRMSE_left) +rRMSE_left
        #print('Distance: ', map_distances[pixels_index], 'Uncertainty: ',map_uncertainty[pixels_index])
    
    d_left  = buffer_radii[0]
    d_right = buffer_radii[-1]
    rRMSE_left  = BLOO_rRMSE[0]
    rRMSE_right = BLOO_rRMSE[-1] 
    outrange_pixels_index = np.where(map_distances >= buffer_radii[-1])
    map_uncertainty[outrange_pixels_index] = rRMSE_right #(map_distances[outrange_pixels_index]-d_left)/(d_right-d_left) * (rRMSE_right - rRMSE_left) +rRMSE_left
    return map_uncertainty

def get_nearest_site_distance_for_each_pixel():
    SATLAT,SATLON = load_NA_GeoLatLon()
    lat_index, lon_index = get_extent_index(Extent)
    extent_lat_map,extent_lon_map = get_extent_lat_lon_map(lat_index=lat_index,lon_index=lon_index,SATLAT=SATLAT,SATLON=SATLON)
    SPECIES, sites_lat,sites_lon = load_monthly_obs_data(species=species)
    landtype = get_landtype(YYYY=2015,extent=Extent)
    interp_start = time.time()
    #interp = NearestNDInterpolator(list(zip(sites_lat,sites_lon)),sites_index)
    #nearest_index_map = interp(tSATLAT_map,tSATLON_map)
    interp_lat = NearestNDInterpolator(list(zip(sites_lat,sites_lon)),sites_lat)
    interp_lon = NearestNDInterpolator(list(zip(sites_lat,sites_lon)),sites_lon)
    nearest_lat_map = interp_lat(extent_lat_map,extent_lon_map)
    nearest_lon_map = interp_lon(extent_lat_map,extent_lon_map)
    
    interp_end   = time.time()

    interp_total = interp_end - interp_start
    print('Finish the nearest interpolation! Time costs:',interp_total,' seconds')
    nearest_distance_map = np.full(nearest_lat_map.shape,1000.0)
    for ix in range(len(lat_index)):
        land_index = np.where(landtype[ix,:] != 0)
        print('It is procceding ' + str(np.round(100*(ix/len(lat_index)),2))+'%.' )
        if len(land_index[0]) == 0:
            print('No lands.')
            None
        else:
            start_time = time.time()
            nearest_distance_map[ix,land_index[0]] = calculate_distance_forArray(nearest_lat_map[ix,land_index[0]],
                                                                        nearest_lon_map[ix,land_index[0]],
                                                                        extent_lat_map[ix,land_index[0]],extent_lon_map[ix,land_index[0]])
            print(nearest_distance_map[ix,land_index[0]])
            end_time = time.time()
            Get_distance_forOneLatitude_time = end_time - start_time
            print('Time for getting distance for one latitude', Get_distance_forOneLatitude_time, 's, the number of pixels is ', len(land_index[0]))

    save_nearest_site_distances_forEachPixel(nearest_distance_map=nearest_distance_map,extent_lat=SATLAT[lat_index],extent_lon=SATLON[lon_index])
    return 
