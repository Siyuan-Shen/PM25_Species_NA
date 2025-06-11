import toml
import numpy as np

cfg = toml.load('./config.toml')

Loss_Accuracy_outdir = cfg['Pathway']['Figures-dir']['Loss_Accuracy_outdir']
Estimation_Map_outdir = cfg['Pathway']['Figures-dir']['Estimation_Map_outdir']
Uncertainty_Map_outdir = cfg['Pathway']['Figures-dir']['Uncertainty_Map_outdir']
SHAP_Analysis_outdir =  cfg['Pathway']['Figures-dir']['SHAP_Analysis_outdir']

def get_length_of_valid_points(obs_data,sitesnumber,kfold,start_year,end_year):
    length_valid_points = np.zeros((sitesnumber), dtype=np.int32)
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for iyear in range(end_year-start_year+1):
        for imonth in range(len(MONTH)):
            for ikfold in range(kfold):
               temp_site =obs_data[str(start_year+iyear)][MONTH[imonth]][ikfold*sitesnumber:(ikfold+1)*sitesnumber]
               temp_index = np.where(~np.isnan(temp_site))
               length_valid_points[temp_index] += 1             
   
    return length_valid_points

def crop_map_data(MapData, lat, lon, Extent):
    bottom_lat = Extent[0]
    top_lat    = Extent[1]
    left_lon   = Extent[2]
    right_lon  = Extent[3]
    lat_start_index = round((bottom_lat - lat[0])* 100 )
    lon_start_index = round((left_lon - lon[0]) * 100 )
    lat_end_index = round((top_lat - lat[0]) * 100 )
    lon_end_index = round((right_lon - lon[0])*100)
    cropped_mapdata = MapData[lat_start_index:lat_end_index+1,lon_start_index:lon_end_index+1]
    return cropped_mapdata

def species_plot_tag_Name(species):
    plot_tag_name_dic = {
        'PM25':'PM$_{2.5}$',
        'SO4' : 'SO$_4^{2-}$',
        'NO3' : 'NO$_3^-$',
        'NH4' : 'NH$_4^+$',
        'OM'  : 'OM',
        'BC'  : 'BC',
        'DUST': 'DUST',
        'SS'  : 'SS'
    }
    return plot_tag_name_dic[species]