import toml

cfg = toml.load('./config.toml')

Loss_Accuracy_outdir = cfg['Pathway']['Figures-dir']['Loss_Accuracy_outdir']
Estimation_Map_outdir = cfg['Pathway']['Figures-dir']['Estimation_Map_outdir']
Uncertainty_Map_outdir = cfg['Pathway']['Figures-dir']['Uncertainty_Map_outdir']
SHAP_Analysis_outdir =  cfg['Pathway']['Figures-dir']['SHAP_Analysis_outdir']
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