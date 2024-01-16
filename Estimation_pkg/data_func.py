import numpy as np

def get_extent_index(extent)->np.array:
    '''
    :param extent:
        The range of the input. [Bottom_Lat, Up_Lat, Left_Lon, Right_Lon]
    :return:
        lat_index, lon_index
    '''
    indir = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/'
    lat_infile = indir + 'tSATLAT_NA.npy'
    lon_infile = indir + 'tSATLON_NA.npy'
    SATLAT = np.load(lat_infile)
    SATLON = np.load(lon_infile)
    lat_index = np.where((SATLAT >= extent[0])&(SATLAT<=extent[1]))
    lon_index = np.where((SATLON >= extent[2])&(SATLON<=extent[3]))
    lat_index = np.squeeze(np.array(lat_index))
    lon_index = np.squeeze(np.array(lon_index))
    return lat_index,lon_index

def get_GL_extent_index(extent)->np.array:
    '''
    :param extent:
        The range of the input. [Bottom_Lat, Up_Lat, Left_Lon, Right_Lon]
    :return:
        lat_index, lon_index
    '''
    SATLAT = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/tSATLAT.npy')
    SATLON = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/tSATLON.npy')
    lat_index = np.where((SATLAT >= extent[0])&(SATLAT<=extent[1]))
    lon_index = np.where((SATLON >= extent[2])&(SATLON<=extent[3]))
    lat_index = np.squeeze(np.array(lat_index))
    lon_index = np.squeeze(np.array(lon_index))
    return lat_index,lon_index

def get_landtype(YYYY,extent)->np.array:
    landtype_infile = '/my-projects/Projects/MLCNN_PM25_2021/data/inputdata/Other_Variables_MAP_INPUT/{}/MCD12C1_LandCoverMap_{}.npy'.format(YYYY,YYYY)
    landtype = np.load(landtype_infile)
    landtype = np.array(landtype,dtype=int)
    lat_index,lon_index = get_GL_extent_index(extent=extent)
    output = np.zeros((len(lat_index),len(lon_index)), dtype=int)
    for ix in range(len(lat_index)):
        output[ix,:] = landtype[lat_index[ix],lon_index]
    return output