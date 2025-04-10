import numpy as np
import xarray as xr
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import RootMeanSquaredError
import os
import joblib

def assign_GPU():
    '''
    This Function must be used before create any tf.tensor data
    '''
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
        # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
            print(e)
assign_GPU()
path = './GFED5/model/global/'
data = np.genfromtxt(path+'UMD_Y_mean_std.csv', delimiter=',')
y_mean = data[0]
y_std  = data[1]
del(data)

latS = -60
latN = 90

#ssp585
model = ["ACCESS-ESM1-5","BCC-CSM2-MR","CanESM5-CanOE","CanESM5","CESM2-WACCM","CESM2","CMCC-CM2-SR5","CMCC-ESM2","CNRM-ESM2-1",\
            "CNRM-CM6-1","E3SM-1-1","EC-Earth3-CC","EC-Earth3-Veg","FGOALS-g3","GFDL-CM4","GFDL-ESM4",\
            "GISS-E2-1-G","HadGEM3-GC31-LL","HadGEM3-GC31-MM","INM-CM4-8","INM-CM5-0","IPSL-CM6A-LR","MPI-ESM1-2-HR","MPI-ESM1-2-LR","MIROC-ES2L",\
            "NorESM2-LM","NorESM2-MM","UKESM1-0-LL"]

nmodel = len(model)
print(nmodel)

models = []

for file in os.listdir(path):
    if file.endswith('.h5'):
        models.append(tf.keras.models.load_model(os.path.join(path, file),custom_objects={'RootMeanSquaredError': RootMeanSquaredError}))
nt = 2100-2003+1
fires_new = np.zeros((nmodel,nt*12,5))
fires_forest = np.zeros((nmodel,nt*12,5))
fires_VPD = np.zeros((nmodel,nt*12,5))
fires_tas = np.zeros((nmodel,nt*12,5))
fires_cropland = np.zeros((nmodel,nt*12,5))

i=0
for m in model:
    forest_GFED5 = np.zeros((2,98,75,180))
    cropland_GFED5 = np.zeros((2,98,75,180))
    ds = xr.open_dataset(f'./LUH2-{m}_ssp585_lightgbm.nc', decode_times=False)
    forest = ds.forest_model[2:,:,:] #2003-2100,
    cropland = ds.cropland_model[2:,:,:] 

    lat_new = ds.lat 
    lon_new = ds.lon
    nlat = len(lat_new)
    nlon = len(lon_new)
    print(nlat)
    print(nlon)
    forest_GFED5[0,:,:,:] = np.array(forest)
    cropland_GFED5[0,:,:,:] = np.array(cropland)
    del forest,cropland

    ds = xr.open_dataset(f'./LUH2-{m}_ssp585_xgboost.nc', decode_times=False)
    forest = ds.forest_model[2:,:,:].sel(lat=slice(latS,latN))
    cropland = ds.cropland_model[2:,:,:].sel(lat=slice(latS,latN)) 

    forest_GFED5[1,:,:,:] = np.array(forest)
    cropland_GFED5[1,:,:,:] = np.array(cropland)
    del forest,cropland

    forest = np.nanmean(forest_GFED5,axis=0)
    cropland = np.nanmean(cropland_GFED5,axis=0)
    del forest_GFED5,cropland_GFED5

    forest2 = np.zeros((98*12,75,180)) 
    cropland2 = np.zeros((98*12,75,180)) 
    for iyear in np.arange(0,98,1) :
        forest2[12*iyear:12*(iyear+1),:,:] = forest[iyear,:,:]
        cropland2[12*iyear:12*(iyear+1),:,:] = cropland[iyear,:,:] 
    del forest,cropland
  
    
    predictor=np.zeros((nt*12,nlat,nlon,4))
    predictor[:,:,:,2] = forest2
    predictor[:,:,:,3] = cropland2
    
    file_path = f'./VPD_{m}_hist_SSP585_corrected.nc'
    ds = xr.open_dataset(file_path,decode_times=False) 
    VPD = ds.VPD #time:2003-2100,lon: -180-180
    time = VPD.time
    VPD_new = VPD.interp(lat=lat_new,lon=lon_new)
    predictor[:,:,:,0]= np.array(VPD_new)
    del(VPD) 
    del(VPD_new)
    
    file_path = f'./tas_{m}_hist_SSP585_corrected.nc'
    ds = xr.open_dataset(file_path,decode_times=False) 
    tas = ds.tas #time:2003-2100,lon:-180-180
    tas_new = tas.interp(lat=lat_new,lon=lon_new)
    predictor[:,:,:,1]= np.array(tas_new)
    del(tas) 
    del(tas_new)
    
    x_mean = np.nanmean(predictor[0:19*12,:,:,:], axis=(0, 1, 2))
    x_std = np.nanstd(predictor[0:19*12,:,:,:], axis=(0, 1, 2))
    predictor = (predictor - x_mean)/x_std
    predictor = np.where(np.isnan(predictor),0,predictor)

#dfifferent conditions
    predictor_VPD=np.zeros((nt*12,nlat,nlon,4))
    predictor_tas=np.zeros((nt*12,nlat,nlon,4))
    predictor_forest=np.zeros((nt*12,nlat,nlon,4))
    predictor_cropland=np.zeros((nt*12,nlat,nlon,4))

    predictor_VPD[:,:,:,0] = predictor[:,:,:,0]
    predictor_VPD[0:19*12,:,:,1:] = predictor[0:19*12,:,:,1:]
    predictor_tas[0:19*12,:,:,0] = predictor[0:19*12,:,:,0]
    #unchanged tas since 2021
    for imon in range(12): 
        predictor_VPD[19*12+imon::12,:,:,1] = predictor[18*12+imon,:,:,1]
        predictor_VPD[19*12+imon::12,:,:,2] = predictor[18*12+imon,:,:,2]
        predictor_VPD[19*12+imon::12,:,:,3] = predictor[18*12+imon,:,:,3]
        predictor_tas[19*12+imon::12,:,:,0] = predictor[18*12+imon,:,:,0]


    predictor_tas[:,:,:,1] = predictor[:,:,:,1]
    predictor_tas[:,:,:,2] = predictor_VPD[:,:,:,2]
    predictor_tas[:,:,:,3] = predictor_VPD[:,:,:,3]

    predictor_forest[:,:,:,0] = predictor_tas[:,:,:,0]
    predictor_forest[:,:,:,1] = predictor_VPD[:,:,:,1]
    predictor_forest[:,:,:,2] = predictor[:,:,:,2]
    predictor_forest[:,:,:,3] = predictor_VPD[:,:,:,3]

    predictor_cropland[:,:,:,0] = predictor_tas[:,:,:,0]
    predictor_cropland[:,:,:,1] = predictor_VPD[:,:,:,1]
    predictor_cropland[:,:,:,2] = predictor_VPD[:,:,:,2]
    predictor_cropland[:,:,:,3] = predictor[:,:,:,3]

    print(i)
    for j in range(5):
        fires_new[i,:,j] = (models[j].predict(predictor).reshape(12*nt))*y_std+y_mean
        fires_forest[i,:,j] = (models[j].predict(predictor_forest).reshape(12*nt))*y_std+y_mean
        fires_VPD[i,:,j] = (models[j].predict(predictor_VPD).reshape(12*nt))*y_std+y_mean
        fires_tas[i,:,j] = (models[j].predict(predictor_tas).reshape(12*nt))*y_std+y_mean
        fires_cropland[i,:,j] = (models[j].predict(predictor_cropland).reshape(12*nt))*y_std+y_mean

    i=i+1
    del predictor,predictor_VPD,predictor_tas,predictor_forest,predictor_cropland
    del(x_mean)
    del(x_std)
    
fires_Hist = (np.nanmean(fires_new,axis=2)).astype(np.float32)
fires_Hist_VPD = (np.nanmean(fires_VPD,axis=2)).astype(np.float32)
fires_Hist_tas = (np.nanmean(fires_tas,axis=2)).astype(np.float32) 
fires_Hist_forest = (np.nanmean(fires_forest,axis=2)).astype(np.float32)
fires_Hist_cropland = (np.nanmean(fires_cropland,axis=2)).astype(np.float32)

filename = 'UMD_LUchange_SSP585_global.nc'

if os.path.exists(filename):
    os.remove(filename)
    
data=xr.Dataset(data_vars={"fires_Hist":(('model','time'),fires_Hist),\
    "fires_Hist_forest":(('model','time'),fires_Hist_forest),\
    "fires_Hist_VPD":(('model','time'),fires_Hist_VPD),\
    "fires_Hist_tas":(('model','time'),fires_Hist_tas),\
    "fires_Hist_cropland":(('model','time'),fires_Hist_cropland)},
                    coords={'model':model,'time':time})
data.to_netcdf(filename)

