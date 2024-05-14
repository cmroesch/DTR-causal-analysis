import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sklearn

import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, RobustParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction

import xarray as xr

def causal_grid_box(tn,tx,qq,clt,aod,lat,lon):
    j1 = 120
    j2 = 243

    yr_nan = qq.reshape(14,360)[:,j1:j2].mean(axis =1)*tn.reshape(14,360)[:,j1:j2].mean(axis =1)*tx.reshape(14,360)[:,j1:j2].mean(axis =1)*clt.reshape(14,360)[:,j1:j2].mean(axis =1)*aod.reshape(14,360)[:,j1:j2].mean(axis =1)

    if np.where(~np.isnan(yr_nan))[0].shape[0] >= 10:
        var_names = ['tmin','tmax','sw','cloudiness','rh','aod']
        dataStacked = np.stack([tn.reshape(14,360)[np.where(~np.isnan(yr_nan))[0],j1:j2],
                                tx.reshape(14,360)[np.where(~np.isnan(yr_nan))[0],j1:j2],
                                qq.reshape(14,360)[np.where(~np.isnan(yr_nan))[0],j1:j2],
                                clt.reshape(14,360)[np.where(~np.isnan(yr_nan))[0],j1:j2],
                                aod.reshape(14,360)[np.where(~np.isnan(yr_nan))[0],j1:j2]])
        
        data = {i: dataStacked[:,i,:].transpose() for i in range(np.where(~np.isnan(yr_nan))[0].shape[0])}

        results = ce_estimation(data)

        np.save('/home/s2135337/carla/results/cmip6/causal-effects-HadGEM3-01-14/'+str(lat[0])+'-'+str(lon[0]),results)

def ce_estimation(dataset):
    
    true_parents = {0:[(1,-1),(3,0),(4,0),(0,-1)],
                   1:[(0,0),(2,0),(3,0),(1,-1)],
                   2:[(3,0),(4,0),(2,-1)],
                   3:[(4,0),(3,-1)],
                   4:[(4,-1)]}
    dataframe = pp.DataFrame(dataset, analysis_mode = 'multiple', var_names = ['tmin','tmax','sw','cloudiness','aod'])
    
    med = LinearMediation(dataframe=dataframe, data_transform = None)
    med.fit_model(all_parents=true_parents, tau_max=1)
    med.fit_model_bootstrap(boot_blocklength=3, seed=42, boot_samples=100)
    
    results = np.zeros((13,3))*np.nan
    
    i_ = [4,4,4,3,3,3,2,1,0]
    j_ = [3,2,0,2,1,0,1,0,1]
    l_ = [0,0,0,0,0,0,0,1,0]
    
    for n in range(9):
        results[n,0] = med.get_coeff(i=i_[n], tau=l_[n],  j=j_[n])
        results[n,1:3] = med.get_bootstrap_of(function='get_coeff', function_args={'i':i_[n], 'tau':l_[n],  'j':j_[n]}, conf_lev=0.9)
    
    results[9,0] = med.get_mce(i=4, tau=0,  j=2, k = 3)
    results[9,1:3] = med.get_bootstrap_of(function='get_mce', function_args={'i':4, 'tau':0,  'j':2, 'k':3}, conf_lev=0.9)
    
    results[10,0] = med.get_ce(i=4, tau=0,  j=0)
    results[10,1:3] = med.get_bootstrap_of(function='get_ce', function_args={'i':4, 'tau':0,  'j':0}, conf_lev=0.9)
    
    results[11,0] = med.get_ce(i=4, tau=0,  j=1)
    results[11,1:3] = med.get_bootstrap_of(function='get_ce', function_args={'i':4, 'tau':0,  'j':1}, conf_lev=0.9)
    
    results[12,0] = med.get_ce(i=4, tau=0,  j=2)
    results[12,1:3] = med.get_bootstrap_of(function='get_ce', function_args={'i':4, 'tau':0,  'j':2}, conf_lev=0.9)
    
    return results



ee = xr.open_dataset('/home/s2135337/carla/cmip6/HadGEM3_all.nc')

# ee = e.sel(time=~((e.time.dt.month == 2) & (e.time.dt.day == 29)))


dime = np.zeros((5040, 2, 41,62))*np.nan
for i in range(41):
    for j in range(62):
        dime[:,0,i,j] = ee.lat.values[i]
        dime[:,1,i,j] = ee.lon.values[j]
        
ee['lattt'] = (['time', 'lat', 'lon'],  dime[:,0,:,:])
ee['lonnn'] = (['time', 'lat', 'lon'],  dime[:,1,:,:])

print(ee)

xr.apply_ufunc(causal_grid_box, ee.tasmin, ee.tasmax, ee.rsds, ee.clt, ee.od550aer, ee.lattt, ee.lonnn, input_core_dims=[["time"],["time"],["time"],["time"],["time"],["time"],["time"]], dask = 'allowed', vectorize = True)
