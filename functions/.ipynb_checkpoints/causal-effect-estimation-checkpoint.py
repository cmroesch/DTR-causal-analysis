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

stn = np.load('/home/s2135337/carla/results/stations/stacked-01-14.npy', allow_pickle = True)
stnClate = np.load('/home/s2135337/carla/results/cmip6/canesm5/stations/stackeddata-2001-2014.npy', allow_pickle = True)
stnHlate = np.load('/home/s2135337/carla/results/cmip6/hadgem3/stations/stackeddata-2001-2014.npy', allow_pickle = True)

stn[0] = stn[0]/10
stn[1] = stn[1]/10
stn[3] = stn[3]/8

dS = np.array([stn[i,:,:] for i in range(5)])
d = {i: dS[:,i,:].transpose() for i in range(4844)}

dSClate = np.array([stnClate[i,:,:] for i in range(5)])
dClate = {i: dSClate[:,i,:].transpose() for i in range(3766)}

dSHlate = np.array([stnHlate[i,:,:] for i in range(5)])
dHlate = {i: dSHlate[:,i,:].transpose() for i in range(10878)}

def ce_estimation(dataset):
    
    true_parents = {0:[(1,-1),(3,0),(4,0),(0,-1)],
                   1:[(0,0),(2,0),(3,0),(1,-1)],
                   2:[(3,0),(4,0),(2,-1)],
                   3:[(4,0),(3,-1)],
                   4:[(4,-1)]}
    dataframe = pp.DataFrame(dataset, analysis_mode = 'multiple', var_names = ['tmin','tmax','sw','cloudiness','aod'])
    
    med = LinearMediation(dataframe=dataframe, data_transform = None)
    med.fit_model(all_parents=true_parents, tau_max=1)
    med.fit_model_bootstrap(boot_blocklength=3, seed=42, boot_samples=1000)
    
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

np.save('/home/s2135337/carla/results/stations/ce-stations-01-14.npy', ce_estimation(d))
np.save('/home/s2135337/carla/results/cmip6/ce-CanESM5-01-14.npy', ce_estimation(dClate))
np.save('/home/s2135337/carla/results/cmip6/ce-HadGEM3-01-14.npy', ce_estimation(dHlate))
