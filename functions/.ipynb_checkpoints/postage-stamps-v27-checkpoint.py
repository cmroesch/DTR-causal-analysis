import geopandas
import geoplot
import pandas as pd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import xarray as xr
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
from tigramite.data_processing import DataFrame
import netCDF4
import os
import re
#import seaborn as sns
#from datetime import datetime
import datetime
#from sklearn.linear_model import LinearRegression

from random import seed
from random import randint

import warnings
warnings.filterwarnings("ignore")

import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec

import scipy.stats as stats
from scipy import signal

## use `%matplotlib notebook` for interactive figures
# plt.style.use('ggplot')
import sklearn

import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys

from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb

import xarray as xr
import netCDF4

import scipy.stats as stats

import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import Models
from tigramite.causal_effects import CausalEffects

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

def causal_grid_box(tn,tx,qq,clt,aod, lat, lon):
    j1 = 120
    j2 = 243

    yr_nan = qq.reshape(21,365)[:,j1:j2].mean(axis =1)*tn.reshape(21,365)[:,j1:j2].mean(axis =1)*tx.reshape(21,365)[:,j1:j2].mean(axis =1)*clt.reshape(21,365)[:,j1:j2].mean(axis =1)*aod.reshape(21,365)[:,j1:j2].mean(axis =1)

    if np.where(~np.isnan(yr_nan))[0].shape[0] >= 10:
        var_names = ['tmin','tmax','sw','cloudiness','rh','aod']
        dataStacked = np.stack([tn.reshape(21,365)[np.where(~np.isnan(yr_nan))[0],j1:j2],
                                tx.reshape(21,365)[np.where(~np.isnan(yr_nan))[0],j1:j2],
                                qq.reshape(21,365)[np.where(~np.isnan(yr_nan))[0],j1:j2],
                                clt.reshape(21,365)[np.where(~np.isnan(yr_nan))[0],j1:j2],
                                aod.reshape(21,365)[np.where(~np.isnan(yr_nan))[0],j1:j2]])
        data = {i: dataStacked[:,i,:].transpose() for i in range(np.where(~np.isnan(yr_nan))[0].shape[0])}
        dataframe = DataFrame(data, analysis_mode = 'multiple', var_names = var_names)

        results = run_and_plot(dataframe, ParCorr())
        
        # print(results)
        plt.savefig('/home/s2135337/carla/results/eobs/v27/graphs/'+str(lat[0])+'-'+str(lon[0])+'.png')
        np.save('/home/s2135337/carla/results/eobs/v27/results/'+str(lat[0])+'-'+str(lon[0]),np.array(list(results.items()))[:,1])
        

def run_and_plot(dataframe, cond_ind_test):
    
    var_names = ['tmin','tmax','sw','cloudiness','aod']
    # Create and run Multidata-PCMCIplus
    pcmci = PCMCI(dataframe = dataframe, cond_ind_test = cond_ind_test)

    link_assumptions = {j:{(i, -tau):'o-o' for i in range(5) for tau in range(1) if (i, -tau) != (j, 0)} for j in range(5)}
    link_assumptions[0] = {(i, -tau):'o-o' if i == 1 else '-->' for i in [0,1,3,4] for tau in range(1) if (i, -tau) != (0, 0)}
    link_assumptions[1] = {(i, -tau):'o-o' if i == 0 else '-->' for i in [0,1,2,3] for tau in range(1) if (i, -tau) != (1, 0)}
    link_assumptions[2] = {(i, -tau):'<--' if i in [1] else '-->' for i in [1,2,3,4] for tau in range(1) if (i, -tau) != (2, 0)}
    link_assumptions[3] = {(i, -tau):'<--' if i in [0,1,2] else '-->' for i in range(5) for tau in range(1) if (i, -tau) != (3, 0)}
    link_assumptions[4] = {(i, -tau):'<--' if i in [0,1,2,3] else '-->' for i in [0,2,3,4] for tau in range(1) if (i, -tau) != (4, 0)}

    link_assumptions[1][(0, -1)] = 'o-o'
    link_assumptions[0][(0, -1)] = 'o-o'
    link_assumptions[1][(1, -1)] = 'o-o'
    link_assumptions[2][(2, -1)] = 'o-o'
    link_assumptions[3][(3, -1)] = 'o-o'
    link_assumptions[4][(4, -1)] = 'o-o'

    results = pcmci.run_pcmciplus(tau_min=0, tau_max=1, pc_alpha=0.01, link_assumptions = link_assumptions)
    
    tp.plot_graph(graph=results['graph'], val_matrix=results['val_matrix'], var_names=var_names)
    
    
    return results



e = xr.open_dataset('/home/s2135337/carla/ceres-clara-merra-eobs27-eobs26-merged-20010101-20211231.nc')

ee = e.sel(time=~((e.time.dt.month == 2) & (e.time.dt.day == 29)))


dime = np.zeros((7665, 2, 140,280))*np.nan
for i in range(140):
    for j in range(280):
        dime[:,0,i,j] = ee.lat.values[i]
        dime[:,1,i,j] = ee.lon.values[j]
        
ee['lattt'] = (['time', 'lat', 'lon'],  dime[:,0,:,:])
ee['lonnn'] = (['time', 'lat', 'lon'],  dime[:,1,:,:])

print(ee)

xr.apply_ufunc(causal_grid_box, ee.tn27, ee.tx27, ee.qq27, ee.ceresCLT, ee.ceresAOD, ee.lattt, ee.lonnn, input_core_dims=[["time"],["time"],["time"],["time"],["time"],["time"],["time"]], dask = 'allowed', vectorize = True)