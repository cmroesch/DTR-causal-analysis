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

# %matplotlib inline     
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

# %matplotlib inline     
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

def ce(dataframe, X,Y,graph):
    causal_effects = CausalEffects(graph, graph_type='stationary_dag', X=X, Y=Y, S=None, 
                               hidden_variables=None, 
                            verbosity=0)
    causal_effects.fit_wright_effect(dataframe=dataframe, mediation = 'direct')
       
    intervention = [0,1]#get_valid_range(x)
    intervention_data = intervention[1]*np.ones((1, 1))
    y1 = causal_effects.predict_wright_effect( 
            intervention_data=intervention_data
            )

    intervention_data = intervention[0]*np.ones((1, 1))
    y2 = causal_effects.predict_wright_effect( 
            intervention_data=intervention_data
            )

    beta = (y1 - y2)#/(intervention[1]-intervention[0])
    
    causal_effects.fit_bootstrap_of(
    method='fit_wright_effect',
    method_args={'dataframe':dataframe, 'mediation':'direct'},
    boot_samples=50)

    intervention_data = 1.*np.ones((1, 1))
    conf = causal_effects.predict_bootstrap_of(
        method='predict_wright_effect',
        method_args={'intervention_data':intervention_data})

    return beta, conf

def ce_estimation_wright(dataset,filepath,filenameresult,filenameci,plotname,graph):
    
    dataframe = DataFrame(dataset, analysis_mode = 'multiple', var_names = ['tmin','tmax','sw','cloudiness','aod'])
    
    results =  np.array([[[0.0,0.0], #tmin
                        [0.0,0.0],
                        [0.0,0.0],
                        [0.0,0.0],
                        [0.0,0.0]],
                       [[0.0,0.0], #tmax
                        [0.0,0.0],
                        [0.0,0.0],
                        [0.0,0.0],
                        [0.0,0.0]],
                       [[0.0,0.0], #SW
                        [0.0,0.0],
                        [0.0,0.0],
                        [0.0,0.0],
                        [0.0,0.0]],
                       [[0.0,0.0], # CC
                        [0.0,0.0],
                        [0.0,0.0],
                        [0.0,0.0],
                        [0.0,0.0]],
                       [[0.0,0.0], #AOD
                        [0.0,0.0],
                        [0.0,0.0],
                        [0.0,0.0],
                        [0.0,0.0]]])
    
    
    ci =  np.array([[[[0.0,0.0],[0.0,0.0]], #tmin
                        [[0.0,0.0],[0.0,0.0]],
                        [[0.0,0.0],[0.0,0.0]],
                        [[0.0,0.0],[0.0,0.0]],
                        [[0.0,0.0],[0.0,0.0]]],
                       [[[0.0,0.0],[0.0,0.0]], #tmax
                        [[0.0,0.0],[0.0,0.0]],
                        [[0.0,0.0],[0.0,0.0]],
                        [[0.0,0.0],[0.0,0.0]],
                        [[0.0,0.0],[0.0,0.0]]],
                       [[[0.0,0.0],[0.0,0.0]], #SW
                        [[0.0,0.0],[0.0,0.0]],
                        [[0.0,0.0],[0.0,0.0]],
                        [[0.0,0.0],[0.0,0.0]],
                        [[0.0,0.0],[0.0,0.0]]],
                       [[[0.0,0.0],[0.0,0.0]], # CC
                        [[0.0,0.0],[0.0,0.0]],
                        [[0.0,0.0],[0.0,0.0]],
                        [[0.0,0.0],[0.0,0.0]],
                        [[0.0,0.0],[0.0,0.0]]],
                       [[[0.0,0.0],[0.0,0.0]], #AOD
                        [[0.0,0.0],[0.0,0.0]],
                        [[0.0,0.0],[0.0,0.0]],
                        [[0.0,0.0],[0.0,0.0]],
                        [[0.0,0.0],[0.0,0.0]]]])

    X = [(0,-1)]
    Y = [(0,0)]
    r = ce(dataframe, X,Y,graph)
    print('Tmin: ' + str(r[0]))
    results[0][0,1] = r[0][0][0]
    ci[0][0,1,:] = r[1][:,0]
    if np.unique(np.sign(r[1][:,0])).shape[0] > 1: graph[0][0,1] = ''
    
    X = [(1,-1)]
    Y = [(1,0)]
    r = ce(dataframe, X,Y,graph)
    print('Tmax: ' + str(r[0]))
    results[1][1,1] = r[0][0][0]
    ci[1][1,1,:] = r[1][:,0]
    if np.unique(np.sign(r[1][:,0])).shape[0] > 1: graph[1][1,1] = ''
    
    X = [(2,-1)]
    Y = [(2,0)]
    r = ce(dataframe, X,Y,graph)
    print('SW: ' + str(r[0]))
    results[2][2,1] = r[0][0][0]
    ci[2][2,1,:] = r[1][:,0]
    if np.unique(np.sign(r[1][:,0])).shape[0] > 1: graph[2][2,1] = ''
    
    X = [(3,-1)]
    Y = [(3,0)]
    r = ce(dataframe, X,Y,graph)
    print('CC: ' + str(r[0]))
    results[3][3,1] = r[0][0][0]
    ci[3][3,1,:] = r[1][:,0]
    if np.unique(np.sign(r[1][:,0])).shape[0] > 1: graph[3][3,1] = ''
    
    X = [(4,-1)]
    Y = [(4,0)]
    r = ce(dataframe, X,Y,graph)
    print('AOD: ' + str(r[0]))
    results[4][4,1] = r[0][0][0]
    ci[4][4,1,:] = r[1][:,0]
    if np.unique(np.sign(r[1][:,0])).shape[0] > 1: graph[4][4,1] = ''
    
    X = [(4,0)]
    Y = [(0,0)]
    r = ce(dataframe, X,Y,graph)
    print('AOD-Tmin: ' + str(r[0]))
    results[4][0,0] = r[0][0][0]
    results[0][4,0] = r[0][0][0]
    ci[4][0,0,:] = r[1][:,0]
    ci[0][4,0,:] = r[1][:,0]
    if np.unique(np.sign(r[1][:,0])).shape[0] > 1: 
        graph[4][0,0] = ''
        graph[0][4,0] = ''

    X = [(4,0)]
    Y = [(3,0)]
    r = ce(dataframe, X,Y,graph)
    print('AOD-CC: ' + str(r[0]))
    results[4][3,0] = r[0][0][0]
    results[3][4,0] = r[0][0][0]
    ci[4][3,0,:] = r[1][:,0]
    ci[3][4,0,:] = r[1][:,0]
    if np.unique(np.sign(r[1][:,0])).shape[0] > 1: 
        graph[4][3,0] = ''
        graph[3][4,0] = ''


    X = [(4,0)]
    Y = [(2,0)]
    r = ce(dataframe, X,Y,graph)
    print('AOD-SW: ' + str(r[0]))
    results[4][2,0] = r[0][0][0]
    results[2][4,0] = r[0][0][0]
    ci[4][2,0,:] = r[1][:,0]
    ci[2][4,0,:] = r[1][:,0]
    if np.unique(np.sign(r[1][:,0])).shape[0] > 1: 
        graph[4][2,0] = ''
        graph[2][4,0] = ''

    X = [(3,0)]
    Y = [(2,0)]
    r = ce(dataframe, X,Y,graph)
    print('CC-SW: ' + str(r[0]))
    results[3][2,0] = r[0][0][0]
    results[2][3,0] = r[0][0][0]
    ci[3][2,0,:] = r[1][:,0]
    ci[2][3,0,:] = r[1][:,0]
    if np.unique(np.sign(r[1][:,0])).shape[0] > 1: 
        graph[3][2,0] = ''
        graph[2][3,0] = ''

    X = [(3,0)]
    Y = [(1,0)]
    r = ce(dataframe, X,Y,graph)
    print('CC-Tmax: ' + str(r[0]))
    results[3][1,0] = r[0][0][0]
    results[1][3,0] = r[0][0][0]
    ci[3][1,0,:] = r[1][:,0]
    ci[1][3,0,:] = r[1][:,0]
    if np.unique(np.sign(r[1][:,0])).shape[0] > 1: 
        graph[3][1,0] = ''
        graph[1][3,0] = ''

    X = [(3,0)]
    Y = [(0,0)]
    r = ce(dataframe, X,Y,graph)
    print('CC-Tmin: ' + str(r[0]))
    results[3][0,0] = r[0][0][0]
    results[0][3,0] = r[0][0][0]
    ci[3][0,0,:] = r[1][:,0]
    ci[0][3,0,:] = r[1][:,0]
    if np.unique(np.sign(r[1][:,0])).shape[0] > 1: 
        graph[3][0,0] = ''
        graph[0][3,0] = ''

    X = [(2,0)]
    Y = [(1,0)]
    r = ce(dataframe, X,Y,graph)
    print('SW-Tmax: ' + str(r[0]))
    results[2][1,0] = r[0][0][0]
    results[1][2,0] = r[0][0][0]
    ci[2][1,0,:] = r[1][:,0]
    ci[1][2,0,:] = r[1][:,0]
    if np.unique(np.sign(r[1][:,0])).shape[0] > 1: 
        graph[2][1,0] = ''
        graph[1][2,0] = ''

    X = [(1,0)]
    Y = [(0,0)]
    r = ce(dataframe, X,Y,graph)
    print('Tmax-Tmin: ' + str(r[0]))
    results[1][0,0] = r[0][0][0]
    results[0][1,0] = r[0][0][0]
    ci[1][0,0,:] = r[1][:,0]
    ci[0][1,0,:] = r[1][:,0]
    if np.unique(np.sign(r[1][:,0])).shape[0] > 1: 
        graph[1][0,0] = ''
        graph[0][1,0] = ''

    X = [(0,-1)]
    Y = [(1,0)]
    r = ce(dataframe, X,Y,graph)
    print('Tmin-Tmax: ' + str(r[0]))
    results[0][1,1] = r[0][0][0]
    results[1][0,1] = r[0][0][0]
    ci[0][1,1,:] = r[1][:,0]
    ci[1][0,1,:] = r[1][:,0]
    if np.unique(np.sign(r[1][:,0])).shape[0] > 1: 
        graph[0][1,1] = ''
        graph[1][0,1] = ''

    X = [(3,-1)]
    Y = [(4,0)]
    r = ce(dataframe, X,Y,graph)
    print('CC-AOD: ' + str(r[0]))
    results[3][4,1] = r[0][0][0]
    results[4][3,1] = r[0][0][0]
    ci[3][4,1,:] = r[1][:,0]
    ci[4][3,1,:] = r[1][:,0]
    if np.unique(np.sign(r[1][:,0])).shape[0] > 1: 
        graph[3][4,1] = ''
        graph[4][3,1] = ''
    
    np.save(filepath+filenameresult, results)
    np.save(filepath+filenameci, ci)
    
    tp.plot_graph(graph = graph,val_matrix = results,var_names=var_names, figsize = (8,8),link_colorbar_label='Norm. Direct Causal Effect', node_colorbar_label='Autocorrelation',vmin_edges=-0.5,
    vmax_edges=0.5); 
    plt.savefig(filepath+plotname)
    return results, ci

stn = pd.read_csv('/home/s2135337/carla/ecad/stations.csv')
ids = [int(i.split('/')[-1].split('.')[0]) for i in np.array(glob.glob('/exports/geos.ed.ac.uk/climate_change/users/carla/ecad/merged-stations-20010101-20211231/*.nc'))]
years = np.load('/home/s2135337/carla/results/stations/years-01-21.npy')


stn['id_active'] = 'no'
for j in range(len(ids)):
        if years[j] > 9:
            k = np.where(stn.id == int(ids[j]))[0][0]
            stn.loc[k,'id_active'] = 'yes'
            
def get_nearest_grid(data, lat, lon):
    abslat = np.abs(data.lat-lat)
    abslon = np.abs(data.lon-lon)
    c = np.maximum(abslon, abslat)

    d = np.where(c == np.min(c))
    
    if d[0].shape[0] > 1:
        xloc = d[0][0]
        yloc = d[1][0]
    else:
        ([xloc], [yloc]) = d

    point_ds = data.sel(lon=data.lon[xloc], lat=data.lat[yloc])
    
    return data.lon.values[xloc], data.lat.values[yloc]


stnf = stn[stn['id_active'] == 'yes']
stnf['mlat'] = np.nan
stnf['mlon'] = np.nan

canlsm = xr.open_dataset('/home/s2135337/carla/cmip6/sftlf_fx_CanESM5_historical_r1i1p1f1_gn.nc')
mask = canlsm.sftlf

clt = xr.open_dataset('/home/s2135337/carla/cmip6/clt_day_CanESM5_r1_18500101-21001231.nc').sel(time = slice('1950-01-01','2014-12-31')).where(mask).clt/100
rsds = xr.open_dataset('//home/s2135337/carla/cmip6/rsds_day_CanESM5_r1_18500101-21001231.nc').sel(time = slice('1950-01-01','2014-12-31')).where(mask).rsds
tasmin = xr.open_dataset('/home/s2135337/carla/cmip6/tasmin_day_CanESM5_r1_18500101-21001231.nc').sel(time = slice('1950-01-01','2014-12-31')).where(mask).tasmin - 273.15
tasmax = xr.open_dataset('//home/s2135337/carla/cmip6/tasmax_day_CanESM5_r1_18500101-21001231.nc').sel(time = slice('1950-01-01','2014-12-31')).where(mask).tasmax - 273.15
aod = xr.open_dataset('/home/s2135337/carla/cmip6/od550aer_day_CanESM5_r1_18500101-20141231.nc').sel(time = slice('1950-01-01','2014-12-31')).where(mask).od550aer

clt = xr.concat([clt.sel(lon = slice(360-180,360), lat = slice(10,80)),clt.sel(lon = slice(0,179), lat = slice(10,80))], dim = 'lon')
rsds = xr.concat([rsds.sel(lon = slice(360-180,360), lat = slice(10,80)),rsds.sel(lon = slice(0,179), lat = slice(10,80))], dim = 'lon')#.where(~np.isnan(mask).transpose(), np.nan)
tasmin = xr.concat([tasmin.sel(lon = slice(360-180,360), lat = slice(10,80)),tasmin.sel(lon = slice(0,179), lat = slice(10,80))], dim = 'lon')#.where(~np.isnan(mask).transpose(), np.nan)
tasmax = xr.concat([tasmax.sel(lon = slice(360-180,360), lat = slice(10,80)),tasmax.sel(lon = slice(0,179), lat = slice(10,80))], dim = 'lon')#.where(~np.isnan(mask).transpose(), np.nan)
aod = xr.concat([aod.sel(lon = slice(360-180,360), lat = slice(10,80)),aod.sel(lon = slice(0,179), lat = slice(10,80))], dim = 'lon')#.where(~np.isnan(mask).transpose(), np.nan)

c = clt.sel(time=~((clt.time.dt.month == 2) & (clt.time.dt.day == 29))).values
r = rsds.sel(time=~((rsds.time.dt.month == 2) & (rsds.time.dt.day == 29))).values
tx = tasmax.sel(time=~((tasmax.time.dt.month == 2) & (tasmax.time.dt.day == 29))).values
tn = tasmin.sel(time=~((tasmin.time.dt.month == 2) & (tasmin.time.dt.day == 29))).values
a = aod.sel(time=~((aod.time.dt.month == 2) & (aod.time.dt.day == 29))).values

cltc = xr.open_dataset('/home/s2135337/carla/cmip6/clt_day_CanESM5_r1_18500101-21001231.nc').sel(time = slice('2001-01-01','2014-12-31')).clt/100
ccc = xr.concat([cltc.sel(lon = slice(180,360), lat = slice(20,80)),cltc.sel(lon = slice(0,179), lat = slice(20,80))], dim = 'lon')

for i in range(186):
    
    lat = stnf.iloc[i,3]
    lon = stnf.iloc[i,4]
    if lon < 0:
        lon = 360 + lon
    stnf.iloc[i,8], stnf.iloc[i,7] = get_nearest_grid(ccc, lat, lon)
    
# latlon = list(set([str(stnf['mlat'].values[i])+' '+str(stnf['mlon'].values[i]) for i in range(186)]))
latlon = list([str(stnf['mlat'].values[i])+' '+str(stnf['mlon'].values[i]) for i in range(186)])

ll = [f.split(' ') for f in latlon]

print(len(ll))

d = 14
j1 = 120
j2 = 243
l = 0

var_names = [r'$Tmin$', r'$Tmax$', r'$SW$', r'$CC$', r'$AOD$']

graph =  np.array([[['', '-->'], #tmin
                    ['<--', '-->'],
                    ['', ''],
                    ['<--', ''],
                    ['<--', '']],
                   [['-->', ''], #tmax
                    ['', '-->'],
                    ['<--', ''],
                    ['<--', ''],
                    ['', '']],
                   [['', ''], #SW
                    ['-->', ''],
                    ['', '-->'],
                    ['<--', ''],
                    ['<--', '']],
                   [['-->', ''], # CC
                    ['-->', ''],
                    ['-->', ''],
                    ['', '-->'],
                    ['<--', '']],
                   [['-->', ''], #AOD
                    ['', ''],
                    ['-->', ''],
                    ['-->', ''],
                    ['', '-->']]], dtype='<U3')

# time_slices = [['1961-01-01','1974-12-31'],['2001-01-01','2014-12-31']]

# for t in range(2):
#     l = 0
#     for j in range(len(ll)):
#         # print(ll[j])
#         cs = clt.sel(time=~((clt.time.dt.month == 2) & (clt.time.dt.day == 29)), lat = ll[j][0], lon = ll[j][1]).sel(time = slice(time_slices[t][0],time_slices[t][1])).values
#         rs = rsds.sel(time=~((rsds.time.dt.month == 2) & (rsds.time.dt.day == 29)), lat = ll[j][0], lon = ll[j][1]).sel(time = slice(time_slices[t][0],time_slices[t][1])).values
#         txs = tasmax.sel(time=~((tasmax.time.dt.month == 2) & (tasmax.time.dt.day == 29)), lat = ll[j][0], lon = ll[j][1]).sel(time = slice(time_slices[t][0],time_slices[t][1])).values
#         tns = tasmin.sel(time=~((tasmin.time.dt.month == 2) & (tasmin.time.dt.day == 29)), lat = ll[j][0], lon = ll[j][1]).sel(time = slice(time_slices[t][0],time_slices[t][1])).values
#         ass = aod.sel(time=~((aod.time.dt.month == 2) & (aod.time.dt.day == 29)), lat = ll[j][0], lon = ll[j][1]).sel(time = slice(time_slices[t][0],time_slices[t][1])).values

#         var_names = ['tmin','tmax','sw','cloudiness','aod']
#         tn_ = tns.reshape(d,365)[:,j1:j2] - np.mean(tns.reshape(d,365)[:,j1:j2])
#         tx_ = txs.reshape(d,365)[:,j1:j2] - np.mean(txs.reshape(d,365)[:,j1:j2])
#         qq_ = rs.reshape(d,365)[:,j1:j2] - np.mean(rs.reshape(d,365)[:,j1:j2])
#         cc_ = cs.reshape(d,365)[:,j1:j2] - np.mean(cs.reshape(d,365)[:,j1:j2])
#         aod_ = ass.reshape(d,365)[:,j1:j2] - np.mean(ass.reshape(d,365)[:,j1:j2])

#         if l == 0:
#             tNs = tn_
#             tXs = tx_
#             qqs = qq_
#             ccs = cc_
#             aods = aod_

#         else:
#             tNs = np.concatenate([tNs,tn_], axis = 0)
#             tXs = np.concatenate([tXs,tx_], axis = 0)
#             qqs = np.concatenate([qqs,qq_], axis = 0)
#             ccs = np.concatenate([ccs,cc_], axis = 0)
#             aods = np.concatenate([aods,aod_], axis = 0)

#         l = l+1

#     dataStackeds = np.stack([tNs,tXs,qqs,ccs,aods])
    
#     if t == 1:
#         np.save('/home/s2135337/scratch/CI-DTR/CanESM5/stations/stackeddata-2001-2014.npy',dataStackeds)
#     elif t == 0:
#         np.save('/home/s2135337/scratch/CI-DTR/CanESM5/stations/stackeddata-1961-1974.npy',dataStackeds)

# l = 0
# d = 10
# ps = ['50-60','55-64','60-70','65-74','70-80','75-84','80-90','85-94','90-00','95-04','00-10','05-14']
# time_slices = [['1950-01-01','1959-12-31'],['1955-01-01','1964-12-31'],['1960-01-01','1969-12-31'],['1965-01-01','1974-12-31'],['1970-01-01','1979-12-31'],['1975-01-01','1984-12-31'],['1980-01-01','1989-12-31'],['1985-01-01','1994-12-31'],['1990-01-01','1999-12-31'],['1995-01-01','2004-12-31'],['2000-01-01','2009-12-31'],['2005-01-01','2014-12-31']] 
# for t in range(12):
#     l = 0
#     for j in range(len(ll)):
#         # print(ll[j])
#         cs = clt.sel(time=~((clt.time.dt.month == 2) & (clt.time.dt.day == 29)), lat = ll[j][0], lon = ll[j][1]).sel(time = slice(time_slices[t][0],time_slices[t][1])).values
#         rs = rsds.sel(time=~((rsds.time.dt.month == 2) & (rsds.time.dt.day == 29)), lat = ll[j][0], lon = ll[j][1]).sel(time = slice(time_slices[t][0],time_slices[t][1])).values
#         txs = tasmax.sel(time=~((tasmax.time.dt.month == 2) & (tasmax.time.dt.day == 29)), lat = ll[j][0], lon = ll[j][1]).sel(time = slice(time_slices[t][0],time_slices[t][1])).values
#         tns = tasmin.sel(time=~((tasmin.time.dt.month == 2) & (tasmin.time.dt.day == 29)), lat = ll[j][0], lon = ll[j][1]).sel(time = slice(time_slices[t][0],time_slices[t][1])).values
#         ass = aod.sel(time=~((aod.time.dt.month == 2) & (aod.time.dt.day == 29)), lat = ll[j][0], lon = ll[j][1]).sel(time = slice(time_slices[t][0],time_slices[t][1])).values

#         var_names = ['tmin','tmax','sw','cloudiness','aod']
#         tn_ = tns.reshape(d,365)[:,j1:j2] - np.mean(tns.reshape(d,365)[:,j1:j2])
#         tx_ = txs.reshape(d,365)[:,j1:j2] - np.mean(txs.reshape(d,365)[:,j1:j2])
#         qq_ = rs.reshape(d,365)[:,j1:j2] - np.mean(rs.reshape(d,365)[:,j1:j2])
#         cc_ = cs.reshape(d,365)[:,j1:j2] - np.mean(cs.reshape(d,365)[:,j1:j2])
#         aod_ = ass.reshape(d,365)[:,j1:j2] - np.mean(ass.reshape(d,365)[:,j1:j2])

#         if l == 0:
#             tNs = tn_
#             tXs = tx_
#             qqs = qq_
#             ccs = cc_
#             aods = aod_

#         else:
#             tNs = np.concatenate([tNs,tn_], axis = 0)
#             tXs = np.concatenate([tXs,tx_], axis = 0)
#             qqs = np.concatenate([qqs,qq_], axis = 0)
#             ccs = np.concatenate([ccs,cc_], axis = 0)
#             aods = np.concatenate([aods,aod_], axis = 0)

#         l = l+1

#     dataStackeds = np.stack([tNs,tXs,qqs,ccs,aods])
    
#     np.save('/home/s2135337/scratch/CI-DTR/CanESM5/stations/stackeddata-'+ps[t]+'.npy',dataStackeds)        

print('starting')
        
time_slice = ['2001-01-01','2014-12-31']
l = 0
for j in ccc.lon.values[np.concatenate([np.where(ccc.lon.values >= 330)[0],np.where(ccc.lon.values <= 30)[0]])]:
    for i in ccc.lat.values:
        # print(ll[j])
        cs = clt.sel(time=~((clt.time.dt.month == 2) & (clt.time.dt.day == 29)), lat = i, lon = j).sel(time = slice(time_slice[0],time_slice[1])).values
        rs = rsds.sel(time=~((rsds.time.dt.month == 2) & (rsds.time.dt.day == 29)), lat = i, lon = j).sel(time = slice(time_slice[0],time_slice[1])).values
        txs = tasmax.sel(time=~((tasmax.time.dt.month == 2) & (tasmax.time.dt.day == 29)), lat = i, lon = j).sel(time = slice(time_slice[0],time_slice[1])).values
        tns = tasmin.sel(time=~((tasmin.time.dt.month == 2) & (tasmin.time.dt.day == 29)), lat = i, lon = j).sel(time = slice(time_slice[0],time_slice[1])).values
        ass = aod.sel(time=~((aod.time.dt.month == 2) & (aod.time.dt.day == 29)),lat = i, lon = j).sel(time = slice(time_slice[0],time_slice[1])).values
        
        if not any(np.isnan(cs)):
            print('yes')
        
            var_names = ['tmin','tmax','sw','cloudiness','aod']
            tn_ = tns.reshape(d,365)[:,j1:j2] - np.mean(tns.reshape(d,365)[:,j1:j2])
            tx_ = txs.reshape(d,365)[:,j1:j2] - np.mean(txs.reshape(d,365)[:,j1:j2])
            qq_ = rs.reshape(d,365)[:,j1:j2] - np.mean(rs.reshape(d,365)[:,j1:j2])
            cc_ = cs.reshape(d,365)[:,j1:j2] - np.mean(cs.reshape(d,365)[:,j1:j2])
            aod_ = ass.reshape(d,365)[:,j1:j2] - np.mean(ass.reshape(d,365)[:,j1:j2])

            dataStacked = np.stack([tn_, tx_, qq_, cc_, aod_])
            data = {i: dataStacked[:,i,:].transpose() for i in range(d)}

            # ce_estimation_wright(data, '/home/s2135337/carla/results/cmip6/canesm5/stations/','results/'+str(i)+'-'+str(j)+'.npy','ci/'+str(i)+'-'+str(j)+'.npy','graphs/'+str(i)+'-'+str(j)+'.png', graph)

            if l == 0:
                tNs = tn_
                tXs = tx_
                qqs = qq_
                ccs = cc_
                aods = aod_

            else:
                tNs = np.concatenate([tNs,tn_], axis = 0)
                tXs = np.concatenate([tXs,tx_], axis = 0)
                qqs = np.concatenate([qqs,qq_], axis = 0)
                ccs = np.concatenate([ccs,cc_], axis = 0)
                aods = np.concatenate([aods,aod_], axis = 0)

            l = l+1
            
dataStackeds = np.stack([tNs,tXs,qqs,ccs,aods])
np.save('/home/s2135337/carla/results/cmip6/canesm5/stations/stackeddata-2001-2014.npy',dataStackeds)
# dimss = dataStackeds.shape
# print(dimss)
# ds = {i: dataStackeds[:,i,:].transpose() for i in range(dimss[1])}
# dataframes = DataFrame(ds, analysis_mode = 'multiple', var_names = ['tmin','tmax','sw','cloudiness','aod'])
# resultss = run_and_plot(dataframes, ParCorr())
# plt.savefig('//home/s2135337/carla/results/cmip6/canesm5/stations/europe-stations-nolag.png')
# np.save('/home/s2135337/carla/results/cmip6/canesm5/stations/europe-stations-nolag.npy',np.array(list(resultss.items()))[:,1])


# def run_and_plot(dataframe, cond_ind_test):
    
#     var_names = ['tmin','tmax','sw','cloudiness','aod']
#     # Create and run Multidata-PCMCIplus
#     pcmci = PCMCI(dataframe = dataframe, cond_ind_test = cond_ind_test)
#     link_assumptions = {j:{(i, -tau):'o-o' for i in range(5) for tau in range(1) if (i, -tau) != (j, 0)} for j in range(5)}
#     link_assumptions[0] = {(i, -tau):'o-o' if i == 1 else '-->' for i in range(5) for tau in range(1) if (i, -tau) != (0, 0)}
#     link_assumptions[1] = {(i, -tau):'o-o' if i == 0 else '-->' for i in range(5) for tau in range(1) if (i, -tau) != (1, 0)}
#     link_assumptions[2] = {(i, -tau):'<--' if i in [0,1] else '-->' for i in range(5) for tau in range(1) if (i, -tau) != (2, 0)}
#     link_assumptions[3] = {(i, -tau):'<--' if i in [0,1,2] else '-->' for i in range(5) for tau in range(1) if (i, -tau) != (3, 0)}
#     link_assumptions[4] = {(i, -tau):'<--' if i in [0,1,2,3] else '-->' for i in range(5) for tau in range(1) if (i, -tau) != (4, 0)}
    
#     link_assumptions[0][(1, -1)] = 'o-o'
#     # link_assumptions[1][(0, -1)] = 'o-o'
#     # link_assumptions[4] = {(i, -tau):'-->' for i in range(5) for tau in range(1 + 1) 
#     #                        if ((i, -tau) != (2, 0))} 
#     link_assumptions[0][(0, -1)] = 'o-o'
#     link_assumptions[1][(1, -1)] = 'o-o'
#     link_assumptions[2][(2, -1)] = 'o-o'
#     link_assumptions[3][(3, -1)] = 'o-o'
#     link_assumptions[4][(4, -1)] = 'o-o'

#     results = pcmci.run_pcmciplus(tau_min=0, tau_max=1, pc_alpha=0.01, link_assumptions = link_assumptions)
    
#     tp.plot_graph(graph=results['graph'], val_matrix=results['val_matrix'], var_names=var_names)
    
    
#     return results
