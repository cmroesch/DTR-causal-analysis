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

canlsm = xr.open_dataset('/home/s2135337/carla/cmip6/sftlf_fx_HadGEM3-GC31-LL_hist-1950_r1i1p1f1_gn.nc')
mask = canlsm.sftlf

clt = xr.open_dataset('/home/s2135337/carla/cmip6/clt_day_HadGEM3-GC31-LL_r1_19500101-20491230.nc').sel(time = slice('1950-01-01','2014-12-30')).where(mask).clt/100
rsds = xr.open_dataset('//home/s2135337/carla/cmip6/rsds_day_HadGEM3-GC31-LL_r1_19500101-20491230.nc').sel(time = slice('1950-01-01','2014-12-30')).where(mask).rsds
tasmin = xr.open_dataset('/home/s2135337/carla/cmip6/tasmin_day_HadGEM3-GC31-LL_r1_19500101-20491230.nc').sel(time = slice('1950-01-01','2014-12-30')).where(mask).tasmin - 273.15
tasmax = xr.open_dataset('//home/s2135337/carla/cmip6/tasmax_day_HadGEM3-GC31-LL_r1_19500101-20491230.nc').sel(time = slice('1950-01-01','2014-12-30')).where(mask).tasmax - 273.15
aod = xr.open_dataset('/home/s2135337/carla/cmip6/od550aer_day_HadGEM3-GC31-LL_r1_19500101-20141230.nc').sel(time = slice('1950-01-01','2014-12-30')).where(mask).od550aer

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

cltc = xr.open_dataset('/home/s2135337/carla/cmip6/clt_day_HadGEM3-GC31-LL_r1_19500101-20491230.nc').sel(time = slice('2001-01-01','2014-12-30')).clt/100
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
        
print('start')
        
time_slice = ['2001-01-01','2014-12-30']
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
            
#             print(ass.shape)
#             print(tn.shape)
        
            var_names = ['tmin','tmax','sw','cloudiness','aod']
            tn_ = tns.reshape(d,359)[:,j1:j2] - np.mean(tns.reshape(d,359)[:,j1:j2])
            tx_ = txs.reshape(d,359)[:,j1:j2] - np.mean(txs.reshape(d,359)[:,j1:j2])
            qq_ = rs.reshape(d,359)[:,j1:j2] - np.mean(rs.reshape(d,359)[:,j1:j2])
            cc_ = cs.reshape(d,359)[:,j1:j2] - np.mean(cs.reshape(d,359)[:,j1:j2])
            aod_ = ass.reshape(d,359)[:,j1:j2] - np.mean(ass.reshape(d,359)[:,j1:j2])

            dataStacked = np.stack([tn_, tx_, qq_, cc_, aod_])
            data = {i: dataStacked[:,i,:].transpose() for i in range(d)}

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
print(l)           
dataStackeds = np.stack([tNs,tXs,qqs,ccs,aods])
np.save('/home/s2135337/carla/results/cmip6/hadgem3/stations/stackeddata-2001-2014.npy',dataStackeds)      

