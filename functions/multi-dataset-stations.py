import numpy as np
import xarray as xr
import glob

ids = [int(i.split('/')[-1].split('.')[0]) for i in np.array(glob.glob('/data/merged-stations-20010101-20211231/*.nc'))]
# print(ids)
j1 = 120
j2 = 243
print(ids)
years = np.zeros(len(ids))
n = 0

for j in range(len(ids)):
    print(j)
    nri = xr.open_dataset('/data/merged-stations-20010101-20211231/'+str(ids[j])+'.nc')
    if (nri.time[0].dt.month.values != 1) | (nri.time[0].dt.day.values != 1):
        time_start = np.datetime64(str(nri.time[0].dt.year.values+1) + '-01-01').astype('datetime64[s]')
        y1 = nri.time[0].dt.year.values+1
    else:
        time_start = nri.time[0].values
        y1 = int(nri.time[0].dt.year.values)

    if (nri.time[-1].dt.month.values != 12) | (nri.time[-1].dt.day.values != 31):
        time_end = np.datetime64(str(nri.time[-1].dt.year.values-1) + '-12-31').astype('datetime64[s]')
        y2 = nri.time[-1].dt.year.values-1
        
    else:
        time_end = nri.time[-1].values
        y2 = int(nri.time[-1].dt.year.values)


    d = (y2-y1)+1
    stn = nri.sel(time=~((nri.time.dt.month == 2) & (nri.time.dt.day == 29))).sel(time = slice(time_start,time_end))    
    yr_nan = stn.TN.values.reshape(d,365)[:,j1:j2].mean(axis =1)*stn.TX.values.reshape(d,365)[:,j1:j2].mean(axis =1)*stn.QQ.values.reshape(d,365)[:,j1:j2].mean(axis =1)*stn.CC.values.reshape(d,365)[:,j1:j2].mean(axis =1)*stn.ceresAOD.values.reshape(d,365)[:,j1:j2].mean(axis =1)

    years[j] = np.where(~np.isnan(yr_nan))[0].shape[0]
    
    # print(years[j])
    
    if np.where(~np.isnan(yr_nan))[0].shape[0]>9:

        var_names = ['tmin','tmax','sw','cloudiness','aod']
        
        tn_ = stn.TN.values.reshape(d,365)[np.where(~np.isnan(yr_nan))[0],j1:j2]
        tx_ = stn.TX.values.reshape(d,365)[np.where(~np.isnan(yr_nan))[0],j1:j2]
        qq_ = stn.QQ.values.reshape(d,365)[np.where(~np.isnan(yr_nan))[0],j1:j2]
        cc_ = stn.CC.values.reshape(d,365)[np.where(~np.isnan(yr_nan))[0],j1:j2]
        aod_ = stn.ceresAOD.values.reshape(d,365)[np.where(~np.isnan(yr_nan))[0],j1:j2]
        
        if n == 0:
            
            tn = (tn_ - np.mean(tn_))/np.std(tn_)
            tx = (tx_ - np.mean(tx_))/np.std(tx_)
            qq = (qq_ - np.mean(qq_))/np.std(qq_)
            cc = (cc_ - np.mean(cc_))/np.std(cc_)
            aod = (aod_ - np.mean(aod_))/np.std(aod_)

        else:
            
            tn = np.concatenate([tn, (tn_-np.mean(tn_))/np.std(tn_)], axis = 0)
            tx = np.concatenate([tx, (tx_-np.mean(tx_))/np.std(tx_)], axis = 0)
            qq = np.concatenate([qq, (qq_-np.mean(qq_))/np.std(qq_)], axis = 0)
            cc = np.concatenate([cc, (cc_-np.mean(cc_))/np.std(cc_)], axis = 0)
            aod = np.concatenate([aod, (aod_-np.mean(aod_))/np.std(aod_)], axis = 0)
        n = n+1

dataStacked = np.stack([tn,tx,qq,cc,aod])
# print(dataStacked.shape)

# print(years.shape)

np.save('/exports/geos.ed.ac.uk/climate_change/users/carla/results/stations/stacked-01-21-std.npy', dataStacked)
np.save('/exports/geos.ed.ac.uk/climate_change/users/carla/results/stations/years-01-21-std.npy', years)
