import xarray as xr
import numpy as np

def get_nearest_grid(ceres, lat, lon):
    abslat = np.abs(ceres.lat-lat)
    abslon = np.abs(ceres.lon-lon)
    c = np.maximum(abslon, abslat)

    d = np.where(c == np.min(c))
    
    if d[0].shape[0] > 1:
        xloc = d[0][0]
        yloc = d[1][0]
    else:
        ([xloc], [yloc]) = d

    point_ds = ceres.sel(lon=ceres.lon[xloc], lat=ceres.lat[yloc])
    
    return point_ds

j1 = 120
j2 = 243
time_slice = ['2001-01-01','2021-12-31']
lat_slice = [35,70]
lon_slice = [-30,40]

EOBStmin27 = xr.open_dataset('/.../eobs/tn_ens_mean_0.25deg_reg_v27.0e.nc').rename({'longitude':'lon','latitude':'lat'}).sel(
    time = slice(time_slice[0], time_slice[1]), lat = slice(lat_slice[0],lat_slice[1]),lon = slice(lon_slice[0],lon_slice[1]))
EOBStmax27 = xr.open_dataset('/.../eobs/tx_ens_mean_0.25deg_reg_v27.0e.nc').rename({'longitude':'lon','latitude':'lat'}).sel(
    time = slice(time_slice[0], time_slice[1]), lat = slice(lat_slice[0],lat_slice[1]),lon = slice(lon_slice[0],lon_slice[1]))
EOBSqq27 = xr.open_dataset('/.../eobs/qq_ens_mean_0.25deg_reg_v27.0e.nc').rename({'longitude':'lon','latitude':'lat'}).sel(
    time = slice(time_slice[0], time_slice[1]), lat = slice(lat_slice[0],lat_slice[1]),lon = slice(lon_slice[0],lon_slice[1]))


EOBStmin26 = xr.open_dataset('/.../eobs/tn_ens_mean_0.25deg_reg_v26.0e.nc').rename({'longitude':'lon','latitude':'lat'}).sel(
    time = slice(time_slice[0], time_slice[1]), lat = slice(lat_slice[0],lat_slice[1]),lon = slice(lon_slice[0],lon_slice[1]))
EOBStmax26 = xr.open_dataset('/.../eobs/tx_ens_mean_0.25deg_reg_v26.0e.nc').rename({'longitude':'lon','latitude':'lat'}).sel(
    time = slice(time_slice[0], time_slice[1]), lat = slice(lat_slice[0],lat_slice[1]),lon = slice(lon_slice[0],lon_slice[1]))
EOBSqq26 = xr.open_dataset('/.../eobs/qq_ens_mean_0.25deg_reg_v26.0e.nc').rename({'longitude':'lon','latitude':'lat'}).sel(
    time = slice(time_slice[0], time_slice[1]), lat = slice(lat_slice[0],lat_slice[1]),lon = slice(lon_slice[0],lon_slice[1]))


ceres= xr.open_dataset('/.../ceres/CERES_SYN1deg-Day_Terra-Aqua-MODIS_Ed4.1_Subset_20000301-20221231.nc')
ceresAOD = ceres.ini_aod55_daily.sel(time = slice(time_slice[0], time_slice[1]))
ceresCLT = ceres.cldarea_total_daily.sel(time = slice(time_slice[0], time_slice[1]))

eobs_all = EOBStmin27
eobs_all['tx27'] = (['time','lat', 'lon'],  EOBStmax27.tx.values)
eobs_all['qq27'] = (['time','lat', 'lon'],  EOBSqq27.qq.values)
eobs_all['tx26'] = (['time','lat', 'lon'],  EOBStmax26.tx.values)
eobs_all['tn26'] = (['time','lat', 'lon'],  EOBStmin26.tn.values)
eobs_all['qq26'] = (['time','lat', 'lon'],  EOBSqq26.qq.values)
eobs_all = eobs_all.rename_vars({'tn':'tn27'})

ceresaod = np.zeros((7670,140,280))*np.nan
ceresclt = np.zeros((7670,140,280))*np.nan

for i in range(140):
    print(i)
    for j in range(280):
        ceresaod[:,i,j] = get_nearest_grid(ceresAOD, EOBStmin26.lat.values[i], EOBStmin26.lon.values[j])
        ceresclt[:,i,j] = get_nearest_grid(ceresCLT, EOBStmin26.lat.values[i], EOBStmin26.lon.values[j])

eobs_all['ceresAOD'] = (['time','lat', 'lon'],  ceresaod)
eobs_all['ceresCLT'] = (['time','lat', 'lon'],  ceresclt)

eobs_all.to_netcdf('/data/ceres-clara-merra-eobs27-eobs26-merged-20010101-20211231.nc')