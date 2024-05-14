import xarray as xr
import numpy as np

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
    
    return point_ds

j1 = 120
j2 = 243
time_slice = ['1982-01-01','2021-12-31']
lat_slice = [35,70]
lon_slice = [-30,40]

EOBStmin27 = xr.open_dataset('/exports/geos.ed.ac.uk/climate_change/users/carla/eobs/tn_ens_mean_0.25deg_reg_v27.0e.nc').rename({'longitude':'lon','latitude':'lat'}).sel(
    time = slice(time_slice[0], time_slice[1]), lat = slice(lat_slice[0],lat_slice[1]),lon = slice(lon_slice[0],lon_slice[1]))
EOBStmax27 = xr.open_dataset('/exports/geos.ed.ac.uk/climate_change/users/carla/eobs/tx_ens_mean_0.25deg_reg_v27.0e.nc').rename({'longitude':'lon','latitude':'lat'}).sel(
    time = slice(time_slice[0], time_slice[1]), lat = slice(lat_slice[0],lat_slice[1]),lon = slice(lon_slice[0],lon_slice[1]))
EOBSqq27 = xr.open_dataset('/exports/geos.ed.ac.uk/climate_change/users/carla/eobs/qq_ens_mean_0.25deg_reg_v27.0e.nc').rename({'longitude':'lon','latitude':'lat'}).sel(
    time = slice(time_slice[0], time_slice[1]), lat = slice(lat_slice[0],lat_slice[1]),lon = slice(lon_slice[0],lon_slice[1]))


EOBStmin26 = xr.open_dataset('/exports/geos.ed.ac.uk/climate_change/users/carla/eobs/tn_ens_mean_0.25deg_reg_v26.0e.nc').rename({'longitude':'lon','latitude':'lat'}).sel(
    time = slice(time_slice[0], time_slice[1]), lat = slice(lat_slice[0],lat_slice[1]),lon = slice(lon_slice[0],lon_slice[1]))
EOBStmax26 = xr.open_dataset('/exports/geos.ed.ac.uk/climate_change/users/carla/eobs/tx_ens_mean_0.25deg_reg_v26.0e.nc').rename({'longitude':'lon','latitude':'lat'}).sel(
    time = slice(time_slice[0], time_slice[1]), lat = slice(lat_slice[0],lat_slice[1]),lon = slice(lon_slice[0],lon_slice[1]))
EOBSqq26 = xr.open_dataset('/exports/geos.ed.ac.uk/climate_change/users/carla/eobs/qq_ens_mean_0.25deg_reg_v26.0e.nc').rename({'longitude':'lon','latitude':'lat'}).sel(
    time = slice(time_slice[0], time_slice[1]), lat = slice(lat_slice[0],lat_slice[1]),lon = slice(lon_slice[0],lon_slice[1]))

claraCLT = xr.open_dataset('/exports/geos.ed.ac.uk/climate_change/users/carla/clara/clara-daily-19820101-20211231-filled-in.nc').sel(time = slice(time_slice[0], time_slice[1])).cfc

merraAOD = xr.open_dataset('/exports/geos.ed.ac.uk/climate_change/users/carla/merra2/daily/merra2-daily-19800101-20211231.nc').sel(time = slice(time_slice[0], time_slice[1])).TOTEXTTAU

eobs_all = EOBStmin27
eobs_all['tx27'] = (['time','lat', 'lon'],  EOBStmax27.tx.values)
eobs_all['qq27'] = (['time','lat', 'lon'],  EOBSqq27.qq.values)
eobs_all['tx26'] = (['time','lat', 'lon'],  EOBStmax26.tx.values)
eobs_all['tn26'] = (['time','lat', 'lon'],  EOBStmin26.tn.values)
eobs_all['qq26'] = (['time','lat', 'lon'],  EOBSqq26.qq.values)
eobs_all = eobs_all.rename_vars({'tn':'tn27'})

ceresaod = np.zeros((14610,140,280))*np.nan
ceresclt = np.zeros((14610,140,280))*np.nan
merraaod = np.zeros((14610,140,280))*np.nan
claraclt = np.zeros((14610,140,280))*np.nan

for i in range(140):
    print(i)
    for j in range(280):
        merraaod[:,i,j] = get_nearest_grid(merraAOD, EOBStmin26.lat.values[i], EOBStmin26.lon.values[j])
        claraclt[:,i,j] = get_nearest_grid(claraCLT, EOBStmin26.lat.values[i], EOBStmin26.lon.values[j])

eobs_all['ceresAOD'] = (['time','lat', 'lon'],  ceresaod)
eobs_all['ceresCLT'] = (['time','lat', 'lon'],  ceresclt)

eobs_all['merraAOD'] = (['time','lat', 'lon'],  merraaod)
eobs_all['claraCLT'] = (['time','lat', 'lon'],  claraclt)

eobs_all.to_netcdf('/exports/geos.ed.ac.uk/climate_change/users/carla/clara-merra-eobs27-eobs26-merged-19820101-20211231.nc')