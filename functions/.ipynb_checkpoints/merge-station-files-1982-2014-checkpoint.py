import datetime
import pandas as pd
import numpy as np
import xarray as xr
import glob

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

listCC = glob.glob('/exports/geos.ed.ac.uk/climate_change/users/carla/ecad/ECA_blend_cc/CC_*.txt')
listQQ = glob.glob('/exports/geos.ed.ac.uk/climate_change/users/carla/ecad/ECA_blend_qq/QQ_*.txt')
listTN = glob.glob('/exports/geos.ed.ac.uk/climate_change/users/carla/ecad/ECA_blend_tn/TN_*.txt')
listTX = glob.glob('/exports/geos.ed.ac.uk/climate_change/users/carla/ecad/ECA_blend_tx/TX_*.txt')

idsCC = [int(j.split('_')[-1].split('.')[0][5:]) for j in listCC]
idsQQ = [int(j.split('_')[-1].split('.')[0][5:]) for j in listQQ]
idsTN = [int(j.split('_')[-1].split('.')[0][5:]) for j in listTN]
idsTX = [int(j.split('_')[-1].split('.')[0][5:]) for j in listTX]

idsmerged = [str(i) for i in list(set(idsCC).intersection(set(idsQQ)).intersection(set(idsTN)).intersection(set(idsTX)))]

stations = pd.read_csv('/exports/geos.ed.ac.uk/climate_change/users/carla/ecad/stations.csv')

clara= xr.open_dataset('/exports/geos.ed.ac.uk/climate_change/users/carla/clara/clara-daily-19820101-20211231-filled-in.nc')#.sel(time = slice(time_slice[0], time_slice[1]))
claraclt = clara.cfc

merra= xr.open_dataset('/exports/geos.ed.ac.uk/climate_change/users/carla/merra2/daily/merra2-daily-19800101-20211231.nc')#.sel(time = slice(time_slice[0], time_slice[1]))
merraaod = merra.TOTEXTTAU

EOBStmin = xr.open_dataset('/exports/geos.ed.ac.uk/climate_change/users/carla/eobs/tn_ens_mean_0.25deg_reg_v27.0e.nc').rename({'longitude':'lon','latitude':'lat'})
EOBStmax = xr.open_dataset('/exports/geos.ed.ac.uk/climate_change/users/carla/eobs/tx_ens_mean_0.25deg_reg_v27.0e.nc').rename({'longitude':'lon','latitude':'lat'})
EOBSsw = xr.open_dataset('/exports/geos.ed.ac.uk/climate_change/users/carla/eobs/qq_ens_mean_0.25deg_reg_v27.0e.nc').rename({'longitude':'lon','latitude':'lat'})

print('whoop whoop things are happening')

for k in idsmerged:
    print(k)
    
    fileCC = glob.glob('/exports/geos.ed.ac.uk/climate_change/users/carla/ecad/ECA_blend_cc/CC_*0'+str(k)+'.txt')[0]
    fileQQ = glob.glob('/exports/geos.ed.ac.uk/climate_change/users/carla/ecad/ECA_blend_qq/QQ_*0'+str(k)+'.txt')[0]
    fileTN = glob.glob('/exports/geos.ed.ac.uk/climate_change/users/carla/ecad/ECA_blend_tn/TN_*0'+str(k)+'.txt')[0]
    fileTX = glob.glob('/exports/geos.ed.ac.uk/climate_change/users/carla/ecad/ECA_blend_tx/TX_*0'+str(k)+'.txt')[0]

    dfCC = open(fileCC, "r")
    linesCC = dfCC.readlines()
    dfCC.close()
    
    dfQQ = open(fileQQ, "r")
    linesQQ = dfQQ.readlines()
    dfQQ.close()
    
    dfTN = open(fileTN, "r")
    linesTN = dfTN.readlines()
    dfTN.close()
    
    dfTX = open(fileTX, "r")
    linesTX = dfTX.readlines()
    dfTX.close()
    
    _cc = np.array([int(l.split(',')[2]) for l in linesCC[21:]])
    _qq = np.array([int(l.split(',')[2]) for l in linesQQ[21:]])
    _tn = np.array([int(l.split(',')[2]) for l in linesTN[21:]])
    _tx = np.array([int(l.split(',')[2]) for l in linesTX[21:]])
    
    if (np.max(_cc) > 19820101) & (np.max(_qq) > 19820101) & (np.max(_tn) > 19820101) & (np.max(_tx) > 19820101) & (np.min(_cc) < 20141231) & (np.min(_qq) < 20141231) & (np.min(_tn) < 20141231) & (np.min(_tx) < 20141231):
    
        _max = np.min([np.max(_cc), np.max(_qq), np.max(_tn), np.max(_tx)])

        if _max > 20141231: _max = 20141231

        if (_max in _cc) & (_max in _qq) & (_max in _tx) & (_max in _tn): 
            ccs = np.where(_cc >= 19820101)[0][0]
            qqs = np.where(_qq >= 19820101)[0][0]
            tns = np.where(_tn >= 19820101)[0][0]
            txs = np.where(_tx >= 19820101)[0][0]

            _min = np.max([_cc[ccs],_qq[qqs],_tn[tns],_tx[txs]])
            
            if (_min in _cc) & (_min in _qq) & (_min in _tx) & (_min in _tn):   

                ccs = np.where(_cc == _min)[0][0]
                qqs = np.where(_qq == _min)[0][0]
                tns = np.where(_tn == _min)[0][0]
                txs = np.where(_tx == _min)[0][0]

                cce = np.where(_cc == _max)[0][0]
                qqe = np.where(_qq == _max)[0][0]
                tne = np.where(_tn == _max)[0][0]
                txe = np.where(_tx == _max)[0][0]      


                dfCC = pd.DataFrame(columns= ['time','CC'])
                dfQQ = pd.DataFrame(columns= ['time','QQ'])
                dfTN = pd.DataFrame(columns= ['time','TN'])
                dfTX = pd.DataFrame(columns= ['time','TX'])

                dfCC['time'] = pd.to_datetime(_cc[ccs:cce+1].astype('str'), format = '%Y%m%d')
                dfCC = dfCC.set_index('time')
                dfQQ['time'] = pd.to_datetime(_qq[qqs:qqe+1].astype('str'), format = '%Y%m%d')
                dfQQ = dfQQ.set_index('time')
                dfTN['time'] = pd.to_datetime(_tn[tns:tne+1].astype('str'), format = '%Y%m%d')
                dfTN = dfTN.set_index('time')
                dfTX['time'] = pd.to_datetime(_tx[txs:txe+1].astype('str'), format = '%Y%m%d')
                dfTX = dfTX.set_index('time')

                dfCC['CC'] = np.array([int(l.split(',')[3]) for l in linesCC[21:][ccs:cce+1]])
                dfQQ['QQ'] = np.array([int(l.split(',')[3]) for l in linesQQ[21:][qqs:qqe+1]])
                dfTN['TN'] = np.array([int(l.split(',')[3]) for l in linesTN[21:][tns:tne+1]])
                dfTX['TX'] = np.array([int(l.split(',')[3]) for l in linesTX[21:][txs:txe+1]])

                df = pd.concat([dfCC,dfQQ,dfTN,dfTX], axis=1)#.fillna(0)

                aod = get_nearest_grid(merraaod, stations.loc[np.where(stations.id == int(k))]['lat'].values[0], stations.loc[np.where(stations.id == int(k))]['lon'].values[0])
                df['merraAOD'] = aod.sel(time = slice(pd.to_datetime(str(_min), format = '%Y%m%d'), pd.to_datetime(str(_max+0.8), format = '%Y%m%d'))).values

                clt = get_nearest_grid(claraclt, stations.loc[np.where(stations.id == int(k))]['lat'].values[0], stations.loc[np.where(stations.id == int(k))]['lon'].values[0])
                df['claraCLT'] = clt.sel(time = slice(pd.to_datetime(str(_min), format = '%Y%m%d'), pd.to_datetime(str(_max+0.8), format = '%Y%m%d'))).values

                df = df.replace(-9999, np.nan)
                df[['CC','QQ','TN','TX','claraCLT','merraAOD']] = df[['CC','QQ','TN','TX','claraCLT','merraAOD']].mask(df.CC.isna() | df.QQ.isna() | df.TN.isna()| df.TX.isna() | df.merraAOD.isna(), np.nan)

                if all(df.count() > 3600):
                       df.to_xarray().to_netcdf('/exports/geos.ed.ac.uk/climate_change/users/carla/ecad/merged-stations-19820101-20141231/'+str(k)+'.nc')