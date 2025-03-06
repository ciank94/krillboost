import logging
import xarray as xr
import numpy as np
import time

def main():
    featureSubset() 

def featureSubset():
    raw_data_path = "input/raw_data"
    output_path = "input/subset_data"
    logger = logging.getLogger(__name__)
    logger.info(f"Subsetting features")
    sstDataset = xr.open_dataset(f"{raw_data_path}/sst.nc")
    sshDataset = xr.open_dataset(f"{raw_data_path}/ssh.nc")
    chlDataset = xr.open_dataset(f"{raw_data_path}/chl.nc")
    ironDataset = xr.open_dataset(f"{raw_data_path}/iron.nc")

    start_time = time.time()
    sst = sstDataset.analysed_sst.sel(time=sstDataset.time.dt.month.isin([1,2,3]))
    sst_yearly_mean = sst.groupby("time.year").mean(dim="time", skipna=True).compute()
    sst_percentile_10 = sst_yearly_mean.quantile(0.1, dim='year', skipna=True)
    sst_percentile_90 = sst_yearly_mean.quantile(0.9, dim='year', skipna=True)
    sst_mean_value = sst_yearly_mean.mean(dim='year', skipna=True)
    sst_percentile_10 = sst_percentile_10.drop_vars("quantile")
    sst_percentile_90 = sst_percentile_90.drop_vars("quantile")
    logger.info(f"sst subset")
    sst_ds = xr.Dataset({"sst_10th_percentile": sst_percentile_10 - 273.15, "sst_90th_percentile": sst_percentile_90 - 273.15, "sst_mean": sst_mean_value - 273.15})

    ssh = sshDataset.sel(time=sshDataset.time.dt.month.isin([1,2,3]))
    ssh_yearly_mean = ssh.groupby("time.year").mean(dim="time", skipna=True).compute()
    ssh_percentile_10 = ssh_yearly_mean.quantile(0.1, dim='year', skipna=True)
    ssh_percentile_90 = ssh_yearly_mean.quantile(0.9, dim='year', skipna=True)
    ssh_percentile_10 = ssh_percentile_10.drop_vars("quantile")
    ssh_percentile_90 = ssh_percentile_90.drop_vars("quantile")
    ssh_mean_value = ssh_yearly_mean.mean(dim='year', skipna=True)
    logger.info(f"ssh subset")
    ssh_ds = xr.Dataset({"ssh_10th_percentile": ssh_percentile_10.adt, "ssh_90th_percentile": ssh_percentile_90.adt, "ssh_mean": ssh_mean_value.adt})
    vel_ds = xr.Dataset({"vel_10th_percentile": np.sqrt(np.power(ssh_percentile_10.ugos,2) + np.power(ssh_percentile_10.vgos,2)), 
                        "vel_90th_percentile": np.sqrt(np.power(ssh_percentile_90.ugos,2) + np.power(ssh_percentile_90.vgos,2)), 
                        "vel_mean": np.sqrt(np.power(ssh_mean_value.ugos,2) + np.power(ssh_mean_value.vgos,2))})

    chl = chlDataset.CHL.sel(time=chlDataset.time.dt.month.isin([1,2,3]))
    chl_yearly_mean = chl.groupby("time.year").mean(dim="time", skipna=True).compute()
    chl_percentile_10 = chl_yearly_mean.quantile(0.1, dim='year', skipna=True)
    chl_percentile_90 = chl_yearly_mean.quantile(0.9, dim='year', skipna=True)
    chl_percentile_10 = chl_percentile_10.drop_vars("quantile")
    chl_percentile_90 = chl_percentile_90.drop_vars("quantile")
    chl_mean_value = chl_yearly_mean.mean(dim='year', skipna=True)
    chl_ds = xr.Dataset({"chl_10th_percentile": chl_percentile_10, "chl_90th_percentile": chl_percentile_90, "chl_mean": chl_mean_value})
    logger.info(f"chl subset")

    iron = ironDataset.fe.sel(time=ironDataset.time.dt.month.isin([1,2,3])).isel(depth=0)
    iron_yearly_mean = iron.groupby("time.year").mean(dim="time", skipna=True).compute()
    iron_percentile_10 = iron_yearly_mean.quantile(0.1, dim='year', skipna=True)
    iron_percentile_90 = iron_yearly_mean.quantile(0.9, dim='year', skipna=True)
    iron_percentile_10 = iron_percentile_10.drop_vars("quantile")
    iron_percentile_90 = iron_percentile_90.drop_vars("quantile")
    iron_mean_value = iron_yearly_mean.mean(dim='year', skipna=True)
    iron_ds = xr.Dataset({"iron_10th_percentile": iron_percentile_10, "iron_90th_percentile": iron_percentile_90, "iron_mean": iron_mean_value})
    logger.info(f"iron subset")

    sst_ds.to_netcdf(f"{output_path}/sst.nc")
    ssh_ds.to_netcdf(f"{output_path}/ssh.nc")
    vel_ds.to_netcdf(f"{output_path}/vel.nc")
    chl_ds.to_netcdf(f"{output_path}/chl.nc")
    iron_ds.to_netcdf(f"{output_path}/iron.nc")
    end_time = time.time()
    print(f"Time: {end_time - start_time}")
    return

if __name__ == "__main__":
    main()