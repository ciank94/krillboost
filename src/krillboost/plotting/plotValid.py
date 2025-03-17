import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import os
import datetime
import xgboost as xgb
import json
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata, RegularGridInterpolator

# Load map parameters from JSON config
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'config', 'map_params.json'), 'r') as f:
    MAP_PARAMS = json.load(f)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Plot validation results')
    parser.add_argument('plot', type=str, default='all', help='Plot to generate validation against real data')
    args = parser.parse_args()

    if args.plot == 'all' or args.plot == 'catch':
        plotCatch()
    return


def plotCatch():
    logger = logging.getLogger(__name__)
    logger.info('Plotting spatial predictions against catch data...')
    
    # Load catch data
    catch_data = pd.read_csv("input/subset_data/catchData.csv")
    
    # Convert time column to datetime
    catch_data['time'] = pd.to_datetime(catch_data['time'])
    
    # Filter for 2012 data
    catch_data_2012 = catch_data[catch_data['time'].dt.year == 2012]
    
    if len(catch_data_2012) == 0:
        logger.warning("No catch data found for 2012. Using the earliest available year instead.")
        # Find the earliest year with data
        earliest_year = catch_data['time'].dt.year.min()
        catch_data_2012 = catch_data[catch_data['time'].dt.year == earliest_year]
        logger.info(f"Using catch data from {earliest_year} instead.")
    
    logger.info(f"Found {len(catch_data_2012)} catch data points for 2012")
    
    # Load environmental data
    logger.info("Loading environmental data...")

    # Filter for the first 3 months of 2012
    start_date = datetime.datetime(2012, 1, 1)
    end_date = datetime.datetime(2012, 3, 31)
    if not os.path.exists(f"input/subset_years/sst_{start_date.year}.nc"):
        logger.info(f"Subsetting features for {start_date.year}")
        featureSubset(start_date, end_date)

    # Load features
    krillDataset = loadFeatures(start_date, end_date)
    logger.info(f"Loading features for {start_date.year}")
   
    # Get coordinates from the dataset
    lons = krillDataset['LONGITUDE'].unique()
    lats = krillDataset['LATITUDE'].unique()
    
    # Create a grid of points for visualization
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Remove NaN values from the dataset
    valid_mask = ~krillDataset.isna().any(axis=1)
    valid_features = krillDataset[valid_mask]
    
    # Create DataFrame for prediction
    X_pred = valid_features.copy()
    
    # Standardize features (using same approach as in training)
    X_pred = (X_pred - X_pred.mean()) / X_pred.std()
    
    # Load the presence model
    logger.info("Loading presence model...")
    presence_model_path = "output/models/presence_model.json"
    
    if not os.path.exists(presence_model_path):
        logger.error(f"Model file not found: {presence_model_path}")
        return
    
    pmod = xgb.XGBClassifier()
    pmod.load_model(presence_model_path)
    
    # Make predictions
    logger.info("Predicting krill presence...")
    presence_probs = pmod.predict_proba(X_pred)[:, 1]  # Probability of presence
    
    # Create a grid for plotting
    presence_grid = np.full((len(lats), len(lons)), np.nan)
    
    # Map the predictions back to the grid
    for i, (idx, row) in enumerate(valid_features.iterrows()):
        lat_idx = np.where(lats == row['LATITUDE'])[0][0]
        lon_idx = np.where(lons == row['LONGITUDE'])[0][0]
        presence_grid[lat_idx, lon_idx] = presence_probs[i]
    
    # Create figure with larger size
    plt.rcParams.update({'font.size': 20})  # Set default font size to 20
    fig = plt.figure(figsize=(14, 12))
    
    # Set up projection - using PlateCarree instead of SouthPolarStereo
    projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    
    # Define map bounds from the config
    lonBounds = [MAP_PARAMS['map_lon_min'], MAP_PARAMS['map_lon_max']]
    latBounds = [MAP_PARAMS['map_lat_min'], MAP_PARAMS['map_lat_max']]
    
    # Set extent to focus on Antarctic Peninsula region using config values
    ax.set_extent(lonBounds + latBounds, crs=ccrs.PlateCarree())
    
    # Add coastlines and land features
    ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=100)
    ax.coastlines(linewidth=1.5, zorder=101)
    
    # Create a custom colormap for krill presence probability
    cmap = LinearSegmentedColormap.from_list(
        'krill_cmap', 
        [(0, 'lightblue'), (0.5, 'yellow'), (1, 'red')]
    )
    
    # Fix the dimension mismatch for pcolormesh with shading='flat'
    # When using shading='flat', C dimensions should be (N-1, M-1) compared to X and Y dimensions (N, M)
    # Option 1: Use the existing grid with shading='auto' instead of 'flat'
    im = ax.pcolormesh(
        lon_grid, lat_grid, presence_grid, 
        transform=ccrs.PlateCarree(),
        cmap=cmap, 
        vmin=0, vmax=1,
        shading='auto',  # Changed from 'flat' to 'auto'
        zorder=1
    )
    
    # Add colorbar for presence probability
    cbar = plt.colorbar(im, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Predicted Krill Presence Probability', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    
    # Overlay catch data as bubbles
    # Scale the catch values for better visualization
    catch_values = catch_data_2012['catch'].values
    max_catch = catch_values.max()
    min_catch = catch_values.min()
    
    # Normalize catch values for bubble size
    normalized_catch = (catch_values - min_catch) / (max_catch - min_catch) * 200 + 20
    
    # Plot catch data as bubbles
    scatter = ax.scatter(
        catch_data_2012['longitude'], 
        catch_data_2012['latitude'],
        s=normalized_catch,  # Size based on catch amount
        c='black',  # Black outline
        edgecolors='white',
        alpha=0.7,
        transform=ccrs.PlateCarree(),
        zorder=102,
        label='Krill Catch'
    )
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Add labels
    ax.set_xlabel('Longitude', fontsize=20)
    ax.set_ylabel('Latitude', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Add title
    year = pd.to_datetime(start_date).year
    plt.title(f'Predicted Krill Presence and Catch Data ({year})', fontsize=22)
    
    # Create legend for catch data
    from matplotlib.lines import Line2D
    catch_levels = [min_catch, (min_catch + max_catch) / 2, max_catch]
    sizes = [(c - min_catch) / (max_catch - min_catch) * 200 + 20 for c in catch_levels]
    
    # Create dummy scatter plots for the legend
    handles = []
    for i, (catch, size) in enumerate(zip(catch_levels, sizes)):
        handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                  markersize=np.sqrt(size/np.pi), alpha=0.7,
                  label=f'Catch: {catch:.1f}')
        )
    
    # Add legend
    legend = ax.legend(handles=handles, loc='upper left', fontsize=14, title='Catch Amount', 
                title_fontsize=16)
    legend.set_zorder(103)
    
    # Save figure
    os.makedirs('output/figures', exist_ok=True)
    plt_path = f'output/figures/krill_presence_catch_{year}.png'
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved krill presence and catch plot to: {plt_path}")
    
    # Close the figure to free memory
    plt.close()
    
    return


def featureSubset(start_date, end_date):
    raw_data_path = "input/raw_data"
    output_path = "input/subset_years"
    logger = logging.getLogger(__name__)
    logger.info(f"Subsetting features")
    sstDataset = xr.open_dataset(f"{raw_data_path}/sst.nc")
    sshDataset = xr.open_dataset(f"{raw_data_path}/ssh.nc")
    chlDataset = xr.open_dataset(f"{raw_data_path}/chl.nc")
    ironDataset = xr.open_dataset(f"{raw_data_path}/iron.nc")

    
    sst = sstDataset.analysed_sst.sel(time= slice(start_date, end_date))
    sst_mean = sst.mean(dim="time", skipna=True).compute()
    sst_percentile_10 = sst.quantile(0.1, dim='time', skipna=True).compute()
    sst_percentile_90 = sst.quantile(0.9, dim='time', skipna=True).compute()
    sst_percentile_10 = sst_percentile_10.drop_vars("quantile")
    sst_percentile_90 = sst_percentile_90.drop_vars("quantile")
    logger.info(f"sst subset")
    sst_ds = xr.Dataset({"sst_10th_percentile": sst_percentile_10 - 273.15, "sst_90th_percentile": sst_percentile_90 - 273.15, "sst_mean": sst_mean - 273.15})

    ssh = sshDataset.sel(time=slice(start_date, end_date))
    ssh_mean = ssh.mean(dim="time", skipna=True).compute()
    ssh_percentile_10 = ssh.quantile(0.1, dim='time', skipna=True).compute()
    ssh_percentile_90 = ssh.quantile(0.9, dim='time', skipna=True).compute()
    ssh_percentile_10 = ssh_percentile_10.drop_vars("quantile")
    ssh_percentile_90 = ssh_percentile_90.drop_vars("quantile")
    logger.info(f"ssh subset")
    ssh_ds = xr.Dataset({"ssh_10th_percentile": ssh_percentile_10.adt, "ssh_90th_percentile": ssh_percentile_90.adt, "ssh_mean": ssh_mean.adt})
    vel_ds = xr.Dataset({"vel_10th_percentile": np.sqrt(np.power(ssh_percentile_10.ugos,2) + np.power(ssh_percentile_10.vgos,2)), 
                        "vel_90th_percentile": np.sqrt(np.power(ssh_percentile_90.ugos,2) + np.power(ssh_percentile_90.vgos,2)), 
                        "vel_mean": np.sqrt(np.power(ssh_mean.ugos,2) + np.power(ssh_mean.vgos,2))})

    chl = chlDataset.CHL.sel(time=slice(start_date, end_date))
    chl_mean = chl.mean(dim="time", skipna=True).compute()
    chl_percentile_10 = chl.quantile(0.1, dim='time', skipna=True).compute()
    chl_percentile_90 = chl.quantile(0.9, dim='time', skipna=True).compute()
    chl_percentile_10 = chl_percentile_10.drop_vars("quantile")
    chl_percentile_90 = chl_percentile_90.drop_vars("quantile")
    chl_ds = xr.Dataset({"chl_10th_percentile": chl_percentile_10, "chl_90th_percentile": chl_percentile_90, "chl_mean": chl_mean})
    logger.info(f"chl subset")

    iron = ironDataset.fe.sel(time=slice(start_date, end_date)).isel(depth=0)
    iron_mean = iron.mean(dim="time", skipna=True).compute()
    iron_percentile_10 = iron.quantile(0.1, dim='time', skipna=True).compute()
    iron_percentile_90 = iron.quantile(0.9, dim='time', skipna=True).compute()
    iron_percentile_10 = iron_percentile_10.drop_vars("quantile")
    iron_percentile_90 = iron_percentile_90.drop_vars("quantile")
    iron_ds = xr.Dataset({"iron_10th_percentile": iron_percentile_10, "iron_90th_percentile": iron_percentile_90, "iron_mean": iron_mean})
    logger.info(f"iron subset")

    # Save datasets
    os.makedirs(output_path, exist_ok=True)
    sst_ds.to_netcdf(f"{output_path}/sst_{start_date.year}.nc")
    ssh_ds.to_netcdf(f"{output_path}/ssh_{start_date.year}.nc")
    chl_ds.to_netcdf(f"{output_path}/chl_{start_date.year}.nc")
    iron_ds.to_netcdf(f"{output_path}/iron_{start_date.year}.nc")
    vel_ds.to_netcdf(f"{output_path}/vel_{start_date.year}.nc")
    logger.info(f"Saved datasets to: {output_path}")

    return

def loadFeatures(start_date, end_date):
    bath = xr.open_dataset(os.path.join(f"input/raw_data/bathymetry.nc"))
    sst = xr.open_dataset(f"input/subset_years/sst_{start_date.year}.nc")
    ssh = xr.open_dataset(f"input/subset_years/ssh_{start_date.year}.nc")
    chl = xr.open_dataset(f"input/subset_years/chl_{start_date.year}.nc")
    iron = xr.open_dataset(f"input/subset_years/iron_{start_date.year}.nc")
    vel = xr.open_dataset(f"input/subset_years/vel_{start_date.year}.nc")

    lon_min = MAP_PARAMS['lon_min']
    lon_max = MAP_PARAMS['lon_max']
    lat_min = MAP_PARAMS['lat_min']
    lat_max = MAP_PARAMS['lat_max']
    grid_step = MAP_PARAMS['grid_step']

    krillDataset = pd.DataFrame()
    lon = np.arange(lon_min, lon_max + grid_step, grid_step)
    lat = np.arange(lat_min, lat_max + grid_step, grid_step)
    lon, lat = np.meshgrid(lon, lat)
    lon = lon.flatten()
    lat = lat.flatten()
    interp_depth = RegularGridInterpolator((bath.lat.values, bath.lon.values), bath.elevation.values, method='linear', bounds_error=False, fill_value=np.nan)
    krillDataset['LONGITUDE'] = lon
    krillDataset['LATITUDE'] = lat
    krillDataset['DEPTH'] = np.abs(interp_depth((lat, lon)))
    interp_chl_mean = RegularGridInterpolator((chl.chl_mean.latitude.values, chl.chl_mean.longitude.values), chl.chl_mean.values, method='linear', bounds_error=False, fill_value=np.nan)
    krillDataset['CHL_MEAN'] = interp_chl_mean((lat, lon))
    interp_chl_min = RegularGridInterpolator((chl.chl_10th_percentile.latitude.values, chl.chl_10th_percentile.longitude.values), chl.chl_10th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
    krillDataset['CHL_MIN'] = interp_chl_min((lat, lon))
    interp_chl_max = RegularGridInterpolator((chl.chl_90th_percentile.latitude.values, chl.chl_90th_percentile.longitude.values), chl.chl_90th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
    krillDataset['CHL_MAX'] = interp_chl_max((lat, lon))
    interp_iron_mean = RegularGridInterpolator((iron.iron_mean.latitude.values, iron.iron_mean.longitude.values), iron.iron_mean.values, method='linear', bounds_error=False, fill_value=np.nan)
    krillDataset['IRON_MEAN'] = interp_iron_mean((lat, lon))
    interp_iron_min = RegularGridInterpolator((iron.iron_10th_percentile.latitude.values, iron.iron_10th_percentile.longitude.values), iron.iron_10th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
    krillDataset['IRON_MIN'] = interp_iron_min((lat, lon))
    interp_iron_max = RegularGridInterpolator((iron.iron_90th_percentile.latitude.values, iron.iron_90th_percentile.longitude.values), iron.iron_90th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
    krillDataset['IRON_MAX'] = interp_iron_max((lat, lon))
    interp_vel_mean = RegularGridInterpolator((vel.vel_mean.latitude.values, vel.vel_mean.longitude.values), vel.vel_mean.values, method='linear', bounds_error=False, fill_value=np.nan)
    interp_ssh_mean = RegularGridInterpolator((ssh.ssh_mean.latitude.values, ssh.ssh_mean.longitude.values), ssh.ssh_mean.values, method='linear', bounds_error=False, fill_value=np.nan)
    krillDataset['SSH_MEAN'] = interp_ssh_mean((lat, lon))
    interp_ssh_min = RegularGridInterpolator((ssh.ssh_10th_percentile.latitude.values, ssh.ssh_10th_percentile.longitude.values), ssh.ssh_10th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
    krillDataset['SSH_MIN'] = interp_ssh_min((lat, lon))
    interp_ssh_max = RegularGridInterpolator((ssh.ssh_90th_percentile.latitude.values, ssh.ssh_90th_percentile.longitude.values), ssh.ssh_90th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
    krillDataset['SSH_MAX'] = interp_ssh_max((lat, lon))
    krillDataset['VEL_MEAN'] = interp_vel_mean((lat, lon))
    interp_vel_min = RegularGridInterpolator((vel.vel_10th_percentile.latitude.values, vel.vel_10th_percentile.longitude.values), vel.vel_10th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
    krillDataset['VEL_MIN'] = interp_vel_min((lat, lon))
    interp_vel_max = RegularGridInterpolator((vel.vel_90th_percentile.latitude.values, vel.vel_90th_percentile.longitude.values), vel.vel_90th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
    krillDataset['VEL_MAX'] = interp_vel_max((lat, lon))
    interp_sst_mean = RegularGridInterpolator((sst.sst_mean.latitude.values, sst.sst_mean.longitude.values), sst.sst_mean.values, method='linear', bounds_error=False, fill_value=np.nan)
    krillDataset['SST_MEAN'] = interp_sst_mean((lat, lon))
    interp_sst_min = RegularGridInterpolator((sst.sst_10th_percentile.latitude.values, sst.sst_10th_percentile.longitude.values), sst.sst_10th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
    krillDataset['SST_MIN'] = interp_sst_min((lat, lon))
    interp_sst_max = RegularGridInterpolator((sst.sst_90th_percentile.latitude.values, sst.sst_90th_percentile.longitude.values), sst.sst_90th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
    krillDataset['SST_MAX'] = interp_sst_max((lat, lon))
    
    return krillDataset

if __name__ == '__main__':
    main()