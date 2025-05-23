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
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

# Load map parameters from JSON config
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'config', 'map_params.json'), 'r') as f:
    MAP_PARAMS = json.load(f)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Plot validation results')
    parser.add_argument('plot', type=str, default='all', help='Plot to generate validation against real data (all, catch, response, multiyear, tree, indices)')
    args = parser.parse_args()

    if args.plot == 'all' or args.plot == 'catch':
        plotCatch()
    if args.plot == 'all' or args.plot == 'response':
        plot_response_curves()
    if args.plot == 'all' or args.plot == 'multiyear':
        plot_multiyear_predictions()
    if args.plot == 'tree':
        plot_xgboost_tree()
    if args.plot == 'all' or args.plot == 'indices':
        plot_yearly_indices()
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

    # NaN-aware gaussian smoothing
    # First create a mask of valid (non-NaN) values
    valid_mask = ~np.isnan(presence_grid)
    
    # Create a copy of the grid with zeros instead of NaNs for filtering
    grid_for_filter = np.copy(presence_grid)
    grid_for_filter[~valid_mask] = 0
    
    # Apply gaussian filter to both the data and the mask
    smooth_grid = gaussian_filter(grid_for_filter, sigma=1)
    smooth_mask = gaussian_filter(valid_mask.astype(float), sigma=1)
    
    # Divide by the mask to normalize only where we have data
    # Avoid division by zero
    smooth_mask[smooth_mask < 1e-10] = 1  # Avoid division by zero
    presence_grid = smooth_grid / smooth_mask
    
    # Restore NaN values where we had no data influence (threshold can be adjusted)
    presence_grid[smooth_mask < 0.2] = np.nan

    # Create figure with larger size
    plt.rcParams.update({'font.size': 20})  # Set default font size to 20
    fig = plt.figure(figsize=(14, 12))
    
    # Set up projection - using PlateCarree instead of SouthPolarStereo
    projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    
    # Load krill data
    try:
        krillData = pd.read_csv("input/fusedData.csv", encoding='latin1')
        logger.info(f"Successfully loaded fusedData.csv with latin1 encoding")
    except UnicodeDecodeError:
        try:
            krillData = pd.read_csv("input/fusedData.csv", encoding='cp1252')
            logger.info(f"Successfully loaded fusedData.csv with cp1252 encoding")
        except Exception as e:
            logger.error(f"Failed to load fusedData.csv: {str(e)}")
            return
    
    # Define dynamic bounds from the data (like in plotClass.py)
    lonBounds = [krillData['LONGITUDE'].min() - 1, krillData['LONGITUDE'].max() + 1]
    latBounds = [krillData['LATITUDE'].min() - 1, krillData['LATITUDE'].max() + 1]
    grid_res = 2.0  # 2-degree grid cells like in plotClass.py
    
    # Set extent to focus on Antarctic Peninsula region using dynamic bounds
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


def plot_response_curves():
    """
    Create response curves for the top 10 features in the krill presence model.
    
    This function:
    1. Loads the trained presence model
    2. Identifies the top 10 features by importance
    3. Generates response curves showing how each feature affects the predicted probability of krill presence
    4. Creates a multi-panel figure with all response curves
    
    Each response curve shows how the predicted probability changes when varying one feature
    while keeping all others at their actual distribution of values from random background points.
    """
    logger = logging.getLogger(__name__)
    logger.info('Plotting response curves for top 10 features...')
    
    # Load model
    presence_model_path = "output/models/presence_model.json"
    
    if not os.path.exists(presence_model_path):
        logger.error(f"Model file not found: {presence_model_path}")
        return
    
    # Load data
    data_path = "input/fusedData.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
        
    krillData = pd.read_csv(data_path)
    
    # Remove unnamed columns if they exist
    unnamed_cols = [col for col in krillData.columns if 'Unnamed' in col]
    if unnamed_cols:
        logger.info(f"Removing unnamed columns: {unnamed_cols}")
        krillData = krillData.drop(columns=unnamed_cols)
    
    # Check for target columns
    required_columns = ['KRILL_PRESENCE', 'KRILL_LOG10', 'KRILL_ORIGINAL']
    missing_columns = [col for col in required_columns if col not in krillData.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return
    
    # Separate features and targets
    target_columns = ['KRILL_PRESENCE', 'KRILL_LOG10', 'KRILL_SQRT', 'KRILL_LOGN', 'KRILL_QUAN', 'KRILL_ORIGINAL']
    available_targets = [col for col in target_columns if col in krillData.columns]
    
    X = krillData.drop(columns=available_targets)
    y_presence = krillData['KRILL_PRESENCE']
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Load model
    try:
        pmod = xgb.XGBClassifier()
        pmod.load_model(presence_model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Get feature importance
    try:
        # Try using the feature_importances_ attribute first
        importances = pmod.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
    except (AttributeError, ValueError) as e:
        logger.warning(f"Could not get feature_importances_ directly: {e}")
        # Fall back to get_score method
        try:
            importance_dict = pmod.get_booster().get_score(importance_type='gain')
            
            # Convert to DataFrame for easier handling
            importance_df = pd.DataFrame({
                'Feature': list(importance_dict.keys()),
                'Importance': list(importance_dict.values())
            })
            
            # Map feature indices (f0, f1, etc.) to actual feature names
            feature_map = {f'f{i}': name for i, name in enumerate(feature_names)}
            importance_df['Feature'] = importance_df['Feature'].map(feature_map)
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return
    
    # Sort by importance and get top 10
    importance_df = importance_df.sort_values('Importance', ascending=False)
    top_features = importance_df.head(10)['Feature'].tolist()
    
    logger.info(f"Top 10 features: {top_features}")
    
    # Feature scaling for better model performance
    X_scaled = (X - X.mean()) / X.std()
    
    # Number of points to evaluate for each feature
    n_points = 100
    
    # Number of background samples to use
    n_background = 500  # Increased for better representation
    
    # Create figure with subplots
    n_cols = 2
    n_rows = (len(top_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    # For each top feature, generate a response curve
    for i, feature in enumerate(top_features):
        logger.info(f"Generating response curve for {feature} ({i+1}/{len(top_features)})...")
        
        # Create range of values to evaluate for this feature
        feature_min = X_scaled[feature].min()
        feature_max = X_scaled[feature].max()
        x_values = np.linspace(feature_min, feature_max, n_points)
        
        # Original scale values for x-axis
        x_orig_scale = x_values * X[feature].std() + X[feature].mean()
        
        # Storage for predictions
        all_predictions = []
        
        # For each x value, create multiple predictions with random background points
        for x_val in x_values:
            # Sample random background points
            background_indices = np.random.choice(X_scaled.shape[0], n_background, replace=True)
            background_samples = X_scaled.iloc[background_indices].copy()
            
            # Set the feature value to the current x value
            background_samples[feature] = x_val
            
            # Make predictions
            preds = pmod.predict_proba(background_samples)[:, 1]
            
            all_predictions.append(preds)
        
        # Convert to numpy array for easier calculations
        all_predictions = np.array(all_predictions)  # shape: (n_points, n_background)
        
        # Calculate mean and confidence intervals
        mean_pred = np.mean(all_predictions, axis=1)
        std_pred = np.std(all_predictions, axis=1)
        lower_bound = mean_pred - 1.96 * std_pred  # 95% confidence interval
        upper_bound = mean_pred + 1.96 * std_pred
        
        # Ensure bounds are within [0, 1]
        lower_bound = np.maximum(0, lower_bound)
        upper_bound = np.minimum(1, upper_bound)
        
        # Plot response curve
        ax = axes[i]
        ax.plot(x_orig_scale, mean_pred, 'b-', linewidth=2)
        ax.fill_between(x_orig_scale, lower_bound, upper_bound, alpha=0.3, color='b')
        
        # Add horizontal line at 0.5 probability
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
        
        # Add labels
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Probability of Presence', fontsize=12)
        ax.set_title(f'Response Curve: {feature}', fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        # Add importance value as text
        importance_value = importance_df[importance_df['Feature'] == feature]['Importance'].values[0]
        importance_percent = 100 * importance_value / importance_df['Importance'].sum()
        ax.text(0.05, 0.95, f'Importance: {importance_percent:.1f}%', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove any unused subplots
    for i in range(len(top_features), len(axes)):
        fig.delaxes(axes[i])
    
    # Add overall title
    plt.suptitle('Response Curves for Top 10 Features (Krill Presence Model)', fontsize=16, y=1.02)
    
    # Save figure
    os.makedirs('output/figures', exist_ok=True)
    plt_path = 'output/figures/response_curves.png'
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved response curves plot to: {plt_path}")
    
    # Close the figure to free memory
    plt.close()
    
    return


def plot_multiyear_predictions():
    """
    Create a multi-year plot of krill presence predictions for years 2011-2016,
    showing contours of prediction probabilities for each year.
    
    This function:
    1. Processes environmental data for years 2011, 2012, 2013, 2014, 2015, and 2016
    2. Makes predictions using the trained presence model for each year
    3. Creates a 6-panel figure showing contours of krill presence probability
    4. Uses a continuous colormap with every 10th quantile (0, 10, 20, ..., 90, 100)
    5. Includes bathymetry contours for better geographic context
    """
    logger = logging.getLogger(__name__)
    logger.info('Creating multi-year krill presence predictions plot with quantiles...')
    
    # Load presence model
    presence_model_path = "output/models/presence_model.json"
    
    if not os.path.exists(presence_model_path):
        logger.error(f"Model file not found: {presence_model_path}")
        return
    
    # Load the model
    try:
        pmod = xgb.XGBClassifier()
        pmod.load_model(presence_model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Load raw krill data for presence/absence points
    krill_raw_data_path = "input/raw_data/krillbase.csv"
    if not os.path.exists(krill_raw_data_path):
        logger.error(f"Raw krill data file not found: {krill_raw_data_path}")
        return
    
    try:
        # Load the raw krill data which should have date information
        # Try different encodings since there's an issue with UTF-8
        try:
            krill_raw_data = pd.read_csv(krill_raw_data_path, encoding='latin1')
            logger.info("Loaded krillbase.csv with latin1 encoding")
        except Exception as e1:
            try:
                krill_raw_data = pd.read_csv(krill_raw_data_path, encoding='cp1252')
                logger.info("Loaded krillbase.csv with cp1252 encoding")
            except Exception as e2:
                logger.error(f"Failed to load krillbase.csv with multiple encodings: {e1}, {e2}")
                return
        
        # Extract year from DATE column (format dd/mm/yyyy)
        if 'DATE' in krill_raw_data.columns:
            # Convert DATE to datetime and extract year
            krill_raw_data['DATE'] = pd.to_datetime(krill_raw_data['DATE'], format='%d/%m/%Y', errors='coerce')
            krill_raw_data['year'] = krill_raw_data['DATE'].dt.year
            logger.info(f"Extracted year information from DATE column")
            
            # Remove rows with invalid dates
            valid_date_mask = ~krill_raw_data['DATE'].isna()
            krill_raw_data = krill_raw_data[valid_date_mask]
            logger.info(f"Removed {sum(~valid_date_mask)} rows with invalid dates")
        else:
            logger.error("DATE column not found in krillbase.csv")
            return
        
        # Ensure we have krill density information to determine presence/absence
        if 'STANDARDISED_KRILL_UNDER_1M2' in krill_raw_data.columns:
            # Based on the memory, values of -2.0 represent absence of krill (zeros)
            krill_raw_data['KRILL_PRESENCE'] = (krill_raw_data['STANDARDISED_KRILL_UNDER_1M2'] > -2.0).astype(int)
            logger.info("Created KRILL_PRESENCE from STANDARDISED_KRILL_UNDER_1M2")
        else:
            logger.error("STANDARDISED_KRILL_UNDER_1M2 column not found in krillbase.csv")
            return
        
        logger.info(f"Loaded raw krill data with {len(krill_raw_data)} records")
    except Exception as e:
        logger.error(f"Error loading raw krill data: {e}")
        return
    
    # Define years to process
    years = [2011, 2012, 2013, 2014, 2015, 2016]
    
    # Define quantile thresholds (every 10th percentile)
    quantile_thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Set up figure with parameters matching envData.png
    plt.rcParams.update({'font.size': 20})  # Set default font size to 20
    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor((0.1216, 0.6510, 0.5804))
    
    # Create a 3x2 grid for subplots with minimal spacing as per memory
    gs = fig.add_gridspec(3, 2, hspace=0.05, wspace=0.05)
    #gs = fig.add_gridspec(3, 2)
    axes = []
    
    # Create subplots with PlateCarree projection
    for i in range(6):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        axes.append(ax)
    
    # Load krill data
    try:
        krillData = pd.read_csv("input/fusedData.csv")
        logger.info(f"Successfully loaded fusedData.csv")
    except Exception as e:
        logger.error(f"Failed to load fusedData.csv: {str(e)}")
        return
    
    # Define map boundaries and grid step based on data
    grid_step = 5
    # Define grid bounds based on data, consistent with plotClass.py
    lon_min = krillData['LONGITUDE'].min() - 0.3
    lon_max = krillData['LONGITUDE'].max() + 0.3
    lat_min = krillData['LATITUDE'].min() - 1
    lat_max = krillData['LATITUDE'].max() + 0.3
    lonBounds = [lon_min, lon_max]
    latBounds = [lat_min, lat_max]
    logger.info(f"Using map bounds: lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}], grid_step={grid_step}")
    
    # Load bathymetry for contours
    try:
        bath = xr.open_dataset(f"input/raw_data/bathymetry.nc")
        # Create masked array for bathymetry where elevation <= 0
        bathymetry = abs(bath.elevation.values)
        # Mask both land (elevation > 0) and invalid points
        masked_bathymetry = np.ma.masked_where((bathymetry <= 0) | (bathymetry > 10000), bathymetry)
        # Create contour levels every 400m
        contour_levels = np.arange(0, 3000, 400)
        has_bathymetry = True
        logger.info("Loaded bathymetry data for contours")
    except Exception as e:
        logger.warning(f"Could not load bathymetry data: {str(e)}")
        has_bathymetry = False
    
    # Create a custom colormap for krill presence probability percentiles
    # Use 'Reds' colormap as specified in memory
    contour_cmap = plt.cm.Reds
    
    # Store all prediction grids to calculate global percentiles
    all_predictions = []
    all_prediction_grids = []
    
    # First pass: process each year and collect predictions
    for i, year in enumerate(years):
        logger.info(f"Processing year {year}...")
        
        # Define date range for this year (first 3 months)
        start_date = datetime.datetime(year, 1, 1)
        end_date = datetime.datetime(year, 3, 31)
        
        # Check if subset data exists, if not create it
        if not os.path.exists(f"input/subset_years/sst_{start_date.year}.nc"):
            logger.info(f"Subsetting features for {start_date.year}")
            featureSubset(start_date, end_date)
        
        # Load features
        krillDataset = loadFeatures(start_date, end_date)
        logger.info(f"Loaded features for {start_date.year}")
        
        # Get coordinates from the dataset
        lons = krillDataset['LONGITUDE'].unique()
        lats = krillDataset['LATITUDE'].unique()
        
        # Create a grid of points for visualization
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Remove NaN values from the dataset
        valid_mask = ~krillDataset.isna().any(axis=1)
        valid_features = krillDataset[valid_mask]

        # Drop latitude and longitude columns
        valid_features = valid_features.drop(['LATITUDE', 'LONGITUDE'], axis=1)
        
        # Create DataFrame for prediction
        X_pred = valid_features.copy()
        
        # Standardize features (using same approach as in training)
        X_pred = (X_pred - X_pred.mean()) / X_pred.std()
        
        # Make predictions
        logger.info(f"Predicting krill presence for {year}...")
        presence_probs = pmod.predict_proba(X_pred)[:, 1]  # Probability of presence
        
        # Store all valid predictions for percentile calculation
        all_predictions.extend(presence_probs)
        
        # Create a grid for plotting
        presence_grid = np.full((len(lats), len(lons)), np.nan)
        
        # Map the predictions back to the grid
        for j, (idx, row) in enumerate(valid_features.iterrows()):
            lat_idx = np.where(lats == krillDataset['LATITUDE'][idx])[0][0]
            lon_idx = np.where(lons == krillDataset['LONGITUDE'][idx])[0][0]
            presence_grid[lat_idx, lon_idx] = presence_probs[j]
        
        original_grid = presence_grid

        # NaN-aware gaussian smoothing
        # First create a mask of valid (non-NaN) values
        valid_mask = ~np.isnan(presence_grid)
        
        # Create a copy of the grid with zeros instead of NaNs for filtering
        grid_for_filter = np.copy(presence_grid)
        grid_for_filter[~valid_mask] = 0
        
        # Apply gaussian filter to both the data and the mask
        smooth_grid = gaussian_filter(grid_for_filter, sigma=1)
        smooth_mask = gaussian_filter(valid_mask.astype(float), sigma=0.5)
        
        # Divide by the mask to normalize only where we have data
        # Avoid division by zero
        smooth_mask[smooth_mask < 1e-10] = 1  # Avoid division by zero
        presence_grid = smooth_grid / smooth_mask
        
        # Restore NaN values where we had no data influence (threshold can be adjusted)
        presence_grid[smooth_mask < 0.2] = np.nan


        presence_grid = original_grid
        

        # Store the grid for this year
        all_prediction_grids.append({
            'year': year,
            'grid': presence_grid,
            'lon_grid': lon_grid,
            'lat_grid': lat_grid
        })
    
    # Calculate global quantile thresholds from all predictions
    quantile_values = {}
    for q in quantile_thresholds:
        if q == 0:
            quantile_values[q] = np.nanmin(all_predictions)
        elif q == 100:
            quantile_values[q] = np.nanmax(all_predictions)
        else:
            quantile_values[q] = np.nanpercentile(all_predictions, q)
    
    logger.info(f"Calculated global quantile thresholds: {quantile_values}")
    
    # Create contour levels based on quantile values
    contour_levels_plot = [quantile_values[q] for q in quantile_thresholds]
    
    # Second pass: plot each year with contour visualization
    for i, year_data in enumerate(all_prediction_grids):
        year = year_data['year']
        presence_grid = year_data['grid']
        lon_grid = year_data['lon_grid']
        lat_grid = year_data['lat_grid']
        
        # Get the current subplot
        ax = axes[i]
        
        # Set extent to focus on Antarctic Peninsula region using specified bounds
        ax.set_extent(lonBounds + latBounds, crs=ccrs.PlateCarree())
        
        # Add coastlines and land features with improved styling
        ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=100)
        ax.coastlines(linewidth=1.5, zorder=101)  # Coastline linewidth of 1.5 as per memory
        
        # Add bathymetry contours if available
        if has_bathymetry:
            cs = ax.contour(bath.lon, bath.lat, 
                          masked_bathymetry, levels=contour_levels,
                          colors='gray', linewidths=0.8, alpha=0.5, transform=ccrs.PlateCarree(),
                          zorder=90)  # Increased z-order to 90, just below the year annotation (1000) and land (100)
        
        # Add gridlines with improved styling - only show labels on edge plots
        gl = ax.gridlines(draw_labels=True, linewidth=1.0, color='gray', alpha=0.5, linestyle='--')
        
        # Configure which axes get labels
        gl.bottom_labels = True
        gl.xlabel_style = {'size': 14}
        
        if i >= 4:  # Bottom row
            # Add x-axis label
            ax.set_xlabel('Longitude', fontsize=20)
        
        if i % 2 == 0:  # Left column
            gl.left_labels = True
            gl.ylabel_style = {'size': 14}
            # Add y-axis label
            ax.set_ylabel('Latitude', fontsize=20)
        else:
            gl.left_labels = False
            
        # Always hide top and right labels
        gl.top_labels = False
        gl.right_labels = False
        
        # Add year as an annotation in the bottom right corner with high z-order to ensure visibility
        ax.annotate(f'{year}', xy=(0.95, 0.05), xycoords='axes fraction', 
                   fontsize=24, fontweight='bold', ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
                   zorder=1000)  # High z-order to ensure it's on top
        
        # Create a categorized grid based on quantiles
        # Instead of using the raw probability values, we'll create a grid with category values
        categorized_grid = np.zeros_like(presence_grid)
        categorized_grid[:] = np.nan  # Start with all NaN
        
        # Fill in the grid with category values based on quantiles
        for q_idx in range(len(quantile_thresholds)-1):
            current_q = quantile_thresholds[q_idx]
            next_q = quantile_thresholds[q_idx+1]
            
            # Get threshold values
            current_threshold = quantile_values[current_q]
            next_threshold = quantile_values[next_q]
            
            # Create mask for values in this quantile range
            if q_idx == 0:  # First quantile includes the minimum
                mask = ~np.isnan(presence_grid) & (presence_grid <= next_threshold)
            else:
                mask = (presence_grid > current_threshold) & (presence_grid <= next_threshold)
            
            # Assign a value representing this quantile range (use the quantile index)
            categorized_grid[mask] = q_idx
        
        # Plot the categorized grid with contourf for smoother transitions
        im = ax.contourf(
            lon_grid, lat_grid, categorized_grid,
            transform=ccrs.PlateCarree(),
            cmap=contour_cmap,
            alpha=0.9,
            zorder=10
        )
        
        # # Add contour lines for better visualization of boundaries between quantiles
        # ax.contour(
        #     lon_grid, lat_grid, categorized_grid,
        #     levels=np.arange(len(quantile_thresholds)-1),  # Contour lines at category boundaries
        #     colors='k',
        #     linewidths=0.3,
        #     alpha=0.3,
        #     transform=ccrs.PlateCarree(),
        #     zorder=11
        # )
    
    # Create a custom colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    
    # Create the colorbar with improved styling
    norm = mcolors.BoundaryNorm(np.arange(len(quantile_thresholds)), contour_cmap.N)
    sm = plt.cm.ScalarMappable(cmap=contour_cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, cax=cbar_ax)
    
    # Create labels for the colorbar with improved styling
    cbar_labels = []
    for q_idx in range(len(quantile_thresholds)-1):
        current_q = quantile_thresholds[q_idx]
        next_q = quantile_thresholds[q_idx+1]
        
        if q_idx == 0:
            cbar_labels.append(f'0-{next_q}th')
        else:
            cbar_labels.append(f'{current_q}-{next_q}th')
    
    # Set the colorbar tick positions
    cbar.set_ticks(np.arange(len(quantile_thresholds)-1) + 0.5)  # Center ticks in each category
    cbar.set_ticklabels(cbar_labels)
    cbar.ax.tick_params(labelsize=14)  # Font size as per memory
    
    # Add colorbar label with improved styling
    cbar.set_label('Krill Presence Probability', fontsize=20)  # Font size as per memory
    
    # Adjust layout to fit labels and colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Save figure with higher resolution
    os.makedirs('output/figures', exist_ok=True)
    plt_path = 'output/figures/multiyear_krill_contours.png'
    plt.savefig(plt_path, format='png', dpi=300)
    # if convert to svg:
    svg = True
    if svg:
        heatmap = plt.imread(plt_path)
        svg_path = 'output/figures/multiyear_krill_contours.svg'
        plt.imsave(svg_path, heatmap, format='svg')
    logger.info(f"Saved high-resolution multi-year krill contours plot to: {plt_path}")
    
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

def plot_xgboost_tree():
    """
    Visualize the structure of the XGBoost classifier tree.
    
    This function:
    1. Loads the trained presence model
    2. Creates a feature importance plot
    3. Saves tree information to text files
    4. Saves the visualizations to the output directory
    """
    logger = logging.getLogger(__name__)
    logger.info('Creating visualization of XGBoost tree structure...')
    
    # Define paths
    presence_model_path = "output/models/presence_model.json"
    
    # Check if model exists
    if not os.path.exists(presence_model_path):
        logger.error(f"Model file not found: {presence_model_path}")
        return
    
    try:
        # Load the model
        pmod = xgb.XGBClassifier()
        pmod.load_model(presence_model_path)
        logger.info("Model loaded successfully")
        
        # Get the booster from the model
        booster = pmod.get_booster()
        
        # Get feature names from the model dump
        dump = booster.get_dump()
        
        # Try to extract feature names from the model
        try:
            # First try to get feature names directly
            feature_names = booster.feature_names
            if not feature_names:
                raise AttributeError("No feature names found in booster")
        except (AttributeError, TypeError):
            # If that fails, try to infer from the first tree
            logger.info("Feature names not found in model, creating generic names")
            # Count unique feature IDs in the first tree
            import re
            first_tree = dump[0]
            feature_ids = set(re.findall(r'f(\d+)', first_tree))
            num_features = max(int(fid) for fid in feature_ids) + 1 if feature_ids else 0
            feature_names = [f'f{i}' for i in range(num_features)]
        
        logger.info(f"Using feature names: {feature_names}")
        
        # Create output directory if it doesn't exist
        os.makedirs('output/figures/trees', exist_ok=True)
        
        # Determine how many trees to visualize (up to 5)
        # Get the number of trees directly from the booster
        num_trees = min(5, len(dump))
        logger.info(f"Visualizing {num_trees} trees out of {len(dump)} total trees")
        
        # Save tree structure to text files
        for i in range(num_trees):
            tree_info = dump[i]
            tree_path = f'output/figures/trees/tree_{i}.txt'
            with open(tree_path, 'w') as f:
                f.write(tree_info)
            logger.info(f"Saved tree {i} structure to {tree_path}")
            
            # Instead of using xgb.plot_tree which requires Graphviz,
            # we'll create a simplified tree visualization using matplotlib
            try:
                # Parse the tree structure
                import re
                nodes = []
                for line in tree_info.split('\n'):
                    if line.strip():
                        # Extract node information
                        match = re.search(r'(\d+):\[(.*)\]', line)
                        if match:
                            node_id = int(match.group(1))
                            node_info = match.group(2)
                            nodes.append((node_id, node_info))
                
                # Create a simple visualization of the tree structure
                plt.figure(figsize=(15, 10))
                
                # Calculate the maximum depth of the tree
                max_depth = 0
                for node_id, _ in nodes:
                    depth = 0
                    while node_id > 0:
                        node_id = (node_id - 1) // 2
                        depth += 1
                    max_depth = max(max_depth, depth)
                
                # Plot each node
                for node_id, node_info in nodes:
                    # Calculate node position
                    depth = 0
                    temp_id = node_id
                    while temp_id > 0:
                        temp_id = (temp_id - 1) // 2
                        depth += 1
                    
                    x = depth / (max_depth + 1)
                    y = node_id / (2**(depth+1) + 1)
                    
                    # Determine if it's a leaf or decision node
                    if 'leaf' in node_info:
                        color = 'lightgreen'
                        # Extract leaf value
                        leaf_match = re.search(r'leaf=([^,]+)', node_info)
                        if leaf_match:
                            leaf_value = leaf_match.group(1)
                            node_label = f"Node {node_id}\nLeaf: {leaf_value}"
                        else:
                            node_label = f"Node {node_id}\nLeaf"
                    else:
                        color = 'lightblue'
                        # Extract feature and threshold
                        feature_match = re.search(r'f(\d+)<([^,]+)', node_info)
                        if feature_match:
                            feature_id = feature_match.group(1)
                            threshold = feature_match.group(2)
                            # Try to get actual feature name if available
                            feature_name = feature_names[int(feature_id)] if int(feature_id) < len(feature_names) else f"f{feature_id}"
                            node_label = f"Node {node_id}\n{feature_name} < {threshold}"
                        else:
                            node_label = f"Node {node_id}\nDecision"
                    
                    # Plot the node
                    plt.scatter(x, y, s=100, c=color, edgecolors='black')
                    plt.text(x, y+0.02, node_label, 
                            ha='center', va='center', fontsize=8)
                    
                    # Add edges to children
                    left_child = 2 * node_id + 1
                    right_child = 2 * node_id + 2
                    
                    left_exists = any(id == left_child for id, _ in nodes)
                    right_exists = any(id == right_child for id, _ in nodes)
                    
                    if left_exists:
                        child_depth = depth + 1
                        child_x = child_depth / (max_depth + 1)
                        child_y = left_child / (2**(child_depth+1) + 1)
                        plt.plot([x, child_x], [y, child_y], 'k-', alpha=0.5)
                    
                    if right_exists:
                        child_depth = depth + 1
                        child_x = child_depth / (max_depth + 1)
                        child_y = right_child / (2**(child_depth+1) + 1)
                        plt.plot([x, child_x], [y, child_y], 'k-', alpha=0.5)
                
                plt.title(f'Simplified Tree {i} Structure', fontsize=16)
                plt.axis('off')
                plt.tight_layout()
                
                # Save the tree visualization
                tree_viz_path = f'output/figures/trees/tree_{i}_viz.png'
                plt.savefig(tree_viz_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved tree {i} visualization to {tree_viz_path}")
                plt.close()
            except Exception as e:
                logger.error(f"Error creating tree visualization for tree {i}: {str(e)}")
        
        # Create a feature importance plot
        plt.figure(figsize=(12, 8))
        
        # Get feature importances
        try:
            # Try to get feature importances by gain
            importance_dict = booster.get_score(importance_type='gain')
            
            # Convert to DataFrame for easier handling
            importance_df = pd.DataFrame({
                'Feature': list(importance_dict.keys()),
                'Importance': list(importance_dict.values())
            })
            
            # Map feature indices (f0, f1, etc.) to actual feature names
            feature_map = {f'f{i}': name for i, name in enumerate(feature_names)}
            importance_df['Feature'] = importance_df['Feature'].map(feature_map)
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return
        
        # Sort by importance and plot
        importance_df = importance_df.sort_values('Importance', ascending=False)
        plt.barh(range(len(importance_df)), importance_df['Importance'], align='center')
        plt.yticks(range(len(importance_df)), importance_df['Feature'])
        plt.xlabel('Importance (gain)')
        plt.ylabel('Feature')
        plt.title('Feature Importance in XGBoost Model')
        plt.tight_layout()
        
        # Save the feature importance plot
        importance_path = 'output/figures/trees/feature_importance.png'
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {importance_path}")
        
        # Create a text file with feature importance rankings
        try:
            importance_txt_path = 'output/figures/trees/feature_importance.txt'
            with open(importance_txt_path, 'w') as f:
                f.write("Feature Importance Rankings (by gain):\n")
                for i, (feature, importance) in enumerate(importance_df.iterrows()):
                    f.write(f"{i+1}. {feature['Feature']}: {importance['Importance']:.4f}\n")
            logger.info(f"Saved feature importance rankings to {importance_txt_path}")
        except Exception as e:
            logger.error(f"Error saving feature importance rankings: {str(e)}")
        
        logger.info("Tree visualization completed successfully")
        
    except Exception as e:
        logger.error(f"Error visualizing tree: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def plot_yearly_indices():
    """
    Create a plot showing yearly indices for krill data:
    1. Survey index based on proportion of krill presence from krillbase.csv
    2. Prediction index based on average probability of presence from the classifier
    3. Catch data shown as dots for available years
    
    Only years with both survey and prediction data are included.
    """
    logger = logging.getLogger(__name__)
    logger.info('Plotting yearly krill indices...')
    
    # Load krillbase data for survey index
    try:
        krillbase = pd.read_csv("input/raw_data/krillbase.csv", encoding='unicode_escape')
        logger.info(f"Successfully loaded krillbase.csv")
    except Exception as e:
        logger.error(f"Failed to load krillbase.csv: {str(e)}")
        return
    
    # Convert date to datetime and extract year
    krillbase['DATE'] = pd.to_datetime(krillbase['DATE'], format='%d/%m/%Y', errors='coerce')
    krillbase['YEAR'] = krillbase['DATE'].dt.year
    
    # Filter out rows with missing dates or krill data
    krillbase = krillbase.dropna(subset=['YEAR', 'STANDARDISED_KRILL_UNDER_1M2'])
    
    # Create presence indicator (1 if krill present, 0 if absent)
    krillbase['KRILL_PRESENT'] = (krillbase['STANDARDISED_KRILL_UNDER_1M2'] > 0).astype(int)
    
    # Calculate proportion of samples with krill presence per year
    presence_counts = krillbase.groupby('YEAR')['KRILL_PRESENT'].sum()
    total_counts = krillbase.groupby('YEAR').size()
    survey_yearly = pd.DataFrame({
        'YEAR': presence_counts.index,
        'PRESENCE_COUNT': presence_counts.values,
        'TOTAL_COUNT': total_counts.values
    })
    
    # Calculate proportion of presence
    survey_yearly['PRESENCE_PROPORTION'] = survey_yearly['PRESENCE_COUNT'] / survey_yearly['TOTAL_COUNT']
    
    # Filter years with sufficient data points (at least 10)
    survey_yearly = survey_yearly[survey_yearly['TOTAL_COUNT'] >= 2]
    survey_yearly = survey_yearly[(survey_yearly['YEAR'] >= 1976) & (survey_yearly['YEAR'] <= 2016)]
    
    # Load fused data for predictions
    try:
        fused_data = pd.read_csv("input/fusedData.csv")
        logger.info(f"Successfully loaded fusedData.csv")
    except Exception as e:
        logger.error(f"Failed to load fusedData.csv: {str(e)}")
        return
    
    # Check if DATE column exists in fused_data
    if 'DATE' not in fused_data.columns:
        logger.info("DATE column not found in fusedData.csv. Trying to match with krillbase data...")
        
        # Try to match with krillbase data to get dates
        # Create a unique identifier based on lat/lon coordinates for joining
        krillbase['COORD_ID'] = krillbase['LONGITUDE'].round(4).astype(str) + '_' + krillbase['LATITUDE'].round(4).astype(str)
        fused_data['COORD_ID'] = fused_data['LONGITUDE'].round(4).astype(str) + '_' + fused_data['LATITUDE'].round(4).astype(str)
        
        # Create a mapping from COORD_ID to YEAR
        coord_to_year = krillbase[['COORD_ID', 'YEAR']].drop_duplicates().set_index('COORD_ID')['YEAR'].to_dict()
        
        # Apply the mapping to fused_data
        fused_data['YEAR'] = fused_data['COORD_ID'].map(coord_to_year)
        
        # Remove rows where YEAR couldn't be mapped
        fused_data = fused_data.dropna(subset=['YEAR'])
        fused_data['YEAR'] = fused_data['YEAR'].astype(int)
        
        logger.info(f"Mapped {len(fused_data)} data points to years")
    
    # Load presence model (we only need the classifier for probability of presence)
    logger.info("Loading presence model...")
    presence_model_path = "output/models/presence_model.json"
    
    if not os.path.exists(presence_model_path):
        logger.error(f"Presence model file not found")
        return
    
    pmod = xgb.XGBClassifier()
    pmod.load_model(presence_model_path)
    
    # Get the exact feature names used by the model to avoid mismatch
    model_features = pmod.feature_names_in_.tolist() if hasattr(pmod, 'feature_names_in_') else []
    
    if not model_features:
        # If feature_names_in_ is not available, try to get from get_booster().feature_names
        try:
            model_features = pmod.get_booster().feature_names
            logger.info(f"Retrieved {len(model_features)} feature names from model booster")
        except Exception as e:
            logger.error(f"Could not get feature names from model: {e}")
            # Use a hardcoded list of features that we know were used in training
            model_features = ['DEPTH', 'CHL_MEAN', 'CHL_MIN', 'CHL_MAX', 'IRON_MEAN', 'IRON_MIN', 'IRON_MAX', 
                             'SSH_MEAN', 'SSH_MIN', 'SSH_MAX', 'VEL_MEAN', 'VEL_MIN', 'VEL_MAX', 
                             'SST_MEAN', 'SST_MIN', 'SST_MAX']
            logger.info(f"Using hardcoded list of {len(model_features)} features")
    
    # Separate features from targets
    feature_cols = [col for col in fused_data.columns if col not in ['KRILL_PRESENCE', 'KRILL_LOG10', 'KRILL_SQRT', 
                                                                     'KRILL_LOGN', 'KRILL_QUAN', 'KRILL_ORIGINAL', 
                                                                     'LONGITUDE', 'LATITUDE', 'YEAR', 'COORD_ID']]
    
    # Filter feature_cols to only include columns that were in the training data
    feature_cols = [col for col in feature_cols if col in model_features]
    
    # Check if we have enough features
    if len(feature_cols) < 5:
        logger.error(f"Not enough feature columns found in fusedData.csv. Found: {feature_cols}")
        return
    
    logger.info(f"Using {len(feature_cols)} features for prediction: {feature_cols[:5]}...")
    
    # Create a DataFrame to store yearly predictions
    prediction_yearly = pd.DataFrame(columns=['YEAR', 'AVG_PRESENCE_PROB', 'SAMPLE_COUNT'])
    
    # Group by year
    for year, year_data in fused_data.groupby('YEAR'):
        logger.info(f"Processing predictions for year {year}")
        
        # Skip if too few data points
        if len(year_data) < 10:
            logger.info(f"Skipping year {year} - too few data points ({len(year_data)})")
            continue
        
        # Extract features
        X_pred = year_data[feature_cols].copy()
        
        # Handle missing values
        X_pred = X_pred.fillna(X_pred.mean())
        
        # Standardize features
        X_pred = (X_pred - X_pred.mean()) / X_pred.std()
        
        # Make predictions - get probability of presence
        presence_probs = pmod.predict_proba(X_pred)[:, 1]
        
        # Calculate average probability of presence for this year
        avg_presence_prob = np.mean(presence_probs)
        
        # Add to yearly predictions DataFrame
        prediction_yearly = pd.concat([prediction_yearly, 
                                      pd.DataFrame({'YEAR': [year], 
                                                   'AVG_PRESENCE_PROB': [avg_presence_prob],
                                                   'SAMPLE_COUNT': [len(year_data)]})],
                                     ignore_index=True)
    
    # Sort by year
    prediction_yearly = prediction_yearly.sort_values('YEAR')
    
    # Apply min-max normalization to the yearly average probabilities
    if len(prediction_yearly) > 1:  # Only normalize if we have more than one year
        min_prob = prediction_yearly['AVG_PRESENCE_PROB'].min()
        max_prob = prediction_yearly['AVG_PRESENCE_PROB'].max()
        prediction_yearly['AVG_PRESENCE_PROB'] = (prediction_yearly['AVG_PRESENCE_PROB'] - min_prob) / (max_prob - min_prob)
    
    # Load catch data
    try:
        catch_data = pd.read_csv("input/subset_data/catchData.csv")
        logger.info(f"Successfully loaded catchData.csv")
        
        # Convert time column to datetime and extract year
        catch_data['time'] = pd.to_datetime(catch_data['time'])
        catch_data['YEAR'] = catch_data['time'].dt.year
        
        # Calculate average catch per year
        catch_yearly = catch_data.groupby('YEAR')['catch'].mean().reset_index()
        
        # Apply log transformation to catch data
        catch_yearly['CATCH_LOG'] = np.log1p(catch_yearly['catch'])
        
        # Normalize the log-transformed values to 0-1 range
        catch_min = catch_yearly['CATCH_LOG'].min()
        catch_max = catch_yearly['CATCH_LOG'].max()
        if catch_max > catch_min:
            catch_yearly['CATCH_INDEX'] = (catch_yearly['CATCH_LOG'] - catch_min) / (catch_max - catch_min)
        else:
            catch_yearly['CATCH_INDEX'] = 0
        
    except Exception as e:
        logger.error(f"Failed to load or process catchData.csv: {str(e)}")
        catch_yearly = None
    
    # Find common years between survey and prediction data
    survey_years = set(survey_yearly['YEAR'])
    prediction_years = set(prediction_yearly['YEAR'])
    common_years = sorted(survey_years.intersection(prediction_years))
    
    if not common_years:
        logger.error("No common years found between survey and prediction data")
        return
    
    logger.info(f"Found {len(common_years)} common years between survey and prediction data")
    
    # Filter data to only include common years
    survey_yearly = survey_yearly[survey_yearly['YEAR'].isin(common_years)]
    prediction_yearly = prediction_yearly[prediction_yearly['YEAR'].isin(common_years)]
    
    if catch_yearly is not None:
        # Only include catch years that are in the common years
        catch_yearly = catch_yearly[catch_yearly['YEAR'].isin(common_years)]
        if catch_yearly.empty:
            catch_yearly = None
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot survey index (proportion of presence)
    plt.plot(survey_yearly['YEAR'], survey_yearly['PRESENCE_PROPORTION'], 'o-', 
             color='blue', linewidth=2, markersize=8, label='Survey Presence Proportion')
    
    # Plot prediction index (average probability of presence)
    plt.plot(prediction_yearly['YEAR'], prediction_yearly['AVG_PRESENCE_PROB'], '--.', 
             color='red', linewidth=1, markersize=8, label='Predicted Presence Probability (normalized)')
    
    # Plot catch index as dots if available
    if catch_yearly is not None and not catch_yearly.empty:
        plt.scatter(catch_yearly['YEAR'], catch_yearly['CATCH_INDEX'], 
                  color='green', s=100, marker='d', label='Catch Index (log transform)')
    
    # Add labels and title
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Probability / Proportion / Index', fontsize=14)
    plt.title('Yearly Krill Presence Comparison', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Set y-axis limits
    plt.ylim(-0.05, 1.05)
    
    # Adjust x-axis to show years clearly
    if common_years:
        min_year = min(common_years)
        max_year = max(common_years)
        plt.xticks(np.arange(min_year - 1, max_year + 2, 5), rotation=45)
    
    # Save the figure
    os.makedirs("output/figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig("output/figures/yearly_krill_indices.png", dpi=300)
    logger.info("Saved figure to output/figures/yearly_krill_indices.png")
    
    # Show the plot
    plt.show()
    
    return


if __name__ == '__main__':
    main()