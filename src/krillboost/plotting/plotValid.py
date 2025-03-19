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
    parser.add_argument('plot', type=str, default='all', help='Plot to generate validation against real data (all, catch, response, multiyear)')
    args = parser.parse_args()

    if args.plot == 'all' or args.plot == 'catch':
        plotCatch()
    if args.plot == 'all' or args.plot == 'response':
        plot_response_curves()
    if args.plot == 'all' or args.plot == 'multiyear':
        plot_multiyear_predictions()
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
    
    # Load krill data
    try:
        krillData = pd.read_csv("input/fusedData.csv", encoding='latin1')
        logger.info(f"Successfully loaded fusedData.csv with latin1 encoding")
    except UnicodeDecodeError:
        try:
            krillData = pd.read_csv("input/fusedData.csv", encoding='cp1252')
            logger.info(f"Successfully loaded fusedData.csv with cp1252 encoding")
        except Exception as e:
            logger.error(f"Failed to load krillbase.csv: {str(e)}")
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
    
    # Adjust layout
    plt.tight_layout()
    
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
    Create a multi-year plot of krill presence predictions for years 2011-2016.
    
    This function:
    1. Processes environmental data for years 2011, 2012, 2013, 2014, 2015, and 2016
    2. Makes predictions using the trained presence model for each year
    3. Creates a 6-panel figure showing predicted probability values
    4. Overlays krill presence (dots) and absence (x) points from the raw dataset for each year
    """
    logger = logging.getLogger(__name__)
    logger.info('Creating multi-year krill presence predictions plot...')
    
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
    
    # Set up figure with parameters matching envData.png
    plt.rcParams.update({'font.size': 20})  # Set default font size to 20
    fig = plt.figure(figsize=(14, 12))
    
    # Create a 3x2 grid for subplots
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)
    axes = []
    
    # Create subplots with PlateCarree projection
    for i in range(6):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        axes.append(ax)
    
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
    
    # Create a custom colormap for krill presence probability
    cmap = plt.get_cmap('Reds')
    
    # Process each year
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
        
        # Create DataFrame for prediction
        X_pred = valid_features.copy()
        
        # Standardize features (using same approach as in training)
        X_pred = (X_pred - X_pred.mean()) / X_pred.std()
        
        # Make predictions
        logger.info(f"Predicting krill presence for {year}...")
        presence_probs = pmod.predict_proba(X_pred)[:, 1]  # Probability of presence
        
        # Create a grid for plotting
        presence_grid = np.full((len(lats), len(lons)), np.nan)
        
        # Map the predictions back to the grid
        for j, (idx, row) in enumerate(valid_features.iterrows()):
            lat_idx = np.where(lats == row['LATITUDE'])[0][0]
            lon_idx = np.where(lons == row['LONGITUDE'])[0][0]
            presence_grid[lat_idx, lon_idx] = presence_probs[j]
        
        # Get the current subplot
        ax = axes[i]
        
        # Set extent to focus on Antarctic Peninsula region using dynamic bounds
        ax.set_extent(lonBounds + latBounds, crs=ccrs.PlateCarree())
        
        # Add coastlines and land features
        ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=100)
        ax.coastlines(linewidth=1.5, zorder=101)
        
        # Plot the prediction grid
        im = ax.pcolormesh(
            lon_grid, lat_grid, presence_grid, 
            transform=ccrs.PlateCarree(),
            cmap=cmap, 
            vmin=0, vmax=0.8,
            shading='auto',
            zorder=1
        )
        
        # Krill presence points plotting is turned off as requested
        
        # Add gridlines (simplified for subplots)
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}
        
        # Add title for this subplot - just the year
        ax.set_title(f'{year}', fontsize=20)
        
        # Only add axis labels for edge subplots
        if i >= 4:  # Bottom row
            ax.set_xlabel('Longitude', fontsize=14)
        if i % 2 == 0:  # Left column
            ax.set_ylabel('Latitude', fontsize=14)
    
    # Add a colorbar that applies to all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Predicted Krill Presence Probability', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    
    # No legend since we're not showing krill presence points
    
    # Add overall title
    plt.suptitle('Predicted Krill Presence Probability (2011-2016)', fontsize=24, y=0.98)
    
    # Save figure with higher resolution
    os.makedirs('output/figures', exist_ok=True)
    plt_path = 'output/figures/multiyear_krill_predictions.png'
    plt.savefig(plt_path, dpi=600, bbox_inches='tight')
    logger.info(f"Saved high-resolution multi-year krill predictions plot to: {plt_path}")
    
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