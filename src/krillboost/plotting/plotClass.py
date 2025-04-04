import logging
import argparse
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from scipy.stats import gaussian_kde, norm
from sklearn.metrics import mean_squared_error
from scipy import interpolate
logging.basicConfig(level=logging.INFO)

def plotQQ():
    """
    Create a scatter plot comparing predicted abundance vs. observed abundance for krill.
    
    The plot includes a 1:1 reference line, density-based coloring, and performance metrics (R² and RMSE).
    Similar to the plot_performance method in krillPredict.py.
    
    Only includes abundance values within the range of -2.0 to 2.0, matching the training range.
    """
    logger = logging.getLogger(__name__)
    logger.info('Plotting predicted vs. observed abundance...')
    
    # Load models
    presence_model_path = "output/models/presence_model.json"
    abundance_model_path = "output/models/abundance_model.json"
    
    # Load data
    krillData = pd.read_csv("input/fusedData.csv")
    
    # Remove unnamed columns if they exist
    unnamed_cols = [col for col in krillData.columns if 'Unnamed' in col]
    if unnamed_cols:
        logger.info(f"Removing unnamed columns: {unnamed_cols}")
        krillData = krillData.drop(columns=unnamed_cols)
    
    # Separate features and targets
    X = krillData.drop(columns=['KRILL_PRESENCE', 'KRILL_LOG10', 'KRILL_SQRT', 'KRILL_LOGN', 'KRILL_QUAN', 'KRILL_ORIGINAL'])
    y_presence = krillData['KRILL_PRESENCE']
    y_abundance = krillData['KRILL_ORIGINAL']
    
    # Feature scaling for better model performance
    X = (X - X.mean()) / X.std()
    
    # Load models
    pmod = xgb.XGBClassifier()
    pmod.load_model(presence_model_path)
    amod = xgb.XGBRegressor()
    amod.load_model(abundance_model_path)
    
    # Generate predictions
    presence_pred_prob = pmod.predict_proba(X)[:, 1]  # Use probability instead of binary prediction
    abundance_pred = amod.predict(X)
    
    # Filter to only include observations where krill is present
    # This focuses on the samples where we actually have abundance data
    present_mask = y_presence == 1
    obs_present = y_abundance[present_mask]
    abundance_pred_present = abundance_pred[present_mask]
    
    # Filter to only include abundance values within -0.5 to 20.0 range
    # This matches the range used for training the abundance model
    logger.info("Filtering to focus on abundance values within -0.5 to 20.0 range")
    total_samples = len(obs_present)
    in_range_mask = (obs_present >= np.percentile(obs_present,20)) & (obs_present <= np.percentile(obs_present,80))
    in_range_count = in_range_mask.sum()
    out_range_count = total_samples - in_range_count
    logger.info(f"Total samples with krill present: {total_samples}")
    logger.info(f"Samples within range [{np.percentile(obs_present,20):.3f}, {np.percentile(obs_present,80):.3f}]: {in_range_count} ({in_range_count/total_samples*100:.2f}%)")
    logger.info(f"Samples outside range: {out_range_count} ({out_range_count/total_samples*100:.2f}%)")
    
    # Apply the range filter
    obs_filtered = obs_present[in_range_mask]
    pred_filtered = abundance_pred_present[in_range_mask]
    
    # Log the original and filtered ranges
    logger.info(f"Original abundance range: [{obs_present.min():.3f}, {obs_present.max():.3f}]")
    logger.info(f"Filtered abundance range: [{obs_filtered.min():.3f}, {obs_filtered.max():.3f}]")
    logger.info(f"Filtered prediction range: [{pred_filtered.min():.3f}, {pred_filtered.max():.3f}]")
    
    # Convert to numpy arrays
    y_true_np = obs_filtered.to_numpy()
    y_pred_np = pred_filtered
    
    # Log data statistics
    logger.info(f"Data statistics for filtered data:")
    logger.info(f"  Number of points: {len(y_true_np)}")
    
    # Calculate performance metrics
    r2 = np.corrcoef(y_true_np, y_pred_np)[0,1]**2
    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    
    logger.info(f"Model Performance Metrics (filtered data):")
    logger.info(f"  R²: {r2:.3f}")
    logger.info(f"  RMSE: {rmse:.3f}")
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Calculate point density for coloring
    xy = np.vstack([y_true_np, y_pred_np])
    z = gaussian_kde(xy)(xy)
    
    # Sort points by density for better visualization
    idx = z.argsort()
    x, y, z = y_true_np[idx], y_pred_np[idx], z[idx]
    
    # Create scatter plot colored by density
    scatter = plt.scatter(x, y, c=z, s=50, alpha=0.5, cmap='inferno')
    plt.colorbar(scatter, label='Point density')
    
    # Add perfect prediction line
    min_val = min(min(y_true_np), min(y_pred_np))
    max_val = max(max(y_true_np), max(y_pred_np))
    
    # Plot 1:1 line and add empty line for stats
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1')
    plt.plot([], [], ' ', label=f'R² = {r2:.3f}\nRMSE = {rmse:.3f}')
    
    # Add labels and title
    plt.xlabel('Observed Log10 Abundance (Range: -2.0 to 2.0)', fontsize=14)
    plt.ylabel('Predicted Abundance', fontsize=14)
    plt.title('Predicted vs Observed Krill Abundance (Filtered Range)', fontsize=16)
    plt.legend(fontsize=12)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Equal aspect ratio
    plt.axis('equal')
    
    # Set axis limits to focus on the filtered range
    plt.xlim(np.percentile(obs_present,20), np.percentile(obs_present,80))
    plt.ylim(np.percentile(obs_present,20), np.percentile(obs_present,80))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs('output/figures', exist_ok=True)
    plt_path = 'output/figures/abundance_performance_filtered.png'
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved filtered abundance performance plot to: {plt_path}")
    
    # Close the figure to free memory
    plt.close()
    
    return

def plot_confusion_matrix():
    """
    Create a confusion matrix for the presence/absence model to show classification performance.
    """
    logger = logging.getLogger(__name__)
    logger.info('Plotting confusion matrix for presence/absence model...')
    
    # Load model
    presence_model_path = "output/models/presence_model.json"
    
    # Load data
    krillData = pd.read_csv("input/fusedData.csv")
    
    # Remove unnamed columns if they exist
    unnamed_cols = [col for col in krillData.columns if 'Unnamed' in col]
    if unnamed_cols:
        logger.info(f"Removing unnamed columns: {unnamed_cols}")
        krillData = krillData.drop(columns=unnamed_cols)
    
    # Separate features and targets
    X = krillData.drop(columns=['KRILL_PRESENCE', 'KRILL_LOG10'])
    y_presence = krillData['KRILL_PRESENCE']
    
    # Feature scaling for better model performance
    X = (X - X.mean()) / X.std()
    
    # Load model
    pmod = xgb.XGBClassifier()
    pmod.load_model(presence_model_path)
    
    # Generate predictions
    y_pred_proba = pmod.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_presence, y_pred)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Absent', 'Present'],
                yticklabels=['Absent', 'Present'])
    
    # Add labels and title
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.title('Krill Presence/Absence Confusion Matrix', fontsize=16)
    
    # Save figure
    os.makedirs('output/figures', exist_ok=True)
    plt_path = 'output/figures/confusion_matrix.png'
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved confusion matrix to: {plt_path}")
    
    # Close the figure to free memory
    plt.close()
    
    return

def plot_roc_curve():
    """
    Create a ROC curve for the presence/absence model to show the tradeoff 
    between true positive rate and false positive rate at different thresholds.
    """
    logger = logging.getLogger(__name__)
    logger.info('Plotting ROC curve for presence/absence model...')
    
    # Load model
    presence_model_path = "output/models/presence_model.json"
    
    # Load data
    krillData = pd.read_csv("input/fusedData.csv")
    
    # Remove unnamed columns if they exist
    unnamed_cols = [col for col in krillData.columns if 'Unnamed' in col]
    if unnamed_cols:
        logger.info(f"Removing unnamed columns: {unnamed_cols}")
        krillData = krillData.drop(columns=unnamed_cols)
    
    # Separate features and targets
    X = krillData.drop(columns=['KRILL_PRESENCE', 'KRILL_LOG10', 'KRILL_SQRT', 'KRILL_LOGN', 'KRILL_QUAN', 'KRILL_ORIGINAL', 'LATITUDE', 'LONGITUDE'])
    y_presence = krillData['KRILL_PRESENCE']
    
    # Feature scaling for better model performance
    X = (X - X.mean()) / X.std()
    
    # Load model
    pmod = xgb.XGBClassifier()
    pmod.load_model(presence_model_path)
    
    # Generate predictions
    y_pred_proba = pmod.predict_proba(X)[:, 1]
    
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_presence, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.3f})')
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Add labels and title
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    
    # Set axis limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs('output/figures', exist_ok=True)
    plt_path = 'output/figures/roc_curve.png'
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved ROC curve to: {plt_path}")
    
    # Close the figure to free memory
    plt.close()
    
    return

def plot_precision_recall_curve():
    """
    Create a precision-recall curve for the presence/absence model.
    This is particularly useful for imbalanced datasets.
    """
    logger = logging.getLogger(__name__)
    logger.info('Plotting precision-recall curve for presence/absence model...')
    
    # Load model
    presence_model_path = "output/models/presence_model.json"
    
    # Load data
    krillData = pd.read_csv("input/fusedData.csv")
    
    # Remove unnamed columns if they exist
    unnamed_cols = [col for col in krillData.columns if 'Unnamed' in col]
    if unnamed_cols:
        logger.info(f"Removing unnamed columns: {unnamed_cols}")
        krillData = krillData.drop(columns=unnamed_cols)
    
    # Separate features and targets
    X = krillData.drop(columns=['KRILL_PRESENCE', 'KRILL_LOG10'])
    y_presence = krillData['KRILL_PRESENCE']
    
    # Feature scaling for better model performance
    X = (X - X.mean()) / X.std()
    
    # Load model
    pmod = xgb.XGBClassifier()
    pmod.load_model(presence_model_path)
    
    # Generate predictions
    y_pred_proba = pmod.predict_proba(X)[:, 1]
    
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_presence, y_pred_proba)
    average_precision = average_precision_score(y_presence, y_pred_proba)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot precision-recall curve
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {average_precision:.3f})')
    
    # Add labels and title
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.legend(loc="best", fontsize=12)
    
    # Set axis limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs('output/figures', exist_ok=True)
    plt_path = 'output/figures/precision_recall_curve.png'
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved precision-recall curve to: {plt_path}")
    
    # Close the figure to free memory
    plt.close()
    
    return

def plot_probability_distribution():
    """
    Plot the distribution of predicted probabilities for the presence/absence model,
    separated by actual class (present/absent).
    """
    logger = logging.getLogger(__name__)
    logger.info('Plotting probability distribution for presence/absence model...')
    
    # Load model
    presence_model_path = "output/models/presence_model.json"
    
    # Load data
    krillData = pd.read_csv("input/fusedData.csv")
    
    # Remove unnamed columns if they exist
    unnamed_cols = [col for col in krillData.columns if 'Unnamed' in col]
    if unnamed_cols:
        logger.info(f"Removing unnamed columns: {unnamed_cols}")
        krillData = krillData.drop(columns=unnamed_cols)
    
    # Separate features and targets
    X = krillData.drop(columns=['KRILL_PRESENCE', 'KRILL_LOG10'])
    y_presence = krillData['KRILL_PRESENCE']
    
    # Feature scaling for better model performance
    X = (X - X.mean()) / X.std()
    
    # Load model
    pmod = xgb.XGBClassifier()
    pmod.load_model(presence_model_path)
    
    # Generate predictions
    y_pred_proba = pmod.predict_proba(X)[:, 1]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot histograms of predicted probabilities for each class
    plt.hist(y_pred_proba[y_presence == 0], bins=20, alpha=0.5, color='red', 
             label='Actual: Absent', density=True)
    plt.hist(y_pred_proba[y_presence == 1], bins=20, alpha=0.5, color='blue', 
             label='Actual: Present', density=True)
    
    # Add vertical line at threshold 0.5
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    
    # Add labels and title
    plt.xlabel('Predicted Probability of Presence', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Distribution of Predicted Probabilities by Actual Class', fontsize=16)
    plt.legend(fontsize=12)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs('output/figures', exist_ok=True)
    plt_path = 'output/figures/probability_distribution.png'
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved probability distribution plot to: {plt_path}")
    
    # Close the figure to free memory
    plt.close()
    
    return

def plot_spatial_predictions():
    """
    Create a spatial heatmap of predicted probabilities binned into grid cells,
    showing both the number of samples and the average predicted probability in each cell.
    Style matches envData.png with bathymetry contours.
    """
    logger = logging.getLogger(__name__)
    logger.info('Plotting spatial predictions heatmap...')
    
    # Load model
    presence_model_path = "output/models/presence_model.json"
    
    # Load data
    krillData = pd.read_csv("input/fusedData.csv")
    
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
    except Exception as e:
        logger.warning(f"Could not load bathymetry data: {str(e)}")
        has_bathymetry = False
    
    # Remove unnamed columns if they exist
    unnamed_cols = [col for col in krillData.columns if 'Unnamed' in col]
    if unnamed_cols:
        logger.info(f"Removing unnamed columns: {unnamed_cols}")
        krillData = krillData.drop(columns=unnamed_cols)
    
    # Check if latitude and longitude columns exist
    if 'LATITUDE' not in krillData.columns or 'LONGITUDE' not in krillData.columns:
        logger.warning("Latitude and/or longitude columns not found. Skipping spatial prediction plot.")
        return
    
    # Separate features and targets
    X = krillData.drop(columns=['KRILL_PRESENCE', 'KRILL_LOG10'])
    y_presence = krillData['KRILL_PRESENCE']
    
    # Feature scaling for better model performance
    X = (X - X.mean()) / X.std()
    
    # Load model
    pmod = xgb.XGBClassifier()
    pmod.load_model(presence_model_path)
    
    # Generate predictions
    y_pred_proba = pmod.predict_proba(X)[:, 1]
    
    # Add predictions to the dataframe
    krillData['PREDICTED_PROBABILITY'] = y_pred_proba
    
    # Define grid bounds and resolution
    lonBounds = [krillData['LONGITUDE'].min() - 1, krillData['LONGITUDE'].max() + 1]
    latBounds = [krillData['LATITUDE'].min() - 1, krillData['LATITUDE'].max() + 1]
    grid_res = 2.0  # 2-degree grid cells
    
    # Create grid
    lon_grid = np.arange(lonBounds[0], lonBounds[1] + grid_res, grid_res)
    lat_grid = np.arange(latBounds[0], latBounds[1] + grid_res, grid_res)
    
    # Initialize arrays to store prediction probabilities and sample counts
    prob_sums = np.zeros((len(lat_grid)-1, len(lon_grid)-1))
    sample_counts = np.zeros((len(lat_grid)-1, len(lon_grid)-1))
    
    # Calculate average prediction probability for each grid cell
    for i in range(len(lat_grid)-1):
        for j in range(len(lon_grid)-1):
            # Find points in this grid cell
            mask = ((krillData['LONGITUDE'] >= lon_grid[j]) & 
                    (krillData['LONGITUDE'] < lon_grid[j+1]) & 
                    (krillData['LATITUDE'] >= lat_grid[i]) & 
                    (krillData['LATITUDE'] < lat_grid[i+1]))
            
            # Count samples and sum probabilities in this cell
            points_in_cell = mask.sum()
            if points_in_cell > 0:
                sample_counts[i, j] = points_in_cell
                prob_sums[i, j] = krillData.loc[mask, 'PREDICTED_PROBABILITY'].sum()
    
    # Calculate average probability (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_probability = np.where(sample_counts > 0, prob_sums / sample_counts, np.nan)
    
    # Set up figure
    plt.rcParams.update({'font.size': 20})  # Set default font size to 20
    fig = plt.figure(figsize=(14, 12))
    
    # Create map with PlateCarree projection (matching envData.png)
    projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    
    # Plot the colormesh
    mesh = ax.pcolormesh(
        lon_grid, lat_grid, avg_probability, 
        transform=projection,
        cmap=plt.get_cmap('YlOrRd'), 
        vmin=0, vmax=1,
        shading='flat',
        zorder=1
    )
    
    # Add bathymetry contours if available
    if has_bathymetry:
        cs = ax.contour(bath.lon, bath.lat, 
                       masked_bathymetry, levels=contour_levels,
                       colors='grey', alpha=0.3, transform=projection,
                       zorder=2)
    
    # Add land and coastlines
    ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=100)
    ax.coastlines(linewidth=1.5, zorder=101)
    
    # Add colorbar
    cbar = plt.colorbar(mesh, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Average Predicted Probability', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    
    # Add a scatter plot showing sample density
    # Only show cells with data
    valid_cells = ~np.isnan(avg_probability)
    if np.any(valid_cells):
        # Convert grid edges to centers for scatter plot
        lon_centers = lon_grid[:-1] + grid_res/2
        lat_centers = lat_grid[:-1] + grid_res/2
        
        # Create meshgrid for scatter
        lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
        
        # Scale the sizes based on log of count to prevent extremely large points
        sizes = np.log1p(sample_counts[valid_cells]) * 30
        scatter = ax.scatter(
            lon_mesh[valid_cells], 
            lat_mesh[valid_cells],
            s=sizes,
            c='black', 
            alpha=0.3,
            transform=projection,
            edgecolor='white',
            linewidth=0.5,
            zorder=102
        )
        
        # Add legend for sample count
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', alpha=0.3,
                   markersize=8, label='10 samples'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', alpha=0.3,
                   markersize=12, label='100 samples'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', alpha=0.3,
                   markersize=16, label='1000 samples')
        ]
        legend = ax.legend(handles=handles, loc='upper left', fontsize=14, title='Sample Count', 
                title_fontsize=16)
        legend.set_zorder(103)
    
    # Set map bounds to focus on regions of interest
    ax.set_extent(lonBounds + latBounds, crs=projection)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Add labels
    ax.set_xlabel('Longitude', fontsize=20)
    ax.set_ylabel('Latitude', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    plt.title('Spatial Distribution of Predicted Krill Presence Probability', fontsize=22)
    
    # Save figure
    os.makedirs('output/figures', exist_ok=True)
    plt_path = 'output/figures/spatial_predictions_heatmap.png'
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved spatial prediction heatmap to: {plt_path}")
    
    # Close the figure to free memory
    plt.close()
    
    return

def plot_calibration_curve():
    """
    Create a calibration curve to assess how well the predicted probabilities
    match the actual frequencies of positive outcomes.
    """
    logger = logging.getLogger(__name__)
    logger.info('Plotting calibration curve for presence/absence model...')
    
    # Load model
    presence_model_path = "output/models/presence_model.json"
    
    # Load data
    krillData = pd.read_csv("input/fusedData.csv")
    
    # Remove unnamed columns if they exist
    unnamed_cols = [col for col in krillData.columns if 'Unnamed' in col]
    if unnamed_cols:
        logger.info(f"Removing unnamed columns: {unnamed_cols}")
        krillData = krillData.drop(columns=unnamed_cols)
    
    # Separate features and targets
    X = krillData.drop(columns=['KRILL_PRESENCE', 'KRILL_LOG10'])
    y_presence = krillData['KRILL_PRESENCE']
    
    # Feature scaling for better model performance
    X = (X - X.mean()) / X.std()
    
    # Load model
    pmod = xgb.XGBClassifier()
    pmod.load_model(presence_model_path)
    
    # Generate predictions
    y_pred_proba = pmod.predict_proba(X)[:, 1]
    
    # Create bins for calibration curve
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Ensure indices are within range
    
    # Calculate fraction of positives and mean predicted probability in each bin
    bin_sums = np.bincount(bin_indices, weights=y_presence, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_counts = np.where(bin_counts == 0, 1, bin_counts)  # Avoid division by zero
    fraction_of_positives = bin_sums / bin_counts
    
    mean_predicted_proba = np.bincount(bin_indices, weights=y_pred_proba, minlength=n_bins) / bin_counts
    
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Plot calibration curve
    plt.plot(mean_predicted_proba, fraction_of_positives, 's-', color='blue', 
             markersize=8, label='Calibration curve')
    
    # Plot diagonal perfect calibration line
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
    
    # Add labels and title
    plt.xlabel('Mean Predicted Probability', fontsize=14)
    plt.ylabel('Fraction of Positives', fontsize=14)
    plt.title('Calibration Curve (Reliability Diagram)', fontsize=16)
    plt.legend(fontsize=12)
    
    # Set axis limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    
    # Make the plot square
    plt.axis('square')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs('output/figures', exist_ok=True)
    plt_path = 'output/figures/calibration_curve.png'
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved calibration curve to: {plt_path}")
    
    # Close the figure to free memory
    plt.close()
    
    return

def plot_feature_importance():
    """
    Create a figure showing the most significant features for predicting krill presence.
    
    This function loads the trained presence model, extracts feature importance scores,
    and creates a horizontal bar chart of the top features.
    """
    logger = logging.getLogger(__name__)
    logger.info('Plotting feature importance for krill presence prediction...')
    
    # Load model
    presence_model_path = "output/models/presence_model.json"
    
    # Load data to get feature names
    krillData = pd.read_csv("input/fusedData.csv")
    
    # Remove unnamed columns if they exist
    unnamed_cols = [col for col in krillData.columns if 'Unnamed' in col]
    if unnamed_cols:
        logger.info(f"Removing unnamed columns: {unnamed_cols}")
        krillData = krillData.drop(columns=unnamed_cols)
    
    # Get feature names (excluding target columns)
    feature_names = [col for col in krillData.columns if col not in ['KRILL_PRESENCE', 'KRILL_LOG10', 'KRILL_SQRT', 'KRILL_LOGN', 'KRILL_QUAN', 'KRILL_ORIGINAL', 'LATITUDE', 'LONGITUDE', 'DATE', 'YEAR', 'MONTH', 'DAY']]
    
    # Load model
    pmod = xgb.XGBClassifier()
    pmod.load_model(presence_model_path)
    
    # Get feature importance directly using feature_importances_ attribute
    try:
        # Try using the feature_importances_ attribute first
        importances = pmod.feature_importances_
        importance_df = pd.DataFrame({
            'Feature_Name': feature_names,
            'Importance': importances
        })
    except (AttributeError, ValueError) as e:
        logger.warning(f"Could not get feature_importances_ directly: {e}")
        # Fall back to get_score method
        importance_dict = pmod.get_booster().get_score(importance_type='gain')
        
        # Convert to DataFrame for easier handling
        importance_df = pd.DataFrame({
            'Feature': list(importance_dict.keys()),
            'Importance': list(importance_dict.values())
        })
        
        # Map feature indices (f0, f1, etc.) to actual feature names
        feature_map = {f'f{i}': name for i, name in enumerate(feature_names)}
        importance_df['Feature_Name'] = importance_df['Feature'].map(feature_map)
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Take top 10 features
    top_features = importance_df.head(10).copy()
    
    # Normalize importance values for better visualization
    total_importance = top_features['Importance'].sum()
    top_features['Normalized_Importance'] = top_features['Importance'] / total_importance * 100
    
    # Create figure with adjusted size
    plt.figure(figsize=(10, 6))
    
    # Truncate long feature names safely
    short_names = []
    for name in top_features['Feature_Name']:
        if isinstance(name, str) and len(name) > 25:
            short_names.append(name[:25] + '...')
        else:
            short_names.append(str(name))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(short_names))
    bars = plt.barh(y_pos, top_features['Normalized_Importance'].values, color='teal')
    
    # Set y-tick labels to the shortened feature names
    plt.yticks(y_pos, short_names)
    
    # Add values to the end of each bar (percentage format)
    for i, value in enumerate(top_features['Normalized_Importance']):
        plt.text(value + 0.5, i, f'{value:.1f}%', va='center')
    
    # Add labels and title
    plt.xlabel('Relative Importance (%)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    #plt.title('Top 10 Features for Predicting Krill Presence', fontsize=14)
    
    # Add grid lines for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Ensure directory exists before saving
    os.makedirs('output/figures', exist_ok=True)
    
    try:
        # Save figure with adjusted parameters
        plt_path = 'output/figures/presence_feature_importance.png'
        plt.savefig(plt_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        logger.info(f"Saved feature importance plot to: {plt_path}")
    except Exception as e:
        logger.error(f"Error saving figure: {e}")
    
    # Close the figure to free memory
    plt.close()
    
    return

def plot_presence_vs_abundance():
    """
    Create a Q-Q plot comparing the quantiles of continuous probability predictions from the presence model
    against the quantiles of log10 abundance values.
    
    This visualization helps assess how well the distributions of presence probabilities and abundance values align,
    which is useful for understanding the relationship between the two-step modeling approach:
    1. Presence/absence prediction (probability)
    2. Abundance prediction (for positive cases)
    """
    logger = logging.getLogger(__name__)
    logger.info('Creating Q-Q plot of presence probability vs. log10 abundance...')
    
    # Load model
    presence_model_path = "output/models/presence_model.json"
    
    # Load data
    krillData = pd.read_csv("input/fusedData.csv")
    
    # Remove unnamed columns if they exist
    unnamed_cols = [col for col in krillData.columns if 'Unnamed' in col]
    if unnamed_cols:
        logger.info(f"Removing unnamed columns: {unnamed_cols}")
        krillData = krillData.drop(columns=unnamed_cols)
    
    # Separate features and targets
    X = krillData.drop(columns=['KRILL_PRESENCE', 'KRILL_LOG10', 'KRILL_SQRT', 'KRILL_LOGN', 'KRILL_QUAN', 'KRILL_ORIGINAL'])
    y_presence = krillData['KRILL_PRESENCE']
    y_log10 = krillData['KRILL_LOG10']
    
    # Feature scaling for better model performance
    X = (X - X.mean()) / X.std()
    
    # Load model
    pmod = xgb.XGBClassifier()
    pmod.load_model(presence_model_path)
    
    # Generate probability predictions
    presence_prob = pmod.predict_proba(X)[:, 1]  # Probability of class 1 (presence)
    
    # Filter to only include observations where krill is present (for log10 values)
    present_mask = y_presence == 1
    log10_values = y_log10[present_mask]
    prob_values_present = presence_prob[present_mask]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Calculate quantiles for both distributions
    # We need to normalize the log10 values to be between 0 and 1 for fair comparison with probabilities
    log10_min = log10_values.min()
    log10_max = log10_values.max()
    log10_normalized = (log10_values - log10_min) / (log10_max - log10_min)
    
    # Sort both arrays
    log10_sorted = np.sort(log10_normalized)
    prob_sorted = np.sort(prob_values_present)
    
    # Generate theoretical quantiles (using percentiles)
    quantiles = np.linspace(0, 1, len(log10_sorted))
    
    # Create Q-Q plot
    plt.scatter(log10_sorted, prob_sorted, alpha=0.6, color='teal', s=30)
    
    # Add reference line (y=x)
    min_val = min(log10_sorted.min(), prob_sorted.min())
    max_val = max(log10_sorted.max(), prob_sorted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Reference Line (y=x)')
    
    # Calculate correlation coefficient
    corr = np.corrcoef(log10_sorted, prob_sorted)[0, 1]
    logger.info(f"Correlation between quantiles: {corr:.3f}")
    
    # Add labels and title
    plt.xlabel('Normalized Log10 Abundance Quantiles', fontsize=14)
    plt.ylabel('Presence Probability Quantiles', fontsize=14)
    plt.title('Q-Q Plot: Presence Probability vs. Log10 Abundance', fontsize=16)
    
    # Add correlation annotation
    plt.annotate(f'Correlation: {corr:.3f}', 
                 xy=(0.05, 0.95), 
                 xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                 fontsize=12)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    # Set equal aspect ratio for better visualization
    plt.axis('equal')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs('output/figures', exist_ok=True)
    plt_path = 'output/figures/presence_probability_qq_plot.png'
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved Q-Q plot to: {plt_path}")
    
    # Close the figure to free memory
    plt.close()
    
    return

def main():
    """Main function to run all plotting methods."""
    parser = argparse.ArgumentParser(description='Generate plots for krill prediction models.')
    parser.add_argument('plot', type=str, default='all', help='Plot to generate (all, qq, confusion, roc, pr, prob, spatial, calibration, feature_importance, presence_abundance)')
    args = parser.parse_args()
    
    if args.plot == 'all' or args.plot == 'qq':
        plotQQ()
    if args.plot == 'all' or args.plot == 'confusion':
        plot_confusion_matrix()
    if args.plot == 'all' or args.plot == 'roc':
        plot_roc_curve()
    if args.plot == 'all' or args.plot == 'pr':
        plot_precision_recall_curve()
    if args.plot == 'all' or args.plot == 'prob':
        plot_probability_distribution()
    if args.plot == 'all' or args.plot == 'spatial':
        plot_spatial_predictions()
    if args.plot == 'all' or args.plot == 'calibration':
        plot_calibration_curve()
    if args.plot == 'all' or args.plot == 'feature_importance':
        plot_feature_importance()
    if args.plot == 'all' or args.plot == 'presence_abundance':
        plot_presence_vs_abundance()
    
    return

if __name__ == '__main__':
    main()
