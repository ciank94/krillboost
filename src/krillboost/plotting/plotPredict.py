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
logging.basicConfig(level=logging.INFO)

def plotQQ():
    """
    Create a Q-Q plot comparing conditional abundance predictions (presence * abundance) 
    vs. observations, with adjustments to improve the relationship.
    
    The plot includes a 1:1 reference line to evaluate model performance.
    """
    logger = logging.getLogger(__name__)
    logger.info('Plotting Q-Q plot for conditional abundance predictions...')
    
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
    X = krillData.drop(columns=['KRILL_PRESENCE', 'KRILL_LOG10'])
    y_presence = krillData['KRILL_PRESENCE']
    y_abundance = krillData['KRILL_LOG10']
    
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
    
    # Create conditional predictions (presence_prob * abundance)
    # This creates a smoother transition than binary presence * abundance
    conditional_pred = presence_pred_prob * abundance_pred
    
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Filter to only include observations where krill is present
    # This focuses on the samples where we actually have abundance data
    present_mask = y_presence == 1
    obs_present = y_abundance[present_mask]
    cond_pred_present = conditional_pred[present_mask]
    
    # Calculate quantiles
    quantiles = np.linspace(0, 1, 100)  # Deciles  # 101 points from 0 to 100
    obs_quantiles = np.quantile(obs_present, quantiles)
    pred_quantiles = np.quantile(cond_pred_present, quantiles)
    
    # Plot quantiles with larger markers and improved styling
    plt.plot(obs_quantiles, pred_quantiles, 'o', 
             color='#1f77b4', alpha=0.8, markersize=6, 
             label='Conditional Abundance')

    # Add 1:1 reference line
    min_val = min(obs_quantiles.min(), pred_quantiles.min())
    max_val = max(obs_quantiles.max(), pred_quantiles.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='1:1 Line')
    
    # Add labels and title
    plt.xlabel('Observed Log10 Abundance (Present Only)', fontsize=14)
    plt.ylabel('Predicted Conditional Abundance', fontsize=14)
    plt.title('Calibrated Two-Step Model Q-Q Plot', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Make the plot square to better visualize the 1:1 relationship
    plt.axis('square')
    
    # Add R² value to the plot
    # from scipy import stats
    # slope, intercept, r_value, p_value, std_err = stats.linregress(obs_quantiles, pred_quantiles)
    # r_squared = r_value**2
    # plt.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=plt.gca().transAxes, 
    #          fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs('output/figures', exist_ok=True)
    plt_path = 'output/figures/conditionalQQ.png'
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved conditional Q-Q plot to: {plt_path}")
    
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
    X = krillData.drop(columns=['KRILL_PRESENCE', 'KRILL_LOG10'])
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

def main():
    parser = argparse.ArgumentParser(description='Plot predicted krill presence and abundance.')
    parser.add_argument('--figure', type=str, 
                        choices=['fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6', 'fig7', 'all'], 
                        default='all', help='Select which figure to plot: \n'
                                           'fig1: Q-Q plot for conditional abundance\n'
                                           'fig2: Confusion matrix\n'
                                           'fig3: ROC curve\n'
                                           'fig4: Precision-recall curve\n'
                                           'fig5: Probability distribution\n'
                                           'fig6: Calibration curve\n'
                                           'fig7: Spatial predictions\n'
                                           'all: Generate all plots (default)')
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    
    # Plot the selected figure(s)
    if args.figure == 'fig1' or args.figure == 'all':
        plotQQ()
    
    if args.figure == 'fig2' or args.figure == 'all':
        plot_confusion_matrix()
    
    if args.figure == 'fig3' or args.figure == 'all':
        plot_roc_curve()
    
    if args.figure == 'fig4' or args.figure == 'all':
        plot_precision_recall_curve()
    
    if args.figure == 'fig5' or args.figure == 'all':
        plot_probability_distribution()
    
    if args.figure == 'fig6' or args.figure == 'all':
        plot_calibration_curve()
    
    if args.figure == 'fig7' or args.figure == 'all':
        plot_spatial_predictions()
    
    if args.figure == 'all':
        logger.info("All plots generated successfully.")
    else:
        logger.info(f"Plot {args.figure} generated successfully.")
    
    return

if __name__ == '__main__':
    main()
