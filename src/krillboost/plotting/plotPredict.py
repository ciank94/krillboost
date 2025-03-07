import logging
import argparse
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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

def main():
    parser = argparse.ArgumentParser(description='Plot predicted krill presence and abundance.')
    args = parser.parse_args()
    
    plotQQ()
    return

if __name__ == '__main__':
    main()
