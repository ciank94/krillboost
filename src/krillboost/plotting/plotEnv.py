import logging
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import pandas as pd
import cmocean
import argparse
import os
import glob
from matplotlib.gridspec import GridSpec


def main():
    logging.basicConfig(level=logging.INFO)
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot environmental data figures')
    parser.add_argument('figure', type=str, choices=['envData', 'subsetData', 'krillDistributions'], 
                        default='envData', help='Select which figure to plot (default: envData)')
    args = parser.parse_args()
    
    # Plot the selected figure
    if args.figure == 'envData':
        plotEnvData()
    elif args.figure == 'subsetData':
        plotVariableDistributions()
    elif args.figure == 'krillDistributions':
        plotKrillDistributions()
    return

def plotEnvData():
    """Create a figure showing bathymetry with krill locations and environmental variables"""
    logger = logging.getLogger(__name__)
    logger.info("Plotting environmental data...")

    # load data
    bath = xr.open_dataset(f"input/raw_data/bathymetry.nc")
    krillData = pd.read_csv("input/fusedData.csv")
    sst = xr.open_dataset(f"input/subset_data/sst.nc")
    ssh = xr.open_dataset(f"input/subset_data/ssh.nc")
    chl = xr.open_dataset(f"input/subset_data/chl.nc")
    iron = xr.open_dataset(f"input/subset_data/iron.nc")
    vel = xr.open_dataset(f"input/subset_data/vel.nc")
    logger.info("Data loaded")
        
    # Create masked array for bathymetry where elevation <= 0
    bathymetry = abs(bath.elevation.values)
    # Mask both land (elevation > 0) and invalid points
    masked_bathymetry = np.ma.masked_where((bathymetry <= 0) | (bathymetry > 10000), bathymetry)
        
    # Create contour levels every 500m
    contour_levels = np.arange(0, 3000, 400)
        
    # Create figure with 3x2 subplots
    plt.rcParams.update({'font.size': 20})  # Set default font size to 20
    fig = plt.figure(figsize=(28, 24))  # Increased height for better spacing
    gs = fig.add_gridspec(3, 2, hspace=0.001, wspace=0.25)  # Increased hspace from 0.005 to 0.4
        
    projection = ccrs.PlateCarree()
        
    # Plot 1: Bathymetry with krill locations
    ax1 = plt.subplot(gs[0, 0], projection=projection)
    im1 = ax1.pcolormesh(bath.lon, bath.lat, 
                      masked_bathymetry, shading='auto', 
                      cmap='Blues', transform=projection, zorder=1)
    im1.set_clim(0, 5000)
    ax1.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=100)  # Increased zorder
    ax1.coastlines(zorder=101)  # Coastlines on top
    ax1.scatter(krillData.LONGITUDE, 
               krillData.LATITUDE,
               c='red', s=10, edgecolor='black', linewidth=0.3,  # Reduced size
               transform=projection, zorder=102)  # Points on top of everything
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.025, pad=0.04)
    cbar1.set_label('Depth (m)', fontsize=20)
    cbar1.ax.tick_params(labelsize=20)
    ax1.set_xlabel('Longitude', fontsize=20)
    ax1.set_ylabel('Latitude', fontsize=20)
    ax1.tick_params(axis='both', labelsize=20)
    cs1 = ax1.contour(bath.lon, bath.lat, 
                   masked_bathymetry, levels=contour_levels,
                   colors='grey', alpha=0.3, transform=projection)
    logger.info("Completed bathymetry subplot (1/6)")

    # Plot 2: SST
    ax2 = plt.subplot(gs[0, 1], projection=projection)
    im2 = ax2.pcolormesh(sst.longitude.values, sst.latitude.values, 
                      sst.sst_mean.values, shading='auto',
                      cmap=cmocean.cm.thermal, transform=projection)
    ax2.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=100)
    ax2.coastlines(zorder=101)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.025, pad=0.04)
    cbar2.set_label('Temperature (°C)', fontsize=20)
    cbar2.ax.tick_params(labelsize=20)
    ax2.set_xlabel('Longitude', fontsize=20)
    ax2.set_ylabel('Latitude', fontsize=20)
    ax2.tick_params(axis='both', labelsize=20)
    cs2 = ax2.contour(bath.lon, bath.lat, 
                   masked_bathymetry, levels=contour_levels,
                   colors='grey', alpha=0.3, transform=projection)
    logger.info("Completed SST subplot (2/6)")

    # Plot 3: SSH
    ax3 = plt.subplot(gs[1, 0], projection=projection)
    im3 = ax3.pcolormesh(ssh.longitude.values, ssh.latitude.values, 
                      ssh.ssh_mean.values, shading='auto',
                      cmap=cmocean.cm.balance, transform=projection)
    ax3.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=100)
    ax3.coastlines(zorder=101)
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.025, pad=0.04)
    cbar3.set_label('Height (m)', fontsize=20)
    cbar3.ax.tick_params(labelsize=20)
    ax3.set_xlabel('Longitude', fontsize=20)
    ax3.set_ylabel('Latitude', fontsize=20)
    ax3.tick_params(axis='both', labelsize=20)
    cs3 = ax3.contour(bath.lon, bath.lat, 
                   masked_bathymetry, levels=contour_levels,
                   colors='grey', alpha=0.3, transform=projection)
    logger.info("Completed SSH subplot (3/6)")

    # Plot 4: Net velocity
    ax4 = plt.subplot(gs[1, 1], projection=projection)
    im4 = ax4.pcolormesh(vel.longitude.values, vel.latitude.values, 
                      vel.vel_mean.values, shading='auto',
                      cmap=cmocean.cm.speed, transform=projection)
    ax4.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=100)
    ax4.coastlines(zorder=101)
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.025, pad=0.04)
    cbar4.set_label('Velocity (m/s)', fontsize=20)
    cbar4.ax.tick_params(labelsize=20)
    ax4.set_xlabel('Longitude', fontsize=20)
    ax4.set_ylabel('Latitude', fontsize=20)
    ax4.tick_params(axis='both', labelsize=20)
    cs4 = ax4.contour(bath.lon, bath.lat, 
                   masked_bathymetry, levels=contour_levels,
                   colors='grey', alpha=0.3, transform=projection)
    ax4.legend().set_visible(False)  # Remove legend
    logger.info("Completed velocity subplot (4/6)")

    # Plot 5: CHL
    ax5 = plt.subplot(gs[2, 0], projection=projection)
    im5 = ax5.pcolormesh(chl.longitude.values, chl.latitude.values, 
                      chl.chl_mean.values, shading='auto',
                      cmap=cmocean.cm.algae, transform=projection, zorder=1)
    im5.set_clim(0, 2)  # Set max to 2 for CHL
    ax5.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=100)
    ax5.coastlines(zorder=101)
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.025, pad=0.04)
    cbar5.set_label('Chlorophyll (mg/m$^3$)', fontsize=20)
    cbar5.ax.tick_params(labelsize=20)
    ax5.set_xlabel('Longitude', fontsize=20)
    ax5.set_ylabel('Latitude', fontsize=20)
    ax5.tick_params(axis='both', labelsize=20)
    cs5 = ax5.contour(bath.lon, bath.lat, 
                   masked_bathymetry, levels=contour_levels,
                   colors='grey', alpha=0.3, transform=projection)
    logger.info("Completed chlorophyll subplot (5/6)")

    # Plot 6: Iron
    ax6 = plt.subplot(gs[2, 1], projection=projection)
    im6 = ax6.pcolormesh(iron.longitude.values, iron.latitude.values, 
                      iron.iron_mean.values, shading='auto',
                      cmap=cmocean.cm.matter, transform=projection, zorder=1)
    im6.set_clim(0, 0.001)  # Set color limits on the plot object
    ax6.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=100)
    ax6.coastlines(zorder=101)
    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.025, pad=0.04)
    cbar6.set_label('Iron (mmol/m$^3$)', fontsize=20)
    cbar6.ax.tick_params(labelsize=20)
    ax6.set_xlabel('Longitude', fontsize=20)
    ax6.set_ylabel('Latitude', fontsize=20)
    ax6.tick_params(axis='both', labelsize=20)
    cs6 = ax6.contour(bath.lon, bath.lat, 
                   masked_bathymetry, levels=contour_levels,
                   colors='grey', alpha=0.3, transform=projection)
    logger.info("Completed iron subplot (6/6)")

    # Set common gridlines for all subplots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 20}
        gl.ylabel_style = {'size': 20}
        
    plotName = f"output/figures/envData.png"
    os.makedirs(os.path.dirname(plotName), exist_ok=True)
    plt.savefig(plotName, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved environmental data plot to: {plotName}")
    return

def plotVariableDistributions():
    """
    Create a figure showing distributions of all variables in the subset folder.
    For each variable, plot the min, mean, and max values.
    """
    logger = logging.getLogger(__name__)
    logger.info("Plotting variable distributions...")

    # Load the fused data which contains all variables used in training
    data = pd.read_csv("input/fusedData.csv")
    logger.info("Loaded fused data")

    # Create a figure with subplots for each variable
    plt.rcParams.update({'font.size': 16})  # Increased base font size
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)  # Tighter layout

    # Variables to plot with proper names and units
    variables = {
        'DEPTH': {
            'columns': ['DEPTH'],
            'xlabel': 'Depth (m)'
        },
        'SST': {
            'columns': ['SST_MIN', 'SST_MEAN', 'SST_MAX'],
            'xlabel': 'Sea Surface Temperature (°C)'
        },
        'SSH': {
            'columns': ['SSH_MIN', 'SSH_MEAN', 'SSH_MAX'],
            'xlabel': 'Sea Surface Height (m)'
        },
        'VEL': {
            'columns': ['VEL_MIN', 'VEL_MEAN', 'VEL_MAX'],
            'xlabel': 'Velocity (m/s)'
        },
        'CHL': {
            'columns': ['CHL_MIN', 'CHL_MEAN', 'CHL_MAX'],
            'xlabel': 'Chlorophyll (mg/m³)'
        },
        'IRON': {
            'columns': ['IRON_MIN', 'IRON_MEAN', 'IRON_MAX'],
            'xlabel': 'Iron (mmol/m³)'
        }
    }

    # Set specific x-axis limits for each variable
    x_limits = {
        'SST': (-2, 5),
        'SSH': (-1.5, -0.75),
        'CHL': (0, 1.5),
        'IRON': (0, 0.001),
        'VEL': (0, 0.25),
        'DEPTH': (0, 5000)
    }

    # New color scheme - using a blue-orange-purple palette
    # Order is important: mean should be plotted last to be on top
    # We'll use different alphas: mean (0.8), min (0.6), max (0.4)
    colors = {
        'MIN': {'color': '#4575b4', 'alpha': 0.6},  # Blue
        'MEAN': {'color': '#d73027', 'alpha': 0.8},  # Red
        'MAX': {'color': '#91bfdb', 'alpha': 0.4}    # Light blue
    }
    
    # Plot order to ensure mean is on top
    plot_order = ['MAX', 'MIN', 'MEAN']
    
    # Plot each variable
    row, col = 0, 0
    for i, (var_name, var_info) in enumerate(variables.items()):
        ax = fig.add_subplot(gs[row, col])
        
        if var_name == 'DEPTH':
            # For depth, we only have one column - no legend
            valid_data = data['DEPTH'].dropna()
            
            # Apply limits
            min_val, max_val = x_limits['DEPTH']
            valid_data = valid_data[(valid_data >= min_val) & (valid_data <= max_val)]
            
            # Calculate number of bins
            bin_count = min(30, max(10, int(len(valid_data) / 50)))
            
            # Plot depth with the mean color
            ax.hist(valid_data, bins=bin_count, alpha=colors['MEAN']['alpha'], 
                   color=colors['MEAN']['color'], edgecolor='black', linewidth=0.5)
        else:
            # For variables with min/mean/max, plot in specific order
            col_dict = {col.split('_')[-1]: col for col in var_info['columns']}
            
            for stat in plot_order:
                if stat in col_dict:
                    col_name = col_dict[stat]
                    valid_data = data[col_name].dropna()
                    
                    # Apply limits
                    min_val, max_val = x_limits[var_name]
                    valid_data = valid_data[(valid_data >= min_val) & (valid_data <= max_val)]
                    
                    # Calculate number of bins
                    bin_count = min(30, max(10, int(len(valid_data) / 50)))
                    
                    # Plot with appropriate color and transparency
                    ax.hist(valid_data, bins=bin_count, 
                           alpha=colors[stat]['alpha'], 
                           color=colors[stat]['color'], 
                           label=stat.capitalize(), 
                           edgecolor='black', linewidth=0.5)
        
        # Set x-axis label with full name and units
        ax.set_xlabel(var_info['xlabel'], fontsize=18)
        ax.set_ylabel('Frequency', fontsize=18)
        
        # Increase tick size
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Set x-axis limits
        if var_name in x_limits:
            ax.set_xlim(x_limits[var_name])
        
        # Add legend only for variables with multiple columns (not for depth)
        if var_name != 'DEPTH':
            ax.legend(loc='upper right', fontsize=16)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        # Move to next subplot position
        col += 1
        if col > 1:
            col = 0
            row += 1
    
    # Save the figure with tight layout
    plt.tight_layout()
    os.makedirs('output/figures', exist_ok=True)
    plt_path = 'output/figures/subsetData.png'
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved variable distributions plot to: {plt_path}")
    return

def plotKrillDistributions():
    """
    Create a figure showing distributions of krill data from krillbase.csv and fusedData.csv,
    focusing on the log-normal distribution and two-step modeling approach.
    
    This visualization specifically highlights:
    1. The log-normal distribution of non-zero values
    2. The two-step modeling approach (presence/absence followed by abundance)
    3. Temporal distribution of samples from 1976 onwards
    4. Spatial distribution of krill samples in the regions of interest
    """
    logger = logging.getLogger(__name__)
    logger.info("Plotting krill data distributions...")

    try:
        # Import necessary libraries
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        
        # Load the raw and fused data
        raw_krill = pd.read_csv("input/raw_data/krillbase.csv", encoding='unicode_escape', low_memory=False)
        fused_krill = pd.read_csv("input/fusedData.csv")
        logger.info(f"Loaded raw krill data: {raw_krill.shape} and fused data: {fused_krill.shape}")

        # Create a figure with subplots - now with only 2x2 grid and tighter spacing
        plt.rcParams.update({'font.size': 16})  # Increased base font size
        fig = plt.figure(figsize=(20, 16))  # Larger figure size
        
        # Create grid with uneven column widths to make the map larger
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3, width_ratios=[1, 1.2])  # Make right column wider for the map

        # Extract standardized krill values from raw data
        std_krill = raw_krill['STANDARDISED_KRILL_UNDER_1M2'].dropna()
        
        # Count zeros and non-zeros for logging
        zeros = (std_krill == 0).sum()
        non_zeros = (std_krill > 0).sum()
        total = len(std_krill)
        logger.info(f"Zero-inflation analysis: {zeros} zeros, {non_zeros} non-zeros out of {total} total")
        
        # Get non-zero values and apply log10 transformation
        non_zero_krill = std_krill[std_krill > 0]
        log10_krill = np.log10(non_zero_krill)
        logger.info(f"Log10 distribution: min={log10_krill.min():.2f}, max={log10_krill.max():.2f}, mean={log10_krill.mean():.2f}")

        # 1. Log10 distribution for non-zero values (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Plot histogram of log10 values
        ax1.hist(log10_krill, bins=30, color='#b2182b', edgecolor='black', alpha=0.8)
        
        # Add normal distribution fit
        from scipy import stats
        mu, sigma = stats.norm.fit(log10_krill)
        x = np.linspace(log10_krill.min(), log10_krill.max(), 100)
        y = stats.norm.pdf(x, mu, sigma) * len(log10_krill) * (log10_krill.max() - log10_krill.min()) / 30
        ax1.plot(x, y, 'k--', linewidth=2, label=f'Normal Fit\nμ={mu:.2f}, σ={sigma:.2f}')
        
        ax1.set_xlabel('Log10(Standardised Krill Under 1m²)', fontsize=20)
        ax1.set_ylabel('Frequency', fontsize=20)
        ax1.legend(loc='upper right', fontsize=16)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='both', which='major', labelsize=16)

        # 2. Two-step model visualization (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Get the data from fused dataset
        presence_data = fused_krill['KRILL_PRESENCE']
        abundance_data = fused_krill.loc[fused_krill['KRILL_PRESENCE'] == 1, 'KRILL_LOG10']
        
        logger.info(f"Two-step model: {presence_data.sum()} presence samples out of {len(presence_data)} total")
        
        # Create a stacked bar for the two-step model
        model_data = [
            presence_data.sum(),  # Number of presence samples
            len(presence_data) - presence_data.sum()  # Number of absence samples
        ]
        
        # Make bars wider and closer together
        bar_width = 0.7  # Increased width from 0.5
        bar_positions = [0, 1.3]  # Closer together (was [0, 1])
        
        # Create the first bar showing presence/absence split
        ax2.bar([bar_positions[0]], [model_data[0]], color='#b2182b', edgecolor='black', width=bar_width, 
               label='Presence (Step 1)')
        ax2.bar([bar_positions[0]], [model_data[1]], bottom=[model_data[0]], color='#4575b4', 
               edgecolor='black', width=bar_width, label='Absence (Step 1)')
        
        # Add text annotations - separate for presence and absence
        presence_count = model_data[0]
        absence_count = model_data[1]
        total = sum(model_data)
        
        # Text for presence (red bar)
        ax2.text(bar_positions[0], presence_count/2, f'Presence\n{presence_count}\n({presence_count/total*100:.1f}%)', 
                 ha='center', va='center', fontsize=18, color='white', fontweight='bold')
        
        # Text for absence (blue bar)
        ax2.text(bar_positions[0], presence_count + absence_count/2, f'Absence\n{absence_count}\n({absence_count/total*100:.1f}%)', 
                 ha='center', va='center', fontsize=18, color='white', fontweight='bold')
        
        # Create histogram for the abundance values (Step 2)
        ax2.bar([bar_positions[1]], [len(abundance_data)], color='#b2182b', edgecolor='black', width=bar_width,
               alpha=0.7, label='Abundance (Step 2)')
        
        # Add text annotations
        ax2.text(bar_positions[1], len(abundance_data)/2, f'Abundance\n{len(abundance_data)}', 
                 ha='center', va='center', fontsize=18, color='white', fontweight='bold')
        
        ax2.set_xticks(bar_positions)
        ax2.set_xticklabels(['Step 1:\nPresence/Absence', 'Step 2:\nAbundance'], fontsize=16)
        ax2.set_ylabel('Number of Samples', fontsize=20)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.tick_params(axis='both', which='major', labelsize=16)

        # 3. Temporal distribution (bottom left) - only from 1976 onwards
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Convert date to datetime and extract year
        raw_krill['DATE'] = pd.to_datetime(raw_krill['DATE'], format='%d/%m/%Y', errors='coerce')
        raw_krill['YEAR'] = raw_krill['DATE'].dt.year
        
        # Filter data from 1976 onwards
        filtered_krill = raw_krill[raw_krill['YEAR'] >= 1976]
        
        # Count samples per year
        year_counts = filtered_krill['YEAR'].value_counts().sort_index()
        
        # Calculate presence ratio per year
        presence_by_year = filtered_krill.groupby('YEAR')['STANDARDISED_KRILL_UNDER_1M2'].apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0
        ).reindex(year_counts.index)
        
        logger.info(f"Temporal distribution: data spans from {year_counts.index.min()} to {year_counts.index.max()}")
        
        # Plot bar chart of sample counts
        bars = ax3.bar(year_counts.index, year_counts, color='#4393c3', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Year', fontsize=20)
        ax3.set_ylabel('Number of Samples', fontsize=20)
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.tick_params(axis='both', which='major', labelsize=16)
        
        # Add line showing presence ratio with extended y-axis
        ax3_twin = ax3.twinx()
        ax3_twin.plot(presence_by_year.index, presence_by_year, 'r-', linewidth=3, marker='o', markersize=6)
        ax3_twin.set_ylabel('Presence Ratio', fontsize=20, color='#b2182b')
        ax3_twin.tick_params(axis='y', colors='#b2182b', labelsize=16)
        
        # Extend y-axis to see presence ratio more clearly with buffer above 1.0
        # Find the min and max of presence ratio to set appropriate limits
        min_ratio = presence_by_year.min()
        max_ratio = presence_by_year.max()
        # Set limits with padding to make the line more visible, ensuring buffer above 1.0
        ax3_twin.set_ylim(max(0, min_ratio - 0.1), 1.05)  # Buffer above 1.0 to show high values
        
        # Add legend
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='#4393c3', lw=0, marker='s', markersize=12),
            Line2D([0], [0], color='#b2182b', lw=3, marker='o', markersize=8)
        ]
        #ax3.legend(custom_lines, ['Sample Count', 'Presence Ratio'], loc='upper left', fontsize=16)

        # 4. Spatial distribution (bottom right) - with specific bounds and grid cells
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            
            ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.SouthPolarStereo())
            
            # Set map bounds to focus on regions of interest using the specified bounds
            lonBounds = [-70, -31]
            latBounds = [-73, -50]
            ax4.set_extent(lonBounds + latBounds, crs=ccrs.PlateCarree())
            
            # Add coastlines and gridlines
            ax4.coastlines(linewidth=1.5)
            gl = ax4.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            ax4.add_feature(cfeature.LAND, facecolor='lightgray')
            
            # Create grid for presence ratio calculation
            # Define grid resolution (in degrees)
            grid_res = 2.0  # 2-degree grid cells
            
            # Create grid
            lon_grid = np.arange(lonBounds[0], lonBounds[1] + grid_res, grid_res)
            lat_grid = np.arange(latBounds[0], latBounds[1] + grid_res, grid_res)
            
            # Initialize arrays to store presence counts and total counts
            presence_counts = np.zeros((len(lat_grid)-1, len(lon_grid)-1))
            total_counts = np.zeros((len(lat_grid)-1, len(lon_grid)-1))
            
            # Calculate presence ratio for each grid cell
            for i in range(len(lat_grid)-1):
                for j in range(len(lon_grid)-1):
                    # Find points in this grid cell
                    mask = ((raw_krill['LONGITUDE'] >= lon_grid[j]) & 
                            (raw_krill['LONGITUDE'] < lon_grid[j+1]) & 
                            (raw_krill['LATITUDE'] >= lat_grid[i]) & 
                            (raw_krill['LATITUDE'] < lat_grid[i+1]))
                    
                    # Count total points and presence points in this cell
                    points_in_cell = mask.sum()
                    if points_in_cell > 0:
                        total_counts[i, j] = points_in_cell
                        presence_counts[i, j] = (raw_krill.loc[mask, 'STANDARDISED_KRILL_UNDER_1M2'] > 0).sum()
            
            # Calculate presence ratio (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                presence_ratio = np.where(total_counts > 0, presence_counts / total_counts, np.nan)
            
            # Create a custom colormap from blue to red
            colors = ['#4575b4', '#91bfdb', '#e0f3f8', '#fee090', '#fc8d59', '#d73027']
            cmap = LinearSegmentedColormap.from_list('presence_ratio', colors)
            
            # Plot the colormesh
            # Convert grid edges to centers for pcolormesh
            lon_centers = lon_grid[:-1] + grid_res/2
            lat_centers = lat_grid[:-1] + grid_res/2
            
            # Create meshgrid for pcolormesh
            lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
            
            # Plot the colormesh
            mesh = ax4.pcolormesh(
                lon_grid, lat_grid, presence_ratio, 
                transform=ccrs.PlateCarree(),
                cmap=plt.get_cmap('YlOrRd'), 
                vmin=0, vmax=1,
                shading='flat'
            )
            
            # Add colorbar
            cbar = plt.colorbar(mesh, ax=ax4, orientation='vertical', pad=0.02, shrink=0.8)
            cbar.set_label('Presence Ratio', fontsize=18)
            cbar.ax.tick_params(labelsize=14)
            
            # Add a scatter plot showing sample density (size of points represents number of samples)
            # Only show cells with data
            valid_cells = ~np.isnan(presence_ratio)
            if np.any(valid_cells):
                # Scale the sizes based on log of count to prevent extremely large points
                sizes = np.log1p(total_counts[valid_cells]) * 30
                scatter = ax4.scatter(
                    lon_mesh[valid_cells], 
                    lat_mesh[valid_cells],
                    s=sizes,
                    c='black', 
                    alpha=0.3,
                    transform=ccrs.PlateCarree(),
                    edgecolor='white',
                    linewidth=0.5
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
                ax4.legend(handles=handles, loc='lower left', fontsize=14, title='Sample Count', title_fontsize=16)
            
        except ImportError:
            # If cartopy is not available, create a simple heatmap instead
            logger.warning("Cartopy not available, using simple heatmap for spatial distribution")
            ax4 = fig.add_subplot(gs[1, 1])
            
            # Use the specified bounds
            lonBounds = [-70, -31]
            latBounds = [-73, -50]
            
            # Create grid for presence ratio calculation
            # Define grid resolution (in degrees)
            grid_res = 2.0  # 2-degree grid cells
            
            # Create grid
            lon_grid = np.arange(lonBounds[0], lonBounds[1] + grid_res, grid_res)
            lat_grid = np.arange(latBounds[0], latBounds[1] + grid_res, grid_res)
            
            # Initialize arrays to store presence counts and total counts
            presence_counts = np.zeros((len(lat_grid)-1, len(lon_grid)-1))
            total_counts = np.zeros((len(lat_grid)-1, len(lon_grid)-1))
            
            # Calculate presence ratio for each grid cell
            for i in range(len(lat_grid)-1):
                for j in range(len(lon_grid)-1):
                    # Find points in this grid cell
                    mask = ((raw_krill['LONGITUDE'] >= lon_grid[j]) & 
                            (raw_krill['LONGITUDE'] < lon_grid[j+1]) & 
                            (raw_krill['LATITUDE'] >= lat_grid[i]) & 
                            (raw_krill['LATITUDE'] < lat_grid[i+1]))
                    
                    # Count total points and presence points in this cell
                    points_in_cell = mask.sum()
                    if points_in_cell > 0:
                        total_counts[i, j] = points_in_cell
                        presence_counts[i, j] = (raw_krill.loc[mask, 'STANDARDISED_KRILL_UNDER_1M2'] > 0).sum()
            
            # Calculate presence ratio (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                presence_ratio = np.where(total_counts > 0, presence_counts / total_counts, np.nan)
            
            # Create a custom colormap from blue to red
            #colors = ['#4575b4', '#91bfdb', '#e0f3f8', '#fee090', '#fc8d59', '#d73027']
            #cmap = LinearSegmentedColormap.from_list('presence_ratio', colors)
            #cmap = plt.get_cmap('plasma')
            
            # Plot the heatmap
            mesh = ax4.pcolormesh(
                lon_grid, lat_grid, presence_ratio, 
                cmap=plt.get_cmap('plasma'), 
                vmin=0, vmax=1,
                shading='flat'
            )
            
            # Add colorbar
            cbar = plt.colorbar(mesh, ax=ax4, orientation='vertical', pad=0.02, shrink=0.8)
            cbar.set_label('Presence Ratio', fontsize=18)
            cbar.ax.tick_params(labelsize=14)
            
            # Set limits to focus on regions of interest
            ax4.set_xlim(lonBounds[0], lonBounds[1])
            ax4.set_ylim(latBounds[0], latBounds[1])
            
            ax4.set_xlabel('Longitude', fontsize=20)
            ax4.set_ylabel('Latitude', fontsize=20)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='both', which='major', labelsize=16)
            
        # Apply tight layout to make the figure more compact
        plt.tight_layout()
        
        # Save the figure
        os.makedirs('output/figures', exist_ok=True)
        plt_path = 'output/figures/krillData.png'
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved krill distributions plot to: {plt_path}")
        
    except Exception as e:
        logger.error(f"Error in plotKrillDistributions: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    return


if __name__ == "__main__":
    main()