import logging
import xarray as xr
import numpy as np
import pandas as pd
import os
import argparse
from scipy.interpolate import RegularGridInterpolator
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

def main():
    DataFusion()

class DataFusion:
    logging.basicConfig(level=logging.INFO)

    def __init__(self):
        self.feature_path = 'input/subset_data'
        self.raw_path = 'input/raw_data'
        self.output_file = 'input/fusedData.csv'
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        # open datasets
        self.krillData = pd.read_table(os.path.join(self.raw_path, 'krillbase.csv'), sep=',', encoding='unicode_escape')
        self.chl = xr.open_dataset(os.path.join(self.feature_path, 'chl.nc'))
        self.iron = xr.open_dataset(os.path.join(self.feature_path, 'iron.nc'))
        self.sst = xr.open_dataset(os.path.join(self.feature_path, 'sst.nc'))
        self.ssh = xr.open_dataset(os.path.join(self.feature_path, 'ssh.nc'))
        self.vel = xr.open_dataset(os.path.join(self.feature_path, 'vel.nc'))
        self.bath = xr.open_dataset(os.path.join(self.raw_path, 'bathymetry.nc'))
        self.logger.info(f"Datasets loaded from {self.feature_path} and {self.raw_path}")

        # algorithm:
        self.subsetKrillData()
        self.fuse_data()

    def subsetKrillData(self):
        # subset krill data
        self.krillDataSubset = self.krillData.loc[:, ["LONGITUDE", "LATITUDE", "STANDARDISED_KRILL_UNDER_1M2", "DATE"]]
        self.krillDataSubset.DATE = pd.to_datetime(self.krillDataSubset.DATE, format='%d/%m/%Y')
        self.krillDataSubset = self.krillDataSubset[(self.krillDataSubset.DATE.dt.year >= 1976) & (self.krillDataSubset.DATE.dt.year <= 2016)]
        self.logger.info(f"Subset to date range 1976-2016")
        lonRange = (-70, -31)
        latRange = (-73, -50)
        self.krillDataSubset = self.krillDataSubset[(self.krillDataSubset.LONGITUDE >= lonRange[0]) & (self.krillDataSubset.LONGITUDE <= lonRange[1]) & \
                              (self.krillDataSubset.LATITUDE >= latRange[0]) & (self.krillDataSubset.LATITUDE <= latRange[1])]
        self.logger.info(f"Subset to longitude range: {lonRange} and latitude range: {latRange}")
        # Create presence/absence column
        self.krillDataSubset['KRILL_PRESENCE'] = (self.krillDataSubset['STANDARDISED_KRILL_UNDER_1M2'] > 0).astype(int)
        
        # Create log10 transformed column for values above 0
        self.krillDataSubset['KRILL_LOG10'] = self.krillDataSubset['STANDARDISED_KRILL_UNDER_1M2'].apply(lambda x: np.nan if x <= 0 else np.log10(x))
        
        #SQRT transformation
        self.krillDataSubset['KRILL_SQRT'] = self.krillDataSubset['STANDARDISED_KRILL_UNDER_1M2'].apply(lambda x: np.nan if x <= 0 else np.sqrt(x))

        #natural log transformation
        self.krillDataSubset['KRILL_LOGN'] = self.krillDataSubset['STANDARDISED_KRILL_UNDER_1M2'].apply(lambda x: np.nan if x <= 0 else np.log1p(x))
      
        # quantile transformer (stabilize variance):
        # Extract positive values for quantile transformation
        positive_krill_values = self.krillDataSubset['STANDARDISED_KRILL_UNDER_1M2'][self.krillDataSubset['STANDARDISED_KRILL_UNDER_1M2'] > 0].values.reshape(-1, 1)
        
        # Apply quantile transformer only to positive values
        quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=42)
        transformed_values = quantile_transformer.fit_transform(positive_krill_values).flatten()
        
        # Create a new column with transformed values
        self.krillDataSubset['KRILL_QUAN'] = np.nan
        self.krillDataSubset.loc[self.krillDataSubset['STANDARDISED_KRILL_UNDER_1M2'] > 0, 'KRILL_QUAN'] = transformed_values

        # original values:
        self.krillDataSubset['KRILL_ORIGINAL'] = self.krillDataSubset['STANDARDISED_KRILL_UNDER_1M2']

        # Drop NaN values
        #self.krillDataSubset.dropna(inplace=True)
        
        # Drop the original column
        self.krillDataSubset.drop(columns=['STANDARDISED_KRILL_UNDER_1M2'], inplace=True)
        self.krill = self.krillDataSubset.drop('DATE', axis=1)
        return

    def fuse_data(self):
         # fuse new data
        lon = self.krill.LONGITUDE
        lat = self.krill.LATITUDE

        bath_interp = RegularGridInterpolator((self.bath.lat.values, self.bath.lon.values), self.bath.elevation.values, method='linear', bounds_error=False, fill_value=np.nan)
        bath = np.abs(bath_interp((lat, lon)))
        self.krill['DEPTH'] = bath

        chl_mean_interp = RegularGridInterpolator((self.chl.latitude.values, self.chl.longitude.values), self.chl.chl_mean.values, method='linear', bounds_error=False, fill_value=np.nan)
        chl_min_interp = RegularGridInterpolator((self.chl.latitude.values, self.chl.longitude.values), self.chl.chl_10th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
        chl_max_interp = RegularGridInterpolator((self.chl.latitude.values, self.chl.longitude.values), self.chl.chl_90th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
        chl_mean = chl_mean_interp((lat, lon))
        chl_min = chl_min_interp((lat, lon))
        chl_max = chl_max_interp((lat, lon))
        self.krill['CHL_MEAN'] = chl_mean
        self.krill['CHL_MIN'] = chl_min
        self.krill['CHL_MAX'] = chl_max
        self.logger.info(f"CHL fused")

        iron_mean_interp = RegularGridInterpolator((self.iron.latitude.values, self.iron.longitude.values), self.iron.iron_mean.values, method='linear', bounds_error=False, fill_value=np.nan)
        iron_min_interp = RegularGridInterpolator((self.iron.latitude.values, self.iron.longitude.values), self.iron.iron_10th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
        iron_max_interp = RegularGridInterpolator((self.iron.latitude.values, self.iron.longitude.values), self.iron.iron_90th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
        iron_mean = iron_mean_interp((lat, lon))
        iron_min = iron_min_interp((lat, lon))
        iron_max = iron_max_interp((lat, lon))
        self.krill['IRON_MEAN'] = iron_mean
        self.krill['IRON_MIN'] = iron_min
        self.krill['IRON_MAX'] = iron_max
        self.logger.info(f"IRON fused")

        ssh_mean_interp = RegularGridInterpolator((self.ssh.latitude.values, self.ssh.longitude.values), self.ssh.ssh_mean.values, method='linear', bounds_error=False, fill_value=np.nan)
        ssh_min_interp = RegularGridInterpolator((self.ssh.latitude.values, self.ssh.longitude.values), self.ssh.ssh_10th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
        ssh_max_interp = RegularGridInterpolator((self.ssh.latitude.values, self.ssh.longitude.values), self.ssh.ssh_90th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
        ssh_mean = ssh_mean_interp((lat, lon))
        ssh_min = ssh_min_interp((lat, lon))
        ssh_max = ssh_max_interp((lat, lon))
        self.krill['SSH_MEAN'] = ssh_mean
        self.krill['SSH_MIN'] = ssh_min
        self.krill['SSH_MAX'] = ssh_max
        self.logger.info(f"SSH fused")

        vel_mean_interp = RegularGridInterpolator((self.vel.latitude.values, self.vel.longitude.values), self.vel.vel_mean.values, method='linear', bounds_error=False, fill_value=np.nan)
        vel_min_interp = RegularGridInterpolator((self.vel.latitude.values, self.vel.longitude.values), self.vel.vel_10th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
        vel_max_interp = RegularGridInterpolator((self.vel.latitude.values, self.vel.longitude.values), self.vel.vel_90th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
        vel_mean = vel_mean_interp((lat, lon))
        vel_min = vel_min_interp((lat, lon))
        vel_max = vel_max_interp((lat, lon))
        self.krill['VEL_MEAN'] = vel_mean
        self.krill['VEL_MIN'] = vel_min
        self.krill['VEL_MAX'] = vel_max
        self.logger.info(f"VEL fused")

        sst_mean_interp = RegularGridInterpolator((self.sst.latitude.values, self.sst.longitude.values), self.sst.sst_mean.values, method='linear', bounds_error=False, fill_value=np.nan)
        sst_min_interp = RegularGridInterpolator((self.sst.latitude.values, self.sst.longitude.values), self.sst.sst_10th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
        sst_max_interp = RegularGridInterpolator((self.sst.latitude.values, self.sst.longitude.values), self.sst.sst_90th_percentile.values, method='linear', bounds_error=False, fill_value=np.nan)
        sst_mean = sst_mean_interp((lat, lon))
        sst_min = sst_min_interp((lat, lon))
        sst_max = sst_max_interp((lat, lon))
        self.krill['SST_MEAN'] = sst_mean
        self.krill['SST_MIN'] = sst_min
        self.krill['SST_MAX'] = sst_max
        self.logger.info(f"SST fused")
        self.save_fused_data()

    def save_fused_data(self):
        # Example logic to save data
        self.krill.to_csv(self.output_file)
        self.logger.info(f"Fused data saved to {self.output_file}")

if __name__ == '__main__':
    main()