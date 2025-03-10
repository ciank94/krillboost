# krillboost
SDM predicting krill abundance using XGBoost

## Navigation
- [Installation](#installation)
- [Preprocess Data](#preprocess-data)

## Installation
Install from source: in development mode:
```bash
git clone https://github.com/ciank/krillboost.git
cd krillboost
python -m pip install -e .
```
## Preprocess Data
### Raw Data
Input data should be stored in the `input` directory. The following datasets are used:
- ssh.nc
- sst.nc
- chl.nc
- iron.nc
- bathymetry.nc

To download copernicus datasets:
```bash
downloadCop <dataKey>
```
where dataKey is a key as listed in `config/download_settings.json`, for example:
```bash
downloadCop test
```
will download SSH data from Copernicus Marine Service and save to `input/raw_data/test.nc`

To explore a NetCDF dataset:
```bash
exploreNC <filename>
```
where filename is the path to the NetCDF dataset to explore e.g. `input/raw_data/test.nc`.

### Subset Data
There is a command line tool to subset Copernicus datasets, with a template for configuration:
```bash
subsetCop
```
where files are saved to `input/subset_data`.

### Fuse Data
To fuse Copernicus environmental datasets with krill data:
```bash
fuseData
```
where files are saved to `input/fusedData.csv`.

### Train Models
```bash
trainXG
```
where models are saved to `output/models`.

### Plotting
For plotting environment variables & initial preprocessed data, we can use the `plotEnv` command with a flag for different figure options in numeric format:
```bash
plotEnv --figure <figureKey>
```
for example `plotEnv --figure fig1`, where plots are saved to `output/figures`.

Similarly, for plotting predictions:
```bash
plotPredict --figure <figureKey>
```




