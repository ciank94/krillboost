# krillboost
SDM predicting krill abundance using XGBoost

## Installation
Install from source: in development mode:
```bash
git clone https://github.com/ciank/krillboost.git
cd krillboost
pip install -e .
```
## Input Data
### Raw Data
Input data should be stored in the `input` directory. The following datasets are used:
- ssh.nc
- sst.nc
- chl.nc
- iron.nc
- bathymetry.nc

To download copernicus datasets:
```bash
python downloadCop <dataKey>
```
where dataKey is a key as listed in `config/download_settings.json`, for example:
```bash
python downloadCop test
```
will download SSH data from Copernicus Marine Service and save to `input/raw_data/test.nc`

To explore a NetCDF dataset:
```bash
exploreNC <filename>
```
where filename is the path to the NetCDF dataset to explore e.g. `input/raw_data/test.nc`.

### Subset Data



