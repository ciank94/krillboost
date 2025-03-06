import logging
import xarray as xr
import argparse

def main():
    parser = argparse.ArgumentParser(description='Explore contents of a NetCDF/xarray dataset.')
    parser.add_argument('filename', type=str, help='Path to the NetCDF file to explore')
    args = parser.parse_args()
    ExploreNC(args.filename)

class ExploreNC:
    logging.basicConfig(level=logging.INFO)
    loggerDescription = "\nExploreNC class description:\n\
        Explores the contents of NetCDF/xarray datasets including: \n\
        - Global attributes \n\
        - Variables and their attributes \n\
        - Coordinates and dimensions \n\
        - Units and other metadata\n"

    def __init__(self, filename):
        self.filename = filename
        self.initLogger()
        self.explore_dataset()
        
    def initLogger(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self.logger.info(self.loggerDescription)
        self.logger.info(f"=================File Information=====================")
        self.logger.info(f"Exploring file: {self.filename}")
        return

    def explore_dataset(self):
        try:
            with xr.open_dataset(self.filename) as ds:
                # Print global attributes
                self.logger.info(f"\n=================Global Attributes=====================")
                for attr_name, attr_value in ds.attrs.items():
                    self.logger.info(f"{attr_name}: {attr_value}")

                # Print dimensions
                self.logger.info(f"\n=================Dimensions=====================")
                #for dim_name, dim_size in ds.dims.items():
                for dim_name, dim_size in ds.sizes.items():
                    self.logger.info(f"{dim_name}: {dim_size}")

                # Print coordinates
                self.logger.info(f"\n=================Coordinates=====================")
                for coord_name, coord in ds.coords.items():
                    self.logger.info(f"\nCoordinate: {coord_name}")
                    self.logger.info(f"Shape: {coord.shape}")
                    if hasattr(coord, 'units'):
                        self.logger.info(f"Units: {coord.units}")

                # Print variables
                self.logger.info(f"\n=================Variables=====================")
                for var_name, var in ds.variables.items():
                    if var_name not in ds.coords:
                        self.logger.info(f"\nVariable: {var_name}")
                        self.logger.info(f"Shape: {var.shape}")
                        self.logger.info(f"Dtype: {var.dtype}")
                        if hasattr(var, 'units'):
                            self.logger.info(f"Units: {var.units}")
                        if var.attrs:
                            self.logger.info("Attributes:")
                            for attr_name, attr_value in var.attrs.items():
                                self.logger.info(f"  {attr_name}: {attr_value}")

        except Exception as e:
            self.logger.error(f"Error exploring dataset: {str(e)}")

if __name__ == '__main__':
    main()
    
