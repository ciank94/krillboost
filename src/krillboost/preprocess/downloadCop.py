import logging
import json
import copernicusmarine as cop
import argparse

#Entry point
def main():
    parser = argparse.ArgumentParser(description='Download data from Copernicus using specified configuration.')
    parser.add_argument('dataKey', type=str, help='Key from download_params.json specifying which dataset to download')
    args = parser.parse_args()
    DownloadCop(args.dataKey)

class DownloadCop:
    logging.basicConfig(level=logging.INFO)
    loggerDescription = "\nDownloadCop class description:\n\
        downloads data from Copernicus: SSH, SST, CHL, OXY etc. \n\
        parameters are loaded from config/download_params.json\n \
        NOTE: requires credentials file \n \
        NOTE: bathymetry must be manually downloaded from GEBCO or other source and saved to input/bathymetry.nc\n"

    def __init__(self, dataKey):
        # load request parameters from config file
        with open(f'config/download_settings.json', 'r') as f:
            request = json.load(f)
        params = request[dataKey] # name of dataset
        self.dataID = params['dataID']
        self.configurePath = params['configurePath']
        self.outputFilename = params['outputFilename']
        self.startDate = params['startDate']
        self.endDate = params['endDate']
        self.lonBounds = params['lonBounds']
        self.latBounds = params['latBounds']
        self.variables = params['variables']

        # log request details and process request;
        self.initLogger()
        self.downloadCop()
        return

    def initLogger(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"================={self.__class__.__name__}=====================")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self.logger.info(self.loggerDescription)
        self.logger.info(f"=================Request parameters=====================")
        self.logger.info(f"DataID: {self.dataID}")    
        self.logger.info(f"Configure path: {self.configurePath}")
        self.logger.info(f"Output path: {self.outputFilename}")
        self.logger.info(f"Start date: {self.startDate}")
        self.logger.info(f"End date: {self.endDate}")
        self.logger.info(f"Longitude bounds: {self.lonBounds}")
        self.logger.info(f"Latitude bounds: {self.latBounds}")
        self.logger.info(f"Variables: {self.variables}")   
        return

    def downloadCop(self):
        self.logger.info(f"Downloading data...")
        cop.subset(
            dataset_id=self.dataID,
            variables=self.variables,
            start_datetime=self.startDate,
            end_datetime=self.endDate,
            minimum_longitude=self.lonBounds[0],
            maximum_longitude=self.lonBounds[1],
            minimum_latitude=self.latBounds[0],
            maximum_latitude=self.latBounds[1],
            output_filename=self.outputFilename,
            credentials_file=self.configurePath,
            force_download=True,
        )
        self.logger.info(f"Data downloaded to {self.outputFilename}")
        return

if __name__ == '__main__':
    main()
    