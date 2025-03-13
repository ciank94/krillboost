
import logging
import pandas as pd
import numpy as np

def main( ):
    sub680()


def sub680():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Subsetting catch data")
    catchData = pd.read_csv("input/raw_data/C1_680.csv")
    time = pd.to_datetime(catchData.datetime_haul_start)
    latitude = catchData.latitude_haul_start
    longitude = catchData.longitude_haul_start
    catch = np.log10(catchData.krill_greenweight_kg)
    catchData = catchData[time.dt.month.isin([1,2,3])]
    catchData['time'] = time
    catchData['latitude'] = latitude
    catchData['longitude'] = longitude
    catchData['catch'] = catch
    catchData = catchData[['time', 'latitude', 'longitude', 'catch']]
    catchData.to_csv("input/subset_data/catchData.csv", index=False)
    logger.info(f"Catch data subsetted and saved to input/subset_data/catchData.csv")

    
if __name__ == "__main__":
    main()
    

    