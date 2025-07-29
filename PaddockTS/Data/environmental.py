"""
Description:
This script downloads climate, soils and elevation data as netcdfs, tiffs and csv files

Steps for running locally:
1. Download and install Miniconda from https://www.anaconda.com/download/success. Note the filepath where it gets downloaded, e.g. /opt/miniconda3
2. Add the miniconda filepath to your ~/.zhrc, e.g. export PATH="/opt/miniconda3/bin:$PATH" 
3. brew install gdal
4. git clone https://github.com/johnburley3000/PaddockTS.git
5. conda create --name PaddockTS python=3.11
6. conda activate PaddockTS
7. pip install rasterio scipy pyproj owslib rioxarray requests pandas netcdf4
8. python3 PaddockTS/Code/04_environmental.py --stub Test --outdir ~/Desktop --tmpdir ~/Downloads --lat -34.3890 --lon 148.4695 --buffer 0.01 --start_time '2020-01-01' --end_time '2020-03-31'

Requirements for running on NCI:
- Projects: ub8 (ANU Water and Landscape Dynamics)
- Modules: gdal/3.6.4  (for terrain_tiles gdalwarp)
- Environment base: /g/data/xe2/John/geospatenv

Inputs:
- stub name
- output directory
- temporary directory
- coordinates
- buffer (in degrees in a single direction. For example, 0.01 degrees is about 1km so it would give a 2kmx2km area)
- start/end date
- flag to run locally or on NCI

Outputs:
- NetCDF files of climate variables from OzWald and SILO
- Tiff files of soil variables from SLGA and elevation from Terrain Tiles
- csv files of median climate and soil variables for input into DAESim

"""
import argparse
import logging
import os
import sys

# Change directory to this repo - this should work on gadi or locally via python or jupyter.
# Unfortunately, this needs to be in all files that can be run directly & use local imports.

from DAESIM_preprocess.terrain_tiles import terrain_tiles
from DAESIM_preprocess.slga_soils import slga_soils
from DAESIM_preprocess.ozwald_8day import ozwald_8day
from DAESIM_preprocess.ozwald_daily import ozwald_daily
from DAESIM_preprocess.silo_daily import silo_daily
from DAESIM_preprocess.daesim_forcing import daesim_forcing, daesim_soils
from PaddockTS.query import Query
from PaddockTS.legend import *

# Adjust logging configuration for the script
logging.basicConfig(level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Download environmental variables for the region of interest and save as lots of .tif and .nc files
        
Example usage on NCI:
python3 Code/04_environmental.py --stub Test --outdir /g/data/xe2/cb8590 --tmpdir /scratch/xe2/cb8590 --lat -34.3890 --lon 148.4695 --buffer 0.01 --start_time '2020-01-01' --end_time '2020-03-31' --nci

Example usage locally:
python3 Code/04_environmental.py --stub Test --outdir ~/Desktop --tmpdir ~/Downloads --lat -34.3890 --lon 148.4695 --buffer 0.01 --start_time '2020-01-01' --end_time '2020-03-31'""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--stub", type=str, required=True, help="Stub name for file naming")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for saved files")
    parser.add_argument("--tmpdir", type=str, required=True, help="Directory for temporary intermediate files")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of the center of the area of interest")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of the center of the area of interest")
    parser.add_argument("--buffer", type=float, required=True, help="Buffer in degrees to define the area around the center point")
    parser.add_argument("--start_time", type=str, required=True, help="Start time for the data query (YYYY-MM-DD)")
    parser.add_argument("--end_time", type=str, required=True, help="End time for the data query (YYYY-MM-DD)")
    parser.add_argument("--nci", action="store_true", help="Flag to run on NCI or in a local python environment")
    return parser.parse_args()


def main(args):
    print("Starting 04_environmental.py")

    lat = args.lat
    lon = args.lon
    buffer = args.buffer
    stub = args.stub
    start_year = args.start_time[:4]
    end_year = args.end_time[:4]
    outdir = args.tmpdir  # Using tmpdir instead of outdir so just the visuals go to outdir when running the full pipeline
    tmpdir = args.tmpdir
    thredds = ~args.nci  # Thredds is a public facing interface that can be used when not running the code directly on NCI

    if args.nci:
        silo_folder = "/g/data/xe2/datasets/Climate_SILO/"  # We have predownloaded all of the silo variables from 2017 to 2024 in this folder
    else:
        silo_folder = tmpdir
    
    # A smaller buffer than this messes up some of the API requests
    buffer = max(0.00001, buffer)

    # Download from OzWald wind and vapour pressure at 5km resolution, rainfall at 4km, and temperature at 250m resolution
    ozwald_daily(["Uavg", "VPeff"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)
    ozwald_daily(["Tmax", "Tmin"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)
    ozwald_daily(["Pg"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)

    # Download from OzWald soil moisture, runoff, leaf area index and gross primary productivity at 500m resolution
    variables = ["Ssoil", "Qtot", "LAI", "GPP"]
    ozwald_8day(variables, lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)

    # Download from SILO radiation, vapour pressure, temperature, rainfall, and evapotranspiration at 5km resolution
    # Note this requires downloading an Australia wide file of ~400MB per variables per year, so takes a long time if not predownloaded
    variables = ["radiation", "vp", "max_temp", "min_temp", "daily_rain", "et_morton_actual", "et_morton_potential"]
    ds_silo_daily = silo_daily(variables, lat, lon, buffer, start_year, end_year, outdir, stub, silo_folder)

    # Merge the SILO and OzWald climate data into DAESim_forcing.csv
    # By default, for variables available in both datasets (vapour pressure, temperature, rainfall), the OzWald variables get used for consistency with the 8day variables
    df_climate = daesim_forcing(outdir, stub)

    # Download soil variables from SLGA at 90m resolution
    variables = ['Clay', 'Silt', 'Sand', 'pH_CaCl2', 'Bulk_Density', 'Available_Water_Capacity', 'Effective_Cation_Exchange_Capacity', 'Total_Nitrogen', 'Total_Phosphorus']
    depths=['5-15cm', '15-30cm', '30-60cm', '60-100cm']
    slga_soils(variables, lat, lon, buffer, tmpdir, stub, depths)

    # Merge the soil data into a csv for input into DAESim
    df_soils = daesim_soils(outdir, stub, tmpdir)

    # Download Terrain Tiles elevation data for calculating topographic variables at 10m resolution
    terrain_tiles(lat, lon, buffer, outdir, stub, tmpdir)


def download_environmental_data(stub: str, query: Query):
    print('Starting 04_environmental.py')
    silo_folder = SILO_DIR

    lat = query.lat
    lon = query.lon
    buffer = query.buffer
    start_year = str(query.start_time)[:4]
    end_year = str(query.end_time)[:4]
    outdir = OUT_DIR
    tmpdir = TMP_DIR

    thredds = True

    # Download from OzWald wind and vapour pressure at 5km resolution, rainfall at 4km, and temperature at 250m resolution
    ozwald_daily(["Uavg", "VPeff"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)
    ozwald_daily(["Tmax", "Tmin"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)
    ozwald_daily(["Pg"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)

    # Download from OzWald soil moisture, runoff, leaf area index and gross primary productivity at 500m resolution
    variables = ["Ssoil", "Qtot", "LAI", "GPP"]
    ozwald_8day(variables, lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)

    # Download from SILO radiation, vapour pressure, temperature, rainfall, and evapotranspiration at 5km resolution
    # Note this requires downloading an Australia wide file of ~400MB per variables per year, so takes a long time if not predownloaded
    variables = ["radiation", "vp", "max_temp", "min_temp", "daily_rain", "et_morton_actual", "et_morton_potential"]
    ds_silo_daily = silo_daily(variables, lat, lon, buffer, start_year, end_year, outdir, stub, silo_folder)

    # Merge the SILO and OzWald climate data into DAESim_forcing.csv
    # By default, for variables available in both datasets (vapour pressure, temperature, rainfall), the OzWald variables get used for consistency with the 8day variables
    df_climate = daesim_forcing(outdir, stub)

    # Download soil variables from SLGA at 90m resolution
    variables = ['Clay', 'Silt', 'Sand', 'pH_CaCl2', 'Bulk_Density', 'Available_Water_Capacity', 'Effective_Cation_Exchange_Capacity', 'Total_Nitrogen', 'Total_Phosphorus']
    depths=['5-15cm', '15-30cm', '30-60cm', '60-100cm']
    slga_soils(variables, lat, lon, buffer, tmpdir, stub, depths)

    # Merge the soil data into a csv for input into DAESim
    df_soils = daesim_soils(outdir, stub, tmpdir)

    # Download Terrain Tiles elevation data for calculating topographic variables at 10m resolution
    terrain_tiles(lat, lon, buffer, outdir, stub, tmpdir)

def test():
    from datetime import date
    from PaddockTS.query import get_example_query
    query = get_example_query()
    # query = Query(
    #     lat=-34.3890427,
    #     lon=148.469499,
    #     buffer=0.01,
    #     start_time= date(2020, 1, 1),
    #     end_time=date(2021, 12, 31),
    #     collections=[],
    #     bands=[]
    # )
    stub = query.get_stub()
    download_environmental_data(stub, query)
    
if __name__ == "__main__":
    test()
    # Example args for debugging
    # args = argparse.Namespace(
    #     lat=-34.3890427,
    #     lon=148.469499,
    #     buffer=0.1,
    #     stub="Test",
    #     start_time="2020-01-01",
    #     end_time="2021-12-31",
    #     outdir=".",
    #     tmpdir="."
    # )


    # args = parse_arguments()
    # main(args)

