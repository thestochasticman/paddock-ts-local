import os
import rasterio

api_key = 'VjBCeUNPcldRYzdWaE9RbS4ybnRURCFKSTFuV3BcYlRhcCU5dls2Vzd8VlY9TURrZU9MdE1bOHcqeiEsYz5jV1RBOXshWX0qLWhWRDhiRi9O'

os.environ['GDAL_HTTP_USERPWD'] = f"apikey:{api_key}"
os.environ['GDAL_HTTP_MAX_RETRY'] = '3'
os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = '.tif'
os.environ['GDAL_HTTP_COOKIEFILE'] = '/tmp/cookies.txt'
os.environ['GDAL_HTTP_COOKIEJAR'] = '/tmp/cookies.txt'

url = "/vsicurl/https://data.tern.org.au/landscapes/slga/NationalMaps/SoilAndLandscapeGrid/CLY/CLY_005_015_EV_N_P_AU_NAT_C.tif"

with rasterio.open(url) as src:
    print("SUCCESS:", src.shape)