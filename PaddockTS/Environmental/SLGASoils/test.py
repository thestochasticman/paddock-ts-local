from .load_tern_api import load_tern_api
import os
import rasterio

api_key = load_tern_api()
url = "/vsicurl/https://data.tern.org.au/landscapes/slga/NationalMaps/SoilAndLandscapeGrid/CLY/CLY_005_015_EV_N_P_AU_NAT_C.tif"

# Method 1: Basic auth (what we tried)
print("Method 1: GDAL_HTTP_USERPWD")
os.environ['GDAL_HTTP_USERPWD'] = f"apikey:{api_key}"
try:
    with rasterio.open(url) as src:
        print("  SUCCESS:", src.shape)
except Exception as e:
    print("  FAILED:", e)

# Method 2: Custom header
print("\nMethod 2: X-API-Key header")
os.environ.pop('GDAL_HTTP_USERPWD', None)
os.environ['GDAL_HTTP_HEADERS'] = f"X-API-Key: {api_key}"
try:
    with rasterio.open(url) as src:
        print("  SUCCESS:", src.shape)
except Exception as e:
    print("  FAILED:", e)

# Method 3: Authorization Bearer
print("\nMethod 3: Authorization Bearer")
os.environ['GDAL_HTTP_HEADERS'] = f"Authorization: Bearer {api_key}"
try:
    with rasterio.open(url) as src:
        print("  SUCCESS:", src.shape)
except Exception as e:
    print("  FAILED:", e)

# Method 4: Just the key as username (no "apikey:" prefix)
print("\nMethod 4: Just key as password")
os.environ.pop('GDAL_HTTP_HEADERS', None)
os.environ['GDAL_HTTP_USERPWD'] = f":{api_key}"
try:
    with rasterio.open(url) as src:
        print("  SUCCESS:", src.shape)
except Exception as e:
    print("  FAILED:", e)