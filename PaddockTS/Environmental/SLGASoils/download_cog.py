from rasterio.errors import RasterioIOError
from rasterio.windows import from_bounds
from .utils import _setup_tern_auth
from .utils import load_tern_api_key
from .utils import get_cog_url
import rasterio

def download_cog(
    bbox: tuple,
    attribute: str,
    depth: str,
    filename: str,
    api_key: str = None
) -> None:
    
    api_key = load_tern_api_key(api_key) if api_key is None else api_key
    _setup_tern_auth(api_key)
    url = get_cog_url(attribute, depth)
    
    try:
        with rasterio.open(f'/vsicurl/{url}') as src:
            window = from_bounds(*bbox, src.transform)
            data = src.read(1, window=window)
            window_transform = src.window_transform(window)
            profile = src.profile.copy()
            profile.update({
                'height': data.shape[0],
                'width': data.shape[1],
                'transform': window_transform,
            })
            with rasterio.open(filename, 'w', **profile) as dst:
                dst.write(data, 1)
    except RasterioIOError as e:
        raise RuntimeError(f'Failed to access COG for {attribute} {depth}: {e}')    
    

if __name__ == '__main__':
    bbox = (147.35, -35.12, 147.36, -35.11)  # ~1km x 1km

    download_cog(
        bbox=bbox,
        attribute='Clay',
        depth='5-15cm',
        filename='test_clay.tif',
    )