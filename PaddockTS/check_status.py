from os.path import exists
from PaddockTS.query import Query

def check_status(query: Query) -> bool:
    s = query.stub
    return {
        'sentinel2_video': exists(f'{query.out_dir}/{s}_sentinel2.mp4'),
        'sentinel2_paddocks_video': exists(f'{query.out_dir}/{s}_sentinel2_paddocks.mp4'),
        'fractional_cover_video': exists(f'{query.out_dir}/{s}_fractional_cover.mp4'),
        'fractional_cover_paddocks_video': exists(f'{query.out_dir}/{s}_fractional_cover_paddocks.mp4'),
    }

if __name__ == '__main__':
    from PaddockTS.utils import get_example_query
    print(check_status(get_example_query()))
