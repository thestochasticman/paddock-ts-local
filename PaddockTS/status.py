from os.path import exists
from PaddockTS.query import Query


def status(query: Query) -> dict[str, bool]:
    s = query.stub
    return {
        'sentinel2_video': exists(f'{query.out_dir}/{s}_sentinel2.mp4'),
        'sentinel2_paddocks_video': exists(f'{query.out_dir}/{s}_sentinel2_paddocks.mp4'),
        'vegfrac_video': exists(f'{query.out_dir}/{s}_vegfrac.mp4'),
        'vegfrac_paddocks_video': exists(f'{query.out_dir}/{s}_vegfrac_paddocks.mp4'),
    }
