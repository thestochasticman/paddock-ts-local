from os.path import exists
from PaddockTS.query import Query

def check_status(query: Query) -> bool:
    return all([
        exists(f'{query.out_dir}/{query.stub}_sentinel2.mp4'),
        exists(f'{query.out_dir}/{query.stub}_sentinel2_paddocks.mp4'),
        exists(f'{query.out_dir}/{query.stub}_vegfrac.mp4'),
        exists(f'{query.out_dir}/{query.stub}_vegfrac_paddocks.mp4'),
    ])

if __name__ == '__main__':
    from PaddockTS.utils import get_example_query
    check_status(get_example_query())
    