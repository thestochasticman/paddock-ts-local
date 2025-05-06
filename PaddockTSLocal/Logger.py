from dataclasses import field
from os.path import expanduser
from types import NoneType
from dataclasses import dataclass
from typing_extensions import Self
from PaddockTSLocal.Query import Query
from argparse import ArgumentParser
from datetime import date
from os.path import exists
from os import makedirs
from os.path import join
from json import dump
from json import load


@dataclass(frozen=True)
class Logger:
    dir             : str = expanduser('~/PaddockTSLocalData')
    downloads_dir   : str = field(init=False)
    samgeo_dir      : str = field(init=False)

    path_stubs_mapping  = property(lambda s: join(s.downloads_dir, 'stubs_mapping.json'))
    stubs_mapping       = property(lambda s: load(open(s.path_stubs_mapping)))

    def ensure_dir_exists(s: Self)->None:
        if not exists(s.downloads_dir): makedirs(s.downloads_dir)

    def ensure_stubs_mapping_exists(s: Self)->None:
        if not exists(s.path_stubs_mapping):
            dump({}, open(s.path_stubs_mapping, 'w+'))
    

    def __post_init__(s: Self):
        object.__setattr__(s, 'downloads_dir', join(s.dir, 'Downloads'))
        object.__setattr__(s, 'samgeo_dir', join(s.dir, 'Samgeo'))
        s.ensure_dir_exists()
        s.ensure_stubs_mapping_exists()

    def get_path_dataset(s: Self, stub: str | NoneType, query: Query | NoneType)->str:
        if stub is None:
            stubs_mapping: dict = s.stubs_mapping
            query_string = query.__str__()
            stub = next((k for k, v in stubs_mapping.items() if v == query_string), None)
            if not stub:
                stub = str(len(stubs_mapping) + 1)
                stubs_mapping[stub] = query_string
                dump(stubs_mapping, open(s.path_stubs_mapping, 'w+'))
            
        path_out = join(s.downloads_dir, f"{stub}_raw_ds2.pkl")
        return path_out
    
    def get_path_presegment_tiff(s: Self, stub: str | NoneType, query: Query | None)->str:
        if stub is None:
            stubs_mapping: dict = s.stubs_mapping
            query_string = query.__str__()
            stub = next((k for k, v in stubs_mapping.items() if v == query_string), None)
            if not stub:
                stub = str(len(stubs_mapping) + 1)
                stubs_mapping[stub] = query_string
                dump(stubs_mapping, open(s.path_stubs_mapping, 'w+'))
            
        path_out = join(s.downloads_dir, f"{stub}.tif")
        return path_out
    
    def get_path_samgeo_mask(s: Self, stub: str | NoneType, query: Query | NoneType)->str:
        if stub is None:
            stubs_mapping: dict = s.stubs_mapping
            query_string = query.__str__()
            stub = next((k for k, v in stubs_mapping.items() if v == query_string), None)
            if not stub:
                stub = str(len(stubs_mapping) + 1)
                stubs_mapping[stub] = query_string
                dump(stubs_mapping, open(s.path_stubs_mapping, 'w+'))
            
        path_out = join(s.samgeo_dir, f"{stub}.tif")
        return path_out

    @classmethod
    def from_cli(cls):
        parser = ArgumentParser()
        parser.add_argument('--dir', type=str, default=expanduser('~/PaddockTSLocalData'),
                            help="Directory to store output files and stub mappings")
        args, _ = parser.parse_known_args()
        return cls(dir=args.dir)
    
def t():
    logger = Logger.from_cli()
    # query = Query.from_cli()
    query = Query(
        lat=-35.28,
        lon=149.13,
        buffer=0.1,
        start_time=date(2023, 1, 1),
        end_time=date(2023, 1, 31),
        collections=["ga_s2am_ard_3", "ga_s2bm_ard_3"],
        bands=["nbart_red", "nbart_green", "nbart_blue"]
    )
    return logger.get_path_query_dataset(None, query)

if __name__ == '__main__': print(t())