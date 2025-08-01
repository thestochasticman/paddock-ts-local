from dataclasses import dataclass

@dataclass
class DS2Conf:
    num_workers: int = 4
    threads_per_worker: int = 2
    tile_width: int = 1024
    tile_height: int = 1024
    tile_time_series_length: int = 1


