from attrs import frozen

@frozen
class SMIPS:
    wms_url: str = 'https://landscapes-mapserver.tern.org.au/smips/'
    layer: str = 'TotalBucketRaw'
    resolution_deg: float = 0.01  # ~1 km native resolution
    server_max_dim: int = 4110
    timeout: int = 60
    layers: tuple[str, ...] = (
        'TotalBucketRaw',
        'SMIndexRaw',
        'TotalBucketColoured',
        'SMIndexColoured',
    )
