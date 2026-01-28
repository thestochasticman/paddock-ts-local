from attrs import frozen

@frozen
class FMask:
    """Sentinel-2 fmask classification values."""
    NODATA: int = 0
    VALID: int = 1
    CLOUD: int = 2
    SHADOW: int = 3
    SNOW: int = 4
    WATER: int = 5