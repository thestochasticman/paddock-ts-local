from attrs import frozen, Factory as F

@frozen
class Sentinel2:
    stub: str = 'Sentinel2'
    stac_url: str = "https://explorer.dea.ga.gov.au/stac"
    collections: tuple[str, ...] = (
        "ga_s2am_ard_3",
        "ga_s2bm_ard_3",
    )
    bands: tuple[str, ...] = (
        "nbart_blue",
        "nbart_green",
        "nbart_red",
        "nbart_red_edge_1",
        "nbart_red_edge_2",
        "nbart_red_edge_3",
        "nbart_nir_1",
        "nbart_nir_2",
        "nbart_swir_2",
        "nbart_swir_3",
    )

    cloud_mask_band: str = "oa_fmask"
    cloud_cover_filter: dict = {"op": "<", "args": [{"property": "eo:cloud_cover"}, 0.10],}
    crs: str = "EPSG:6933"
    resolution: int = 10
    groupby: str = "solar_day"

defaultsentinel2 = Sentinel2()

print(defaultsentinel2.cloud_cover_filter)
