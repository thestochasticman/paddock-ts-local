from attrs import frozen

@frozen
class SLGASoils:
    # Attribute codes follow the TERN SLGA COG layout under
    # data.tern.org.au/model-derived/slga/NationalMaps/SoilAndLandscapeGrid/<CODE>/
    # Codes verified against the directory listing on 2026-05-24. The old
    # 'PHC' (pH in CaCl2) and 'SOC' codes from the legacy esoil.io tree are
    # not present in this layout; use 'PHW' (pH in water) and consult the
    # upstream catalog if you need them.
    attribute_codes = {
        'Clay': 'CLY',
        'Silt': 'SLT',
        'Sand': 'SND',
        'pH_Water': 'PHW',
        'Bulk_Density': 'BDW',
        'Available_Water_Capacity': 'AWC',
        'Cation_Exchange_Capacity': 'CEC',
        'Effective_Cation_Exchange_Capacity': 'ECE',
        'Total_Nitrogen': 'NTO',
        'Total_Phosphorus': 'PTO',
        'Coarse_Fragments': 'CFG',
        'Depth_of_Soil': 'DES',
        'Depth_to_Rock': 'DER',
        'Drained_Upper_Limit': 'DUL',
        'L15': 'L15',
        'Available_Phosphorus': 'AVP',
    }
    depth_codes = {
        '0-5cm': ('000', '005'),
        '5-15cm': ('005', '015'),
        '15-30cm': ('015', '030'),
        '30-60cm': ('030', '060'),
        '60-100cm': ('060', '100'),
        '100-200cm': ('100', '200'),
    }
    # SLGA is published in versioned releases under each attribute directory:
    #   {attr}/v1/{attr}_{ds}_{de}_EV_N_P_AU_NAT_C_20140801.tif   (Release 1, 2014)
    #   {attr}/v2/{attr}_{ds}_{de}_EV_N_P_AU_TRN_N_20210902.tif   (Release 2, 2021)
    # Release 2 is the newer/recommended dataset; switch ``url_template`` to
    # the v1 line if you specifically need the original 2014 layer.
    url_template = 'https://data.tern.org.au/model-derived/slga/NationalMaps/SoilAndLandscapeGrid/{attr_code}/v2/{attr_code}_{depth_start}_{depth_end}_EV_N_P_AU_TRN_N_20210902.tif'