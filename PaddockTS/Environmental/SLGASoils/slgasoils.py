from attrs import frozen

@frozen
class SLGASoils:
    attribute_codes = {
        'Clay': 'CLY',
        'Silt': 'SLT',
        'Sand': 'SND',
        'pH_CaCl2': 'PHC',
        'Bulk_Density': 'BDW',
        'Available_Water_Capacity': 'AWC',
        'Effective_Cation_Exchange_Capacity': 'ECE',
        'Total_Nitrogen': 'NTO',
        'Total_Phosphorus': 'PTO',
        'Organic_Carbon': 'SOC',
        'Depth_of_Soil': 'DES',
    }
    depth_codes = {
        '0-5cm': ('000', '005'),
        '5-15cm': ('005', '015'),
        '15-30cm': ('015', '030'),
        '30-60cm': ('030', '060'),
        '60-100cm': ('060', '100'),
        '100-200cm': ('100', '200'),
    }
    
    url_template = 'https://data.tern.org.au/model-derived/slga/NationalMaps/SoilAndLandscapeGrid/{attr_code}/v2/{attr_code}_{depth_start}_{depth_end}_EV_N_P_AU_TRN_N_20210902.tif'