from attrs import frozen

@frozen
class SILO:
    variables = [
        'daily_rain',
        'max_temp',
        'min_temp',
        'radiation',
        'vp',
        'vp_deficit',
        'evap_pan',
        'evap_syn',
        'evap_comb',
        'evap_morton_lake',
        'et_short_crop',
        'et_tall_crop',
        'et_morton_actual',
        'et_morton_potential',
        'et_morton_wet',
        'mslp',
        'rh_tmax',
        'rh_tmin',
    ]
