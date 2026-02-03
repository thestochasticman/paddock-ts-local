from attrs import frozen

@frozen
class SLGASoils:
    abbreviations: dict = {
        "Clay": "https://www.asris.csiro.au/arcgis/services/TERN/CLY_ACLEP_AU_NAT_C/MapServer/WCSServer",
        "Silt": "https://www.asris.csiro.au/arcgis/services/TERN/SLT_ACLEP_AU_NAT_C/MapServer/WCSServer",
        "Sand": "https://www.asris.csiro.au/arcgis/services/TERN/SND_ACLEP_AU_NAT_C/MapServer/WCSServer",
        "pH_CaCl2": "https://www.asris.csiro.au/arcgis/services/TERN/PHC_ACLEP_AU_NAT_C/MapServer/WCSServer",
        "Bulk_Density": "https://www.asris.csiro.au/arcgis/services/TERN/BDW_ACLEP_AU_NAT_C/MapServer/WCSServer",
        "Available_Water_Capacity": "https://www.asris.csiro.au/arcgis/services/TERN/AWC_ACLEP_AU_NAT_C/MapServer/WCSServer",
        "Effective_Cation_Exchange_Capacity": "https://www.asris.csiro.au/arcgis/services/TERN/ECE_ACLEP_AU_NAT_C/MapServer/WCSServer",
        "Total_Nitrogen": "https://www.asris.csiro.au/arcgis/services/TERN/NTO_ACLEP_AU_NAT_C/MapServer/WCSServer",
        "Total_Phosphorus": "https://www.asris.csiro.au/arcgis/services/TERN/PTO_ACLEP_AU_NAT_C/MapServer/WCSServer"
    }
    identifiers = {
        "5-15cm": '4',
        "15-30cm":'8',
        "30-60cm":'12',
        "60-100cm":'16',
    }