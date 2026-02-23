from attrs import frozen

@frozen
class OzWALD:
    base_url = 'https://thredds.nci.org.au/thredds/dodsC/ub8/au/OzWALD'

    daily_meteo = {
        'Pg':      {'path': 'daily/meteo/Pg',      'file': 'OzWALD.daily.Pg.{year}.nc'},
        'Tmax':    {'path': 'daily/meteo/Tmax',     'file': 'OzWALD.Tmax.{year}.nc'},
        'Tmin':    {'path': 'daily/meteo/Tmin',     'file': 'OzWALD.Tmin.{year}.nc'},
        'Uavg':    {'path': 'daily/meteo/Uavg',     'file': 'OzWALD.Uavg.{year}.nc'},
        'Ueff':    {'path': 'daily/meteo/Ueff',     'file': 'OzWALD.Ueff.{year}.nc'},
        'VPeff':   {'path': 'daily/meteo/VPeff',    'file': 'OzWALD.VPeff.{year}.nc'},
        'kTavg':   {'path': 'daily/meteo/kTavg',    'file': 'OzWALD.kTavg.{year}.nc'},
        'kTeff':   {'path': 'daily/meteo/kTeff',    'file': 'OzWALD.kTeff.{year}.nc'},
        'DWLReff': {'path': 'daily/meteo/DWLReff',  'file': 'OzWALD.DWLReff.{year}.nc'},
    }

    eight_day = {
        'LAI':   {'path': '8day/LAI',   'file': 'OzWALD.LAI.{year}.nc'},
        'GPP':   {'path': '8day/GPP',   'file': 'OzWALD.GPP.{year}.nc'},
        'NDVI':  {'path': '8day/NDVI',  'file': 'OzWALD.NDVI.{year}.nc'},
        'EVI':   {'path': '8day/EVI',   'file': 'OzWALD.EVI.{year}.nc'},
        'PV':    {'path': '8day/PV',    'file': 'OzWALD.PV.{year}.nc'},
        'NPV':   {'path': '8day/NPV',   'file': 'OzWALD.NPV.{year}.nc'},
        'BS':    {'path': '8day/BS',    'file': 'OzWALD.BS.{year}.nc'},
        'FMC':   {'path': '8day/FMC',   'file': 'OzWALD.FMC.{year}.nc'},
        'Qtot':  {'path': '8day/Qtot',  'file': 'OzWALD.Qtot.{year}.nc'},
        'Ssoil': {'path': '8day/Ssoil', 'file': 'OzWALD.Ssoil.{year}.nc'},
        'OW':    {'path': '8day/OW',    'file': 'OzWALD.OW.{year}.nc'},
        'SN':    {'path': '8day/SN',    'file': 'OzWALD.SN.{year}.nc'},
        'Alb':   {'path': '8day/Alb',   'file': 'OzWALD.Alb.{year}.nc'},
    }

    def get_url(self, variable: str, year: int) -> str:
        info = self.daily_meteo.get(variable) or self.eight_day.get(variable)
        filename = info['file'].format(year=year)
        return f'{self.base_url}/{info["path"]}/{filename}'
