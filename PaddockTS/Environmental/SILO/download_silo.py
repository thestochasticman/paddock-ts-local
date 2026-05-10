"""Download SILO climate data for the AOI centre point.

`SILO <https://www.longpaddock.qld.gov.au/silo/>`_ is the Queensland
Government's daily climate database for Australia, with a continuous
record from 1889 onward via spatial interpolation of station data.
This module hits the SILO DataDrillDataset CGI endpoint at the centre
of ``query.bbox``, requests every available variable, drops the
``_source`` columns and metadata, and writes a tidy CSV.

Requires an email address to be registered with SILO via
``~/.config/PaddockTS.json`` (``"email": "..."``) or passed explicitly
to :func:`download_silo`.
"""

from PaddockTS.query import Query
from PaddockTS.config import config
from .silo import SILO
from os import makedirs
from os.path import exists
from urllib.request import urlopen
from io import StringIO
import pandas as pd

silo = SILO()
get_filename = lambda q: f'{q.tmp_dir}/Environmental/{q.stub}_silo.csv'

BASE_URL = 'https://www.longpaddock.qld.gov.au/cgi-bin/silo/DataDrillDataset.php'
ALL_CODES = 'RXNJVDESCLFTAPWMHG'


def download_silo(query: Query, email: str = None):
    """Fetch SILO daily climate data for the centre of ``query.bbox``.

    Cached: if the output CSV already exists at
    ``{query.tmp_dir}/Environmental/{query.stub}_silo.csv``, it is loaded
    and returned without contacting SILO.

    Args:
        query: The :class:`PaddockTS.query.Query`.
        email: SILO registration email. If ``None``, falls back to
            ``config.email`` (read from ``~/.config/PaddockTS.json``).
            SILO uses this as the username; ``apirequest`` is sent as
            the password.

    Returns:
        pandas.DataFrame: One row per day in ``[query.start, query.end]``
        with a ``YYYY-MM-DD`` column and one column per climate variable
        (``daily_rain``, ``max_temp``, ``radiation``, ``vp``,
        ``et_short_crop``, etc.).

    Raises:
        ValueError: If no email is configured.
    """
    email = email or config.email
    if not email:
        raise ValueError('Set email in ~/.config/PaddockTS.json or pass email parameter')
    makedirs(f'{query.tmp_dir}/Environmental', exist_ok=True)
    filename = get_filename(query)

    if exists(filename):
        print(f'  cached: {filename}')
        return pd.read_csv(filename, parse_dates=['YYYY-MM-DD'])

    lat = (query.bbox[1] + query.bbox[3]) / 2
    lon = (query.bbox[0] + query.bbox[2]) / 2
    start = query.start.strftime('%Y%m%d')
    finish = query.end.strftime('%Y%m%d')

    url = (
        f'{BASE_URL}?lat={lat}&lon={lon}'
        f'&start={start}&finish={finish}'
        f'&format=csv&comment={ALL_CODES}'
        f'&username={email}&password=apirequest'
    )

    print(f'  fetching SILO data for ({lat:.2f}, {lon:.2f})...', flush=True)
    response = urlopen(url)
    text = response.read().decode('utf-8')

    df = pd.read_csv(StringIO(text))
    # Drop source columns and metadata
    source_cols = [c for c in df.columns if c.endswith('_source')]
    df = df.drop(columns=source_cols + ['metadata', 'latitude', 'longitude'], errors='ignore')

    df.to_csv(filename, index=False)
    print(f'  saved: {filename} ({len(df)} days, {len(df.columns)-1} variables)')
    return df


def test():
    from PaddockTS.utils import get_example_query
    df = download_silo(get_example_query())
    print(df.head(10))
    print(f'\nShape: {df.shape}')
    print(f'Columns: {list(df.columns)}')


if __name__ == '__main__':
    test()
