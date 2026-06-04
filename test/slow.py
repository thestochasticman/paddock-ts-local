from datetime import date
from PaddockTS.query import Query
from PaddockTS.get_outputs import get_outputs

paddocks_fp = "artifacts/Milgadara_paddock-polygons_2024-12-17_12-45-58.json"

q = Query.build_from_paddocks(
    paddocks_filepath=paddocks_fp,
    start=date(2018, 1, 1),
    end=date(2025, 12, 31),
    stub="Migadara_2018-25",
    label_col="title",
)


from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
from PaddockTS.SpectralIndices.indices import compute_indices
from PaddockTS.FractionalCover import compute_fractional_cover
from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks
from PaddockTS.Phenology.make_paddock_time_series import make_paddock_time_series
from PaddockTS.Phenology.make_smoothed_paddock_time_series import make_smoothed_paddock_time_series
from PaddockTS.Phenology.make_yearly_paddock_time_series import make_yearly_paddock_time_series
from PaddockTS.Phenology.make_smoothed_paddock_time_series import make_smoothed_paddock_time_series

ds = download_sentinel2(q)                          # worked
ds = compute_indices(q, ds_sentinel2=ds)            # worked
fc = compute_fractional_cover(q, ds_sentinel2=ds)   # worked
#paddocks = get_paddocks(q, ds_sentinel2=ds)        #
ts = make_paddock_time_series(q, paddocks_filepath = paddocks_fp)      # worked
yearly = make_yearly_paddock_time_series(q, paddocks_filepath = paddocks_fp)	    #
smoothed = make_smoothed_paddock_time_series(      
    q, paddocks_filepath = paddocks_fp,
    days=10,          # 10-day median resample
    window_length=7,  # Savitzky-Golay window (odd; coerced if even)
    polyorder=2,      # SG polynomial order (< window_length)
)						    # worked

# did not proceed, test this after interactively. 



