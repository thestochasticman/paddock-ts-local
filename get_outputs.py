from PaddockTS.query import Query
from PaddockTS.paddock_conf import PaddockConf
from PaddockTS.ds2_conf import DS2Conf

from PaddockTS.Data.download_ds2 import download_ds2
from PaddockTS.Data.environmental import download_environmental_data
from PaddockTS.PaddockSegmentation.segment_paddocks import get_paddocks
from PaddockTS.IndicesAndVegFrac.add_indices_and_veg_frac import add_indices_and_veg_frac

from PaddockTS.PaddockTS.get_paddock_ts import get_paddock_ts
from PaddockTS.Plotting.checkpoint_plots import plot as plot_checkpoints
from PaddockTS.Plotting.topographic_plots import plot_topography

def get_outputs( 
    stub: str,
    query: Query
    ):
    download_ds2(stub, query)
    download_environmental_data(stub, query)
    get_paddocks(stub, min_area_ha=0.1)
    add_indices_and_veg_frac(stub)
    get_paddock_ts(stub)
    plot_checkpoints(stub)
    plot_topography(stub)

def test():
    from PaddockTS.query import get_example_query
    stub = 'test_example_query'
    get_outputs(stub, get_example_query())

if __name__ == '__main__':
    test()