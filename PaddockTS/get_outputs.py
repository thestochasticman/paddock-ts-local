from PaddockTS.query import Query
from PaddockTS.Data.download_ds2 import download_ds2
from PaddockTS.Data.environmental import download_environmental_data
from PaddockTS.PaddockSegmentation.get_paddocks import get_paddocks
from PaddockTS.IndicesAndVegFrac.add_indices_and_veg_frac import add_indices_and_veg_frac
from PaddockTS.PaddockTS.get_paddock_ts import get_paddock_ts
from PaddockTS.Plotting.checkpoint_plots import plot as plot_checkpoints
from PaddockTS.Plotting.topographic_plots import plot_topography

def get_outputs( 
    query: Query,
):
    download_ds2(query)
    download_environmental_data(query)
    get_paddocks(query)
    add_indices_and_veg_frac(query)
    get_paddock_ts(query)
    plot_checkpoints(query)
    plot_topography(query)

def test():
    from PaddockTS.query import get_example_query
    query = get_example_query()
    get_outputs(query)

if __name__ == '__main__':
    test()