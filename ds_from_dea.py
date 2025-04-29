
from dea_tools.datahandling import load_ard
from dea_tools.dask import create_local_dask_cluster
from Args import Args


##  /opt/homebrew/opt/postgresql@14/bin/postgres -D /opt/homebrew/var/postgresql@14
    

def load_and_process_data(dc, query):
    query = {k: v for k, v in query.items() if k not in ['centre', 'buffer']} # this takes centre out of the query	
    ds = load_ard(
        dc=dc,
        products=['ga_s2am_ard_3', 'ga_s2bm_ard_3'],
        cloud_mask='s2cloudless',
        min_gooddata=0.9,
        measurements=['nbart_blue', 'nbart_green', 'nbart_red', 
                      'nbart_red_edge_1', 'nbart_red_edge_2', 'nbart_red_edge_3',
                      'nbart_nir_1', 'nbart_nir_2',
                      'nbart_swir_2', 'nbart_swir_3'],
        **query
    )
    return ds

def f(args: Args):
    load_and_process_data(args.dc, args.query)

def t():
    args = Args.from_cli()
    f(args)

if __name__ == '__main__':
    t()