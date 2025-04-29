from Args import Args
import pystac_client
import odc.stac
import pickle

def f():
    args= Args.from_cli()

    catalog = pystac_client.Client.open("https://explorer.dea.ga.gov.au/stac")
    odc.stac.configure_rio(
        cloud_defaults=True,
        aws={"aws_unsigned": True},
    )
    collections = ['ga_s2am_ard_3']
    query = catalog.search(
        bbox=args.bbox,
        collections=collections,
        datetime=f'{str(args.start_time)}/{str(args.end_time)}'
    )
    items = list(query.items())
    ds = odc.stac.load(
        items,
        bands=["nbart_red"],
        crs="utm",
        resolution=10,
        groupby="solar_day",
        bbox=args.bbox,
    )
    
    with open(args.path_out, 'wb') as handle:
        pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(ds)
    return ds

def t(): f()

if __name__ == '__main__':
    t()
