import pystac_client
import odc.stac
import matplotlib
from matplotlib import pyplot as plt

catalog = pystac_client.Client.open("https://explorer.dea.ga.gov.au/stac")

odc.stac.configure_rio(
    cloud_defaults=True,
    aws={"aws_unsigned": True},
)

# Set a bounding box
# [xmin, ymin, xmax, ymax] in latitude and longitude
bbox = [149.05, -35.32, 149.17, -35.25]

# Set a start and end date
start_date = "2021-12-01"
end_date = "2021-12-31"

# Set product ID as the STAC "collection"
collections = ["ga_ls8c_ard_3"]


# Build a query with the parameters above
query = catalog.search(
    bbox=bbox,
    collections=collections,
    datetime=f"{start_date}/{end_date}",
)

# Search the STAC catalog for all items matching the query
items = list(query.items())
print(f"Found: {len(items):d} datasets")


ds = odc.stac.load(
    items,
    bands=["nbart_red"],
    crs="utm",
    resolution=30,
    groupby="solar_day",
    bbox=bbox,
)


ds.nbart_red.plot(col="time", robust=True)
plt.show()