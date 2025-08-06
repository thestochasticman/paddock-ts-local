from PaddockTS.filter import Filter

"""
PaddockTS/filter.py

Apart from datetime, geographical coordinates and collections, there are 
many other criteria based on which the user might want to select data.

Some common properties are

eo:cloud_cover            : Estimated cloud coverage percentage (0–100)
datetime                  : Day-Start/Month-Start/Year-Start/Day-end/Month-end/Year-end
platform                  : one of landsat-8, 'sentinel-2a, etc
gsd                       : ground sampling distance(meters)
dea:dataset_maturity      : final, interim, etc
dea:product_level         : level3, level2
dea:product               : ga_s2am_ard_3, etc
odc:processing_datetime   : When the dataset was processed
proj:shape                : Shape
proj:transform            : Transform

There are many more properties, we will try to add example usage for all soon.
"""

### The filter object to create a filter where we want the
### eo:cloud_cover to be les than 10 can be created as follows
### f1, f2 are same

f1 = Filter.lt('eo:cloud_cover', 10)
f2 = Filter.from_string('eo:cloud_cover < 10')

### The filter object to choose sentinel-2a data can selected as follows
### Wrap the string value of the property in quotes when using Filter.from_string
### f4, f4, f5 are same

f3 = Filter.eq('platform', 'sentinel-2a')
f4 = Filter.from_string('platform == "sentinel-2a"')
f5 = Filter.from_string('platform == "sentinel-2a"')


### To combine a filter object which uses sentinel-2a platform and contains
### records with eo:cloud_cover < 10

f6 = Filter.AND(f1, f3)
f7 = f1.AND(f3)
f8 = Filter.from_string('platform == "sentinel-2a" and en:cloud_cover < 10')
### Hope this is clear now
