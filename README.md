# PaddockTS

A Python toolkit for paddock-level remote sensing workflows. Features include querying data, calculating indices, fractional cover, segmentation, and visualising results as static maps and animations.

## Package Overview


```
├── env.yml                             # Conda environment specification
├── pyproject.toml                      # Package metadata & dependencies
├── PaddockTS/                          # Core library modules
│   ├── Data/                           # Data acquisition utilities (download, environmental)
│   │   ├── download_ds2.py               # Download Sentinel 2 data
│   │   └── environmental.py              # Download Environment Data(Silo, etc)
│   │
│   ├── IndicesAndVegFrac/              # Index and fractional cover calculations
│   │   ├── indices.py                    # Calculate Indices            
│   │   ├── veg_frac.py                   # Add fractional cover score per pixel using a pretrained model
│   │   ├── add_indices_and_veg_frac.py   # Run the above 2 steps
│   │   └── utils.py
│   │ 
│   ├── PaddockSegmentation/            # Paddock boundary segmentation routines
│   │   ├── _1_presegment.py              # Calculate NDWI Time Series and convert to a fourier image
│   │   ├── _2_segment.py                 # Take the fourier image and segment to get paddocks(maks or polygons).
│   │   ├── segment_paddocks.py           # Run the above 2 steps
│   │   └── utils.py                      # Some utilities for paddock_ts
│   │
│   ├── PaddockTS                       # Generate Paddock Time Series Data
│   │    ├──get_paddock_ts.py              # Generate PaddockTime Series Data
│   │
│   ├── Plotting/                         # Static plotting functions
│   │   ├── plotting_functions.py       # Plotting
│   │   ├── checkpoint_plots.py           # Checkpoint Plots
│   │   └── topographic_plots.py          # Topographic Plots
│   │ 
│   ├── filter.py           # STAC‐API filter builder
│   ├── legend.py           # File paths & configuration management
│   ├── query.py            # Query class to define area of interest
│   ├── get_outputs.py      # Wrapper to get all outputs from a given query
│   └── __init__.py
├── dist/                   # Built distributions
└── README.md               # This documentation

```

## Installation

Using Conda (recommended)

``conda env create -f env.yml``

``conda activate PaddockTSEnv``


## Configuration

By default, [legend.get_config()](/PaddockTS/legend.py) writes a JSON at ~/.configs/PaddockTSLocal.json with paths:

* **out_dir**: where outputs are saved
* **tmp_dir**: where intermediate files (DS2I, shapefiles) are stored
* **scratch_dir**: scratch workspace

Adjust these after first run or via environment variables.

