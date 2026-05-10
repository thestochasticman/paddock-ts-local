# Fractional cover

Spectral unmixing of Sentinel-2 surface reflectance into bare ground (`bg`),
green vegetation (`pv`), and non-green vegetation (`npv`) fractions using
a TFLite MLP model adapted from
[fractionalcover3](https://github.com/jrsrp/fractionalcover3) by Robert
Denham (MIT-licensed; see `PaddockTS/LICENSES/fractionalcover3.LICENSE`).

Four model variants are bundled, indexed `n=1..4` by complexity. Default
is `n=4` (largest, highest accuracy).

::: PaddockTS.FractionalCover.compute_fractional_cover
