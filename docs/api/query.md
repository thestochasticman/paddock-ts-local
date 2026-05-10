# Query

The central data object that flows through the pipeline. A `Query` describes
a region (bounding box) and a time window; every stage reads inputs and
writes outputs under paths derived from it.

::: PaddockTS.query
