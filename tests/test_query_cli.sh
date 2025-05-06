#!/bin/bash

# test_query.sh
python PaddockTSLocal/Query.py \
  --lat -35.28 \
  --lon 149.13 \
  --buffer 0.1 \
  --start_time 2023-01-01 \
  --end_time 2023-01-31 \
  --collections ga_s2am_ard_3 ga_s2bm_ard_3 \
  --bands nbart_red nbart_green nbart_blue