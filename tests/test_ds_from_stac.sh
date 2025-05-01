##  /opt/homebrew/opt/postgresql@14/bin/postgres -D /opt/homebrew/var/postgresql@14


python3 ds_from_stac.py \
  --lat -34.3 \
  --lon 148.4 \
  --buffer 0.01 \
  --start_time 2020-01-01 \
  --end_time 2020-03-31 \
  --out_dir 'Data/shelter' \
  --stub 'yas'