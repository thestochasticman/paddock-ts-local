from PaddockTS.query import Query
from PaddockTS.paddock_conf import PaddockConf


class PaddockConf:
  ha: int = 10
  input2: int = 5

  user_mask_file: str = 'path'

def get_outputs(query: Query, paddock_conf: PaddockConf):
    download_ds2(stub, query)
    if not paddock_conf: user_mask_file:
        