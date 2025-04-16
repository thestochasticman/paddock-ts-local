from dea_tools.datahandling import load_ard
from .Args import Args


def f(args: Args):
    query = {k: v for k, v in args.query.items() if k != 'centre'}
    ds = load_ard(
        
    )