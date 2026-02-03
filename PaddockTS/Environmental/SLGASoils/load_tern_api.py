from json import load
from os.path import expanduser

def load_tern_api(path: str=None):
    return load(open(path)) if path else load(open(expanduser('~/.configs/tern.json')))