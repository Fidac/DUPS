import json
from types import SimpleNamespace

def parse_config(config_path):
    f = open(config_path)
    data = f.read()

    # Parse JSON into an object with attributes corresponding to dict keys.
    return json.loads(data, object_hook=lambda d: SimpleNamespace(**d))