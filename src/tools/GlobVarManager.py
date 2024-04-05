def _init():
    global _global_dict
    _global_dict = {}

def set(key, value):
    _global_dict[key] = value

def get(key):
    return _global_dict[key]