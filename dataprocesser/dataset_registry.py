# dataset_registry.py
DATASET_REGISTRY = {}

def register_dataset(name):
    def decorator(fn):
        DATASET_REGISTRY[name] = fn
        return fn
    return decorator
