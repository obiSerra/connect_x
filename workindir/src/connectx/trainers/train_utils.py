import importlib


def load_module(module_name):
    module = importlib.import_module(f"connectx.models.{module_name}")
    return module
