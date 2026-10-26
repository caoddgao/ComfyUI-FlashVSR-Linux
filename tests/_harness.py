"""Load the plugin like ComfyUI does (folder imported as a package), with stub ComfyUI modules.

Only for the CPU tests in this folder; inside ComfyUI the real modules are used.
"""
import importlib.util
import os
import sys
import tempfile
import types

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = "flashvsr_linux_under_test"
TMP = tempfile.mkdtemp(prefix="flashvsr_test_")


def _stub_comfy():
    if "comfy" in sys.modules:
        return
    import torch
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: torch.device("cpu")
    mm.unload_all_models = lambda: None
    mm.soft_empty_cache = lambda: None
    mm.throw_exception_if_processing_interrupted = lambda: None
    utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total):
            self.calls = []

        def update_absolute(self, value, total=None):
            self.calls.append((value, total))

    utils.ProgressBar = ProgressBar
    comfy.model_management, comfy.utils = mm, utils
    fp = types.ModuleType("folder_paths")
    fp.models_dir = os.path.join(TMP, "models")
    fp.get_output_directory = lambda: os.path.join(TMP, "output")
    fp.get_temp_directory = lambda: os.path.join(TMP, "temp")
    for d in (fp.models_dir, fp.get_output_directory(), fp.get_temp_directory()):
        os.makedirs(d, exist_ok=True)
    sys.modules.update({"comfy": comfy, "comfy.model_management": mm, "comfy.utils": utils, "folder_paths": fp})


def load_plugin():
    if PACKAGE in sys.modules:
        return sys.modules[PACKAGE]
    _stub_comfy()
    spec = importlib.util.spec_from_file_location(PACKAGE, os.path.join(PLUGIN_DIR, "__init__.py"),
                                                  submodule_search_locations=[PLUGIN_DIR])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)
    return module


def sub(name):
    load_plugin()
    return importlib.import_module(f"{PACKAGE}.{name}")
