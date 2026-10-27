import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
