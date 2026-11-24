"""Run ray tasks with one shared RBF matrix per process pool."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from copy import copy
from multiprocessing import shared_memory

import numpy as np
from scipy.sparse import csr_array


# These methods use the grid geometry or velocity grid, but never the RBF matrix.
_METHODS_WITHOUT_RBF = {"crkernel_tensor", "safe_crkernel_tensor", "safe_ttime_only"}


def _init_ray_worker(grid, method_name, rbf_spec):
    global _ray_method, _rbf_blocks
    _rbf_blocks = []
    if rbf_spec is not None:
        matrix_shape, array_specs = rbf_spec
        arrays = []
        for name, shape, dtype in array_specs:
            block = shared_memory.SharedMemory(name=name)
            _rbf_blocks.append(block)  # Keep the views alive for the worker's lifetime.
            array = np.ndarray(shape, dtype=dtype, buffer=block.buf)
            array.flags.writeable = False
            arrays.append(array)
        grid.rbf_weights = csr_array(tuple(arrays), shape=matrix_shape, copy=False)

    _ray_method = getattr(grid, method_name)


def _process_ray(path):
    return _ray_method(path)


@contextmanager
def ray_results(grid, method_name, paths, nu_cpu):
    """Send only paths as tasks and release shared memory after the workers exit."""
    worker_grid = copy(grid)
    worker_grid.rbf_weights = None
    with ExitStack() as stack:
        matrix = grid.rbf_weights
        rbf_spec = None
        if matrix is not None and method_name not in _METHODS_WITHOUT_RBF:
            array_specs = []
            for array in (matrix.data, matrix.indices, matrix.indptr):
                block = shared_memory.SharedMemory(create=True, size=max(1, array.nbytes))
                stack.callback(block.close)
                stack.callback(block.unlink)
                view = np.ndarray(array.shape, dtype=array.dtype, buffer=block.buf)
                view[...] = array
                del view
                array_specs.append((block.name, array.shape, array.dtype.str))
            rbf_spec = (matrix.shape, array_specs)

        with ProcessPoolExecutor(
            max_workers=nu_cpu, initializer=_init_ray_worker,
            initargs=(worker_grid, method_name, rbf_spec),
        ) as executor:
            yield executor.map(_process_ray, paths, chunksize=1)
