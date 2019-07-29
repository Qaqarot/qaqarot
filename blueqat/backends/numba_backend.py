def numba_backend_lazy():
    from ._numba_backend_impl import NumbaBackend
    return NumbaBackend()
