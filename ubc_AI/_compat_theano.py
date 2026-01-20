def patch_numpy_distutils_blas_opt_info():
    """
    Patch for Theano expecting numpy.distutils.__config__.blas_opt_info
    under newer numpy where it may be missing.
    Must run BEFORE importing theano.
    """
    try:
        import numpy as np
        # numpy.distutils may be missing in very new numpy builds
        import numpy.distutils  # noqa: F401
        from numpy.distutils import __config__  # noqa: F401
    except Exception:
        return  # can't patch, will handle via error message elsewhere

    try:
        cfg = np.distutils.__config__
        if not hasattr(cfg, "blas_opt_info"):
            # fall back candidates seen in some numpy builds
            if hasattr(cfg, "blas_ilp64_opt_info"):
                cfg.blas_opt_info = cfg.blas_ilp64_opt_info
            else:
                cfg.blas_opt_info = {}
    except Exception:
        return

