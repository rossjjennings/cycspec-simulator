from numba import cuda

try:
    import cupy
    current_gpu = cuda.gpus.current
except (ImportError, cuda.CudaSupportError) as e:
    have_cuda = False
    cuda_failure = e
else:
    have_cuda = True
    cuda_failure = None
