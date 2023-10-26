import sys
from cuda.cuda import (
    CUresult,
    CUdevice_attribute,
    cuInit,
    cuDeviceGetCount,
    cuDeviceGetName,
    cuDeviceTotalMem,
    cuDeviceGetAttribute,
)

class CudaError(Exception):
    def __init__(self, err):
        message = err.name.removeprefix('CUDA_ERROR_')
        super().__init__(message)
        self.message = message

def get_cuda_attribute(attr_name, device_id):
    attr_id = getattr(CUdevice_attribute, f'CU_DEVICE_ATTRIBUTE_{attr_name}')
    err, attr = cuDeviceGetAttribute(attr_id, device_id)
    if err:
        raise CudaError(err)
    else:
        return attr

def print_device_info():
    try:
        err, = cuInit(0)
    except RuntimeError as e:
        print(f"Could not locate CUDA libraries ({e}).")
        sys.exit(0)
    if err:
        if err.name == 'CUDA_ERROR_NO_DEVICE':
            print(f"No CUDA devices found")
            sys.exit(0)
        else:
            raise CudaError(err)
    err, device_count = cuDeviceGetCount()
    if err:
        raise CudaError(err)

    print(f"Found {device_count} CUDA device" + ("s" if device_count != 1 else "") + ".")
    for device_id in range(device_count):
        has_maxblocks = True
        err, device_name = cuDeviceGetName(128, device_id)
        if err:
            raise CudaError(err)
        device_name = device_name.strip(b'\x00')
        device_name = device_name.decode('ascii')
        err, device_total_mem = cuDeviceTotalMem(device_id)
        if err:
            raise CudaError(err)
        clock_rate = get_cuda_attribute('CLOCK_RATE', device_id)
        memory_clock_rate = get_cuda_attribute('MEMORY_CLOCK_RATE', device_id)
        bus_width = get_cuda_attribute('GLOBAL_MEMORY_BUS_WIDTH', device_id)
        memory_bandwidth = memory_clock_rate * 2 * bus_width / 8 # DDR, then bits -> bytes
        cc_major = get_cuda_attribute('COMPUTE_CAPABILITY_MAJOR', device_id)
        cc_minor = get_cuda_attribute('COMPUTE_CAPABILITY_MINOR', device_id)
        n_smp = get_cuda_attribute('MULTIPROCESSOR_COUNT', device_id)
        try:
            max_blocks_per_smp = get_cuda_attribute('MAX_BLOCKS_PER_MULTIPROCESSOR', device_id)
        except CudaError as error:
            if error.message == 'INVALID_VALUE':
                has_maxblocks = False
            else:
                raise
        max_threads_per_block = get_cuda_attribute('MAX_THREADS_PER_BLOCK', device_id)
        max_threads_per_smp = get_cuda_attribute('MAX_THREADS_PER_MULTIPROCESSOR', device_id)
        max_sharedmem_per_block = get_cuda_attribute('MAX_SHARED_MEMORY_PER_BLOCK', device_id)
        max_sharedmem_per_smp = get_cuda_attribute('MAX_SHARED_MEMORY_PER_MULTIPROCESSOR', device_id)
        print()
        print(f"Device {device_id}:")
        print(f"  Name: {device_name}")
        print(f"  Total global memory: {device_total_mem/2**30:g} GiB")
        print(f"  Maximum clock speed: {clock_rate/1e6:g} GHz")
        print(f"  Maximum effective memory clock: {memory_clock_rate/1e6:g} GHz")
        print(f"  Global memory bus width: {bus_width} bits")
        print(f"  Memory bandwidth: {memory_bandwidth/1e6:g} GB/s")
        print(f"  Compute capability: {cc_major}.{cc_minor}")
        print(f"  Number of multiprocessors: {n_smp}")
        if has_maxblocks:
            print(f"  Max. blocks per multiprocessor: {max_blocks_per_smp}")
        print(f"  Max. threads per block: {max_threads_per_block}")
        print(f"  Max. threads per multiprocessor: {max_threads_per_smp}")
        print(f"  Max. shared memory per block: {max_sharedmem_per_block/1024:g} kiB")
        print(f"  Max. shared memory per multiprocessor: {max_sharedmem_per_smp/1024:g} kiB")

if __name__ == '__main__':
    print_device_info()
    
