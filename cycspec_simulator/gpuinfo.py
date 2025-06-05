import sys
try:
    from cuda.bindings.driver import (
        CUresult,
        CUdevice_attribute,
        cuInit,
        cuDeviceGetCount,
        cuDeviceGetName,
        cuDeviceTotalMem,
        cuDeviceGetAttribute,
    )
    have_cuda = True
except ModuleNotFoundError:
    have_cuda = False
try:
    from hip.hip import (
        hipError_t,
        hipDeviceAttribute_t,
        hipInit,
        hipGetDeviceCount,
        hipDeviceGetName,
        hipDeviceTotalMem,
        hipDeviceGetAttribute,
    )
    have_hip = True
except ModuleNotFoundError:
    have_hip = False

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

def find_nvidia_gpus():
    try:
        err, = cuInit(0)
    except RuntimeError as e:
        print(f"Could not locate CUDA libraries ({e}).")
        return 0
    if err:
        if err.name == 'CUDA_ERROR_NO_DEVICE':
            print(f"No CUDA devices found")
            return 0
        else:
            raise CudaError(err)
    err, device_count = cuDeviceGetCount()
    if err:
        raise CudaError(err)

    return device_count

def print_nvidia_gpu_info(device_id):
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
    memory_speed_gbps = 2 * memory_clock_rate # double data rate
    bus_width = get_cuda_attribute('GLOBAL_MEMORY_BUS_WIDTH', device_id)
    memory_bandwidth = memory_speed_gbps * bus_width / 8 # bits -> bytes
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
    max_total_threads = max_threads_per_smp*n_smp
    max_sharedmem_per_block = get_cuda_attribute('MAX_SHARED_MEMORY_PER_BLOCK', device_id)
    max_sharedmem_per_smp = get_cuda_attribute('MAX_SHARED_MEMORY_PER_MULTIPROCESSOR', device_id)
    print(f"  Name: {device_name}")
    print(f"  Compute capability: {cc_major}.{cc_minor}")
    print(f"  Total device memory: {device_total_mem/2**30:g} GiB")
    print(f"  Device memory speed: {memory_speed_gbps/1e6:g} Gb/s")
    print(f"  Device memory bus width: {bus_width} bits")
    print(f"  Device memory bandwidth: {memory_bandwidth/1e6:g} GB/s")
    print(f"  Maximum clock speed: {clock_rate/1e6:g} GHz")
    print(f"  Number of multiprocessors: {n_smp}")
    print(f"  Max. threads per block: {max_threads_per_block}")
    print(f"  Max. threads per multiprocessor: {max_threads_per_smp}")
    if has_maxblocks:
        print(f"  Max. blocks per multiprocessor: {max_blocks_per_smp}")
    print(f"  Max. total in-flight threads: {max_total_threads}")
    print(f"  Max. shared memory per block: {max_sharedmem_per_block/1024:g} kiB")
    print(f"  Max. shared memory per multiprocessor: {max_sharedmem_per_smp/1024:g} kiB")

class HipError(Exception):
    def __init__(self, err):
        message = err.name.removeprefix('hipError')
        super().__init__(message)
        self.message = message

def get_hip_attribute(attr_name, device_id):
    attr_id = getattr(hipDeviceAttribute_t, f'hipDeviceAttribute{attr_name}')
    err, attr = hipDeviceGetAttribute(attr_id, device_id)
    if err:
        raise HipError(err)
    else:
        return attr

def find_amd_gpus():
    try:
        err, = hipInit(0)
    except RuntimeError as e:
        print(f"Could not locate HIP libraries ({e}).")
        return 0
    if err:
        if err.name == 'hipErrorNoDevice':
            print(f"No ROCm devices found")
            return 0
        else:
            raise HipError(err)
    err, device_count = hipGetDeviceCount()
    if err:
        raise HipError(err)

    return device_count

def print_amd_gpu_info(device_id):
    has_maxblocks = True
    err, device_name = hipDeviceGetName(128, device_id)
    if err:
        raise HipError(err)
    device_name = str(device_name)
    err, device_total_mem = hipDeviceTotalMem(device_id)
    if err:
        raise HipError(err)
    clock_rate = get_hip_attribute('ClockRate', device_id)
    memory_clock_rate = get_hip_attribute('MemoryClockRate', device_id)
    memory_speed_gbps = 8 * 2 * memory_clock_rate # double data rate, 8 transfers/cycle
    bus_width = get_hip_attribute('MemoryBusWidth', device_id)
    memory_bandwidth = memory_speed_gbps * bus_width / 8 # bits -> bytes
    gfx_major = get_hip_attribute('ComputeCapabilityMajor', device_id)
    gfx_minor = get_hip_attribute('ComputeCapabilityMinor', device_id)
    n_wgp = get_hip_attribute('MultiprocessorCount', device_id)
    try:
        max_blocks_per_wgp = get_hip_attribute('MaxBlocksPerMultiProcessor', device_id)
    except CudaError as error:
        if error.message == 'InvalidValue':
            has_maxblocks = False
        else:
            raise
    max_threads_per_block = get_hip_attribute('MaxThreadsPerBlock', device_id)
    max_threads_per_wgp = get_hip_attribute('MaxThreadsPerMultiProcessor', device_id)
    max_total_threads = max_threads_per_wgp*n_wgp
    max_sharedmem_per_block = get_hip_attribute('MaxSharedMemoryPerBlock', device_id)
    max_sharedmem_per_wgp = get_hip_attribute('MaxSharedMemoryPerMultiprocessor', device_id)
    print(f"  Name: {device_name}")
    print(f"  GFX version: {gfx_major}.{gfx_minor}")
    print(f"  Total device memory: {device_total_mem/2**30:g} GiB")
    print(f"  Device memory speed: {memory_speed_gbps/1e6:g} Gb/s")
    print(f"  Device memory bus width: {bus_width} bits")
    print(f"  Device memory bandwidth: {memory_bandwidth/1e6:g} GB/s")
    print(f"  Maximum clock speed: {clock_rate/1e6:g} GHz")
    print(f"  Number of multiprocessors: {n_wgp}")
    print(f"  Max. threads per block: {max_threads_per_block}")
    print(f"  Max. threads per multiprocessor: {max_threads_per_wgp}")
    if has_maxblocks:
        print(f"  Max. blocks per multiprocessor: {max_blocks_per_wgp}")
    print(f"  Max. total in-flight threads: {max_total_threads}")
    print(f"  Max. shared memory per block: {max_sharedmem_per_block/1024:g} kiB")
    print(f"  Max. shared memory per multiprocessor: {max_sharedmem_per_wgp/1024:g} kiB")

def find_gpus():
    if have_cuda:
        nvidia_count = find_nvidia_gpus()
        print(f"Found {nvidia_count} CUDA device" + ("s" if nvidia_count != 1 else "") + ".")
        if nvidia_count >= 0:
            for device_id in range(nvidia_count):
                print()
                print(f"Device {device_id}:")
                print_nvidia_gpu_info(device_id)
    else:
        nvidia_count = 0

    if have_hip:
        amd_count = find_amd_gpus()
        print(f"Found {amd_count} ROCm device" + ("s" if amd_count != 1 else "") + ".")
        if amd_count >= 0:
            for device_id in range(amd_count):
                print()
                print(f"Device {device_id}:")
                print_amd_gpu_info(device_id)
    else:
        amd_count = 0

    if nvidia_count + amd_count == 0:
        print("No GPU found.")

if __name__ == '__main__':
    main()
