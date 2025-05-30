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
    from amdsmi import (
        amdsmi_init,
        amdsmi_get_processor_handles,
        amdsmi_get_clk_freq,
        amdsmi_get_clock_info,
        amdsmi_get_gpu_asic_info,
        amdsmi_get_gpu_memory_total,
        amdsmi_get_gpu_subsystem_name,
        amdsmi_get_gpu_vram_info,
        AmdSmiClkType,
        AmdSmiMemoryType,
    )
    have_amdsmi = True
except ModuleNotFoundError:
    have_amdsmi = False

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

def find_amd_gpus():
    amdsmi_init()
    return amdsmi_get_processor_handles()

def print_amd_gpu_info(handle):
    device_name = amdsmi_get_gpu_subsystem_name(handle)
    device_total_mem = amdsmi_get_gpu_memory_total(handle, AmdSmiMemoryType.VRAM)
    asic_info = amdsmi_get_gpu_asic_info(handle)
    vram_info = amdsmi_get_gpu_vram_info(handle)
    sys_clock_info = amdsmi_get_clock_info(handle, AmdSmiClkType.SYS)
    mem_clock_info = amdsmi_get_clock_info(handle, AmdSmiClkType.MEM)
    bus_width = vram_info['vram_bit_width']
    memory_speed_gbps = 8 * 2 * mem_clock_info['max_clk'] # DDR, 8 transfers/cycle
    memory_bandwidth = memory_speed_gbps * bus_width / 8 # bits -> bytes
    print(f"  Name: {device_name}")
    if 'target_graphics_version' in asic_info:
        print(f"  Graphics version: {asic_info['target_graphics_version']}")
    print(f"  Total device memory: {device_total_mem/2**30:g} GiB")
    print(f"  Device memory speed: {memory_speed_gbps/1e3:g} Gb/s")
    print(f"  Device memory bus width: {bus_width} bits")
    print(f"  Device memory bandwidth: {memory_bandwidth/1e3:g} GB/s")
    print(f"  Maximum clock speed: {sys_clock_info['max_clk']/1e3:g} GHz")
    if 'num_compute_units' in asic_info:
        print(f"  Number of multiprocessors: {asic_info['num_compute_units']}")

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

    if have_amdsmi:
        amd_handles = find_amd_gpus()
        amd_count = len(amd_handles)
        print(f"Found {amd_count} ROCm device" + ("s" if amd_count != 1 else "") + ".")
        if amd_count >= 0:
            for i, handle in enumerate(amd_handles):
                print()
                print(f"Device {i}:")
                print_amd_gpu_info(handle)
    else:
        amd_count = 0

    if nvidia_count + amd_count == 0:
        print("No GPU found.")

if __name__ == '__main__':
    main()
