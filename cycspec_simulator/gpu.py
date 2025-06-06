from abc import ABCMeta, abstractmethod
from io import StringIO
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
if have_cuda:
    from numba import cuda
if have_hip:
    from numba import hip

def get_current_device():
    if have_hip:
        return ROCmDevice(hip.get_current_device().id)
    elif have_cuda:
        return CUDADevice(cuda.get_current_device().id)
    else:
        return None

def get_cuda_devices():
    err, = cuInit(0)
    CUDAError.raise_if_necessary(err, "cuInit")
    err, device_count = cuDeviceGetCount()
    CUDAError.raise_if_necessary(err, "cuDeviceGetCount")

    devices = []
    for device_id in range(device_count):
        devices.append(CUDADevice(device_id))
    return devices

def get_rocm_devices():
    err, = hipInit(0)
    HIPError.raise_if_necessary(err, "hipInit")
    err, device_count = hipGetDeviceCount()
    HIPError.raise_if_necessary(err, "hipGetDeviceCount")

    devices = []
    for device_id in range(device_count):
        devices.append(ROCmDevice(device_id))
    return devices

def get_devices():
    devices = []
    if have_cuda:
        devices.extend(get_cuda_devices())
    if have_hip:
        devices.extend(get_rocm_devices())
    return devices

def print_devices():
    if have_hip:
        rocm_devices = get_rocm_devices()
        n_rocm = len(rocm_devices)
        print(f"Found {n_rocm} ROCm device{'s' if n_rocm != 1 else ''}.")
        for device in cuda_devices:
            print()
            print(device.get_description())
    elif have_cuda:
        cuda_devices = get_cuda_devices()
        n_cuda = len(cuda_devices)
        print(f"Found {n_cuda} CUDA device{'s' if n_cuda != 1 else ''}.")
        for device in cuda_devices:
            print()
            print(device.get_description())

class CUDAError(Exception):
    """
    Represents an error raised by CUDA libraries.
    """
    @classmethod
    def raise_if_necessary(self, err, func_name):
        if err != CUresult.CUDA_SUCCESS:
            name = err.name.removeprefix('CUDA_ERROR_')
            raise CUDAError(f"{name} in call to {call_name}")

class HIPError(Exception):
    """
    Represents an error raised by HIP libraries.
    """
    @classmethod
    def raise_if_necessary(self, err, func_name):
        if err != hipError_t.hipSuccess:
            name = err.name.removeprefix('hipError')
            raise HIPError(f"{name} in call to {func_name}")

class Device(metaclass=ABCMeta):
    """
    An object representing a GPU. Provides a nicer, more Pythonic interface
    for various properties that can be obtained through cuda-python or hip-python.
    """
    def get_description(self):
        """
        Print a short description of the GPU device, including values of various properties.
        """
        out = StringIO()
        print(f"{self.device_type} device {self.device_id}:")
        print(f"  Name: {self.name}", file=out)
        major, minor = self.hardware_version
        print(f"  {self.hardware_version_name}: {major}.{minor}", file=out)
        print(f"  Number of multiprocessors: {self.multiprocessor_count}", file=out)
        print(f"  Maximum clock speed: {self.max_clock_rate/1e6:g} GHz", file=out)
        print(f"  Total device memory: {self.total_memory/2**30:g} GiB", file=out)
        print(f"  Device memory speed: {self.memory_speed/1e6:g} Gb/s", file=out)
        print(f"  Device memory bus width: {self.memory_bus_width} bits", file=out)
        print(f"  Device memory bandwidth: {self.memory_bandwidth/1e6:g} GB/s", file=out)
        print(f"  Max. threads per block: {self.max_threads_per_block}", file=out)
        print(f"  Max. threads per multiprocessor: {self.max_threads_per_multiprocessor}", file=out)
        print(f"  Max. shared memory per block: {self.max_shared_mem_per_block/1024:g} kiB", file=out)
        print(f"  Max. shared memory per multiprocessor: {self.max_shared_mem_per_multiprocessor/1024:g} kiB", file=out)
        return out.getvalue().strip("\n")

    def describe(self):
        print(self.get_description())

    @property
    def memory_bandwidth(self):
        """
        Global memory bandwidth, in kB/s.
        """
        bandwidth = self.memory_speed * self.memory_bus_width
        bandwidth /= 8 # bits -> bytes
        return bandwidth

    @property
    @abstractmethod
    def device_type(self):
        pass

    @property
    @abstractmethod
    def name(self):
        """
        Device name, as reported by the driver.
        """
        pass

    @property
    @abstractmethod
    def total_memory(self):
        """
        Total global memory in bytes, as reported by the driver.
        """
        pass

    @property
    @abstractmethod
    def max_clock_rate(self):
        """
        Maximum GPU clock rate in kHz.
        """
        pass

    @property
    @abstractmethod
    def memory_speed(self):
        """
        Memory speed in kilobits per second.
        """
        pass

    @property
    @abstractmethod
    def memory_bus_width(self):
        """
        Memory bus width in bits.
        """
        pass

    @property
    @abstractmethod
    def multiprocessor_count(self):
        """
        Number of simultaneous multiprocessors (Nvidia) or workgroup processors (AMD).
        """
        pass

    @property
    @abstractmethod
    def max_threads_per_block(self):
        """
        Maximum allowed number of threads per block.
        """
        pass

    @property
    @abstractmethod
    def max_threads_per_multiprocessor(self):
        """
        Maximum allowed number of threads per multiprocessor.
        """
        pass

    @property
    @abstractmethod
    def max_shared_mem_per_block(self):
        """
        Maximum amount of shared memory allowed per thread block, in bytes.
        """
        pass

    @property
    @abstractmethod
    def max_shared_mem_per_multiprocessor(self):
        """
        Maximum amount of shared memory allowed per multiprocessor, in bytes.
        """
        pass

    @property
    @abstractmethod
    def hardware_version_name(self):
        pass

    @property
    @abstractmethod
    def hardware_version(self):
        """
        Compute capability (Nvidia) or GFX version (AMD) of the chip.
        """
        pass

class CUDADevice(Device):
    """
    A CUDA-capable Nvidia GPU.
    """
    def __init__(self, device_id):
        """
        A CUDA-capable Nvidia GPU.

        Parameters
        ----------
        device_id: The GPU's numerical ID.
        """
        cuInit(0)
        self.device_id = device_id

    @property
    def device_type(self):
        return "CUDA"

    @property
    def name(self):
        """
        Device name, as reported by the driver.
        """
        err, device_name = cuDeviceGetName(64, self.device_id)
        CUDAError.raise_if_necessary(err, "cuDeviceGetName")
        device_name = device_name.strip(b'\x00')
        device_name = device_name.decode('ascii')
        return device_name

    @property
    def total_memory(self):
        """
        Total global memory in bytes, as reported by the driver.
        """
        err, device_total_mem = cuDeviceTotalMem(self.device_id)
        CUDAError.raise_if_necessary(err, "cuDeviceTotalMem")
        return device_total_mem

    @property
    def max_clock_rate(self):
        """
        Maximum GPU clock rate in kHz.
        """
        err, clock_rate = cuDeviceGetAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
            self.device_id,
        )
        CUDAError.raise_if_necessary(err, "cuDeviceGetAttribute")
        return clock_rate

    @property
    def memory_speed(self):
        """
        Memory speed in kilobits per second.
        """
        err, mem_clock_rate = cuDeviceGetAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
            self.device_id,
        )
        CUDAError.raise_if_necessary(err, "cuDeviceGetAttribute")
        mem_clock_rate *= 2 # account for double data rate
        return mem_clock_rate

    @property
    def memory_bus_width(self):
        """
        Memory bus width in bits.
        """
        err, bus_width = cuDeviceGetAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
            self.device_id,
        )
        CUDAError.raise_if_necessary(err, "cuDeviceGetAttribute")
        return bus_width

    @property
    def multiprocessor_count(self):
        """
        Number of simultaneous multiprocessors on the GPU.
        """
        err, n_sm = cuDeviceGetAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            self.device_id,
        )
        CUDAError.raise_if_necessary(err, "cuDeviceGetAttribute")
        return n_sm

    @property
    def max_threads_per_block(self):
        """
        Maximum allowed number of threads per block.
        """
        err, n_threads = cuDeviceGetAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            self.device_id,
        )
        CUDAError.raise_if_necessary(err, "cuDeviceGetAttribute")
        return n_threads

    @property
    def max_threads_per_multiprocessor(self):
        """
        Maximum allowed number of threads per block.
        """
        err, n_threads = cuDeviceGetAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            self.device_id,
        )
        CUDAError.raise_if_necessary(err, "cuDeviceGetAttribute")
        return n_threads

    @property
    def max_shared_mem_per_block(self):
        """
        Maximum amount of shared memory allowed per thread block, in bytes.
        """
        err, n_bytes = cuDeviceGetAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            self.device_id,
        )
        CUDAError.raise_if_necessary(err, "cuDeviceGetAttribute")
        return n_bytes

    @property
    def max_shared_mem_per_multiprocessor(self):
        """
        Maximum amount of shared memory allowed per multiprocessor, in bytes.
        """
        err, n_bytes = cuDeviceGetAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
            self.device_id,
        )
        CUDAError.raise_if_necessary(err, "cuDeviceGetAttribute")
        return n_bytes

    @property
    def hardware_version_name(self):
        return "Compute capability"

    @property
    def hardware_version(self):
        """
        Major and minor parts of the compute capability version number.
        """
        err, cc_major = cuDeviceGetAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            self.device_id,
        )
        CUDAError.raise_if_necessary(err, "cuDeviceGetAttribute")
        err, cc_minor = cuDeviceGetAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            self.device_id,
        )
        CUDAError.raise_if_necessary(err, "cuDeviceGetAttribute")
        return cc_major, cc_minor


class ROCmDevice(Device):
    """
    A ROCm-capable AMD GPU.
    """
    def __init__(self, device_id):
        """
        A ROCm-capable AMD GPU.

        Parameters
        ----------
        device_id: The GPU's numerical ID.
        """
        hipInit(0)
        self.device_id = device_id

    @property
    def device_type(self):
        return "ROCm"

    @property
    def name(self):
        """
        Device name, as reported by the driver.
        """
        err, device_name = hipDeviceGetName(128, self.device_id)
        HIPError.raise_if_necessary(err, "hipDeviceGetName")
        device_name = str(device_name)
        return device_name

    @property
    def total_memory(self):
        """
        Total global memory in bytes, as reported by the driver.
        """
        err, device_total_mem = hipDeviceTotalMem(self.device_id)
        HIPError.raise_if_necessary(err, "hipDeviceTotalMem")
        return device_total_mem

    @property
    def max_clock_rate(self):
        """
        Maximum GPU clock rate in kHz.
        """
        err, clock_rate = hipDeviceGetAttribute(
            hipDeviceAttribute_t.hipDeviceAttributeClockRate,
            self.device_id,
        )
        HIPError.raise_if_necessary(err, "hipDeviceGetAttribute")
        return clock_rate

    @property
    def memory_speed(self):
        """
        Memory speed in kilobits per second.
        """
        err, mem_clock_rate = hipDeviceGetAttribute(
            hipDeviceAttribute_t.hipDeviceAttributeMemoryClockRate,
            self.device_id,
        )
        HIPError.raise_if_necessary(err, "hipDeviceGetAttribute")
        mem_clock_rate *= 8 # AMD reports this in MHz, not Mb/s
        mem_clock_rate *= 2 # account for double data rate
        return mem_clock_rate

    @property
    def memory_bus_width(self):
        """
        Memory bus width in bits.
        """
        err, bus_width = hipDeviceGetAttribute(
            hipDeviceAttribute_t.hipDeviceAttributeMemoryBusWidth,
            self.device_id,
        )
        HIPError.raise_if_necessary(err, "hipDeviceGetAttribute")
        return bus_width

    @property
    def multiprocessor_count(self):
        """
        Number of simultaneous multiprocessors on the GPU.
        """
        err, n_sm = hipDeviceGetAttribute(
            hipDeviceAttribute_t.hipDeviceAttributeMultiprocessorCount,
            self.device_id,
        )
        HIPError.raise_if_necessary(err, "hipDeviceGetAttribute")
        return n_sm

    @property
    def max_threads_per_block(self):
        """
        Maximum allowed number of threads per block.
        """
        err, n_threads = hipDeviceGetAttribute(
            hipDeviceAttribute_t.hipDeviceAttributeMaxThreadsPerBlock,
            self.device_id,
        )
        HIPError.raise_if_necessary(err, "hipDeviceGetAttribute")
        return n_threads

    @property
    def max_threads_per_multiprocessor(self):
        """
        Maximum allowed number of threads per block.
        """
        err, n_threads = hipDeviceGetAttribute(
            hipDeviceAttribute_t.hipDeviceAttributeMaxThreadsPerMultiProcessor,
            self.device_id,
        )
        HIPError.raise_if_necessary(err, "hipDeviceGetAttribute")
        return n_threads

    @property
    def max_shared_mem_per_block(self):
        """
        Maximum amount of shared memory allowed per thread block, in bytes.
        """
        err, n_bytes = hipDeviceGetAttribute(
            hipDeviceAttribute_t.hipDeviceAttributeMaxSharedMemoryPerBlock,
            self.device_id,
        )
        HIPError.raise_if_necessary(err, "hipDeviceGetAttribute")
        return n_bytes

    @property
    def max_shared_mem_per_multiprocessor(self):
        """
        Maximum amount of shared memory allowed per multiprocessor, in bytes.
        """
        err, n_bytes = hipDeviceGetAttribute(
            hipDeviceAttribute_t.hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
            self.device_id,
        )
        HIPError.raise_if_necessary(err, "hipDeviceGetAttribute")
        return n_bytes

    @property
    def hardware_version_name(self):
        return "GFX version"

    @property
    def hardware_version(self):
        """
        Major and minor parts of the compute capability version number.
        """
        err, gfx_major = hipDeviceGetAttribute(
            hipDeviceAttribute_t.hipDeviceAttributeComputeCapabilityMajor,
            self.device_id,
        )
        HIPError.raise_if_necessary(err, "hipDeviceGetAttribute")
        err, gfx_minor = hipDeviceGetAttribute(
            hipDeviceAttribute_t.hipDeviceAttributeComputeCapabilityMinor,
            self.device_id,
        )
        HIPError.raise_if_necessary(err, "hipDeviceGetAttribute")
        return gfx_major, gfx_minor
