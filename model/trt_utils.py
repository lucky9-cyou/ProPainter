from typing import Optional, List
import numpy as np
import tensorrt as trt
from cuda import cuda, cudart

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
import ctypes
from math import prod
import torch
import cupy as cp


def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


class DeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""

    def __init__(
        self,
        size: int,
        dtype: np.dtype,
        name: Optional[str] = None,
        shape: Optional[trt.Dims] = None,
        format: Optional[trt.TensorFormat] = None,
    ):
        nbytes = size * dtype.itemsize
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes
        self._name = name
        self._shape = shape
        self._format = format
        self._dtype = dtype

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def shape(self) -> Optional[trt.Dims]:
        return self._shape

    @property
    def format(self) -> Optional[trt.TensorFormat]:
        return self._format

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __str__(self):
        return f"Device:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))


def allocate_buffers(engine: trt.ICudaEngine, profile_idx: Optional[int] = None):
    inputs = []
    outputs = []
    bindings = []
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for binding in tensor_names:
        # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
        # Pick out the max shape to allocate enough memory for the binding.

        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            continue

        shape = engine.get_tensor_shape(binding)
        shape_valid = np.all([s >= 0 for s in shape])

        if not shape_valid and profile_idx is None:
            # hard code the shape to 24 for now
            shape[0] = 18
            print(
                f"Binding {binding} has dynamic shape, "
                + "but no profile was specified."
            )
        size = trt.volume(shape)

        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(binding)))

        # Allocate host and device buffers
        bindingMemory = DeviceMem(size, dtype, name=binding, shape=shape, format=format)

        # Append the device buffer to device bindings.
        bindings.append(int(bindingMemory.device))

        # Append to the appropriate list.
        outputs.append(bindingMemory)
    return inputs, outputs, bindings


def free_outputs(outputs: List[DeviceMem], stream: cudart.cudaStream_t):
    for mem in outputs:
        mem.free()
    cuda_call(cudart.cudaStreamDestroy(stream))


# Frees the resources allocated in allocate_buffers
def free_buffers(
    inputs: List[DeviceMem], outputs: List[DeviceMem], stream: cudart.cudaStream_t
):
    for mem in inputs + outputs:
        mem.free()
    cuda_call(cudart.cudaStreamDestroy(stream))


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
    )


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
    )


def _do_inference_base(outputs, execute_sync):
    execute_sync()
    return [out.device for out in outputs]


# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, outputs):

    def execute_sync():
        context.execute_v2(bindings=bindings)

    return _do_inference_base(outputs, execute_sync)


def ptr_to_tensor(device_ptr: int, nbytes: int, shape: tuple):
    mem = cp.cuda.UnownedMemory(device_ptr, nbytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, offset=0)
    arr = cp.ndarray(shape, dtype=cp.float16, memptr=memptr)
    return torch.as_tensor(arr, device="cuda")
