import torch
import ctypes
import numpy as np
from functools import partial
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

from functools import reduce


lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def pre_gather(inputShape, indexShape, axis):
    result_tuple = inputShape[:axis] + indexShape + inputShape[axis+1:]
    total_data=reduce(lambda x, y: x * y, inputShape)
    total_ind=reduce(lambda x, y: x * y, indexShape)
    total_out=reduce(lambda x, y: x * y, result_tuple)
    
    result_tensor = torch.tensor(result_tuple)
    return result_tensor ,total_out,total_data,total_ind

def gather(rank, axis, inputTensor, indexTensor):
    indices = [slice(None)] * rank
    indices[axis] = indexTensor
    outTensor = inputTensor[tuple(indices)]
    return outTensor
def test(inputShape, indexShape, axis, test_dtype, device):
    print(
        f"Testing gather on {device} with x_shape:{inputShape} , indice_shape:{indexShape}, axis:{axis} ,dtype:{test_dtype}"
    )
    M=inputShape[0]
    N=inputShape[1]
    m=indexShape[0]
    n=indexShape[1]
    inputTensor = torch.rand(inputShape, device=device, dtype=test_dtype)

    index = np.random.randint(0, inputShape[axis], indexShape).astype(np.int32)
    indexTensor = torch.from_numpy(index).to(torch.int64).to(device)

    array_gather,total_out,total_data,total_ind=pre_gather(inputShape,indexShape,axis)
    array_gather = array_gather.to(device)
    rank = len(inputShape)
    outTensor = gather(rank, axis, inputTensor, indexTensor)#

    Q_output = torch.zeros(outTensor.shape, device=device, dtype=test_dtype)
    input_ptr = ctypes.cast(inputTensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    index_ptr = ctypes.cast(indexTensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(Q_output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))


    if test_dtype == torch.float32:
        if device == "cuda":
            lib.gather_my.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
            
        ]
            torch_gather_time = performance.CudaProfile((gather, (rank, axis, inputTensor, indexTensor)))
            custom_gather_time = performance.CudaProfile((
            lib.gather_my,
            (input_ptr, index_ptr, output_ptr, M,N, m,n,axis)
        ))
    if test_dtype == torch.float16:
        if device == "cuda":
            lib.gather_my_16.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int

            
        ]
            torch_gather_time = performance.CudaProfile((gather, (rank, axis, inputTensor, indexTensor)))
            custom_gather_time = performance.CudaProfile((
            lib.gather_my_16,
            (input_ptr, index_ptr, output_ptr, M,N, m,n,axis)
        ))
    performance.logBenchmark(torch_gather_time, custom_gather_time)

    tmpa = outTensor.to('cpu').numpy().flatten()
    tmpb = Q_output.to('cpu').numpy().flatten()

    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)

    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))


def test2(inputShape, indexShape, axis, test_dtype, device):
    print(
        f"Testing high_dim gather on {device} with x_shape:{inputShape} , indice_shape:{indexShape}, axis:{axis} ,dtype:{test_dtype}"
    )
    N=len(inputShape)
    M=len(indexShape)
    inputTensor = torch.rand(inputShape, device=device, dtype=test_dtype)

    index = np.random.randint(0, inputShape[axis], indexShape).astype(np.int32)
    indexTensor = torch.from_numpy(index).to(torch.int64).to(device)

    array_gather,total_out,total_data,total_ind=pre_gather(inputShape,indexShape,axis)
    array_gather = array_gather.to(device)
    rank = len(inputShape)
    outTensor = gather(rank, axis, inputTensor, indexTensor)#

    Q_output = torch.zeros(outTensor.shape, device=device, dtype=test_dtype)
    input_ptr = ctypes.cast(inputTensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    index_ptr = ctypes.cast(indexTensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(Q_output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    array_gather_ptr=ctypes.cast(array_gather.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    j=total_data*total_ind//total_out

    if test_dtype == torch.float32:
        if device == "cuda":
            lib.gather_dim_h.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
            
        ]
            torch_gather_time = performance.CudaProfile((gather, (rank, axis, inputTensor, indexTensor)))
            custom_gather_time = performance.CudaProfile((
            lib.gather_dim_h,
            (input_ptr, index_ptr, output_ptr, array_gather_ptr,M,N, total_out,total_ind,total_data,j,axis)
        ))
    if test_dtype == torch.float16:
        if device == "cuda":
            lib.gather_dim_h.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
            
        ]
            torch_gather_time = performance.CudaProfile((gather, (rank, axis, inputTensor, indexTensor)))
            custom_gather_time = performance.CudaProfile((
            lib.gather_dim_h_16,
            (input_ptr, index_ptr, output_ptr, array_gather_ptr,M,N, total_out,total_ind,total_data,j,axis)
        ))
    performance.logBenchmark(torch_gather_time, custom_gather_time)

    tmpa = outTensor.to('cpu').numpy().flatten()
    tmpb = Q_output.to('cpu').numpy().flatten()

    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)

    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
parser = argparse.ArgumentParser(description="Test softmax on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    
test_cases = [
        # inputShape , indexShape, axis, test_dtype, device
        ((3, 2), (2, 2), 0, torch.float32, "cuda"),
        ((3, 2), (1, 2), 1, torch.float32, "cuda"),
        ((50257, 768), (16, 1024), 0, torch.float32, "cuda"),

        ((3, 2), (2, 2), 0, torch.float16, "cuda"),
        ((3, 2), (1, 2), 1, torch.float16, "cuda"),
        ((50257, 768), (16, 1024), 0, torch.float16, "cuda"),
         
]

test_cases2 = [
        # inputShape , indexShape, axis, test_dtype, device
        ((3, 2,2,4), (2, 2), 1, torch.float16, "cuda"),
        ((3, 2,2,5), (2, 2), 0, torch.float16, "cuda"),
        ((5, 2,2,4), (1,2, 2), 1, torch.float16, "cuda"),
        ((3, 2,2), (2, 2), 2, torch.float16, "cuda"),

        ((3, 2,2,4), (2, 2), 1, torch.float16, "cuda"),
        ((3, 2,2,5), (2, 2), 0, torch.float16, "cuda"),
        ((5, 2,2,4), (1,2, 2), 1, torch.float16, "cuda"),
        ((3, 2,2), (2, 2), 2, torch.float16, "cuda"),
         
]

filtered_test_cases = [
    (inputShape , indexShape, axis, test_dtype, device)
    for inputShape , indexShape, axis, test_dtype, device in test_cases
    if device == args.device
]

filtered_test_cases2 = [
    (inputShape , indexShape, axis, test_dtype, device)
    for inputShape , indexShape, axis, test_dtype, device in test_cases2
    if device == args.device
]

if args.device == 'mlu':
    import torch_mlu
# 执行过滤后的测试用例
for inputShape , indexShape, axis, test_dtype, device in filtered_test_cases:
    test(inputShape , indexShape, axis, test_dtype, device)

for inputShape , indexShape, axis, test_dtype, device in filtered_test_cases2:
    test2(inputShape , indexShape, axis, test_dtype, device)