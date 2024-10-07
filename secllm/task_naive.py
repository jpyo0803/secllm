from typing import Any
import torch
import torch.cuda.nvtx as nvtx

# from torch_int.functional.quantization import (
#     dynamic_quantize_activation_per_token_absmax,
# )

from transformers.models.llama.modeling_llama import (
    repeat_kv,
)

import cupy
from secllm.time_collector import TimeStamp

# import threading

MEASURE_TIME_WITH_NVTX = True

task_description = {
    0: 'Print Test',
    1: 'Broadcast 1',
    2: 'RMSNorm 1',
    3: 'Broadcast 2',
    4: '[Q proj] Move weight to GPU',
    5: '[K proj] Move weight to GPU',
    6: '[V proj] Move weight to GPU',
    7: '[Q proj] Quantize input',
    8: '[K proj] Quantize input',
    9: '[V proj] Quantize input',
    10: '[Q proj] Encrypt input',
    11: '[K proj] Encrypt input',
    12: '[V proj] Encrypt input',
    13: '[Q proj] Move input to GPU',
    14: '[K proj] Move input to GPU',
    15: '[V proj] Move input to GPU',
    16: '[Q proj] Matmul',
    17: '[K proj] Matmul',
    18: '[V proj] Matmul',
    19: '[Q proj] Move result to CPU',
    20: '[K proj] Move result to CPU',
    21: '[V proj] Move result to CPU',
    22: '[Q proj] Decrypt result',
    23: '[K proj] Decrypt result',
    24: '[V proj] Decrypt result',
    25: '[Q proj] Dequantize result',
    26: '[K proj] Dequantize result',
    27: '[V proj] Dequantize result',
    28: 'RoPE',
    29: '[QK^T] Key Generation',
    30: '[QK^T] Quantize Q input',
    31: '[QK^T] Quantize K input',
    32: '[QK^T] Shift Q input',
    33: '[QK^T] Shift K input',
    34: '[QK^T] Dec. Key Preproc.',
    35: '[QK^T] Encrypt Q input',
    36: '[QK^T] Encrypt K input',
    37: '[QK^T] Move Q input to GPU',
    38: '[QK^T] Move K input to GPU',
    39: '[QK^T] Matmul CPU',
    40: '[QK^T] Move result to CPU',
    41: '[QK^T] Decrypt result',
    42: '[QK^T] Unshift result',
    43: '[QK^T] Dequantize result',
    44: 'Softmax',
    45: '[PV] Key Generation',
    46: '[PV] Quantize P input',
    47: '[PV] Quantize V input',
    48: '[PV] Shift P input',
    49: '[PV] Shift V input',
    50: '[PV] Dec. Key Preproc.',
    51: '[PV] Encrypt P input',
    52: '[PV] Encrypt V input',
    53: '[PV] Move P input to GPU',
    54: '[PV] Move V input to GPU',
    55: '[PV] Matmul CPU',
    56: '[PV] Move result to CPU',
    57: '[PV] Decrypt result',
    58: '[PV] Unshift result',
    59: '[PV] Dequantize result',
    60: '[O proj] Move weight to GPU',
    61: '[O proj] Quantize input',
    62: '[O proj] Encrypt input',
    63: '[O proj] Move input to GPU',
    64: '[O proj] Matmul',
    65: '[O proj] Move result to CPU',
    66: '[O proj] Decrypt result',
    67: '[O proj] Dequantize result',
    68: 'Residual Add 1',
    69: 'Broadcast 3',
    70: 'RMSNorm 2',
    71: 'Broadcast 4',
    72: '[Gate proj] Move weight to GPU',
    73: '[Up proj] Move weight to GPU',
    74: '[Gate proj] Quantize input',
    75: '[Up proj] Quantize input',
    76: '[Gate proj] Encrypt input',
    77: '[Up proj] Encrypt input',
    78: '[Gate proj] Move input to GPU',
    79: '[Up proj] Move input to GPU',
    80: '[Gate proj] Matmul',
    81: '[Up proj] Matmul',
    82: '[Gate proj] Move result to CPU',
    83: '[Up proj] Move result to CPU',
    84: '[Gate proj] Decrypt result',
    85: '[Up proj] Decrypt result',
    86: '[Gate proj] Dequantize result',
    87: '[Up proj] Dequantize result',
    88: 'SwiGLU',
    89: '[Down proj] Move weight to GPU',
    90: '[Down proj] Quantize input',
    91: '[Down proj] Encrypt input',
    92: '[Down proj] Move input to GPU',
    93: '[Down proj] Matmul',
    94: '[Down proj] Move result to CPU',
    95: '[Down proj] Decrypt result',
    96: '[Down proj] Dequantize result',
    97: 'Residual Add 2',
    98: '[QK^T] Dec. Compute Add Factor',
    99: '[QK^T] Dec. Compute Mult. Factor',
    100: '[QK^T] Dec. Compute Unshift Factor',
    101: '[PV] Dec Compute Add Factor',
    102: '[PV] Dec Compute Mult Factor',
    103: '[PV] Dec Compute Unshift Factor',
    104: '[QK^T] Move K Cache to GPU',
    105: '[PV] Move V Cache to GPU',
}

def GetBookKeeperLinearIndex(layer_index, operation_index, input_index):
  # NOTE(jpyo0803): debugging purpose
  # Assume there are 105 operations in a layer  
  return layer_index * 330 + input_index * 110 + operation_index

class Task:
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        self.name = name
        self.layer_idx = layer_idx
        self.task_id = task_id
        self.next_task_ids = next_task_ids
        self.secllm_cpp_wrapper = secllm_cpp_wrapper
        self.model = model
        self.task_description = task_description[task_id]
        self.time_collector = time_collector


    def run(self):
        print(f"Task: {self.name, self.task_id, self.next_task_ids}")

    def print_info(self):
        print(f'Task Description: {self.task_description}, Task ID: {self.task_id}, layer_idx: {self.layer_idx})')

    def to_string_info(self):
        return f'Task Description: {self.task_description}, Task ID: {self.task_id}, layer_idx: {self.layer_idx})'

    def __call__(self, worker_id):
        self.run()

class Task0(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self, worker_id):
        self.run()

class Task1(Task):
    # Copy 
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        # Input is float
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0) # float
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self.secllm_cpp_wrapper.BroadcastTensor_Float(src, dst)

    def __call__(self, worker_id):
        self.run()

class Task2(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        # input is float
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        # input_layernorm = self.model.layers[self.layer_idx].input_layernorm
        self.secllm_cpp_wrapper.RMSNorm(self.layer_idx, src, dst, 0)

    def __call__(self, worker_id):
        self.run()

class Task3(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        # input is float
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        # Replicate from 3 to 7, 8, 9
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self.secllm_cpp_wrapper.BroadcastTensor_Float(src, dst)
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self, worker_id):
        self.run()

class Task4(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True # it does not depend on anything

    def run(self):
        # Move Q weight to GPU
        # def async_task():
        self.model.layers[self.layer_idx].q_proj_weight_buffer = self.model.layers[self.layer_idx].q_proj.weight
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        self.model.layers[self.layer_idx].q_proj_weight_buffer = self.model.layers[self.layer_idx].q_proj_weight_buffer.to('cuda:0')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        assert self.model.layers[self.layer_idx].q_proj_weight_buffer.dtype == torch.int8
        # threading.Thread(target=async_task).start()
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task5(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True # it does not depend on anything

    def run(self):
        # Move K weight to GPU
        # def async_task():
        self.model.layers[self.layer_idx].k_proj_weight_buffer = self.model.layers[self.layer_idx].k_proj.weight
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        self.model.layers[self.layer_idx].k_proj_weight_buffer = self.model.layers[self.layer_idx].k_proj_weight_buffer.to('cuda:0')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        assert self.model.layers[self.layer_idx].k_proj_weight_buffer.dtype == torch.int8
        # threading.Thread(target=async_task).start()

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task6(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True # it does not depend on anything

    def run(self):
        # Move V weight to GPU
        # def async_task():
        self.model.layers[self.layer_idx].v_proj_weight_buffer = self.model.layers[self.layer_idx].v_proj.weight
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        self.model.layers[self.layer_idx].v_proj_weight_buffer = self.model.layers[self.layer_idx].v_proj_weight_buffer.to('cuda:0')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        assert self.model.layers[self.layer_idx].v_proj_weight_buffer.dtype == torch.int8
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task7(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Q
        self.secllm_cpp_wrapper.QuantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task8(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.K
        self.secllm_cpp_wrapper.QuantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task9(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.V
        self.secllm_cpp_wrapper.QuantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task10(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Q
        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task11(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.K
        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task12(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.V
        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task13(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        # def async_task():
        act = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        act = act.to('cuda:0')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
        self.model.tensor_buffer[dst] = act
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task14(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        # def async_task():
        act = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        act = act.to('cuda:0')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
        self.model.tensor_buffer[dst] = act
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task15(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        # def async_task():
        act = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        act = act.to('cuda:0')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
        self.model.tensor_buffer[dst] = act
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task16(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        ready = True
        ready &= (self.model.layers[self.layer_idx].q_proj_weight_buffer is not None) and (self.model.layers[self.layer_idx].q_proj_weight_buffer.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
        return ready

    def run(self):
        # def async_task():
        act_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        assert self.model.tensor_buffer[act_loc] is not None

        act = self.model.tensor_buffer[act_loc]
        self.model.tensor_buffer[act_loc] = None

        act_shape = act.shape
        act = act.view(-1, act_shape[-1])

        weight = self.model.layers[self.layer_idx].q_proj_weight_buffer
        self.model.layers[self.layer_idx].q_proj_weight_buffer = None

        # At this point, act and weight are on GPU

        act = act.to(torch.int32)
        weight = weight.to(torch.int32)

        assert act.dtype == torch.int32
        assert weight.dtype == torch.int32

        act_cupy = cupy.from_dlpack(act)
        weight_T_cupy = cupy.from_dlpack(weight.transpose(-2, -1))

        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(f"Layer idx: {self.layer_idx}, Q proj Matmul")
        result_cupy = cupy.matmul(act_cupy, weight_T_cupy)
        cupy.cuda.Stream.null.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()

        result = torch.from_dlpack(result_cupy)

        result = result.view(*act_shape[:-1], -1)

        dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
        self.model.tensor_buffer[dst] = result
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task17(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        ready = True
        ready &= (self.model.layers[self.layer_idx].k_proj_weight_buffer is not None) and (self.model.layers[self.layer_idx].k_proj_weight_buffer.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
        return ready

    def run(self):
        # def async_task():
        act_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        assert self.model.tensor_buffer[act_loc] is not None

        act = self.model.tensor_buffer[act_loc]
        self.model.tensor_buffer[act_loc] = None

        act_shape = act.shape
        act = act.view(-1, act_shape[-1])

        weight = self.model.layers[self.layer_idx].k_proj_weight_buffer
        self.model.layers[self.layer_idx].k_proj_weight_buffer = None

        # At this point, act and weight are on GPU

        act = act.to(torch.int32)
        weight = weight.to(torch.int32)

        assert act.dtype == torch.int32
        assert weight.dtype == torch.int32

        act_cupy = cupy.from_dlpack(act)
        weight_T_cupy = cupy.from_dlpack(weight.transpose(-2, -1))

        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        result_cupy = cupy.matmul(act_cupy, weight_T_cupy)
        cupy.cuda.Stream.null.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()

        result = torch.from_dlpack(result_cupy)

        result = result.view(*act_shape[:-1], -1)

        dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
        self.model.tensor_buffer[dst] = result
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task18(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        ready = True
        ready &= (self.model.layers[self.layer_idx].v_proj_weight_buffer is not None) and (self.model.layers[self.layer_idx].v_proj_weight_buffer.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
        return ready

    def run(self):
        # def async_task():
        act_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        assert self.model.tensor_buffer[act_loc] is not None

        act = self.model.tensor_buffer[act_loc]
        self.model.tensor_buffer[act_loc] = None

        act_shape = act.shape
        act = act.view(-1, act_shape[-1])

        weight = self.model.layers[self.layer_idx].v_proj_weight_buffer
        self.model.layers[self.layer_idx].v_proj_weight_buffer = None

        # At this point, act and weight are on GPU

        act = act.to(torch.int32)
        weight = weight.to(torch.int32)

        assert act.dtype == torch.int32
        assert weight.dtype == torch.int32

        act_cupy = cupy.from_dlpack(act)
        weight_T_cupy = cupy.from_dlpack(weight.transpose(-2, -1))

        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        result_cupy = cupy.matmul(act_cupy, weight_T_cupy)
        cupy.cuda.Stream.null.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()

        result = torch.from_dlpack(result_cupy)

        result = result.view(*act_shape[:-1], -1)

        dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
        self.model.tensor_buffer[dst] = result
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task19(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
    
    def run(self):
        # def async_task():
        result_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        result = self.model.tensor_buffer[result_loc]
        self.model.tensor_buffer[result_loc] = None
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        result = result.to('cpu')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        self.secllm_cpp_wrapper.BookKeeperStore_Int32(self.layer_idx, self.next_task_ids[0], 0, result)
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task20(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
    
    def run(self):
        # def async_task():
        result_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        result = self.model.tensor_buffer[result_loc]
        self.model.tensor_buffer[result_loc] = None
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        result = result.to('cpu')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        self.secllm_cpp_wrapper.BookKeeperStore_Int32(self.layer_idx, self.next_task_ids[0], 0, result)
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task21(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
    
    def run(self):
        # def async_task():
        result_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        result = self.model.tensor_buffer[result_loc]
        self.model.tensor_buffer[result_loc] = None
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        result = result.to('cpu')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        self.secllm_cpp_wrapper.BookKeeperStore_Int32(self.layer_idx, self.next_task_ids[0], 0, result)
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task22(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Q
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task23(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.K
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task24(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.V
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task25(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)]
        type = self.secllm_cpp_wrapper.ProjectionType.Q
        self.secllm_cpp_wrapper.DequantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task26(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 1)]
        type = self.secllm_cpp_wrapper.ProjectionType.K
        self.secllm_cpp_wrapper.DequantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task27(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)]
        type = self.secllm_cpp_wrapper.ProjectionType.V
        self.secllm_cpp_wrapper.DequantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task28(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    # inputs are all floats
    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 1)
        return ready

    def run(self):
        query_states = self.secllm_cpp_wrapper.BookKeeperLoad_Float(self.layer_idx, self.task_id, 0)
        key_states = self.secllm_cpp_wrapper.BookKeeperLoad_Float(self.layer_idx, self.task_id, 1)
        # value_states = self.secllm_cpp_wrapper.BookKeeperLoad_Float(self.layer_idx, self.task_id, 2)

        bsz = self.model.layers[self.layer_idx].bsz
        q_len = self.model.layers[self.layer_idx].q_len
        num_heads = self.model.layers[self.layer_idx].num_heads
        num_key_value_heads = self.model.layers[self.layer_idx].num_key_value_heads
        head_dim = self.model.layers[self.layer_idx].head_dim
        inv_freq = self.model.layers[self.layer_idx].rotary_emb.inv_freq
        position_ids = self.model.layers[self.layer_idx].position_ids

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2).contiguous()
        # value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2).contiguous()

        cos, sin = self.secllm_cpp_wrapper.LlamaRotaryEmbedding(inv_freq, position_ids, torch.float32)
        query_states, key_states = self.secllm_cpp_wrapper.ApplyRotaryPosEmb(query_states, key_states, cos, sin)
        
        self.model.fake_cache = torch.empty_like(key_states, dtype=torch.int8)

        # Update KV cache with fake one to generate a correct causal mask
        self.model.layers[self.layer_idx].past_key_value.update_key(self.model.fake_cache, self.layer_idx)
        self.model.layers[self.layer_idx].past_key_value.update_value(self.model.fake_cache, self.layer_idx) 
        
        assert self.next_task_ids[0] == 30
        assert self.next_task_ids[1] == 31
        # assert self.next_task_ids[2] == 47

        self.secllm_cpp_wrapper.BookKeeperStore_Float(self.layer_idx, self.next_task_ids[0], 0, query_states.to(torch.float32))
        self.secllm_cpp_wrapper.BookKeeperStore_Float(self.layer_idx, self.next_task_ids[1], 0, key_states.to(torch.float32))
        # self.secllm_cpp_wrapper.BookKeeperStore_Float(self.layer_idx, self.next_task_ids[2], 0, value_states.to(torch.float32))

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task29(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task30(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)]
        assert self.next_task_ids[0] == 39

        self.secllm_cpp_wrapper.QuantizeQ_QK(self.layer_idx, src, dst)

    def __call__(self, worker_id):
        self.run()

class Task31(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 1)]
        assert self.next_task_ids[0] == 39

        self.secllm_cpp_wrapper.QuantizeK_QK(self.layer_idx, src, dst)

    def __call__(self, worker_id):
        self.run()

class Task32(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True

    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task33(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task34(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task35(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task36(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task37(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True

    def run(self):
        pass

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task38(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task39(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 1)
        return ready
    
    def run(self):
        q_src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        k_src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)
        assert self.next_task_ids[0] == 43
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)]

        self.secllm_cpp_wrapper.Matmul_CPU_QK(self.layer_idx, q_src, k_src, dst)
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        self.run()

class Task40(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task41(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task42(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True

    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task43(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        self.secllm_cpp_wrapper.Dequantize_QK(self.layer_idx, src, dst)

    def __call__(self, worker_id):
        self.run()

class Task44(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        dst.append(GetBookKeeperLinearIndex(self.layer_idx, 98, 1))
        self.secllm_cpp_wrapper.Softmax(self.layer_idx, src, dst)

    def __call__(self, worker_id):
        self.run()

class Task45(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task46(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)]
        self.secllm_cpp_wrapper.QuantizeP_PV(self.layer_idx, src, dst)

    def __call__(self, worker_id):
        self.run()

class Task47(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)
        
    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 1)]
        self.secllm_cpp_wrapper.QuantizeV_PV(self.layer_idx, src, dst)

    def __call__(self, worker_id):
        self.run()

class Task48(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)
        
    def is_ready(self):
        return True

    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task49(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)
        
    def is_ready(self):
        return True

    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task50(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task51(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task52(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task53(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task54(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task55(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 1)
        return ready

    def run(self):
        p_src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        q_src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)

        assert self.next_task_ids[0] == 59
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)]
        self.secllm_cpp_wrapper.Matmul_CPU_PV(self.layer_idx, p_src, q_src, dst)

    def __call__(self, worker_id):
        self.run()

class Task56(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task57(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True

    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task58(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task59(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self.secllm_cpp_wrapper.Dequantize_PV(self.layer_idx, src, dst)

    def __call__(self, worker_id):
        self.run()

class Task60(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True

    def run(self):
        # def async_task():
        self.model.layers[self.layer_idx].o_proj_weight_buffer = self.model.layers[self.layer_idx].o_proj.weight
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        self.model.layers[self.layer_idx].o_proj_weight_buffer = self.model.layers[self.layer_idx].o_proj_weight_buffer.to('cuda:0')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        assert self.model.layers[self.layer_idx].o_proj_weight_buffer.dtype == torch.int8
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task61(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.O
        self.secllm_cpp_wrapper.QuantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task62(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.O
        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task63(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        # def async_task():
        act = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        act = act.to('cuda:0')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
        self.model.tensor_buffer[dst] = act
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task64(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        ready = True
        ready &= (self.model.layers[self.layer_idx].o_proj_weight_buffer is not None) and (self.model.layers[self.layer_idx].o_proj_weight_buffer.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
        return ready
    
    def run(self):
        # def async_task():
        act_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        assert self.model.tensor_buffer[act_loc] is not None

        act = self.model.tensor_buffer[act_loc]
        self.model.tensor_buffer[act_loc] = None

        act_shape = act.shape
        act = act.view(-1, act_shape[-1])

        weight = self.model.layers[self.layer_idx].o_proj_weight_buffer
        self.model.layers[self.layer_idx].o_proj_weight_buffer = None

        act = act.to(torch.int32)
        weight = weight.to(torch.int32)

        assert act.dtype == torch.int32
        assert weight.dtype == torch.int32

        act_cupy = cupy.from_dlpack(act)
        weight_cupy = cupy.from_dlpack(weight.transpose(-2, -1))

        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        result_cupy = cupy.matmul(act_cupy, weight_cupy)
        cupy.cuda.Stream.null.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()

        result = torch.from_dlpack(result_cupy)

        result = result.view(*act_shape[:-1], -1)

        dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
        self.model.tensor_buffer[dst] = result
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task65(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None

    def run(self):
        # def async_task():
        result_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        result = self.model.tensor_buffer[result_loc]
        self.model.tensor_buffer[result_loc] = None
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        result = result.to('cpu')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        self.secllm_cpp_wrapper.BookKeeperStore_Int32(self.layer_idx, self.next_task_ids[0], 0, result)
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task66(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.O
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task67(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 1)]

        type = self.secllm_cpp_wrapper.ProjectionType.O
        self.secllm_cpp_wrapper.DequantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task68(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 1)
        return ready

    def run(self):
        src1 = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        src2 = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self.secllm_cpp_wrapper.ElementwiseAdd(self.layer_idx, src1, src2, dst, 0)

    def __call__(self, worker_id):
        self.run()

class Task69(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self.secllm_cpp_wrapper.BroadcastTensor_Float(src, dst)

    def __call__(self, worker_id):
        self.run()

class Task70(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        # post_attention_layernorm = self.model.layers[self.layer_idx].post_attention_layernorm
        self.secllm_cpp_wrapper.RMSNorm(self.layer_idx, src, dst, 1)

    def __call__(self, worker_id):
        self.run()

class Task71(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self.secllm_cpp_wrapper.BroadcastTensor_Float(src, dst)
        
    def __call__(self, worker_id):
        self.run()

class Task72(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True

    def run(self):
        # def async_task():
        self.model.layers[self.layer_idx].gate_proj_weight_buffer = self.model.layers[self.layer_idx].gate_proj.weight
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        self.model.layers[self.layer_idx].gate_proj_weight_buffer = self.model.layers[self.layer_idx].gate_proj_weight_buffer.to('cuda:0')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        assert self.model.layers[self.layer_idx].gate_proj_weight_buffer.dtype == torch.int8
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task73(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True

    def run(self):
        # def async_task():
        self.model.layers[self.layer_idx].up_proj_weight_buffer = self.model.layers[self.layer_idx].up_proj.weight
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        self.model.layers[self.layer_idx].up_proj_weight_buffer = self.model.layers[self.layer_idx].up_proj_weight_buffer.to('cuda:0')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        assert self.model.layers[self.layer_idx].up_proj_weight_buffer.dtype == torch.int8
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task74(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Gate
        self.secllm_cpp_wrapper.QuantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task75(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Up
        self.secllm_cpp_wrapper.QuantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task76(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Gate
        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task77(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Up
        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task78(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        # def async_task():
        act = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        act = act.to('cuda:0')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
        self.model.tensor_buffer[dst] = act
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task79(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        # def async_task():
        act = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        act = act.to('cuda:0')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
        self.model.tensor_buffer[dst] = act
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task80(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        ready = True
        ready &= (self.model.layers[self.layer_idx].gate_proj_weight_buffer is not None) and (self.model.layers[self.layer_idx].gate_proj_weight_buffer.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
        return ready
    
    def run(self):
        # def async_task():
        act_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        assert self.model.tensor_buffer[act_loc] is not None

        act = self.model.tensor_buffer[act_loc]
        self.model.tensor_buffer[act_loc] = None

        act_shape = act.shape
        act = act.view(-1, act_shape[-1])

        weight = self.model.layers[self.layer_idx].gate_proj_weight_buffer
        self.model.layers[self.layer_idx].gate_proj_weight_buffer = None

        act = act.to(torch.int32)
        weight = weight.to(torch.int32)

        assert act.dtype == torch.int32
        assert weight.dtype == torch.int32

        act_cupy = cupy.from_dlpack(act)
        weight_cupy = cupy.from_dlpack(weight.transpose(-2, -1))

        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        result_cupy = cupy.matmul(act_cupy, weight_cupy)
        cupy.cuda.Stream.null.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()

        result = torch.from_dlpack(result_cupy)

        result = result.view(*act_shape[:-1], -1)

        dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
        self.model.tensor_buffer[dst] = result
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task81(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        ready = True
        ready &= (self.model.layers[self.layer_idx].up_proj_weight_buffer is not None) and (self.model.layers[self.layer_idx].up_proj_weight_buffer.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
        return ready
    
    def run(self):
        # def async_task():
        act_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        assert self.model.tensor_buffer[act_loc] is not None

        act = self.model.tensor_buffer[act_loc]
        self.model.tensor_buffer[act_loc] = None

        act_shape = act.shape
        act = act.view(-1, act_shape[-1])

        weight = self.model.layers[self.layer_idx].up_proj_weight_buffer
        self.model.layers[self.layer_idx].up_proj_weight_buffer = None

        act = act.to(torch.int32)
        weight = weight.to(torch.int32)

        assert act.dtype == torch.int32
        assert weight.dtype == torch.int32

        act_cupy = cupy.from_dlpack(act)
        weight_cupy = cupy.from_dlpack(weight.transpose(-2, -1))

        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        result_cupy = cupy.matmul(act_cupy, weight_cupy)
        cupy.cuda.Stream.null.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()

        result = torch.from_dlpack(result_cupy)

        result = result.view(*act_shape[:-1], -1)

        dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
        self.model.tensor_buffer[dst] = result
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task82(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None

    def run(self):
        # def async_task():
        result_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        result = self.model.tensor_buffer[result_loc]
        self.model.tensor_buffer[result_loc] = None
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        result = result.to('cpu')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        self.secllm_cpp_wrapper.BookKeeperStore_Int32(self.layer_idx, self.next_task_ids[0], 0, result)
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task83(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None

    def run(self):
        # def async_task():
        result_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        result = self.model.tensor_buffer[result_loc]
        self.model.tensor_buffer[result_loc] = None
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        result = result.to('cpu')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        self.secllm_cpp_wrapper.BookKeeperStore_Int32(self.layer_idx, self.next_task_ids[0], 0, result)
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task84(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        
        type = self.secllm_cpp_wrapper.ProjectionType.Gate
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task85(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        
        type = self.secllm_cpp_wrapper.ProjectionType.Up
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task86(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)]
        
        type = self.secllm_cpp_wrapper.ProjectionType.Gate
        self.secllm_cpp_wrapper.DequantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task87(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 1)]
        
        type = self.secllm_cpp_wrapper.ProjectionType.Up
        self.secllm_cpp_wrapper.DequantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task88(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 1)
        return ready

    def run(self):
        src1 = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        src2 = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self. secllm_cpp_wrapper.SwiGLU(self.layer_idx, src1, src2, dst)

    def __call__(self, worker_id):
        self.run()

class Task89(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True

    def run(self):
        # def async_task():
        self.model.layers[self.layer_idx].down_proj_weight_buffer = self.model.layers[self.layer_idx].down_proj.weight
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        self.model.layers[self.layer_idx].down_proj_weight_buffer = self.model.layers[self.layer_idx].down_proj_weight_buffer.to('cuda:0')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        assert self.model.layers[self.layer_idx].down_proj_weight_buffer.dtype == torch.int8
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task90(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Down
        self.secllm_cpp_wrapper.QuantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task91(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Down
        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task92(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        # def async_task():
        act = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        act = act.to('cuda:0')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
        self.model.tensor_buffer[dst] = act
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task93(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        ready = True
        ready &= (self.model.layers[self.layer_idx].down_proj_weight_buffer is not None) and (self.model.layers[self.layer_idx].down_proj_weight_buffer.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
        return ready

    def run(self):
        # def async_task():
        act_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        assert self.model.tensor_buffer[act_loc] is not None

        act = self.model.tensor_buffer[act_loc]
        self.model.tensor_buffer[act_loc] = None

        act_shape = act.shape
        act = act.view(-1, act_shape[-1])

        weight = self.model.layers[self.layer_idx].down_proj_weight_buffer
        self.model.layers[self.layer_idx].down_proj_weight_buffer = None

        act = act.to(torch.int32)
        weight = weight.to(torch.int32)

        assert act.dtype == torch.int32
        assert weight.dtype == torch.int32

        act_cupy = cupy.from_dlpack(act)
        weight_cupy = cupy.from_dlpack(weight.transpose(-2, -1))

        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        result_cupy = cupy.matmul(act_cupy, weight_cupy)
        cupy.cuda.Stream.null.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()

        result = torch.from_dlpack(result_cupy)

        result = result.view(*act_shape[:-1], -1)

        dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
        self.model.tensor_buffer[dst] = result
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task94(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None

    def run(self):
        # def async_task():
        result_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        result = self.model.tensor_buffer[result_loc]
        self.model.tensor_buffer[result_loc] = None

        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_push(self.to_string_info())
        result = result.to('cpu')
        torch.cuda.synchronize()
        if MEASURE_TIME_WITH_NVTX == True:
            nvtx.range_pop()
        self.secllm_cpp_wrapper.BookKeeperStore_Int32(self.layer_idx, self.next_task_ids[0], 0, result)
        # threading.Thread(target=async_task).start()

    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task95(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Down
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task96(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 1)]
        
        type = self.secllm_cpp_wrapper.ProjectionType.Down
        self.secllm_cpp_wrapper.DequantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self, worker_id):
        self.run()

class Task97(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 1)
        return ready


    def run(self):
        src1 = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        src2 = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)

        dst = [GetBookKeeperLinearIndex(self.layer_idx, 98, 0)]
        self.secllm_cpp_wrapper.ElementwiseAdd(self.layer_idx, src1, src2, dst, 1)
        
    def __call__(self, worker_id):
        self.run()

class Task98(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task99(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True

    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task100(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task101(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task102(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass

    def __call__(self, worker_id):
        self.run()

class Task103(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass
    
    def __call__(self, worker_id):
        self.run()

class Task104(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass
        
    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)

class Task105(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model, time_collector):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model, time_collector)

    def is_ready(self):
        return True
    
    def run(self):
        pass
        
    def __call__(self, worker_id):
        ts = TimeStamp(self.layer_idx, worker_id, self.task_description)
        ts.Start()
        self.run()
        ts.End()
        self.time_collector.Insert(worker_id, ts)