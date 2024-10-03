from typing import Any
import torch

# from torch_int.functional.quantization import (
#     dynamic_quantize_activation_per_token_absmax,
# )

from transformers.models.llama.modeling_llama import (
    repeat_kv,
    apply_rotary_pos_emb,
)

import cupy

import threading

def GetBookKeeperLinearIndex(layer_index, operation_index, input_index):
  # NOTE(jpyo0803): debugging purpose
  return layer_index * 300 + input_index * 100 + operation_index

class Task:
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        self.name = name
        self.layer_idx = layer_idx
        self.task_id = task_id
        self.next_task_ids = next_task_ids
        self.secllm_cpp_wrapper = secllm_cpp_wrapper
        self.model = model


    def run(self):
        print(f"Task: {self.name, self.task_id, self.next_task_ids}")

    def __call__(self):
        self.run()

class Task0(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task1(Task):
    # Copy 
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        # Input is float
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0) # float
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self.secllm_cpp_wrapper.BroadcastTensor_Float(src, dst)

    def __call__(self):
        self.run()

class Task2(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        # input is float
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        input_layernorm = self.model.layers[self.layer_idx].input_layernorm
        self.secllm_cpp_wrapper.RMSNorm(src, dst, input_layernorm.weight, input_layernorm.variance_epsilon)

    def __call__(self):
        self.run()

class Task3(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        # input is float
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        # Replicate from 3 to 7, 8, 9
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self.secllm_cpp_wrapper.BroadcastTensor_Float(src, dst)
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task4(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True # it does not depend on anything

    def run(self):
        # Move Q weight to GPU
        def async_task():
            self.model.layers[self.layer_idx].q_proj_weight_buffer = self.model.layers[self.layer_idx].q_proj.weight
            self.model.layers[self.layer_idx].q_proj_weight_buffer = self.model.layers[self.layer_idx].q_proj_weight_buffer.to('cuda:0')
            assert self.model.layers[self.layer_idx].q_proj_weight_buffer.dtype == torch.int8
        threading.Thread(target=async_task).start()
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task5(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True # it does not depend on anything

    def run(self):
        # Move K weight to GPU
        def async_task():
            self.model.layers[self.layer_idx].k_proj_weight_buffer = self.model.layers[self.layer_idx].k_proj.weight
            self.model.layers[self.layer_idx].k_proj_weight_buffer = self.model.layers[self.layer_idx].k_proj_weight_buffer.to('cuda:0')
            assert self.model.layers[self.layer_idx].k_proj_weight_buffer.dtype == torch.int8
        threading.Thread(target=async_task).start()

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task6(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True # it does not depend on anything

    def run(self):
        # Move V weight to GPU
        def async_task():
            self.model.layers[self.layer_idx].v_proj_weight_buffer = self.model.layers[self.layer_idx].v_proj.weight
            self.model.layers[self.layer_idx].v_proj_weight_buffer = self.model.layers[self.layer_idx].v_proj_weight_buffer.to('cuda:0')
            assert self.model.layers[self.layer_idx].v_proj_weight_buffer.dtype == torch.int8
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task7(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Q
        self.secllm_cpp_wrapper.QuantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task8(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.K
        self.secllm_cpp_wrapper.QuantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task9(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.V
        self.secllm_cpp_wrapper.QuantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task10(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Q
        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task11(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.K
        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task12(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.V
        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task13(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        def async_task():
            act = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
            act = act.to('cuda:0')
            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = act
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task14(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        def async_task():
            act = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
            act = act.to('cuda:0')
            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = act
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task15(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        def async_task():
            act = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
            act = act.to('cuda:0')
            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = act
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task16(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= (self.model.layers[self.layer_idx].q_proj_weight_buffer is not None) and (self.model.layers[self.layer_idx].q_proj_weight_buffer.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
        return ready

    def run(self):
        def async_task():
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

            result_cupy = cupy.matmul(act_cupy, weight_T_cupy)

            result = torch.from_dlpack(result_cupy)

            result = result.view(*act_shape[:-1], -1)

            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = result
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task17(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= (self.model.layers[self.layer_idx].k_proj_weight_buffer is not None) and (self.model.layers[self.layer_idx].k_proj_weight_buffer.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
        return ready

    def run(self):
        def async_task():
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

            result_cupy = cupy.matmul(act_cupy, weight_T_cupy)

            result = torch.from_dlpack(result_cupy)

            result = result.view(*act_shape[:-1], -1)

            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = result
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task18(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= (self.model.layers[self.layer_idx].v_proj_weight_buffer is not None) and (self.model.layers[self.layer_idx].v_proj_weight_buffer.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
        return ready

    def run(self):
        def async_task():
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

            result_cupy = cupy.matmul(act_cupy, weight_T_cupy)

            result = torch.from_dlpack(result_cupy)

            result = result.view(*act_shape[:-1], -1)

            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = result
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task19(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
    
    def run(self):
        def async_task():
            result_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
            result = self.model.tensor_buffer[result_loc]
            self.model.tensor_buffer[result_loc] = None
            result = result.to('cpu')
            self.secllm_cpp_wrapper.BookKeeperStore_Int32(self.layer_idx, self.next_task_ids[0], 0, result)
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task20(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
    
    def run(self):
        def async_task():
            result_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
            result = self.model.tensor_buffer[result_loc]
            self.model.tensor_buffer[result_loc] = None
            result = result.to('cpu')
            self.secllm_cpp_wrapper.BookKeeperStore_Int32(self.layer_idx, self.next_task_ids[0], 0, result)
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task21(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
    
    def run(self):
        def async_task():
            result_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
            result = self.model.tensor_buffer[result_loc]
            self.model.tensor_buffer[result_loc] = None
            result = result.to('cpu')
            self.secllm_cpp_wrapper.BookKeeperStore_Int32(self.layer_idx, self.next_task_ids[0], 0, result)
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task22(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Q
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task23(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.K
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task24(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.V
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task25(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)]
        type = self.secllm_cpp_wrapper.ProjectionType.Q
        self.secllm_cpp_wrapper.DequantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task26(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 1)]
        type = self.secllm_cpp_wrapper.ProjectionType.K
        self.secllm_cpp_wrapper.DequantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task27(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 2)]
        type = self.secllm_cpp_wrapper.ProjectionType.V
        self.secllm_cpp_wrapper.DequantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task28(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    # inputs are all floats
    def is_ready(self):
        ready = True
        for i in range(3):
            ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, i)
        return ready

    def run(self):
        query_states = self.secllm_cpp_wrapper.BookKeeperLoad_Float(self.layer_idx, self.task_id, 0)
        key_states = self.secllm_cpp_wrapper.BookKeeperLoad_Float(self.layer_idx, self.task_id, 1)
        value_states = self.secllm_cpp_wrapper.BookKeeperLoad_Float(self.layer_idx, self.task_id, 2)

        bsz = self.model.layers[self.layer_idx].bsz
        q_len = self.model.layers[self.layer_idx].q_len
        num_heads = self.model.layers[self.layer_idx].num_heads
        num_key_value_heads = self.model.layers[self.layer_idx].num_key_value_heads
        head_dim = self.model.layers[self.layer_idx].head_dim
        inv_freq = self.model.layers[self.layer_idx].rotary_emb.inv_freq
        position_ids = self.model.layers[self.layer_idx].position_ids

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2).contiguous()

        cos, sin = self.secllm_cpp_wrapper.LlamaRotaryEmbedding(inv_freq, position_ids, torch.float32)
        query_states, key_states = self.secllm_cpp_wrapper.ApplyRotaryPosEmb(query_states, key_states, cos, sin)

        assert self.next_task_ids[0] == 30
        assert self.next_task_ids[1] == 31
        assert self.next_task_ids[2] == 47

        self.secllm_cpp_wrapper.BookKeeperStore_Float(self.layer_idx, self.next_task_ids[0], 0, query_states.to(torch.float32))
        self.secllm_cpp_wrapper.BookKeeperStore_Float(self.layer_idx, self.next_task_ids[1], 0, key_states.to(torch.float32))
        self.secllm_cpp_wrapper.BookKeeperStore_Float(self.layer_idx, self.next_task_ids[2], 0, value_states.to(torch.float32))

    def __call__(self):
        self.run()

class Task29(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True
    
    def run(self):
        self.secllm_cpp_wrapper.GenerateSecretKey_QK(self.layer_idx)

    def __call__(self):
        self.run()

class Task30(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        self.secllm_cpp_wrapper.QuantizeQ_QK(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task31(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        self.secllm_cpp_wrapper.QuantizeK_QK(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task32(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 1), GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[1], 1)]

        self.secllm_cpp_wrapper.ShiftQ_QK(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task33(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 2), GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[1], 1)]

        self.secllm_cpp_wrapper.ShiftK_QK(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task34(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.QKKeyIsAvailable(self.layer_idx)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, self.task_id, 1)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, self.task_id, 2)
        return ready
    
    def run(self):
        q_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)
        k_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 2)

        self.secllm_cpp_wrapper.GenerateDecryptionKey_QK(self.layer_idx, q_loc, k_loc)

    def __call__(self):
        self.run()

class Task35(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.QKKeyIsAvailable(self.layer_idx)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, self.task_id, 1)
        return ready
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        self.secllm_cpp_wrapper.EncryptX_QK(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task36(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.QKKeyIsAvailable(self.layer_idx)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, self.task_id, 1)
        return ready
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        self.secllm_cpp_wrapper.EncryptY_QK(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task37(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        def async_task():
            act = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, self.task_id, 0)
            act = act.to('cuda:0')
            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = act
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task38(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        def async_task():
            act = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, self.task_id, 0)
            act = act.to('cuda:0')
            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 1)
            self.model.tensor_buffer[dst] = act
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task39(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)] is not None
        return ready
    
    def run(self):
        def async_task():
            q_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
            k_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)

            q = self.model.tensor_buffer[q_loc]
            self.model.tensor_buffer[q_loc] = None

            k = self.model.tensor_buffer[k_loc]
            self.model.tensor_buffer[k_loc] = None

            assert q.dtype == torch.uint32
            assert k.dtype == torch.uint32
            assert q.shape[-1] == k.shape[-1]

            past_key_value = self.model.layers[self.layer_idx].past_key_value
            if past_key_value is not None:
                k = past_key_value.update_key(k, self.layer_idx)
                
            k = repeat_kv(k, self.model.layers[self.layer_idx].num_key_value_groups)

            q_cupy = cupy.from_dlpack(q)
            k_T_cupy = cupy.from_dlpack(k.transpose(-2, -1))
            
            attn_weights_cupy = cupy.matmul(q_cupy, k_T_cupy)
            attn_weights = torch.from_dlpack(attn_weights_cupy)
            assert attn_weights.dtype == torch.uint32

            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = attn_weights
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task40(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
    
    def run(self):
        def async_task():
            attn_weights_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
            attn_weights = self.model.tensor_buffer[attn_weights_loc]
            self.model.tensor_buffer[attn_weights_loc] = None
            attn_weights = attn_weights.to('cpu')
            self.secllm_cpp_wrapper.BookKeeperStore_Uint32(self.layer_idx, self.next_task_ids[0], 0, attn_weights)
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task41(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.QKDecKeyIsAvailable(self.layer_idx)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, self.task_id, 0)
        return ready
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        self.secllm_cpp_wrapper.Decrypt_QK(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task42(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        self.secllm_cpp_wrapper.Unshift_QK(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task43(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        self.secllm_cpp_wrapper.Dequantize_QK(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task44(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        dst.append(GetBookKeeperLinearIndex(self.layer_idx, 98, 1))
        self.secllm_cpp_wrapper.Softmax(src, dst)

    def __call__(self):
        self.run()

class Task45(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True
    
    def run(self):
        self.secllm_cpp_wrapper.GenerateSecretKey_PV(self.layer_idx)

    def __call__(self):
        self.run()

class Task46(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self.secllm_cpp_wrapper.QuantizeP_PV(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task47(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)
        
    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self.secllm_cpp_wrapper.QuantizeV_PV(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task48(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)
        
    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 1), GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[1], 1)]
        self.secllm_cpp_wrapper.ShiftP_PV(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task49(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)
        
    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 2), GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[1], 1)]
        self.secllm_cpp_wrapper.ShiftV_PV(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task50(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.PVKeyIsAvailable(self.layer_idx)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, self.task_id, 1)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, self.task_id, 2)
        return ready
    
    def run(self):
        p_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)
        v_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 2)

        self.secllm_cpp_wrapper.GenerateDecryptionKey_PV(self.layer_idx, p_loc, v_loc)

    def __call__(self):
        self.run()

class Task51(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.PVKeyIsAvailable(self.layer_idx)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, self.task_id, 1)
        return ready
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        self.secllm_cpp_wrapper.EncryptX_PV(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task52(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.PVKeyIsAvailable(self.layer_idx)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, self.task_id, 1)
        return ready
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        self.secllm_cpp_wrapper.EncryptY_PV(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task53(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        def async_task():
            act = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, self.task_id, 0)
            act = act.to('cuda:0')
            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = act
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task54(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        def async_task():
            act = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, self.task_id, 0)
            act = act.to('cuda:0')
            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 1)
            self.model.tensor_buffer[dst] = act
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task55(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)] is not None
        return ready

    def run(self):
        def aync_task():
            p_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
            v_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)

            p = self.model.tensor_buffer[p_loc]
            self.model.tensor_buffer[p_loc] = None

            v = self.model.tensor_buffer[v_loc]
            self.model.tensor_buffer[v_loc] = None

            assert p.dtype == torch.uint32
            assert v.dtype == torch.uint32
            assert p.shape[-1] == v.shape[-2]

            past_key_value = self.model.layers[self.layer_idx].past_key_value
            if past_key_value is not None:
                v = past_key_value.update_value(v, self.layer_idx)

            v = repeat_kv(v, self.model.layers[self.layer_idx].num_key_value_groups)

            p_cupy = cupy.from_dlpack(p)
            v_cupy = cupy.from_dlpack(v)
        
            attn_output_cupy = cupy.matmul(p_cupy, v_cupy)
            attn_output = torch.from_dlpack(attn_output_cupy)
            assert attn_output.dtype == torch.uint32

            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = attn_output
        threading.Thread(target=aync_task).start()

    def __call__(self):
        self.run()

class Task56(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None

    def run(self):
        def async_task():
            attn_output_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
            attn_output = self.model.tensor_buffer[attn_output_loc]
            self.model.tensor_buffer[attn_output_loc] = None
            attn_output = attn_output.to('cpu')
            self.secllm_cpp_wrapper.BookKeeperStore_Uint32(self.layer_idx, self.next_task_ids[0], 0, attn_output)
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task57(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.PVDecKeyIsAvailable(self.layer_idx)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, self.task_id, 0)
        return ready

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self.secllm_cpp_wrapper.Decrypt_PV(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task58(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self.secllm_cpp_wrapper.Unshift_PV(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task59(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self.secllm_cpp_wrapper.Dequantize_PV(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task60(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        def async_task():
            self.model.layers[self.layer_idx].o_proj_weight_buffer = self.model.layers[self.layer_idx].o_proj.weight
            self.model.layers[self.layer_idx].o_proj_weight_buffer = self.model.layers[self.layer_idx].o_proj_weight_buffer.to('cuda:0')
            assert self.model.layers[self.layer_idx].o_proj_weight_buffer.dtype == torch.int8
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task61(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.O
        self.secllm_cpp_wrapper.QuantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task62(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.O
        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task63(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        def async_task():
            act = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
            act = act.to('cuda:0')
            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = act
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task64(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= (self.model.layers[self.layer_idx].o_proj_weight_buffer is not None) and (self.model.layers[self.layer_idx].o_proj_weight_buffer.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
        return ready
    
    def run(self):
        def async_task():
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

            result_cupy = cupy.matmul(act_cupy, weight_cupy)

            result = torch.from_dlpack(result_cupy)

            result = result.view(*act_shape[:-1], -1)

            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = result
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task65(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None

    def run(self):
        def async_task():
            result_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
            result = self.model.tensor_buffer[result_loc]
            self.model.tensor_buffer[result_loc] = None
            result = result.to('cpu')
            self.secllm_cpp_wrapper.BookKeeperStore_Int32(self.layer_idx, self.next_task_ids[0], 0, result)
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task66(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.O
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task67(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 1)]

        type = self.secllm_cpp_wrapper.ProjectionType.O
        self.secllm_cpp_wrapper.DequantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task68(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 1)
        return ready

    def run(self):
        src1 = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        src2 = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self.secllm_cpp_wrapper.ElementwiseAdd(src1, src2, dst)

    def __call__(self):
        self.run()

class Task69(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self.secllm_cpp_wrapper.BroadcastTensor_Float(src, dst)

    def __call__(self):
        self.run()

class Task70(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        post_attention_layernorm = self.model.layers[self.layer_idx].post_attention_layernorm
        self.secllm_cpp_wrapper.RMSNorm(src, dst, post_attention_layernorm.weight, post_attention_layernorm.variance_epsilon)

    def __call__(self):
        self.run()

class Task71(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self.secllm_cpp_wrapper.BroadcastTensor_Float(src, dst)
        
    def __call__(self):
        self.run()

class Task72(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        def async_task():
            self.model.layers[self.layer_idx].gate_proj_weight_buffer = self.model.layers[self.layer_idx].gate_proj.weight
            self.model.layers[self.layer_idx].gate_proj_weight_buffer = self.model.layers[self.layer_idx].gate_proj_weight_buffer.to('cuda:0')
            assert self.model.layers[self.layer_idx].gate_proj_weight_buffer.dtype == torch.int8
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task73(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        def async_task():
            self.model.layers[self.layer_idx].up_proj_weight_buffer = self.model.layers[self.layer_idx].up_proj.weight
            self.model.layers[self.layer_idx].up_proj_weight_buffer = self.model.layers[self.layer_idx].up_proj_weight_buffer.to('cuda:0')
            assert self.model.layers[self.layer_idx].up_proj_weight_buffer.dtype == torch.int8
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task74(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Gate
        self.secllm_cpp_wrapper.QuantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task75(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
    
    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Up
        self.secllm_cpp_wrapper.QuantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task76(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Gate
        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task77(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Up
        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task78(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        def async_task():
            act = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
            act = act.to('cuda:0')
            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = act
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task79(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        def async_task():
            act = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
            act = act.to('cuda:0')
            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = act
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task80(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= (self.model.layers[self.layer_idx].gate_proj_weight_buffer is not None) and (self.model.layers[self.layer_idx].gate_proj_weight_buffer.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
        return ready
    
    def run(self):
        def async_task():
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

            result_cupy = cupy.matmul(act_cupy, weight_cupy)

            result = torch.from_dlpack(result_cupy)

            result = result.view(*act_shape[:-1], -1)

            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = result
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task81(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= (self.model.layers[self.layer_idx].up_proj_weight_buffer is not None) and (self.model.layers[self.layer_idx].up_proj_weight_buffer.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
        return ready
    
    def run(self):
        def async_task():
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

            result_cupy = cupy.matmul(act_cupy, weight_cupy)

            result = torch.from_dlpack(result_cupy)

            result = result.view(*act_shape[:-1], -1)

            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = result
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task82(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None

    def run(self):
        def async_task():
            result_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
            result = self.model.tensor_buffer[result_loc]
            self.model.tensor_buffer[result_loc] = None
            result = result.to('cpu')
            self.secllm_cpp_wrapper.BookKeeperStore_Int32(self.layer_idx, self.next_task_ids[0], 0, result)
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task83(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None

    def run(self):
        def async_task():
            result_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
            result = self.model.tensor_buffer[result_loc]
            self.model.tensor_buffer[result_loc] = None
            result = result.to('cpu')
            self.secllm_cpp_wrapper.BookKeeperStore_Int32(self.layer_idx, self.next_task_ids[0], 0, result)
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task84(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        
        type = self.secllm_cpp_wrapper.ProjectionType.Gate
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task85(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        
        type = self.secllm_cpp_wrapper.ProjectionType.Up
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task86(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)]
        
        type = self.secllm_cpp_wrapper.ProjectionType.Gate
        self.secllm_cpp_wrapper.DequantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task87(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 1)]
        
        type = self.secllm_cpp_wrapper.ProjectionType.Up
        self.secllm_cpp_wrapper.DequantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task88(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 1)
        return ready

    def run(self):
        src1 = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        src2 = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]
        self. secllm_cpp_wrapper.SwiGLU(src1, src2, dst)

    def __call__(self):
        self.run()

class Task89(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        def async_task():
            self.model.layers[self.layer_idx].down_proj_weight_buffer = self.model.layers[self.layer_idx].down_proj.weight
            self.model.layers[self.layer_idx].down_proj_weight_buffer = self.model.layers[self.layer_idx].down_proj_weight_buffer.to('cuda:0')
            assert self.model.layers[self.layer_idx].down_proj_weight_buffer.dtype == torch.int8
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task90(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Down
        self.secllm_cpp_wrapper.QuantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task91(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int8(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Down
        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task92(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        def async_task():
            act = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
            act = act.to('cuda:0')
            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = act
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task93(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= (self.model.layers[self.layer_idx].down_proj_weight_buffer is not None) and (self.model.layers[self.layer_idx].down_proj_weight_buffer.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None
        return ready

    def run(self):
        def async_task():
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

            result_cupy = cupy.matmul(act_cupy, weight_cupy)

            result = torch.from_dlpack(result_cupy)

            result = result.view(*act_shape[:-1], -1)

            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = result
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task94(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)] is not None

    def run(self):
        def async_task():
            result_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
            result = self.model.tensor_buffer[result_loc]
            self.model.tensor_buffer[result_loc] = None
            result = result.to('cpu')
            self.secllm_cpp_wrapper.BookKeeperStore_Int32(self.layer_idx, self.next_task_ids[0], 0, result)
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task95(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

        type = self.secllm_cpp_wrapper.ProjectionType.Down
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task96(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 1)]
        
        type = self.secllm_cpp_wrapper.ProjectionType.Down
        self.secllm_cpp_wrapper.DequantizeLinearActivation(self.layer_idx, src, dst, type)

    def __call__(self):
        self.run()

class Task97(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 0)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Float(self.layer_idx, self.task_id, 1)
        return ready


    def run(self):
        src1 = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
        src2 = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)

        dst = [GetBookKeeperLinearIndex(self.layer_idx, 98, 0)]
        self.secllm_cpp_wrapper.ElementwiseAdd(src1, src2, dst)
        
    def __call__(self):
        self.run()