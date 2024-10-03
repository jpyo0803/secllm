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
            self.q_proj_weight = self.model.layers[self.layer_idx].q_proj.weight
            self.q_proj_weight = self.q_proj_weight.to('cuda:0')
            assert self.q_proj_weight.dtype == torch.int8
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
           self.k_proj_weight = self.model.layers[self.layer_idx].k_proj.weight
           self.k_proj_weight = self.k_proj_weight.to('cuda:0')
           assert self.k_proj_weight.dtype == torch.int8
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
            self.v_proj_weight = self.model.layers[self.layer_idx].v_proj.weight
            self.v_proj_weight = self.v_proj_weight.to('cuda:0')
            assert self.v_proj_weight.dtype == torch.int8
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
        self.secllm_cpp_wrapper.QuatizeLinearActivation(self.layer_idx, src, dst, type)

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
        self.secllm_cpp_wrapper.QuatizeLinearActivation(self.layer_idx, src, dst, type)

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
        self.secllm_cpp_wrapper.QuatizeLinearActivation(self.layer_idx, src, dst, type)

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
            activation = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
            activation = activation.to('cuda:0')
            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = activation
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
            activation = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
            activation = activation.to('cuda:0')
            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = activation
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
            activation = self.secllm_cpp_wrapper.BookKeeperLoad_Int32(self.layer_idx, self.task_id, 0)
            activation = activation.to('cuda:0')
            dst = GetBookKeeperLinearIndex(self.layer_idx, self.next_task_ids[0], 0)
            self.model.tensor_buffer[dst] = activation
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task16(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= (self.q_proj_weight is not None) and (self.q_proj_weight.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)] is not None
        return ready

    def run(self):
        def async_task():
            act_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)
            assert self.model.tensor_buffer[act_loc] is not None

            act = self.model.tensor_buffer[act_loc]
            self.model.tensor_buffer[act_loc] = None

            act_shape = act.shape
            act = act.view(-1, act_shape[-1])

            weight = self.q_proj_weight
            self.q_proj_weight = None

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
        ready &= (self.k_proj_weight is not None) and (self.k_proj_weight.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)] is not None
        return ready

    def run(self):
        def async_task():
            act_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)
            assert self.model.tensor_buffer[act_loc] is not None

            act = self.model.tensor_buffer[act_loc]
            self.model.tensor_buffer[act_loc] = None

            act_shape = act.shape
            act = act.view(-1, act_shape[-1])

            weight = self.k_proj_weight
            self.k_proj_weight = None

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
        ready &= (self.v_proj_weight is not None) and (self.v_proj_weight.is_cuda)
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)] is not None
        return ready

    def run(self):
        def async_task():
            act_loc = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 1)
            assert self.model.tensor_buffer[act_loc] is not None

            act = self.model.tensor_buffer[act_loc]
            self.model.tensor_buffer[act_loc] = None

            act_shape = act.shape
            act = act.view(-1, act_shape[-1])

            weight = self.v_proj_weight
            self.v_proj_weight = None

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
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
            dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

            type = self.secllm_cpp_wrapper.ProjectionType.Q
            self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task23(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
            dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

            type = self.secllm_cpp_wrapper.ProjectionType.K
            self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task24(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Int32(self.layer_idx, self.task_id, 0)
    
    def run(self):
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, self.task_id, 0)
            dst = [GetBookKeeperLinearIndex(self.layer_idx, next_task_id, 0) for next_task_id in self.next_task_ids]

            type = self.secllm_cpp_wrapper.ProjectionType.V
            self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, type)
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()












