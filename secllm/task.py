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
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 1, 0)

    def run(self):
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)
        src = GetBookKeeperLinearIndex(self.layer_idx, 1, 0) # float
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 2, 0), GetBookKeeperLinearIndex(self.layer_idx, 67, 0)]
        self.secllm_cpp_wrapper.ReplicateTensor(src, dst)

    def __call__(self):
        self.run()

class Task2(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        # input is float
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 2, 0)

    def run(self):
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)
        src = GetBookKeeperLinearIndex(self.layer_idx, 2, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 3, 0)]

        input_layernorm = self.model.layers[self.layer_idx].input_layernorm

        self.secllm_cpp_wrapper.RMSNorm(src, dst, input_layernorm.weight, input_layernorm.variance_epsilon)

    def __call__(self):
        self.run()

class Task3(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        # input is float
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 3, 0)

    def run(self):
        # Replicate from 3 to 7, 8, 9
        src = GetBookKeeperLinearIndex(self.layer_idx, 3, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 7, 0), GetBookKeeperLinearIndex(self.layer_idx, 8, 0), GetBookKeeperLinearIndex(self.layer_idx, 9, 0)]
        self.secllm_cpp_wrapper.ReplicateTensor(src, dst)
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
            self.model.layers[self.layer_idx].q_proj.weight = self.model.layers[self.layer_idx].q_proj.weight.to('cuda:0')
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
            self.model.layers[self.layer_idx].k_proj.weight = self.model.layers[self.layer_idx].k_proj.weight.to('cuda:0')
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
            self.model.layers[self.layer_idx].v_proj.weight = self.model.layers[self.layer_idx].v_proj.weight.to('cuda:0')
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task7(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        # input is float, output is int32
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 7, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, 7, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 10, 0)]

        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, 0) # Q

    def __call__(self):
        self.run()

class Task8(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        # input is float, output is int32
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 8, 0)

    def run(self):
        # Encryption but for not it just bypasses
        src = GetBookKeeperLinearIndex(self.layer_idx, 8, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 11, 0)]

        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, 1) # K

    def __call__(self):
        self.run()

class Task9(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        # input is float, output is int32
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 9, 0)

    def run(self):
        # Encryption but for not it just bypasses

        src = GetBookKeeperLinearIndex(self.layer_idx, 9, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 12, 0)]

        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, 2) # V

    def __call__(self):
        self.run()

class Task10(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 10, 0)

    def run(self):
        # Retrieve input from BookKeeper and move to GPU
        def async_task():
            enc_activation = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, 10, 0)
            enc_activation = enc_activation.to('cuda:0')

            dst = GetBookKeeperLinearIndex(self.layer_idx, 13, 1)
            self.model.tensor_buffer[dst] = enc_activation
        threading.Thread(target=async_task).start()


    def __call__(self):
        self.run()

class Task11(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 11, 0)

    def run(self):
        # Retrieve input from BookKeeper and move to GPU
        def async_task():
            enc_activation = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, 11, 0)
            enc_activation = enc_activation.to('cuda:0')

            dst = GetBookKeeperLinearIndex(self.layer_idx, 14, 1)
            self.model.tensor_buffer[dst] = enc_activation
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task12(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 12, 0)

    def run(self):
        # Retrieve input from BookKeeper and move to GPU
        def async_task():
            enc_activation = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, 12, 0)
            enc_activation = enc_activation.to('cuda:0')

            dst = GetBookKeeperLinearIndex(self.layer_idx, 15, 1)
            self.model.tensor_buffer[dst] = enc_activation
        threading.Thread(target=async_task).start()

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task13(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 13, 1)] is not None
        ready &= self.model.layers[self.layer_idx].q_proj.weight.is_cuda
        return ready

    def run(self):
        # Do Q projection

        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, 13, 1)
            assert self.model.tensor_buffer[src] is not None

            x = self.model.tensor_buffer[src]
            x_shape = x.shape
            x = x.view(-1, x_shape[-1]) # This vertically concatenates batches

            x_cupy = cupy.from_dlpack(x.to(torch.int32))
            weight_T_cupy = cupy.from_dlpack(self.model.layers[self.layer_idx].q_proj.weight.transpose(-2, -1).to(torch.int32))
            y_cupy = cupy.matmul(x_cupy, weight_T_cupy)
            y = torch.from_dlpack(y_cupy)

            y = y.view(*x_shape[:-1], -1)

            self.model.tensor_buffer[src] = None

            dst = GetBookKeeperLinearIndex(self.layer_idx, 16, 0)
            self.model.tensor_buffer[dst] = y
        threading.Thread(target=async_task).start()

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task14(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 14, 1)] is not None
        ready &= self.model.layers[self.layer_idx].k_proj.weight.is_cuda
        return ready

    def run(self):
        # Do K projection
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, 14, 1)
            assert self.model.tensor_buffer[src] is not None

            x = self.model.tensor_buffer[src]
            x_shape = x.shape
            x = x.view(-1, x_shape[-1]) # This vertically concatenates batches

            x_cupy = cupy.from_dlpack(x.to(torch.int32))
            weight_T_cupy = cupy.from_dlpack(self.model.layers[self.layer_idx].k_proj.weight.transpose(-2, -1).to(torch.int32))
            y_cupy = cupy.matmul(x_cupy, weight_T_cupy)
            y = torch.from_dlpack(y_cupy)

            y = y.view(*x_shape[:-1], -1)

            self.model.tensor_buffer[src] = None

            dst = GetBookKeeperLinearIndex(self.layer_idx, 17, 0)
            self.model.tensor_buffer[dst] = y
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task15(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 15, 1)] is not None
        ready &= self.model.layers[self.layer_idx].v_proj.weight.is_cuda
        return ready

    def run(self):
        # Do V projection
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, 15, 1)
            assert self.model.tensor_buffer[src] is not None

            x = self.model.tensor_buffer[src]
            x_shape = x.shape
            x = x.view(-1, x_shape[-1]) # This vertically concatenates batches

            x_cupy = cupy.from_dlpack(x.to(torch.int32))
            weight_T_cupy = cupy.from_dlpack(self.model.layers[self.layer_idx].v_proj.weight.transpose(-2, -1).to(torch.int32))
            y_cupy = cupy.matmul(x_cupy, weight_T_cupy)
            y = torch.from_dlpack(y_cupy)

            y = y.view(*x_shape[:-1], -1)

            self.model.tensor_buffer[src] = None

            dst = GetBookKeeperLinearIndex(self.layer_idx, 18, 0)
            self.model.tensor_buffer[dst] = y
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task16(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 16, 0)] is not None

    def run(self):
        # Move query_states to CPU
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, 16, 0)
            assert self.model.tensor_buffer[src] is not None
            y = self.model.tensor_buffer[src]
            y = y.to('cpu')

            self.model.tensor_buffer[src] = None
            
            self.secllm_cpp_wrapper.BookKeeperStore_Uint32(self.layer_idx, 19, 0, y.to(torch.uint32))
        threading.Thread(target=async_task).start()

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task17(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 17, 0)] is not None

    def run(self):
        # Move key_states to CPU
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, 17, 0)
            assert self.model.tensor_buffer[src] is not None

            y = self.model.tensor_buffer[src]
            y = y.to('cpu')

            self.model.tensor_buffer[src] = None
            self.secllm_cpp_wrapper.BookKeeperStore_Uint32(self.layer_idx, 20, 0, y.to(torch.uint32))
        threading.Thread(target=async_task).start()

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task18(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 18, 0)] is not None

    def run(self):
        # Move value_states to CPU
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, 18, 0)
            assert self.model.tensor_buffer[src] is not None

            y = self.model.tensor_buffer[src]
            y = y.to('cpu')

            self.model.tensor_buffer[src] = None
            self.secllm_cpp_wrapper.BookKeeperStore_Uint32(self.layer_idx, 21, 0, y.to(torch.uint32))
        threading.Thread(target=async_task).start()


    def __call__(self):
        self.run()

class Task19(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 19, 0)

    def run(self):
        # Decryption but for now it just bypasses
        src = GetBookKeeperLinearIndex(self.layer_idx, 19, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 22, 0)]
        
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, 0)

    def __call__(self):
        self.run()

class Task20(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 20, 0)

    def run(self):
        # Decryption but for now it just bypasses
        src = GetBookKeeperLinearIndex(self.layer_idx, 20, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 22, 1)]

        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, 1)

    def __call__(self):
        self.run()

class Task21(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 21, 0)

    def run(self):
        # Decryption but for now it just bypasses
        src = GetBookKeeperLinearIndex(self.layer_idx, 21, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 22, 2)]

        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, 2)

    def __call__(self):
        self.run()

class Task22(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    # inputs are all floats
    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 22, 0)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 22, 1)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 22, 2)
        return ready

    def run(self):
        query_states = self.secllm_cpp_wrapper.BookKeeperLoad(self.layer_idx, 22, 0)
        key_states = self.secllm_cpp_wrapper.BookKeeperLoad(self.layer_idx, 22, 1)
        value_states = self.secllm_cpp_wrapper.BookKeeperLoad(self.layer_idx, 22, 2)

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

        self.secllm_cpp_wrapper.BookKeeperStore(self.layer_idx, 28, 0, query_states.to(torch.float32))
        self.secllm_cpp_wrapper.BookKeeperStore(self.layer_idx, 29, 0, key_states.to(torch.float32))
        self.secllm_cpp_wrapper.BookKeeperStore(self.layer_idx, 47, 0, value_states.to(torch.float32))


    def __call__(self):
        self.run()

class Task23(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task24(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task25(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task26(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task27(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    # Assume its required are prepared before task schedulers are called
    def is_ready(self):
        return True

    def run(self):
        # pass
        self.secllm_cpp_wrapper.GenerateSecretKey_QK(self.layer_idx)
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task28(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    # its input is float
    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 28, 0)

    def run(self):
        # Bypass shift for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 28, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 33, 0), GetBookKeeperLinearIndex(self.layer_idx, 35, 1)]

        self.secllm_cpp_wrapper.QuantizeAndShiftQ(self.layer_idx, src, dst)
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task29(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    # its input is float
    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 29, 0)

    def run(self):
        # Bypass shift for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 29, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 34, 0), GetBookKeeperLinearIndex(self.layer_idx, 35, 2)]

        self.secllm_cpp_wrapper.QuantizeAndShiftK(self.layer_idx, src, dst)
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task30(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task31(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        # Broadcast
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task32(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        # Broadcast
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task33(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.QKKeyIsAvailable(self.layer_idx)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 33, 0)
        return ready

    def run(self):
        # ByPass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 33, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 36, 0)]

        self.secllm_cpp_wrapper.EncryptX_QK(self.layer_idx, src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task34(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.QKKeyIsAvailable(self.layer_idx)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 34, 0)
        return ready

    def run(self):
        # ByPass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 34, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 37, 0)]

        self.secllm_cpp_wrapper.EncryptY_QK(self.layer_idx, src, dst)

        # self.secllm_cpp_wrapper.ReplicateTensor_Uint32(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task35(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.QKKeyIsAvailable(self.layer_idx)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 35, 1)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 35, 2)
        return ready

    def run(self):
        src_x = GetBookKeeperLinearIndex(self.layer_idx, 35, 1)
        src_y = GetBookKeeperLinearIndex(self.layer_idx, 35, 2)

        self.secllm_cpp_wrapper.GenerateDecryptionKey_QK(self.layer_idx, src_x, src_y)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task36(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 36, 0)

    def run(self):
        # Move Enc_Q to GPU
        def async_task():
            enc_q = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, 36, 0)
            
            enc_q = enc_q.to('cuda:0')

            dst = GetBookKeeperLinearIndex(self.layer_idx, 39, 0)
            self.model.tensor_buffer[dst] = enc_q
        threading.Thread(target=async_task).start()


    def __call__(self):
        self.run()

class Task37(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 37, 0)

    def run(self):
        # Move Enc_K to GPU
        def async_task():
            enc_k = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, 37, 0)

            enc_k = enc_k.to('cuda:0')

            dst = GetBookKeeperLinearIndex(self.layer_idx, 39, 1)
            self.model.tensor_buffer[dst] = enc_k
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task38(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task39(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 39, 0)] is not None
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 39, 1)] is not None
        return ready

    def run(self):
        # Matmul Q, K^T
        def async_task():
            num_key_value_groups = self.model.layers[self.layer_idx].num_key_value_groups
            
            src_q = GetBookKeeperLinearIndex(self.layer_idx, 39, 0)
            assert self.model.tensor_buffer[src_q] is not None

            src_k = GetBookKeeperLinearIndex(self.layer_idx, 39, 1)
            assert self.model.tensor_buffer[src_k] is not None

            enc_q = self.model.tensor_buffer[src_q]
            enc_k = self.model.tensor_buffer[src_k]

            assert enc_q.dtype == torch.uint32
            assert enc_k.dtype == torch.uint32
            assert enc_q.shape[-1] == enc_k.shape[-1]

            past_key_value = self.model.layers[self.layer_idx].past_key_value
            if past_key_value is not None:
                enc_k = past_key_value.update_key(enc_k, self.layer_idx)

            enc_k = repeat_kv(enc_k, num_key_value_groups)

            enc_q_cupy = cupy.from_dlpack(enc_q.to(torch.uint32))
            enc_k_T_cupy = cupy.from_dlpack(enc_k.transpose(-2, -1).to(torch.uint32))

            attn_weights_cupy = cupy.matmul(enc_q_cupy, enc_k_T_cupy)
            attn_weights = torch.from_dlpack(attn_weights_cupy).to(torch.uint32)
            dst = GetBookKeeperLinearIndex(self.layer_idx, 40, 0)
            self.model.tensor_buffer[dst] = (attn_weights, enc_k.shape[-2]) # Before transposed

            self.model.tensor_buffer[src_q] = None
            self.model.tensor_buffer[src_k] = None
        threading.Thread(target=async_task).start()


    def __call__(self):
        self.run()

class Task40(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 40, 0)] is not None

    def run(self):
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, 40, 0)
            assert self.model.tensor_buffer[src] is not None

            attn_weights, _ = self.model.tensor_buffer[src]

            attn_weights = attn_weights.to('cpu')
            # assert attn_weights.is_contiguous()
            
            self.secllm_cpp_wrapper.BookKeeperStore_Uint32(self.layer_idx, 41, 0, attn_weights)
            
            self.model.tensor_buffer[src] = None
        threading.Thread(target=async_task).start()


    def __call__(self):
        self.run()

class Task41(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.QKDecKeyIsAvailable(self.layer_idx)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 41, 0)
        return ready

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, 41, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 42, 0)]

        self.secllm_cpp_wrapper.Decrypt_QK(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task42(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 42, 0)

    def run(self):
        # ByPass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 42, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 43, 0)]

        self.secllm_cpp_wrapper.UnshiftAndDequantizeQK(self.layer_idx, src, dst)

        # Do un shift and dequantize
        # self.secllm_cpp_wrapper.ReplicateTensor(src, dst)
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task43(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    # since it is dequantized, it is float
    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 43, 0)

    def run(self):
        # Softmax
        src = GetBookKeeperLinearIndex(self.layer_idx, 43, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 46, 0)]

        self.secllm_cpp_wrapper.Softmax(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task44(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task45(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    # Assume metadata required for key generation is set before task scheduler is called
    def is_ready(self):
        return True

    def run(self):
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)
        self.secllm_cpp_wrapper.GenerateSecretKey_PV(self.layer_idx)

    def __call__(self):
        self.run()

class Task46(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    # output of softmax is float
    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 46, 0)

    def run(self):
        # Bypass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 46, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 51, 0), GetBookKeeperLinearIndex(self.layer_idx, 53, 1)]

        self.secllm_cpp_wrapper.QuantizeAndShiftP(self.layer_idx, src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task47(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    # rotary embedding output is float
    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 47, 0)

    def run(self):
        # Bypass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 47, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 52, 0), GetBookKeeperLinearIndex(self.layer_idx, 53, 2)]

        self.secllm_cpp_wrapper.QuantizeAndShiftV(self.layer_idx, src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task48(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task49(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        # Broadcast
        pass

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task50(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        # Broadcast
        pass

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task51(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.PVKeyIsAvailable(self.layer_idx)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 51, 0)
        return ready

    def run(self):
        # Encrypt P, for now it just bypasses
        src = GetBookKeeperLinearIndex(self.layer_idx, 51, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 54, 0)]

        self.secllm_cpp_wrapper.EncryptX_PV(self.layer_idx, src, dst)

        # self.secllm_cpp_wrapper.ReplicateTensor_Uint32(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task52(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.PVKeyIsAvailable(self.layer_idx)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 52, 0)
        return ready

    def run(self):
        # encrypt V, for now it just bypasses
        src = GetBookKeeperLinearIndex(self.layer_idx, 52, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 55, 0)]

        self.secllm_cpp_wrapper.EncryptY_PV(self.layer_idx, src, dst)

    def __call__(self):
        self.run()

class Task53(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.PVKeyIsAvailable(self.layer_idx)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 53, 1)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 53, 2)
        return ready

    def run(self):
        src_x = GetBookKeeperLinearIndex(self.layer_idx, 53, 1)
        src_y = GetBookKeeperLinearIndex(self.layer_idx, 53, 2)

        self.secllm_cpp_wrapper.GenerateDecryptionKey_PV(self.layer_idx, src_x, src_y)
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task54(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 54, 0)

    def run(self):
        # Move enc_P to GPU
        def async_task():
            enc_q_attn_weights = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, 54, 0)
            enc_q_attn_weights = enc_q_attn_weights.to('cuda:0')
            dst = GetBookKeeperLinearIndex(self.layer_idx, 57, 0)
            self.model.tensor_buffer[dst] = enc_q_attn_weights
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task55(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 55, 0)

    def run(self):
        # Move enc_V to GPU
        def async_task():
            enc_v = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, 55, 0)
            enc_v = enc_v.to('cuda:0')
            dst = GetBookKeeperLinearIndex(self.layer_idx, 57, 1)
            self.model.tensor_buffer[dst] = enc_v
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task56(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task57(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 57, 0)] is not None
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 57, 1)] is not None
        return ready

    def run(self):
        # Matmul PV
        def async_task():
            num_key_value_groups = self.model.layers[self.layer_idx].num_key_value_groups

            src_p = GetBookKeeperLinearIndex(self.layer_idx, 57, 0)
            assert self.model.tensor_buffer[src_p] is not None

            src_v = GetBookKeeperLinearIndex(self.layer_idx, 57, 1)
            assert self.model.tensor_buffer[src_v] is not None

            attn_weights = self.model.tensor_buffer[src_p]
            value_states = self.model.tensor_buffer[src_v]

            past_key_value = self.model.layers[self.layer_idx].past_key_value
            if past_key_value is not None:
                value_states = past_key_value.update_value(value_states, self.layer_idx)

            value_states = repeat_kv(value_states, num_key_value_groups)

            attn_weights_cupy = cupy.from_dlpack(attn_weights.to(torch.uint32))
            value_states_cupy = cupy.from_dlpack(value_states.to(torch.uint32))

            attn_output_cupy = cupy.matmul(attn_weights_cupy, value_states_cupy)

            attn_output = torch.from_dlpack(attn_output_cupy).to(torch.uint32)

            dst = GetBookKeeperLinearIndex(self.layer_idx, 58, 0)
            self.model.tensor_buffer[dst] = attn_output

            self.model.tensor_buffer[src_p] = None
            self.model.tensor_buffer[src_v] = None
        threading.Thread(target=async_task).start()


    def __call__(self):
        self.run()

class Task58(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 58, 0)] is not None

    def run(self):
        # Move result of PV to CPU
        def async_task():
            bsz = self.model.layers[self.layer_idx].bsz
            num_heads = self.model.layers[self.layer_idx].num_heads
            q_len = self.model.layers[self.layer_idx].q_len
            head_dim = self.model.layers[self.layer_idx].head_dim

            src_attn_output = GetBookKeeperLinearIndex(self.layer_idx, 58, 0)
            assert self.model.tensor_buffer[src_attn_output] is not None

            attn_output = self.model.tensor_buffer[src_attn_output]

            attn_output = attn_output.to('cpu')

            if attn_output.size() != (bsz, num_heads, q_len, head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            
            self.secllm_cpp_wrapper.BookKeeperStore_Uint32(self.layer_idx, 59, 0, attn_output, (bsz, q_len, num_heads * head_dim))
            self.model.tensor_buffer[src_attn_output] = None
        threading.Thread(target=async_task).start()

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task59(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.PVDecKeyIsAvailable(self.layer_idx)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 59, 0)
        return ready

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, 59, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 60, 0)]

        self.secllm_cpp_wrapper.Decrypt_PV(self.layer_idx, src, dst)


    def __call__(self):
        self.run()

class Task60(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 60, 0)

    def run(self):
        # Bypass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 60, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 62, 0)]

        self.secllm_cpp_wrapper.UnshiftAndDequantizePV(self.layer_idx, src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task61(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        # Move o_proj weight to GPU
        def async_task():
            self.model.layers[self.layer_idx].o_proj.weight = self.model.layers[self.layer_idx].o_proj.weight.to('cuda:0')
        threading.Thread(target=async_task).start()
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task62(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    # Before encryption, it is float
    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 62, 0)

    def run(self):
        # ByPass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 62, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 63, 0)]

        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, 3) # O

    def __call__(self):
        self.run()

class Task63(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 63, 0)

    def run(self):
        # Move enc_activation to GPU
        def async_task():
            enc_activation = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, 63, 0)
            enc_activation = enc_activation.to('cuda:0')

            dst = GetBookKeeperLinearIndex(self.layer_idx, 64, 1)
            self.model.tensor_buffer[dst] = enc_activation
        threading.Thread(target=async_task).start()

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task64(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 64, 1)] is not None
        ready &= self.model.layers[self.layer_idx].o_proj.weight.is_cuda
        return ready

    def run(self):
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, 64, 1)
            assert self.model.tensor_buffer[src] is not None

            x = self.model.tensor_buffer[src]
            x_shape = x.shape
            x = x.view(-1, x_shape[-1])

            x_cupy = cupy.from_dlpack(x.to(torch.int32))
            weight_T_cupy = cupy.from_dlpack(self.model.layers[self.layer_idx].o_proj.weight.transpose(-2, -1).to(torch.int32))
            
            y_cupy = cupy.matmul(x_cupy, weight_T_cupy)
            y = torch.from_dlpack(y_cupy)

            y = y.view(*x_shape[:-1], -1)

            self.model.tensor_buffer[src] = None

            dst = GetBookKeeperLinearIndex(self.layer_idx, 65, 0)
            self.model.tensor_buffer[dst] = y
        threading.Thread(target=async_task).start()


    def __call__(self):
        self.run()

class Task65(Task): 
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 65, 0)] is not None

    def run(self):
        # Move o_proj output to CPU
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, 65, 0)
            assert self.model.tensor_buffer[src] is not None
            y = self.model.tensor_buffer[src]
            y = y.to('cpu')

            self.model.tensor_buffer[src] = None
            self.secllm_cpp_wrapper.BookKeeperStore_Uint32(self.layer_idx, 66, 0, y.to(torch.uint32))
        threading.Thread(target=async_task).start()

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task66(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 66, 0)

    def run(self):
        # Bypass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 66, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 67, 1), GetBookKeeperLinearIndex(self.layer_idx, 91, 1)]
        
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, 3) # O

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task67(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    # both inputs are float for residual add
    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 67, 0)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 67, 1)
        return ready

    def run(self):
        # Elementwise add
        src1 = GetBookKeeperLinearIndex(self.layer_idx, 67, 0)
        src2 = GetBookKeeperLinearIndex(self.layer_idx, 67, 1)

        dst = [GetBookKeeperLinearIndex(self.layer_idx, 68, 0)]

        self.secllm_cpp_wrapper.ElementwiseAdd(src1, src2, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task68(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 68, 0)

    def run(self):
        # Broadcast
        src = GetBookKeeperLinearIndex(self.layer_idx, 68, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 69, 0), GetBookKeeperLinearIndex(self.layer_idx, 90, 1)]

        self.secllm_cpp_wrapper.ReplicateTensor(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task69(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 69, 0)

    def run(self):
        # RMS 2
        src = GetBookKeeperLinearIndex(self.layer_idx, 69, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 70, 0)]

        post_attention_layernorm = self.model.layers[self.layer_idx].post_attention_layernorm

        self.secllm_cpp_wrapper.RMSNorm(src, dst, post_attention_layernorm.weight, post_attention_layernorm.variance_epsilon)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task70(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 70, 0)

    def run(self):
        # Broadcast
        src = GetBookKeeperLinearIndex(self.layer_idx, 70, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 73, 0), GetBookKeeperLinearIndex(self.layer_idx, 74, 0)]

        self.secllm_cpp_wrapper.ReplicateTensor(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task71(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        # Move gate_proj weight to GPU
        def async_task():
            self.model.layers[self.layer_idx].gate_proj.weight = self.model.layers[self.layer_idx].gate_proj.weight.to('cuda:0')
        threading.Thread(target=async_task).start()

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task72(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        # Move up_proj weight to GPU
        def async_task():
            self.model.layers[self.layer_idx].up_proj.weight = self.model.layers[self.layer_idx].up_proj.weight.to('cuda:0')
        threading.Thread(target=async_task).start()

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task73(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 73, 0)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, 73, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 75, 0)]
        
        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, 5) # Gate

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task74(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 74, 0)

    def run(self):
        # Bypass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 74, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 76, 0)]
        
        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, 4) # Up

    def __call__(self):
        self.run()

class Task75(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 75, 0)

    def run(self):
        def async_task():
            enc_activation = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, 75, 0)
            enc_activation = enc_activation.to('cuda:0')

            dst = GetBookKeeperLinearIndex(self.layer_idx, 77, 1)
            self.model.tensor_buffer[dst] = enc_activation
        threading.Thread(target=async_task).start()

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task76(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 76, 0)

    def run(self):
        def async_task():
            enc_activation = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, 76, 0)
            enc_activation = enc_activation.to('cuda:0')

            dst = GetBookKeeperLinearIndex(self.layer_idx, 78, 1)
            self.model.tensor_buffer[dst] = enc_activation
        threading.Thread(target=async_task).start()

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task77(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 77, 1)] is not None
        ready &= self.model.layers[self.layer_idx].gate_proj.weight.is_cuda
        return ready

    def run(self):
        # gate_proj
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, 77, 1)
            assert self.model.tensor_buffer[src] is not None

            x = self.model.tensor_buffer[src]
            x_shape = x.shape
            x = x.view(-1, x_shape[-1])

            x_cupy = cupy.from_dlpack(x.to(torch.int32))
            weight_T_cupy = cupy.from_dlpack(self.model.layers[self.layer_idx].gate_proj.weight.transpose(-2, -1).to(torch.int32))
            y_cupy = cupy.matmul(x_cupy, weight_T_cupy)
            y = torch.from_dlpack(y_cupy)

            y = y.view(*x_shape[:-1], -1)

            self.model.tensor_buffer[src] = None

            dst = GetBookKeeperLinearIndex(self.layer_idx, 79, 0)
            self.model.tensor_buffer[dst] = y
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task78(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 78, 1)] is not None
        ready &= self.model.layers[self.layer_idx].up_proj.weight.is_cuda
        return ready

    def run(self):
        # up_proj
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, 78, 1)
            assert self.model.tensor_buffer[src] is not None

            x = self.model.tensor_buffer[src]
            x_shape = x.shape
            x = x.view(-1, x_shape[-1])

            x_cupy = cupy.from_dlpack(x.to(torch.int32))
            weight_T_cupy = cupy.from_dlpack(self.model.layers[self.layer_idx].up_proj.weight.transpose(-2, -1).to(torch.int32))
            y_cupy = cupy.matmul(x_cupy, weight_T_cupy)
            y = torch.from_dlpack(y_cupy)

            y = y.view(*x_shape[:-1], -1)

            self.model.tensor_buffer[src] = None

            dst = GetBookKeeperLinearIndex(self.layer_idx, 80, 0)
            self.model.tensor_buffer[dst] = y
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task79(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 79, 0)] is not None

    def run(self):
        # Move gate_proj output to CPU
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, 79, 0)
            assert self.model.tensor_buffer[src] is not None
            y = self.model.tensor_buffer[src]
            y = y.to('cpu')

            self.model.tensor_buffer[src] = None

            self.secllm_cpp_wrapper.BookKeeperStore_Uint32(self.layer_idx, 81, 0, y.to(torch.uint32))
        threading.Thread(target=async_task).start()

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task80(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 80, 0)] is not None

    def run(self):
        # Move up_proj output to CPU
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, 80, 0)
            assert self.model.tensor_buffer[src] is not None
            y = self.model.tensor_buffer[src]
            y = y.to('cpu')

            self.model.tensor_buffer[src] = None

            self.secllm_cpp_wrapper.BookKeeperStore_Uint32(self.layer_idx, 82, 0, y.to(torch.uint32))
        threading.Thread(target=async_task).start()
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task81(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 81, 0)

    def run(self):
        # Decrypt gate_proj output
        src = GetBookKeeperLinearIndex(self.layer_idx, 81, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 83, 0)]
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, 5) # Gate

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task82(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 82, 0)

    def run(self):
        # Decrypt up_proj output
        src = GetBookKeeperLinearIndex(self.layer_idx, 82, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 83, 1)]
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, 4) # Up

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task83(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 83, 0)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 83, 1)
        return ready

    def run(self):
        # SwiGLU
        src1 = GetBookKeeperLinearIndex(self.layer_idx, 83, 0)
        src2 = GetBookKeeperLinearIndex(self.layer_idx, 83, 1)

        dst = [GetBookKeeperLinearIndex(self.layer_idx, 85, 0)]

        self.secllm_cpp_wrapper.SwiGLU(src1, src2, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task84(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return True

    def run(self):
        # Move down_proj weight to GPU
        def async_task():
            self.model.layers[self.layer_idx].down_proj.weight = self.model.layers[self.layer_idx].down_proj.weight.to('cuda:0')
        threading.Thread(target=async_task).start()

    def __call__(self):
        self.run()

class Task85(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 85, 0)

    def run(self):
        # Encrypt down_proj input
        src = GetBookKeeperLinearIndex(self.layer_idx, 85, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 86, 0)]

        self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, dst, 6) # Down

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task86(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 86, 0)

    def run(self):
        # Move enc_x to GPU
        def async_task():
            enc_activation = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, 86, 0)
            enc_activation = enc_activation.to('cuda:0')

            dst = GetBookKeeperLinearIndex(self.layer_idx, 87, 1)
            self.model.tensor_buffer[dst] = enc_activation
        threading.Thread(target=async_task).start()
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task87(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 87, 1)] is not None
        ready &= self.model.layers[self.layer_idx].down_proj.weight.is_cuda
        return ready

    def run(self):
        # down_proj
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, 87, 1)
            assert self.model.tensor_buffer[src] is not None

            x = self.model.tensor_buffer[src]
            x_shape = x.shape
            x = x.view(-1, x_shape[-1])

            x_cupy = cupy.from_dlpack(x.to(torch.int32))
            weight_T_cupy = cupy.from_dlpack(self.model.layers[self.layer_idx].down_proj.weight.transpose(-2, -1).to(torch.int32))
            y_cupy = cupy.matmul(x_cupy, weight_T_cupy)
            y = torch.from_dlpack(y_cupy)

            y = y.view(*x_shape[:-1], -1)

            self.model.tensor_buffer[src] = None

            dst = GetBookKeeperLinearIndex(self.layer_idx, 88, 0)
            self.model.tensor_buffer[dst] = y
        threading.Thread(target=async_task).start()


    def __call__(self):
        self.run()

class Task88(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.model.tensor_buffer[GetBookKeeperLinearIndex(self.layer_idx, 88, 0)] is not None

    def run(self):
        # Move down_proj output to CPU
        def async_task():
            src = GetBookKeeperLinearIndex(self.layer_idx, 88, 0)
            assert self.model.tensor_buffer[src] is not None
            y = self.model.tensor_buffer[src]
            y = y.to('cpu')
            self.model.tensor_buffer[src] = None
            self.secllm_cpp_wrapper.BookKeeperStore_Uint32(self.layer_idx, 89, 0, y.to(torch.uint32))
        
        threading.Thread(target=async_task).start()


    def __call__(self):
        self.run()

class Task89(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        return self.secllm_cpp_wrapper.BookKeeperIsAvailable_Uint32(self.layer_idx, 89, 0)

    def run(self):
        # Bypass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 89, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 90, 0)]
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, src, dst, 6)
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task90(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def is_ready(self):
        ready = True
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 90, 0)
        ready &= self.secllm_cpp_wrapper.BookKeeperIsAvailable(self.layer_idx, 90, 1)
        return ready

    def run(self):
        # Elementwise add
        src1 = GetBookKeeperLinearIndex(self.layer_idx, 90, 0)
        src2 = GetBookKeeperLinearIndex(self.layer_idx, 90, 1)

        dst = [GetBookKeeperLinearIndex(self.layer_idx, 91, 0)]
        self.secllm_cpp_wrapper.ElementwiseAdd(src1, src2, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class TaskSubclassA(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        print(f"TaskSubclassA: {self.name, self.task_id, self.next_task_ids}")

    def __call__(self):
        self.run()

class TaskSubclassB(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        print(f"TaskSubclassB: {self.name, self.task_id, self.next_task_ids}")

    def __call__(self):
        self.run()

if __name__ == '__main__':
    task = Task("task 0", 0, [1, 2])
    task()