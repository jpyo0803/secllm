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

import math

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

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task1(Task):
    # Copy 
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)
        src = GetBookKeeperLinearIndex(self.layer_idx, 1, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 2, 0), GetBookKeeperLinearIndex(self.layer_idx, 67, 0)]
        self.secllm_cpp_wrapper.ReplicateTensor(src, dst)

    def __call__(self):
        self.run()

class Task2(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)
        src = GetBookKeeperLinearIndex(self.layer_idx, 2, 0)
        dst = GetBookKeeperLinearIndex(self.layer_idx, 3, 0)

        input_layernorm = self.model.layers[self.layer_idx].input_layernorm

        self.secllm_cpp_wrapper.RMSNorm(src, dst, input_layernorm.weight, input_layernorm.variance_epsilon)

    def __call__(self):
        self.run()

class Task3(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

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

    def run(self):
        # Move Q weight to GPU
        self.model.layers[self.layer_idx].q_proj.weight = self.model.layers[self.layer_idx].q_proj.weight.to('cuda:0')
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task5(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move K weight to GPU
        self.model.layers[self.layer_idx].k_proj.weight = self.model.layers[self.layer_idx].k_proj.weight.to('cuda:0')

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task6(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move V weight to GPU
        self.model.layers[self.layer_idx].v_proj.weight = self.model.layers[self.layer_idx].v_proj.weight.to('cuda:0')

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task7(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # 

        src = GetBookKeeperLinearIndex(self.layer_idx, 7, 0)

        enc_activation = self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, 0) # Q
        
        dst = GetBookKeeperLinearIndex(self.layer_idx, 10, 0)
        self.model.tensor_buffer[dst] = enc_activation
        # dst = [GetBookKeeperLinearIndex(self.layer_idx, 10, 0)]
        # self.secllm_cpp_wrapper.ReplicateTensor(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task8(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Encryption but for not it just bypasses
        
        src = GetBookKeeperLinearIndex(self.layer_idx, 8, 0)

        enc_activation = self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, 1) # K

        dst = GetBookKeeperLinearIndex(self.layer_idx, 11, 0)
        self.model.tensor_buffer[dst] = enc_activation

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task9(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Encryption but for not it just bypasses

        src = GetBookKeeperLinearIndex(self.layer_idx, 9, 0)

        enc_activation = self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, 2) # V

        dst = GetBookKeeperLinearIndex(self.layer_idx, 12, 0)
        self.model.tensor_buffer[dst] = enc_activation

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task10(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Retrieve input from BookKeeper and move to GPU
        src = GetBookKeeperLinearIndex(self.layer_idx, 10, 0)
        assert self.model.tensor_buffer[src] is not None
        enc_activation = self.model.tensor_buffer[src]
        self.model.tensor_buffer[src] = None
        
        enc_activation = enc_activation.to('cuda:0')

        dst = GetBookKeeperLinearIndex(self.layer_idx, 13, 1)
        self.model.tensor_buffer[dst] = (enc_activation)
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)


    def __call__(self):
        self.run()

class Task11(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Retrieve input from BookKeeper and move to GPU
        src = GetBookKeeperLinearIndex(self.layer_idx, 11, 0)
        assert self.model.tensor_buffer[src] is not None
        enc_activation = self.model.tensor_buffer[src]
        self.model.tensor_buffer[src] = None

        enc_activation = enc_activation.to('cuda:0')

        dst = GetBookKeeperLinearIndex(self.layer_idx, 14, 1)
        self.model.tensor_buffer[dst] = (enc_activation)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task12(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Retrieve input from BookKeeper and move to GPU
        src = GetBookKeeperLinearIndex(self.layer_idx, 12, 0)
        assert self.model.tensor_buffer[src] is not None
        enc_activation = self.model.tensor_buffer[src]
        self.model.tensor_buffer[src] = None

        enc_activation = enc_activation.to('cuda:0')

        dst = GetBookKeeperLinearIndex(self.layer_idx, 15, 1)
        self.model.tensor_buffer[dst] = (enc_activation)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task13(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Do Q projection
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

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task14(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Do K projection
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

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task15(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Do V projection
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

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task16(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move query_states to CPU
        src = GetBookKeeperLinearIndex(self.layer_idx, 16, 0)
        assert self.model.tensor_buffer[src] is not None
        y = self.model.tensor_buffer[src]
        y = y.to('cpu')

        self.model.tensor_buffer[src] = None
        
        dst = GetBookKeeperLinearIndex(self.layer_idx, 19, 0)
        self.model.tensor_buffer[dst] = y

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task17(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move key_states to CPU

        src = GetBookKeeperLinearIndex(self.layer_idx, 17, 0)
        assert self.model.tensor_buffer[src] is not None

        y = self.model.tensor_buffer[src]
        y = y.to('cpu')

        self.model.tensor_buffer[src] = None

        dst = GetBookKeeperLinearIndex(self.layer_idx, 20, 0)
        self.model.tensor_buffer[dst] = y

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task18(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move value_states to CPU

        src = GetBookKeeperLinearIndex(self.layer_idx, 18, 0)
        assert self.model.tensor_buffer[src] is not None

        y = self.model.tensor_buffer[src]
        y = y.to('cpu')

        self.model.tensor_buffer[src] = None

        dst = GetBookKeeperLinearIndex(self.layer_idx, 21, 0)
        self.model.tensor_buffer[dst] = y

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task19(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Decryption but for now it just bypasses
        src = GetBookKeeperLinearIndex(self.layer_idx, 19, 0)
        assert self.model.tensor_buffer[src] is not None
        y = self.model.tensor_buffer[src] # need to be decrypted, int32
        self.model.tensor_buffer[src] = None

        dst = GetBookKeeperLinearIndex(self.layer_idx, 22, 0)
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, dst, y, 0)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task20(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Decryption but for now it just bypasses

        src = GetBookKeeperLinearIndex(self.layer_idx, 20, 0)
        assert self.model.tensor_buffer[src] is not None
        y = self.model.tensor_buffer[src] # need to be decrypted, int32
        self.model.tensor_buffer[src] = None

        dst = GetBookKeeperLinearIndex(self.layer_idx, 22, 1)
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, dst, y, 1)

    def __call__(self):
        self.run()

class Task21(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Decryption but for now it just bypasses

        src = GetBookKeeperLinearIndex(self.layer_idx, 21, 0)
        assert self.model.tensor_buffer[src] is not None
        y = self.model.tensor_buffer[src] # need to be decrypted, int32
        self.model.tensor_buffer[src] = None

        dst = GetBookKeeperLinearIndex(self.layer_idx, 22, 2)
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, dst, y, 2)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task22(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

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
        # cache_position = self.model.layers[self.layer_idx].cache_position
        # past_key_value = self.model.layers[self.layer_idx].past_key_value

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2).contiguous()

        cos, sin = self.secllm_cpp_wrapper.LlamaRotaryEmbedding(inv_freq, position_ids, torch.float32)
        query_states, key_states = self.secllm_cpp_wrapper.ApplyRotaryPosEmb(query_states, key_states, cos, sin)
        # cos, sin = self.model.layers[self.layer_idx].rotary_emb(value_states, position_ids)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
            # print("type : ", type(past_key_value))
            # key_states = past_key_value.update_key(key_states, self.layer_idx)
            # value_states = past_key_value.update_value(value_states, self.layer_idx)

        # query_states = query_states.contiguous()
        # key_states = key_states.contiguous()
        # value_states = value_states.contiguous()

        self.secllm_cpp_wrapper.BookKeeperStore(self.layer_idx, 28, 0, query_states.to(torch.float32))
        self.secllm_cpp_wrapper.BookKeeperStore(self.layer_idx, 29, 0, key_states.to(torch.float32))
        self.secllm_cpp_wrapper.BookKeeperStore(self.layer_idx, 47, 0, value_states.to(torch.float32))

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task23(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task24(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task25(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task26(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task27(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # pass
        self.secllm_cpp_wrapper.GenerateSecretKey_QK(self.layer_idx)
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task28(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

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

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task31(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Broadcast
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task32(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Broadcast
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task33(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # ByPass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 33, 0)
        dst = GetBookKeeperLinearIndex(self.layer_idx, 36, 0)

        self.secllm_cpp_wrapper.EncryptX_QK(self.layer_idx, src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task34(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # ByPass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 34, 0)
        dst = GetBookKeeperLinearIndex(self.layer_idx, 37, 0)

        self.secllm_cpp_wrapper.EncryptY_QK(self.layer_idx, src, dst)

        # self.secllm_cpp_wrapper.ReplicateTensor_Uint32(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task35(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # pass
        src_x = GetBookKeeperLinearIndex(self.layer_idx, 35, 1)
        src_y = GetBookKeeperLinearIndex(self.layer_idx, 35, 2)

        self.secllm_cpp_wrapper.GenerateDecryptionKey_QK(self.layer_idx, src_x, src_y)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task36(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move Enc_Q to GPU
        enc_q = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, 36, 0)
        
        enc_q = enc_q.to('cuda:0')

        dst = GetBookKeeperLinearIndex(self.layer_idx, 39, 0)
        self.model.tensor_buffer[dst] = enc_q

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task37(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move Enc_K to GPU
        enc_k = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, 37, 0)

        enc_k = enc_k.to('cuda:0')

        dst = GetBookKeeperLinearIndex(self.layer_idx, 39, 1)
        self.model.tensor_buffer[dst] = enc_k

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task38(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task39(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Matmul Q, K^T
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

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task40(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # q_output_scale = self.model.layers[self.layer_idx].q_output_scale
        # k_output_scale = self.model.layers[self.layer_idx].k_output_scale

        # head_dim = self.model.layers[self.layer_idx].head_dim

        # attention_mask = self.model.layers[self.layer_idx].attention_mask

        src = GetBookKeeperLinearIndex(self.layer_idx, 40, 0)
        assert self.model.tensor_buffer[src] is not None

        attn_weights, _ = self.model.tensor_buffer[src]

        attn_weights = attn_weights.to('cpu')
        assert attn_weights.is_contiguous()

        # if attention_mask is not None:
        #     causal_mask = attention_mask[:, :, :, : enc_k_shape_minus_2]
        #     attn_weights = attn_weights + causal_mask
        
        self.secllm_cpp_wrapper.BookKeeperStore_Uint32(self.layer_idx, 41, 0, attn_weights)
        
        self.model.tensor_buffer[src] = None

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)


    def __call__(self):
        self.run()

class Task41(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # ByPass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 41, 0)
        dst = GetBookKeeperLinearIndex(self.layer_idx, 42, 0)

        self.secllm_cpp_wrapper.Decrypt_QK(self.layer_idx, src, dst)

        # self.secllm_cpp_wrapper.ReplicateTensor_Uint32(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task42(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # ByPass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 42, 0)
        dst = GetBookKeeperLinearIndex(self.layer_idx, 43, 0)

        self.secllm_cpp_wrapper.UnshiftAndDequantizeQK(self.layer_idx, src, dst)

        # Do un shift and dequantize
        # self.secllm_cpp_wrapper.ReplicateTensor(src, dst)
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task43(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Softmax
        src = GetBookKeeperLinearIndex(self.layer_idx, 43, 0)
        dst = GetBookKeeperLinearIndex(self.layer_idx, 46, 0)

        self.secllm_cpp_wrapper.Softmax(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task44(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task45(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)
        self.secllm_cpp_wrapper.GenerateSecretKey_PV(self.layer_idx)

    def __call__(self):
        self.run()

class Task46(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

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

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task49(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Broadcast
        pass

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task50(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Broadcast
        pass

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task51(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Encrypt P, for now it just bypasses
        src = GetBookKeeperLinearIndex(self.layer_idx, 51, 0)
        dst = GetBookKeeperLinearIndex(self.layer_idx, 54, 0)

        self.secllm_cpp_wrapper.EncryptX_PV(self.layer_idx, src, dst)

        # self.secllm_cpp_wrapper.ReplicateTensor_Uint32(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task52(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # encrypt V, for now it just bypasses
        src = GetBookKeeperLinearIndex(self.layer_idx, 52, 0)
        dst = GetBookKeeperLinearIndex(self.layer_idx, 55, 0)

        self.secllm_cpp_wrapper.EncryptY_PV(self.layer_idx, src, dst)
        # self.secllm_cpp_wrapper.ReplicateTensor_Uint32(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task53(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

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

    def run(self):
        # Move enc_P to GPU
        enc_q_attn_weights = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, 54, 0)
        # attn_weights.mul_(127).round_()

        # int8_attn_weights = attn_weights.to(torch.int8)

        # int8_attn_weights = int8_attn_weights.to('cuda:0')
        enc_q_attn_weights = enc_q_attn_weights.to('cuda:0')

        dst = GetBookKeeperLinearIndex(self.layer_idx, 57, 0)

        self.model.tensor_buffer[dst] = enc_q_attn_weights

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task55(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move enc_V to GPU
        # v_output_scale = self.model.layers[self.layer_idx].v_output_scale

        enc_v = self.secllm_cpp_wrapper.BookKeeperLoad_Uint32(self.layer_idx, 55, 0)
        # int8_value_states = (value_states / v_output_scale).round().clamp(-128, 127).to(torch.int8)
        # int8_value_states = int8_value_states.to('cuda:0')
        enc_v = enc_v.to('cuda:0')

        dst = GetBookKeeperLinearIndex(self.layer_idx, 57, 1)

        self.model.tensor_buffer[dst] = enc_v

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task56(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        pass
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task57(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Matmul PV
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

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task58(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move result of PV to CPU
        bsz = self.model.layers[self.layer_idx].bsz
        num_heads = self.model.layers[self.layer_idx].num_heads
        q_len = self.model.layers[self.layer_idx].q_len
        head_dim = self.model.layers[self.layer_idx].head_dim

        src_attn_output = GetBookKeeperLinearIndex(self.layer_idx, 58, 0)
        assert self.model.tensor_buffer[src_attn_output] is not None

        attn_output = self.model.tensor_buffer[src_attn_output]

        attn_output = attn_output.to('cpu')
        # attn_output = attn_output.to(torch.float32)

        # attn_output *= v_output_scale / 127

        if attn_output.size() != (bsz, num_heads, q_len, head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.reshape(bsz, q_len, -1)

        self.secllm_cpp_wrapper.BookKeeperStore_Uint32(self.layer_idx, 59, 0, attn_output, (bsz, q_len, num_heads * head_dim))

        self.model.tensor_buffer[src_attn_output] = None

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task59(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Bypass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 59, 0)
        dst = GetBookKeeperLinearIndex(self.layer_idx, 60, 0)

        self.secllm_cpp_wrapper.Decrypt_PV(self.layer_idx, src, dst)

        # self.secllm_cpp_wrapper.ReplicateTensor_Uint32(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task60(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Bypass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 60, 0)
        dst = GetBookKeeperLinearIndex(self.layer_idx, 62, 0)

        self.secllm_cpp_wrapper.UnshiftAndDequantizePV(self.layer_idx, src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task61(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move o_proj weight to GPU
        self.model.layers[self.layer_idx].o_proj.weight = self.model.layers[self.layer_idx].o_proj.weight.to('cuda:0')
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task62(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # ByPass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 62, 0)

        enc_activation = self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, 3) # O

        dst = GetBookKeeperLinearIndex(self.layer_idx, 63, 0)
        self.model.tensor_buffer[dst] = enc_activation

    def __call__(self):
        self.run()

class Task63(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, 63, 0)
        assert self.model.tensor_buffer[src] is not None
        enc_activation = self.model.tensor_buffer[src]
        self.model.tensor_buffer[src] = None

        enc_activation = enc_activation.to('cuda:0')

        dst = GetBookKeeperLinearIndex(self.layer_idx, 64, 1)
        self.model.tensor_buffer[dst] = enc_activation

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task64(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
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

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task65(Task): 
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move o_proj output to CPU
        src = GetBookKeeperLinearIndex(self.layer_idx, 65, 0)
        assert self.model.tensor_buffer[src] is not None
        y = self.model.tensor_buffer[src]
        y = y.to('cpu')

        self.model.tensor_buffer[src] = None

        dst = GetBookKeeperLinearIndex(self.layer_idx, 66, 0)
        self.model.tensor_buffer[dst] = y

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task66(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Bypass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 66, 0)
        assert self.model.tensor_buffer[src] is not None
        y = self.model.tensor_buffer[src]
        self.model.tensor_buffer[src] = None

        dst = GetBookKeeperLinearIndex(self.layer_idx, 67, 1)
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, dst, y, 3) # O

        # Fix it
        src = GetBookKeeperLinearIndex(self.layer_idx, 67, 1)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 67, 1), GetBookKeeperLinearIndex(self.layer_idx, 91, 1)]

        self.secllm_cpp_wrapper.ReplicateTensor(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task67(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Elementwise add
        src1 = GetBookKeeperLinearIndex(self.layer_idx, 67, 0)
        src2 = GetBookKeeperLinearIndex(self.layer_idx, 67, 1)

        dst = GetBookKeeperLinearIndex(self.layer_idx, 68, 0)

        self.secllm_cpp_wrapper.ElementwiseAdd(src1, src2, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task68(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

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

    def run(self):
        # RMS 2
        src = GetBookKeeperLinearIndex(self.layer_idx, 69, 0)
        dst = GetBookKeeperLinearIndex(self.layer_idx, 70, 0)

        post_attention_layernorm = self.model.layers[self.layer_idx].post_attention_layernorm

        self.secllm_cpp_wrapper.RMSNorm(src, dst, post_attention_layernorm.weight, post_attention_layernorm.variance_epsilon)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task70(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

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

    def run(self):
        # Move gate_proj weight to GPU
        self.model.layers[self.layer_idx].gate_proj.weight = self.model.layers[self.layer_idx].gate_proj.weight.to('cuda:0')

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task72(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move up_proj weight to GPU
        self.model.layers[self.layer_idx].up_proj.weight = self.model.layers[self.layer_idx].up_proj.weight.to('cuda:0')

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task73(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Bypass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 73, 0)
        
        enc_activation = self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, 5) # Gate

        dst = GetBookKeeperLinearIndex(self.layer_idx, 75, 0)
        self.model.tensor_buffer[dst] = enc_activation

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task74(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Bypass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 74, 0)
        
        enc_activation = self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, 4) # Up

        dst = GetBookKeeperLinearIndex(self.layer_idx, 76, 0)
        self.model.tensor_buffer[dst] = enc_activation

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task75(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, 75, 0)
        assert self.model.tensor_buffer[src] is not None
        enc_activation = self.model.tensor_buffer[src]
        self.model.tensor_buffer[src] = None

        enc_activation = enc_activation.to('cuda:0')

        dst = GetBookKeeperLinearIndex(self.layer_idx, 77, 1)
        self.model.tensor_buffer[dst] = enc_activation

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task76(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        src = GetBookKeeperLinearIndex(self.layer_idx, 76, 0)
        assert self.model.tensor_buffer[src] is not None
        enc_activation = self.model.tensor_buffer[src]
        self.model.tensor_buffer[src] = None

        enc_activation = enc_activation.to('cuda:0')

        dst = GetBookKeeperLinearIndex(self.layer_idx, 78, 1)
        self.model.tensor_buffer[dst] = enc_activation

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task77(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # gate_proj
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

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task78(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # up_proj
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
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task79(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move gate_proj output to CPU
        src = GetBookKeeperLinearIndex(self.layer_idx, 79, 0)
        assert self.model.tensor_buffer[src] is not None
        y = self.model.tensor_buffer[src]
        y = y.to('cpu')

        self.model.tensor_buffer[src] = None

        dst = GetBookKeeperLinearIndex(self.layer_idx, 81, 0)
        self.model.tensor_buffer[dst] = y

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task80(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move up_proj output to CPU
        src = GetBookKeeperLinearIndex(self.layer_idx, 80, 0)
        assert self.model.tensor_buffer[src] is not None
        y = self.model.tensor_buffer[src]
        y = y.to('cpu')

        self.model.tensor_buffer[src] = None

        dst = GetBookKeeperLinearIndex(self.layer_idx, 82, 0)
        self.model.tensor_buffer[dst] = y

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task81(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Decrypt gate_proj output
        src = GetBookKeeperLinearIndex(self.layer_idx, 81, 0)
        assert self.model.tensor_buffer[src] is not None
        y = self.model.tensor_buffer[src]
        self.model.tensor_buffer[src] = None

        dst = GetBookKeeperLinearIndex(self.layer_idx, 83, 0)
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, dst, y, 5) # Gate

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task82(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Decrypt up_proj output
        src = GetBookKeeperLinearIndex(self.layer_idx, 82, 0)
        assert self.model.tensor_buffer[src] is not None
        y = self.model.tensor_buffer[src]
        self.model.tensor_buffer[src] = None

        dst = GetBookKeeperLinearIndex(self.layer_idx, 83, 1)
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, dst, y, 4) # Up

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task83(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # SwiGLU
        src1 = GetBookKeeperLinearIndex(self.layer_idx, 83, 0)
        src2 = GetBookKeeperLinearIndex(self.layer_idx, 83, 1)

        dst = GetBookKeeperLinearIndex(self.layer_idx, 85, 0)

        self.secllm_cpp_wrapper.SwiGLU(src1, src2, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task84(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move down_proj weight to GPU
        self.model.layers[self.layer_idx].down_proj.weight = self.model.layers[self.layer_idx].down_proj.weight.to('cuda:0')

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task85(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Encrypt down_proj input
        src = GetBookKeeperLinearIndex(self.layer_idx, 85, 0)

        enc_activation = self.secllm_cpp_wrapper.EncryptLinearActivation(self.layer_idx, src, 6) # Down

        dst = GetBookKeeperLinearIndex(self.layer_idx, 86, 0)
        self.model.tensor_buffer[dst] = enc_activation

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task86(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move enc_x to GPU
        src = GetBookKeeperLinearIndex(self.layer_idx, 86, 0)
        assert self.model.tensor_buffer[src] is not None
        enc_activation = self.model.tensor_buffer[src]
        self.model.tensor_buffer[src] = None

        enc_activation = enc_activation.to('cuda:0')

        dst = GetBookKeeperLinearIndex(self.layer_idx, 87, 1)
        self.model.tensor_buffer[dst] = enc_activation
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task87(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # down_proj
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

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task88(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move down_proj output to CPU
        src = GetBookKeeperLinearIndex(self.layer_idx, 88, 0)
        assert self.model.tensor_buffer[src] is not None

        y = self.model.tensor_buffer[src]
        y = y.to('cpu')

        self.model.tensor_buffer[src] = None

        dst = GetBookKeeperLinearIndex(self.layer_idx, 89, 0)
        self.model.tensor_buffer[dst] = y

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task89(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Bypass for now
        src = GetBookKeeperLinearIndex(self.layer_idx, 89, 0)
        assert self.model.tensor_buffer[src] is not None
        y = self.model.tensor_buffer[src]
        self.model.tensor_buffer[src] = None

        dst = GetBookKeeperLinearIndex(self.layer_idx, 90, 0)
        self.secllm_cpp_wrapper.DecryptLinearActivation(self.layer_idx, dst, y, 6)
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task90(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Elementwise add
        src1 = GetBookKeeperLinearIndex(self.layer_idx, 90, 0)
        src2 = GetBookKeeperLinearIndex(self.layer_idx, 90, 1)

        dst = GetBookKeeperLinearIndex(self.layer_idx, 91, 0)
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