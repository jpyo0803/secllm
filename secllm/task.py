from typing import Any
import torch

from torch_int.functional.quantization import (
    dynamic_quantize_activation_per_token_absmax,
)

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
        q_proj_weight = self.model.layers[self.layer_idx].q_proj.weight
        q_proj_weight = q_proj_weight.to('cuda:0')
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task5(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move K weight to GPU
        k_proj_weight = self.model.layers[self.layer_idx].k_proj.weight
        k_proj_weight = k_proj_weight.to('cuda:0')

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task6(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move V weight to GPU
        v_proj_weight = self.model.layers[self.layer_idx].v_proj.weight
        v_proj_weight = v_proj_weight.to('cuda:0')

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task7(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Encryption but for not it just bypasses

        src = GetBookKeeperLinearIndex(self.layer_idx, 7, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 10, 0)]
        self.secllm_cpp_wrapper.ReplicateTensor(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task8(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Encryption but for not it just bypasses
        
        src = GetBookKeeperLinearIndex(self.layer_idx, 8, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 11, 0)]
        self.secllm_cpp_wrapper.ReplicateTensor(src, dst)
        
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task9(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Encryption but for not it just bypasses

        src = GetBookKeeperLinearIndex(self.layer_idx, 9, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 12, 0)]
        self.secllm_cpp_wrapper.ReplicateTensor(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task10(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Retrieve input from BookKeeper and move to GPU
        activation = self.secllm_cpp_wrapper.BookKeeperLoad(self.layer_idx, 10, 0)

        int8_activation, int8_scales = dynamic_quantize_activation_per_token_absmax(activation)
        int8_activation = int8_activation.to('cuda:0')

        dst = GetBookKeeperLinearIndex(self.layer_idx, 13, 1)
        self.model.tensor_buffer[dst] = (int8_activation, int8_scales)
        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)


    def __call__(self):
        self.run()

class Task11(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Retrieve input from BookKeeper and move to GPU
        activation = self.secllm_cpp_wrapper.BookKeeperLoad(self.layer_idx, 11, 0)
        
        int8_activation, int8_scales = dynamic_quantize_activation_per_token_absmax(activation)
        int8_activation = int8_activation.to('cuda:0')

        dst = GetBookKeeperLinearIndex(self.layer_idx, 14, 1)
        self.model.tensor_buffer[dst] = (int8_activation, int8_scales)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task12(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Retrieve input from BookKeeper and move to GPU
        activation = self.secllm_cpp_wrapper.BookKeeperLoad(self.layer_idx, 12, 0)
        int8_activation, int8_scales = dynamic_quantize_activation_per_token_absmax(activation)
        int8_activation = int8_activation.to('cuda:0')

        dst = GetBookKeeperLinearIndex(self.layer_idx, 15, 1)
        self.model.tensor_buffer[dst] = (int8_activation, int8_scales)

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

        int8_activation, int8_scales = self.model.tensor_buffer[src]

        query_states = self.model.layers[self.layer_idx].q_proj(int8_activation, int8_scales)

        self.model.tensor_buffer[src] = None

        dst = GetBookKeeperLinearIndex(self.layer_idx, 16, 0)
        self.model.tensor_buffer[dst] = query_states

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

        int8_activation, int8_scales = self.model.tensor_buffer[src]

        key_states = self.model.layers[self.layer_idx].k_proj(int8_activation, int8_scales)

        self.model.tensor_buffer[src] = None

        dst = GetBookKeeperLinearIndex(self.layer_idx, 17, 0)
        self.model.tensor_buffer[dst] = key_states

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

        int8_activation, int8_scales = self.model.tensor_buffer[src]

        value_states = self.model.layers[self.layer_idx].v_proj(int8_activation, int8_scales)

        self.model.tensor_buffer[src] = None

        dst = GetBookKeeperLinearIndex(self.layer_idx, 18, 0)
        self.model.tensor_buffer[dst] = value_states

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task16(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move query_states to CPU
        src = GetBookKeeperLinearIndex(self.layer_idx, 16, 0)

        query_states = self.model.tensor_buffer[src]

        query_states = query_states.to('cpu')

        self.secllm_cpp_wrapper.BookKeeperStore(self.layer_idx, 19, 0, query_states.to(torch.float32))

        self.model.tensor_buffer[src] = None

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task17(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move key_states to CPU

        src = GetBookKeeperLinearIndex(self.layer_idx, 17, 0)

        key_states = self.model.tensor_buffer[src]

        key_states = key_states.to('cpu')

        self.secllm_cpp_wrapper.BookKeeperStore(self.layer_idx, 20, 0, key_states.to(torch.float32))

        self.model.tensor_buffer[src] = None

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task18(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Move value_states to CPU

        src = GetBookKeeperLinearIndex(self.layer_idx, 18, 0)

        value_states = self.model.tensor_buffer[src]

        value_states = value_states.to('cpu')

        self.secllm_cpp_wrapper.BookKeeperStore(self.layer_idx, 21, 0, value_states.to(torch.float32))

        self.model.tensor_buffer[src] = None

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task19(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Decryption but for now it just bypasses

        src = GetBookKeeperLinearIndex(self.layer_idx, 19, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 24, 0)]

        self.secllm_cpp_wrapper.ReplicateTensor(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task20(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Decryption but for now it just bypasses

        src = GetBookKeeperLinearIndex(self.layer_idx, 20, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 25, 0)]

        self.secllm_cpp_wrapper.ReplicateTensor(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task21(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        # Decryption but for now it just bypasses

        src = GetBookKeeperLinearIndex(self.layer_idx, 21, 0)
        dst = [GetBookKeeperLinearIndex(self.layer_idx, 47, 0)]

        self.secllm_cpp_wrapper.ReplicateTensor(src, dst)

        # self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task22(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task23(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task24(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task25(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task26(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task27(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task28(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task29(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task30(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task31(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task32(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task33(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task34(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task35(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task36(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task37(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task38(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task39(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task40(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task41(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task42(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task43(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task44(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task45(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task46(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task47(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task48(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task49(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task50(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task51(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task52(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task53(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task54(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task55(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task56(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task57(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task58(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task59(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task60(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task61(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task62(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task63(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task64(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task65(Task): 
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task66(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task67(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task68(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task69(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task70(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task71(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task72(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task73(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task74(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task75(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task76(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task77(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task78(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task79(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task80(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task81(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task82(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task83(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task84(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task85(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task86(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task87(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task88(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task89(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

    def __call__(self):
        self.run()

class Task90(Task):
    def __init__(self, name: str, layer_idx : int, task_id : int, next_task_ids: list[int], secllm_cpp_wrapper, model):
        super().__init__(name, layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model)

    def run(self):
        self.secllm_cpp_wrapper.PrintTest(self.layer_idx, self.task_id)

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