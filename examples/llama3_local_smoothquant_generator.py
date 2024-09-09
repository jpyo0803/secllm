import torch

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, '..', 'models', 'quantized', 'llama')
sys.path.append(module_path)

from smoothquant_modeling_llama import SqLlamaForCausalLM

from transformers import AutoTokenizer

remote_model_id = 'meta-llama/Meta-Llama-3-8B'
local_model_id = './pretrained_weights/Meta-Llama-3-8B-smoothquant.pt'

tokenizer = AutoTokenizer.from_pretrained(remote_model_id)
model = SqLlamaForCausalLM.from_pretrained(local_model_id, torch_dtype=torch.float16, device_map='cuda', attn_implementation='eager')

text = "Hello, what is the 3+7?"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

from memory_monitor import MemoryMonitor

memory_monitor = MemoryMonitor(0.1)

outputs = model.generate(inputs["input_ids"], max_length=8000)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

memory_monitor.stop()
peak_cpu_mem_used, peak_gpu_mem_used = memory_monitor.get_max_mem_used()
print("Generated Text:", generated_text)
# Print the number of tokens in the output
num_output_tokens = outputs.shape[1]  # The second dimension represents the token count
print(f"Number of output tokens: {num_output_tokens}")

print(f"Peak CPU memory usage: {peak_cpu_mem_used:0.3f} MB")
print(f"Peak GPU memory usage: {peak_gpu_mem_used:0.3f} MB")
