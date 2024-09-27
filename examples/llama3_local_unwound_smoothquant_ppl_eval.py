import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, '..')
sys.path.append(module_path)

from models.quantized.llama.unwound_smoothquant_modeling_llama import UnwoundSqLlamaForCausalLM

from transformers import AutoTokenizer

import ppl_evaluator

remote_model_id = 'meta-llama/Meta-Llama-3-8B'
local_model_id = './pretrained_weights/Meta-Llama-3-8B-unwound-smoothquant.pt'
'''
    When original weights are used for pure model ppl: 5.911846160888672
    When smoothed weights are used for pure model ppl: 5.911849498748779
'''

model = UnwoundSqLlamaForCausalLM.from_pretrained(local_model_id, torch_dtype=torch.float16, device_map='cpu', attn_implementation='eager')
model.my_post_init()

from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(remote_model_id)
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

# NOTE(jpyo0803): For faster evaluation, set n_samples to a smaller number
evaluator = ppl_evaluator.Evaluator(dataset, tokenizer, model.device, n_samples=2)

'''
    n_samples=10, cuda -> ppl=6.267731666564941
    n_samples=10, cpu + cuda -> ppl=6.259809494018555
'''


from memory_monitor import MemoryMonitor

memory_monitor = MemoryMonitor(0.1)

ppl = evaluator.evaluate(model)

memory_monitor.stop()

print(f"Local Smoothquant Llama3-8B (int8) perplexity: {ppl}")

# Output the peak GPU memory usage
peak_cpu_mem_used, peak_gpu_mem_used = memory_monitor.get_max_mem_used()
print(f"Peak CPU memory usage: {peak_cpu_mem_used:0.3f} MB")
print(f"Peak GPU memory usage: {peak_gpu_mem_used:0.3f} MB")


