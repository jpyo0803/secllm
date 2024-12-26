cd secllm_cpp &&
make -f Makefile.sgx clean &&
make clean &&
make -f Makefile.sgx && cd .. && python3 examples/llama3_local_unwound_smoothquant_ppl_eval.py