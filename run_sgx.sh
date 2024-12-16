cd secllm_cpp &&
make SGX_MODE=SIM_RELEASE -f Makefile.sgx clean &&
make clean &&
make SGX_MODE=SIM_RELEASE -f Makefile.sgx && cd .. && python3 examples/llama3_local_unwound_smoothquant_ppl_eval.py