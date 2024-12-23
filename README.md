### Running Environment

#### Hardware info
- Server: deep7
- CPU: Intel(R) Xeon(R) Gold 6326
- GPU: NVIDIA GeForce RTX 4090 24GB GMEM (10G이하의 GMEM으로도 예제를 돌리기에 충분)
- RAM: 882GB (30GB 정도면 예제를 돌리기에 충분)

#### Software info
- Python version: 3.11.7
- Intel SGX PSW / SDK 설치
- 필요 Python 패키지는 requirements.txt 참조

### 실행 예제

#### SGX-enabled 모드 예제 (Simulation Mode)
Top directory에 **run_sgx.sh**을 실행. 이때 secllm_cpp/secllm_cpp_wrapper.py에 최상단에 있는 변수 **USE_SGX = True**로 설정. 

#### 디버깅 모드 예제 (SGX-disabled)
TEE 구역내에 실행과정 관찰을 위해 SGX를 키지 않고 실행하기 위해서는 top directory에 **run_no_sgx.sh**를 실행한다. 이때 secllm_cpp/secllm_cpp_wrapper.py에 최상단에 있는 변수 **USE_SGX = False**로 설정. 

#### SGX 하드웨어 모드 실행
```sh
make -f Makefile.sgx && cd .. && python3 examples/llama3_local_unwound_smoothquant_ppl_eval.py
```