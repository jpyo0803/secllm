### 실행 환경

#### Hardware 정보
- Server: deep7
- CPU: Intel(R) Xeon(R) Gold 6326
- GPU: NVIDIA GeForce RTX 4090 24GB GMEM (10G이하의 GMEM으로도 예제를 돌리기에 충분, 실제로 예제 SGX simulation 모드에서 3GB 요구)
- RAM: 882GB (30GB 정도면 예제를 돌리기에 충분, 실제로 예제 SGX simulation 모드에서 20GB 요구)

#### Software 정보
- Python version: 3.11.7
- Intel SGX PSW / SDK 설치
- 필요 Python 패키지는 requirements.txt 참조

### 실행 예제

#### SGX-enabled Simulation 모드 예제
Top directory에 **run_sgx_sim_mode.sh**을 실행. 이때 secllm_cpp/secllm_cpp_wrapper.py에 최상단에 있는 변수 **USE_SGX = True**로 설정. 

#### SGX-enabled Hardware 모드 예제
Top directory에 **run_sgx_hw_mode.sh**을 실행. 이때 secllm_cpp/secllm_cpp_wrapper.py에 최상단에 있는 변수 **USE_SGX = True**로 설정. 

#### SGX-disabled 모드 예제 (디버깅용)
TEE 구역내에 실행과정 관찰을 위해 SGX를 키지 않고 실행하기 위해서는 top directory에 **run_no_sgx.sh**를 실행한다. 이때 secllm_cpp/secllm_cpp_wrapper.py에 최상단에 있는 변수 **USE_SGX = False**로 설정. 
보
