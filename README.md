
[GPU Offloading Model]   
우리가 LLM 추론을 위해 채택한 computation model은 GPU memory보다 상대적으로 크고 저렴한 CPU host memory에 필요 데이터를 저장하는 FlexGen model (ICML 23)과 유사하다.   
이와 같은 offloading scheme은 평소에는 CPU host memory에 model weight나 KV cache를 저장하고 있다가 matrix multiplication과 같은 가속기가 효율적으로 처리할 수 있는 연산을 해야할때만 GPU에 필요한 데이터를 보내고 연산하여 결과를 돌려받는다.   

[Motivation - Protecting User Input for Privacy]   
본 연구의 주 목적은 앞서 설명한 GPU offloading model에 따라 사용자 민감정보를 담고 있는 입력 데이터가 연산을 위해 GPU로 내보내질때 생기는 개인정보 유출을 적절한 암호화 기법을 통해 방지하는 것이다.   
기존의 머신러닝 모델 (ex. RNN, CNN)은 대부분 (사용자 입력 정보)와 (모델 가중치) 사이의 연산을 의미했다.   
이는 보호해야될 대상이 사용자 입력 데이터와 모델 가중치 중에 하나라면 기존의 Slalom (ICLR 19)의 암호화 기법을 사용하면 효율적으로 사용자 입력 데이터를 공격자로부터 감춘상태에서 효율적인 연산이 가능하다.  
하지만 LLM의 attention mechanism은 연산 특성상 앞서 언급한 기존의 머신러닝 모델과 달리 (사용자 입력 정보)와 (사용자 입력 정보)사이에 연산을 필요로한다.   
이러한 연산 특징은 기존의 Slalom 방식을 사용하지 못하게 된다 (이 경우 offloading model을 사용하지 않고 CPU에서 단독으로 처리하는 것이 언제나 더 효율적이다).   
정리하자면 본 연구가 추구하는 것을 한마디로 표현하자면 "행렬곱셈의 양쪽 피연산자가 둘다가 오프로딩시 보호의 대상인 경우 어떻게 공격자로부터 confidentiality를 지킬까"이다.   

[본 연구가 제안했던 암호화 기법 - 사용불가]   
결론부터 말하자면, 여기서 소개할 encryption scheme은 안전하다고 여기기 위한 bit security level (2^80 이상)을 충족하지 못해 안전하지 않아 사용할 수 없는 것으로 판명.   

Substitution cipher의 한 종류인 affine cipher (a * x + b 형태)를 활용하여 사용자 입력 행렬을 행과 열단위로 각각 곱셈 암호키 a와 덧셈 암호키 b를 공유하여 암호화 시킨다. 자세한 내용은 학위논문 참조 바람.   
이렇게 행과 열단위로 곱셈 및 덧셈 암호키를 공유하는 것은 결국엔 평문 데이터를 감추는데 하나의 암호키를 여러번 재사용하는 것과 동일하며 이는 cryptography분야에서는 결정적인 취약점이라 한다. 본 연구가 제안했던 방식도 결국에는 암호키의 재사용때문에 공격자가 손쉽게 사용자 정보를 탈취할 수 있으므로 위 암호화 기법은 사용하면 안된다.   



### Running Environment
#### Hardware info
- Server: deep7
- CPU: Intel(R) Xeon(R) Gold 6326
- GPU: NVIDIA GeForce RTX 4090 24GB GMEM (10G이하의 GMEM으로도 예제를 돌리기에 충분)
- RAM: 882GB (30GB 정도면 예제를 돌리기에 충분)
#### Software info
- Python version: 3.11.7
- Intel SGX PSW / SDK
- 필요 Python 패키지는 requirements.txt 참조


### Running Example
- SGX-enabled example 실행을 위해서는 top directory에 **run_sgx.sh**을 실행 
- 디버깅을 위해 SGX-disabled example 실행을 위해서는 top directory에 **run_no_sgx.sh**을 실행 (실행전 secllm_cpp/secllm_cpp_wrapper.py에 최상단에 있는 **USE_SGX** 변수를 적절하게 설정해주어야함. SGX를 사용하지 않을 것이라면 **False**로 설정)