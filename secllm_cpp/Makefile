CC = g++
# CXXFLAGS = -fPIC -Wall -O2
CXXFLAGS = -fPIC -Wall -O2 -DEIGEN_USE_MKL_ALL -I/home/jpyo0803/secllm/include  # Include path to mkl.h
LDFLAGS = -shared /home/jpyo0803/secllm/lib/libmkl_intel_lp64.so.2 \
/home/jpyo0803/secllm/lib/libmkl_core.so.2 \
/home/jpyo0803/secllm/lib/libmkl_sequential.so.2 -lm -ldl  # Direct full path to MKL .so files

SRCS = secllm.cpp aes_stream.cpp func_utils.cpp decoder_layer.cpp secllm_interface.cpp thread_pool.cpp
OBJS = $(SRCS:.cpp=.o)

TARGET_LIB = libsecllm.so
GTEST = /usr/local/lib/libgtest.a

# New variables for libtest.so
LIBTEST_SRC = func_utils.cpp aes_stream.cpp test_interface.cpp
LIBTEST_OBJ = $(LIBTEST_SRC:.cpp=.o)
LIBTEST_TARGET = libtest.so

.PHONY: all
all: $(TARGET_LIB) $(LIBTEST_TARGET)

$(TARGET_LIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

$(LIBTEST_TARGET): $(LIBTEST_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

$(SRCS:.cpp=.d):%.d:%.cpp
	$(CC) $(CXXFLAGS) -MM $< >$@

include $(SRCS:.cpp:.d)

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET_LIB) $(SRCS:.cpp:.d) $(LIBTEST_OBJ) $(LIBTEST_TARGET)

.PHONY: test
test:
	$(CC) -o secllm_unittest secllm_unittest.cpp -L. -lsecllm -Wl,-rpath,. $(GTEST)
