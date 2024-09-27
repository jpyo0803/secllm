CC = g++
CXXFLAGS = -fPIC -Wall -O2
LDFLAGS = -shared

SRCS = secllm.cpp aes_stream.cpp func_utils.cpp decoder_layer.cpp
OBJS = $(SRCS:.cpp=.o)

TARGET_LIB = libsecllm.so
GTEST = /usr/local/lib/libgtest.a

.PHONY: all
all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

$(SRCS:.cpp=.d):%.d:%.cpp
	$(CC) $(CXXFLAGS) -MM $< >$@

include $(SRCS:.cpp=.d)

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET_LIB) $(SRCS:.cpp=.d)

.PHONY: test
test:
	$(CC) -o secllm_unittest secllm_unittest.cpp -L. -lsecllm -Wl,-rpath,. $(GTEST)