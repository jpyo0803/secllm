CC = g++
CXXFLAGS = -fPIC -Wall -O2
LDFLAGS = -shared

SRCS = secllm.cpp
OBJS = $(SRCS:.cpp=.o)

TARGET_LIB = secllm.so

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