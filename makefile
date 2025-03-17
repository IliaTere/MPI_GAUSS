TARGET = functions.h functions.cpp main.cpp
CXX = mpicxx
CXXFLAGS = -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/ -O3 -mfpmath=sse -fstack-protector-all -g -W -Wall -Wextra -Wunused -Wcast-align -Werror -pedantic -pedantic-errors -Wfloat-equal -Wpointer-arith -Wformat-security -Wmissing-format-attribute -Wformat=1 -Wwrite-strings -Wcast-align -Wno-long-long -Woverloaded-virtual -Wnon-virtual-dtor -Wcast-qual -Wno-suggest-attribute=format

all:
	$(CXX) $(CXXFLAGS) $(TARGET) -o a.out
