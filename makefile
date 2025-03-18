SOURCES = utils.cpp matrix_io.cpp matrix_operations.cpp matrix_inversion.cpp memory.cpp main.cpp
HEADERS = utils.h matrix_io.h matrix_operations.h matrix_inversion.h memory.h
CXX = mpicxx
CXXFLAGS = -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/ -O3 -mfpmath=sse -fstack-protector-all -g -W -Wall -Wextra -Wunused -Wcast-align -Werror -pedantic -pedantic-errors -Wfloat-equal -Wpointer-arith -Wformat-security -Wmissing-format-attribute -Wformat=1 -Wwrite-strings -Wcast-align -Wno-long-long -Woverloaded-virtual -Wnon-virtual-dtor -Wcast-qual -Wno-suggest-attribute=format

all:
	$(CXX) $(CXXFLAGS) $(SOURCES) -o a.out

clean:
	rm -f *.o
	rm -f a.out
