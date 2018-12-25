CPP=$(shell (which icpc 2>/dev/null || which g++) | tail -n 1)
# OPT=-O3 -Wall -DNDEBUG -Wno-c++11-extensions
OPT=-O2 -Wall -std=c++11
INCLUDE=

all: stoch_block_sampling.out

HEADERS=network.hpp sampling.hpp
SRCS=network.cpp sampling.cpp
OBJS=network.o sampling.o

network.o: network.cpp network.hpp
	$(CPP) $(OPT) -c $(INCLUDE) network.cpp

sampling.o: sampling.cpp $(HEADERS)
	$(CPP) $(OPT) -c $(INCLUDE) sampling.cpp

stoch_block_sampling.out: $(OBJS) $(HEADERS) stoch_block_sampling.cpp Makefile
	$(CPP) $(OPT) $(INCLUDE) ${OBJS} stoch_block_sampling.cpp -o $@

clean:
	rm -f *.out *~ *.bak *.o

