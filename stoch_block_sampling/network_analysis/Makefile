OPT ?= -O2 -Wall -std=c++11
INCLUDE ?=

all: analyzer.out 

HEADERS=network.hpp
SRCS=analysis_main.cpp network.cpp
analyzer.out: $(SRCS) $(HEADERS) Makefile
	$(CXX) $(OPT) $(INCLUDE) ${SRCS} -o $@

clean:
	rm -f *.out *~ *.bak

