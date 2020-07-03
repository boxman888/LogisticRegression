CXX = g++
CXXFLAGS = -std=c++11 -Wall -fpic -O2 -I ../../eigen
SRCS = regression.cpp USPS.cpp
HEADERS = USPS.h
OBJS = regression.o USPS.o

all: ${SRCS} ${HEADERS}
	${CXX} ${CXXFLAGS} ${SRCS} -o regression

${OBJS}: ${SRCS}
	${CXX} -c $(@:.o=.cpp)

clear:
	rm -f *.o regression
