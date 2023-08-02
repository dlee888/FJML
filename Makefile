.PHONY: all init install docs clean coverage

CC = g++
CFLAGS = -O3 -std=c++17 -march=native -Wall -pedantic -fopenmp

ifeq ($(debug), true)
	CFLAGS += -g
else
	CFLAGS += -DNDEBUG
endif

ifeq ($(arch), cuda)
	CFLAGS = -ccbin g++ -Xcompiler -fopenmp -O3 --std=c++17 -DCUDA -lcublas
	CC = nvcc
	ifeq ($(debug), true)
		CFLAGS += -g -G
	else
		CFLAGS += -DNDEBUG
	endif
	CFLAGS += --compiler-options
endif

HEADERS = include/FJML/activations.h \
		  include/FJML/data.h \
		  include/FJML/layers.h \
		  include/FJML/linalg.h \
		  include/FJML/loss.h \
		  include/FJML/metrics.h \
		  include/FJML/mlp.h \
		  include/FJML/optimizers.h \
		  include/FJML/tensor.h
CFILES = bin/activations.o \
		 bin/data.o \
		 bin/dense.o bin/layers.o bin/softmax.o \
		 bin/linalg.o bin/tensor.o \
		 bin/loss.o \
		 bin/metrics.o \
		 bin/mlp.o \
		 bin/adam.o bin/SGD.o 

default: install

bin/%.o: src/%.cpp $(HEADERS) init
	$(CC) -c $(CFLAGS) -fPIC $< -o $@

libFJML.so: $(CFILES)
	$(CC) $(CFLAGS) -shared $(CFILES) -o libFJML.so

install: libFJML.so
	sudo rm -rf /usr/local/include/FJML
	sudo cp -r include/* /usr/local/include
	sudo cp libFJML.so /usr/local/lib
	sudo ldconfig /usr/local/lib

init:
	mkdir -p bin

docs: src/* include/**/* doxygen.conf
	doxygen doxygen.conf

coverage:
	lcov --directory . --no-external --exclude `pwd`/tests/\* --exclude `pwd`/tests/\*\*/\* --capture --output-file coverage.info
	genhtml coverage.info --output-directory docs/html/coverage

clean:
	rm -rf bin/* bin/**/* libFJML.so
