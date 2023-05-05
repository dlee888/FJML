.PHONY: all init install docs clean coverage

CC = g++
CFLAGS = -O3 -std=c++17 -march=native

release: CFLAGS += -DNDEBUG
release: all libFJML.so
	sudo make install

cuda: CFLAGS = -ccbin g++ -O3 --std=c++17 -DNDEBUG -DCUDA -lcublas --compiler-options
cuda: CC = nvcc
cuda: all libFJML.so
	sudo make install

debug: CFLAGS += -coverage -g -fsanitize=undefined
debug: all libFJML.so
	sudo make install

HEADERS = include/FJML/activations.h \
		  include/FJML/data.h \
		  include/FJML/layers.h \
		  include/FJML/tensor.h \
		  include/FJML/loss.h \
		  include/FJML/mlp.h \
		  include/FJML/optimizers.h
CFILES = bin/activations.o \
		 bin/data.o \
		 bin/dense.o bin/layers.o bin/softmax.o \
		 bin/linalg.o bin/tensor.o \
		 bin/loss.o \
		 bin/mlp.o \
		 bin/adam.o bin/SGD.o 

bin/%.o: src/%.cpp $(HEADERS) init
	$(CC) -c $(CFLAGS) -fPIC $< -o $@

all: $(CFILES)
	$(CC) $(CFLAGS) -shared $(CFILES) -o libFJML.so

install: libFJML.so
	rm -rf /usr/local/include/FJML
	cp -r include/* /usr/local/include
	cp libFJML.so /usr/local/lib
	ldconfig /usr/local/lib

init:
	mkdir -p bin

docs: src/* include/**/* doxygen.conf
	doxygen doxygen.conf

coverage:
	lcov --directory . --no-external --exclude `pwd`/tests/\* --exclude `pwd`/tests/\*\*/\* --capture --output-file coverage.info
	genhtml coverage.info --output-directory docs/html/coverage

clean:
	rm -rf bin/* bin/**/* libFJML.so
