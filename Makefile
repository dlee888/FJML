.PHONY: all init install docs clean coverage

CC = g++
CFLAGS = -O3 -std=c++17 -march=native

release: CFLAGS += -DNDEBUG
release: all libFJML.so
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
CFILES = bin/activations/activations.o \
		 bin/data/data.o \
		 bin/layers/dense.o bin/layers/layers.o bin/layers/softmax.o \
		 bin/loss/loss.o \
		 bin/mlp/mlp.o \
		 bin/optimizers/SGD.o bin/optimizers/adam.o 

bin/%.o: src/FJML/%.cpp $(HEADERS) init
	$(CC) -c $(CFLAGS) -fPIC $< -o $@

all: $(CFILES)
	$(CC) -shared $(CFLAGS) $(CFILES) -o libFJML.so

install: libFJML.so
	rm -rf /usr/local/include/FJML
	cp -r include/* /usr/local/include
	cp libFJML.so /usr/local/lib
	ldconfig /usr/local/lib

init:
	mkdir -p bin/activations bin/data bin/layers bin/loss bin/mlp bin/optimizers

docs: src/**/* include/**/* doxygen.conf
	doxygen doxygen.conf

coverage:
	lcov --directory . --capture --output-file coverage.info
	genhtml coverage.info --output-directory docs/html/coverage

clean:
	rm -rf bin/* bin/**/* libFJML.so
