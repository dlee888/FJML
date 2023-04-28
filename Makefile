CC = g++
CFLAGS = -O3 -std=c++17 -march=native

release: init
release: CFLAGS += -DNDEBUG
release: all libFJML.so
	sudo make install

debug: init
debug: CFLAGS += -coverage -g -fsanitize=undefined
debug: all libFJML.so
	sudo make install

HEADERS = src/FJML/activations/activations.h \
		  src/FJML/data/data.h \
		  src/FJML/layers/layers.h \
		  src/FJML/linalg/linalg.h src/FJML/linalg/tensor.h \
		  src/FJML/loss/loss.h \
		  src/FJML/mlp/mlp.h \
		  src/FJML/optimizers/optimizers.h
CFILES = bin/activations/activations.o \
		 bin/layers/dense.o bin/layers/layers.o bin/layers/softmax.o \
		 bin/loss/loss.o \
		 bin/mlp/mlp.o \
		 bin/optimizers/SGD.o bin/optimizers/adam.o 

bin/%.o: src/FJML/%.cpp $(HEADERS)
	$(CC) -c $(CFLAGS) -fPIC $< -o $@

all: $(CFILES)
	$(CC) -shared $(CFLAGS) $(CFILES) -o libFJML.so

install: libFJML.so
	rm -rf /usr/local/include/FJML
	cp -r src/FJML /usr/include
	cp src/FJML.h /usr/include
	cp libFJML.so /usr/local/lib
	ldconfig /usr/local/lib

init:
	mkdir -p bin/layers bin/activations bin/loss bin/util bin/mlp bin/optimizers

docs: src/FJML/**/* doxygen.conf
	doxygen doxygen.conf

coverage:
	lcov --directory . --capture --output-file coverage.info
	genhtml coverage.info --output-directory docs/html/coverage

clean:
	-rm -rf bin/ libFJML.so
