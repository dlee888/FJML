CC = g++
CFLAGS = -O3 -std=c++17 -fsanitize=undefined $(FLAGS)

HEADERS = FJML/activations/activations.h \
		  FJML/data/data.h \
		  FJML/layers/layers.h \
		  FJML/linalg/linalg.h FJML/linalg/tensor.h \
		  FJML/loss/loss.h \
		  FJML/mlp/mlp.h \
		  FJML/optimizers/optimizers.h
CFILES = bin/activations/activations.o \
		 bin/layers/dense.o bin/layers/layers.o bin/layers/softmax.o \
		 bin/loss/loss.o \
		 bin/mlp/mlp.o

bin/%.o: FJML/%.cpp $(HEADERS)
	$(CC) -c $(CFLAGS) -fPIC $< -o $@

all: $(CFILES)
	$(CC) -shared $(CFLAGS) $(CFILES) -o libFJML.so

install: libFJML.so
	rm -rf /usr/local/include/FJML
	cp -r FJML /usr/include
	cp FJML.h /usr/include
	cp libFJML.so /usr/local/lib
	ldconfig /usr/local/lib

init:
	mkdir -p bin/layers bin/activations bin/loss bin/util bin/mlp

clean:
	rm bin/*.o bin/**/*.o libFJML.so
