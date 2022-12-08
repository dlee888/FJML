CC = g++
CFLAGS = -O3 -std=c++17 -fsanitize=undefined $(FLAGS)

HEADERS = FJML/activations/activations.h \
		  FJML/data/data.h \
		  FJML/layers/layers.h \
		  FJML/loss/loss.h \
		  FJML/mlp/mlp.h \
		  FJML/optimizers/optimizers.h \
		  FJML/util/linalg.h FJML/util/types.h FJML/util/util.h
CFILES = bin/activations/activations.o \
		 bin/layers/dense.o bin/layers/layers.o bin/layers/softmax.o \
		 bin/loss/loss.o \
		 bin/mlp/mlp.o \
		 bin/optimizers/sgd.o bin/optimizers/adam.o


ifeq ($(TARGET),LOCAL)
	CFLAGS += -D LOCAL -I ~/templates/templates
endif

bin/%.o : FJML/%.cpp $(HEADERS)
	$(CC) -c $(CFLAGS) $< -o $@

all: $(CFILES)
	echo "Done"

init:
	mkdir -p bin/layers bin/activations bin/loss bin/util bin/mlp bin/optimizers

clear:
	rm bin/*.o bin/**/*.o
