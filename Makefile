CC = nvcc
FLAGS = -O3
LIBS = -lcuda -lcudart -lGL -lglut -lGLU -lcutil_x86_64 -lAntTweakBar -lGLEW
INCLUDES = -I/opt/cuda/sdk/C/common/inc/ -L/opt/cuda/sdk/C/lib/ -L./ -I./
PROJECT = raytracer

all: $(PROJECT)

$(PROJECT): raytracer.cu main.cpp
	$(CC) -o $(PROJECT) raytracer.cu main.cpp $(INCLUDES) $(LIBS) $(FLAGS)

clean:
	rm -f *.o $(PROJECT)
