###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda-12.2
#CUDA_ROOT_DIR=/gpfs/loomis/apps/avx/software/CUDAcore/11.3.1

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=/usr/bin/g++
#CC=/gpfs/loomis/apps/avx/software/GCCcore/10.2.0/bin/g++
CC_FLAGS= -O3
CC_LIBS= -lstdc++fs

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS= -O3 -Wno-deprecated-gpu-targets --expt-extended-lambda --expt-relaxed-constexpr -diag-suppress 550 #-g -G
NVCC_LIBS=

LFLAGS= -lm -Wno-deprecated-gpu-targets -fstack-protector

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##

# Target executable name:

#EXE = testDPM
#EXE = testInteraction

# make packings
#EXE = makeDPM
#EXE = compressFIRE
#EXE = compressNVE
#EXE = compressNVT
#EXE = compressRigidFIRE
#EXE = compressRigidNVT

# run dynamics
EXE = runNVE
#EXE = runNVT
#EXE = runActive
#EXE = runActivePlastic
#EXE = runRigidNVE
#EXE = runRigidNVT

# Object files:
OBJS = $(OBJ_DIR)/$(EXE).o $(OBJ_DIR)/DPM2D.o $(OBJ_DIR)/BumpyEllipse.o $(OBJ_DIR)/FIRE.o $(OBJ_DIR)/Simulator.o $(OBJ_DIR)/DPM3D.o

##########################################################

## Compile ##

# Compiler-specific flags:
GENCODE_SM30 = -gencode=arch=compute_30,code=\"sm_30,compute_30\"
GENCODE_SM60 = -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE = $(GENCODE_SM60)

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(NVCC) $(GENCODE) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(CC_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/$(EXE).o : $(EXE).cpp
	$(NVCC) $(GENCODE) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.h
	$(NVCC) $(GENCODE) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cuh
	$(NVCC) $(GENCODE) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)
