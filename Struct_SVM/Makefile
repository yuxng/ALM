################################################################################
# MPI
################################################################################
MPI_INCLUDES := /usr/local/include
MPI_LIBS := /usr/local/lib

################################################################################
# CUDA
################################################################################

include ./findcudalib.mk

# Location of the CUDA Toolkit
CUDA_PATH ?= "/usr/local/cuda-5.5"

# internal flags
NVCCFLAGS   := -m${OS_SIZE}
CCFLAGS     :=
NVCCLDFLAGS :=
LDFLAGS     :=

# Extra user flags
EXTRA_NVCCFLAGS   ?=
EXTRA_NVCCLDFLAGS ?=
EXTRA_LDFLAGS     ?=
EXTRA_CCFLAGS     ?=

# OS-specific build flags
ifneq ($(DARWIN),) 
  LDFLAGS += -rpath $(CUDA_PATH)/lib
  CCFLAGS += -arch $(OS_ARCH) $(STDLIB)
else
  ifeq ($(OS_ARCH),armv7l)
    ifeq ($(abi),gnueabi)
      CCFLAGS += -mfloat-abi=softfp
    else
      # default to gnueabihf
      override abi := gnueabihf
      LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
      CCFLAGS += -mfloat-abi=hard
    endif
  endif
endif

ifeq ($(ARMv7),1)
NVCCFLAGS += -target-cpu-arch ARM
ifneq ($(TARGET_FS),) 
CCFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += -rpath-link=$(TARGET_FS)/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-$(abi)
endif
endif

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      TARGET := debug
else
      TARGET := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(NVCCLDFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(EXTRA_NVCCLDFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  :=
LIBRARIES :=

################################################################################

# CUDA code generation flags
ifneq ($(OS_ARCH),armv7l)
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
endif
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30)

################################################################################

CC = gcc
LD = gcc

CFLAGS = -O3 -Wall
LDCFLAGS = -O3 -lm -Wall

INCLUDES += -I$(MPI_INCLUDES)
LIBRARIES += -lmpi

# Target rules
SRC_CUDA = convolve.cu matrix.cu hog.cu rectify.cu select_gpu.cu 
SRC_C = cad.c tree.c distance_transform.c svm_struct_learn.c svm_struct_api.c svm_common.c svm_learn.c svm_hideo.c svm_struct_common.c

OBJ_CUDA = $(addsuffix .o, $(basename $(SRC_CUDA)))
OBJ_C = $(addsuffix .o, $(basename $(SRC_C)))

all: svm_struct_learn svm_struct_classify

svm_struct_learn: $(OBJ_CUDA) $(OBJ_C) svm_struct_main.o
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARIES) $(LDFLAGS) -L$(MPI_LIBS)

svm_struct_classify: $(OBJ_CUDA) $(OBJ_C) svm_struct_classify.o
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARIES) $(LDFLAGS) -L$(MPI_LIBS)

%.o: %.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

%.o: %.c
	$(CC) $(INCLUDES) $(CFLAGS) -o $@ -c $<

clean: $(OBJ)
	rm -f $(OBJ_CUDA)
	rm -f $(OBJ_C)
	rm -f svm_struct_classify svm_struct_learn

