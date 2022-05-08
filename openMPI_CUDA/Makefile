CC = /usr/bin/gcc
NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -Xcompiler -g -w -I..  
MPI = -I /usr/local/software/openmpi/include/ 
LIB = -L /usr/local/software/openmpi/lib -lmpi  

BINS:= FW

FW: Floyd_Warshallal.cu
	${NVCC} ${NVCCFLAGS} ${MPI} ${LIB} --std=c++11 Floyd_Warshallal.cu -o FW

case_1: $(BINS)
	@mpiexec -n 9 --hostfile hostfile -x LD_LIBRARY_PATH=/usr/local/software/openmpi/lib:$LD_LIBRARY_PATH ./FW case_1.txt
case_2: $(BINS)
	@mpiexec -n 9 --hostfile hostfile -x LD_LIBRARY_PATH=/usr/local/software/openmpi/lib:$LD_LIBRARY_PATH ./FW case_2.txt
case_3: $(BINS)
	@mpiexec -n 9 --hostfile hostfile -x LD_LIBRARY_PATH=/usr/local/software/openmpi/lib:$LD_LIBRARY_PATH ./FW case_3.txt
case_4: $(BINS)
	@mpiexec -n 9 --hostfile hostfile -x LD_LIBRARY_PATH=/usr/local/software/openmpi/lib:$LD_LIBRARY_PATH ./FW case_4.txt
case_4: $(BINS)
	@mpiexec -n 9 --hostfile hostfile -x LD_LIBRARY_PATH=/usr/local/software/openmpi/lib:$LD_LIBRARY_PATH ./FW case_5.txt

clean:
	-rm FW