BINS:= FW

FW: Floyd_Warshallal.cpp
	mpic++ --std=c++11 -o FW Floyd_Warshallal.cpp

case_1: $(BINS)
	mpiexec -n 4 --hostfile hostfile FW case_1.txt

case_2: $(BINS)
	mpiexec -n 4 --hostfile hostfile FW case_2.txt

case_3: $(BINS)
	mpiexec -n 4 --hostfile hostfile FW case_3.txt

clean:
	-rm FW