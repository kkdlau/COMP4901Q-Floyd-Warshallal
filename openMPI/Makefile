BINS:= FW

FW: Floyd_Warshallal.cpp
	mpic++ --std=c++11 -o FW Floyd_Warshallal.cpp

case_1: $(BINS)
	mpiexec -n 9 --hostfile hostfile FW ../case_1.txt

case_2: $(BINS)
	mpiexec -n 9 --hostfile hostfile FW ../case_2.txt

case_3: $(BINS)
	mpiexec -n 9 --hostfile hostfile FW ../case_3.txt

case_4: $(BINS)
	mpiexec -n 9 --hostfile hostfile FW ../case_4.txt

case_5: $(BINS)
	mpiexec -n 9 --hostfile hostfile FW ../case_5.txt

case_6: $(BINS)
	mpiexec -n 9 --hostfile hostfile FW ../case_6.txt

case_7: $(BINS)
	mpiexec -n 9 --hostfile hostfile FW ../case_7.txt

clean:
	-rm FW