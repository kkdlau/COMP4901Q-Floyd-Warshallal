BINS:= FW

FW: Floyd_Warshallal.cpp
	g++ --std=c++11 -o FW -fopenmp Floyd_Warshallal.cpp

case_1: $(BINS)
	./FW case_1.txt

case_2: $(BINS)
	./FW case_2.txt

case_3: $(BINS)
	./FW case_3.txt

clean:
	-rm FW