BINS:= FW

FW: Floyd_Warshallal.cpp
	g++ --std=c++11 -o FW -fopenmp Floyd_Warshallal.cpp

case_1: $(BINS)
	./FW ../case_1.txt

case_2: $(BINS)
	./FW ../case_2.txt

case_3: $(BINS)
	./FW ../case_3.txt

case_4: $(BINS)
	./FW ../case_4.txt

case_5: $(BINS)
	./FW ../case_5.txt

case_6: $(BINS)
	./FW ../case_6.txt

case_7: $(BINS)
	./FW ../case_7.txt

clean:
	-rm FW