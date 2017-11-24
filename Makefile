TINY = ~/Documents/learn_machine_learning/tiny-dnn
JSON= ~/Documents/learn_machine_learning/jsoncpp/include
BOOST = ~/Documents/learn_machine_learning/boost_1_65_1
BINS = -L /usr/local/lib
INC=-I $(BOOST) -I $(JSON) -I $(TINY)

default: main

main: jsoncpp.o gym_binding.o
	g++ -Wall -std=c++14 $(INC) $(BINS) -o agent agent.cpp gym_binding.o jsoncpp.o -lcurl

gym_binding.o:
	g++ -Wall -std=c++14 $(INC) -c gym_binding.cpp

jsoncpp.o:
	g++ -Wall -std=c++14 $(INC) -c ./dist/jsoncpp.cpp

clean:
	rm -f agent $$(find . -name '*.o')
