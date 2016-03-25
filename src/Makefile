CXX = g++
CXXFLAGS = -g -O3 -ansi -Wall -I../inc $(DEFS)

TARGET = ../lib/libzfp.a

$(TARGET): libzfp.o
	rm -f $@
	ar rc $@ $^

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<

clean:
	@rm -f libzfp.o

cleanall: clean
	@rm -f $(TARGET)
