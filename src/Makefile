include ../Config

TARGET = ../lib/libzfp.a

all: $(TARGET)

$(TARGET): libzfp.o
	rm -f $@
	ar rc $@ $^

libzfp.o: ../Config

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f libzfp.o

cleanall: clean
	rm -f $(TARGET)
