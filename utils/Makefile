include ../Config

TARGET = ../bin/zfp
INCS = -I../include
LIBS = -L../lib -lzfp $(LDFLAGS) -lm

all: $(TARGET)

$(TARGET): zfp.c ../lib/$(LIBZFP)
	mkdir -p ../bin
	$(CC) $(CFLAGS) $(INCS) zfp.c $(LIBS) -o $(TARGET)

clean:
	rm -f $(TARGET) fields.o
