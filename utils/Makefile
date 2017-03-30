include ../Config

TARGET = ../bin/zfp

all: $(TARGET)

$(TARGET): zfp.c ../lib/libzfp.a
	mkdir -p ../bin
	$(CC) $(CFLAGS) zfp.c -L../lib -lzfp -lm -o $(TARGET)

clean:
	rm -f $(TARGET) fields.o
