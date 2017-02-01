include ../Config

TARGETS = zfp

all: $(TARGETS)

zfp: zfp.c ../lib/libzfp.a
	$(CC) $(CFLAGS) zfp.c ../lib/libzfp.a -lm -o zfp

clean:
	rm -f $(TARGETS)
