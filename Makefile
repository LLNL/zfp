MAKEFLAGS += --no-print-directory

all:
	@cd src; $(MAKE) clean all
	@cd examples; $(MAKE) clean all

clean:
	@cd src; $(MAKE) cleanall
	@cd examples; $(MAKE) clean

test:
	@cd examples; $(MAKE) test
