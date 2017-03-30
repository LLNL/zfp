# see Config file for compile-time settings

MAKEFLAGS += --no-print-directory

all:
	@cd src; $(MAKE) clean static
	@cd utils; $(MAKE) clean all
	@cd tests; $(MAKE) clean all
	@cd examples; $(MAKE) clean all

shared:
	@cd src; $(MAKE) shared

test:
	@cd tests; $(MAKE) test

clean:
	@cd src; $(MAKE) clean
	@cd utils; $(MAKE) clean
	@cd tests; $(MAKE) clean
	@cd examples; $(MAKE) clean
