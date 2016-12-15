MAKEFLAGS += --no-print-directory

all:
	@cd src; $(MAKE)
	@cd utils; $(MAKE)
	@cd examples; $(MAKE)
	@cd tests; $(MAKE)

shared:
	@cd src; $(MAKE) shared

test:
	@cd tests; $(MAKE) test

clean:
	@cd src; $(MAKE) clean
	@cd utils; $(MAKE) clean
	@cd examples; $(MAKE) clean
	@cd tests; $(MAKE) clean
