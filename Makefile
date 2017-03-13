#MAKEFLAGS += --no-print-directory

all: static shared utils examples tests

static:
	@cd src; $(MAKE) static

shared:
	@cd src; $(MAKE) shared

utils: static
	@cd utils; $(MAKE)

examples: static
	@cd examples; $(MAKE)

tests: static
	@cd tests; $(MAKE)

test: tests static
	@cd tests; $(MAKE) test

clean:
	@cd src; $(MAKE) clean
	@cd utils; $(MAKE) clean
	@cd examples; $(MAKE) clean
	@cd tests; $(MAKE) clean
	rm -rf lib
