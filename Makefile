# see Config file for compile-time settings
include Config

MAKEFLAGS += --no-print-directory


# default: build all targets enabled in Config
all:
	@echo $(LIBRARY)
	@cd src; $(MAKE) clean $(LIBRARY)
ifneq ($(BUILD_CFP),0)
	@cd cfp/src; $(MAKE) clean $(LIBRARY)
endif
ifneq ($(BUILD_ZFORP),0)
	@cd fortran; $(MAKE) clean $(LIBRARY)
endif
ifneq ($(BUILD_UTILITIES),0)
	@cd utils; $(MAKE) clean all
endif
ifneq ($(BUILD_TESTING),0)
	@cd tests; $(MAKE) clean all
endif
ifneq ($(BUILD_EXAMPLES),0)
	@cd examples; $(MAKE) clean all
endif


# run basic regression tests
test:
	@cd tests; $(MAKE) test


# clean all
clean:
	@cd src; $(MAKE) clean
	@cd cfp/src; $(MAKE) clean
	@cd fortran; $(MAKE) clean
	@cd utils; $(MAKE) clean
	@cd tests; $(MAKE) clean
	@cd examples; $(MAKE) clean
