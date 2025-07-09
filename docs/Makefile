# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = /usr/bin/python3 -m sphinx
SPHINXPROJ    = zfp
SOURCEDIR     = source
BUILDDIR      = build

# Needed for spell checking on macOS
PYENCHANT_LIBRARY_PATH = /opt/homebrew/lib/libenchant-2.dylib

# Build HTML by default
all:
	@$(MAKE) html

# List targets
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Run spell checker
spell:
	@PYENCHANT_LIBRARY_PATH=$(PYENCHANT_LIBRARY_PATH) $(SPHINXBUILD) -b spelling -d "$(BUILDDIR)/doctrees" "$(SOURCEDIR)" "$(BUILDDIR)/spelling"

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
