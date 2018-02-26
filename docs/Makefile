# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = invertnn
SOURCEDIR     = .
BUILDDIR      = _build
SERVE_PORT    = 10002
SERVE_HOST    = localhost

# Make sure the output is not cluttered with tensorflow logging.


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

serve:
	@echo "Serving documentation on http://${SERVE_HOST}:${SERVE_PORT}/"
	@echo "Press CTRL-C to stop"
	@python -c 'from twisted.web.server import Site; from twisted.web.static import File; from twisted.internet import reactor; reactor.listenTCP(${SERVE_PORT}, Site(File("./_build/html")), interface="${SERVE_HOST}"); reactor.run()'


.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

