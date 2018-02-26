# invertible-neural-flows

Open Source (MIT License)

## Scope of this Project

This repository is a private collection of Pytorch modules, which implement functionality around invertible neural networks and normalizing flows. The code has been developed as part of private research efforts by it's author Kai Londenberg ( Kai.Londenberg@googlemail.com )

## Documentation

The documentation is made with sphinx and resides within `docs/`. To build it, make sure you have sphinx and nbsphinx installed and run make from the documentation directory


```
conda install sphinx
conda install sphinx_rtd_theme
conda install docutils
conda install -c conda-forge nbsphinx pandoc
cd docs
make html
```


After that, point your browser to `docs/_build/html/index.html` to get started.

See [NBSphinx](http://nbsphinx.readthedocs.io/en/latest/) for more info about NBSphinx. Also, take a look at the example Notebook in the documentation.

In order to serve the documentation directly using a simple static webserver, also install twisted and service_identity:

```
conda install twisted service_identity
cd docs
make serve
```

By default, it will only serve from http://localhost:10002/, so you need to either configure it (in the docs/Makefile) to use another listening host,
use an SSH tunnel, or access it from the local machine.
