# Overview

High Performance Plasticity is a library of optimised crystal plasticity implementations for high-performance hardware. It has been primarily developed as a reference implementation for the GPU-accelerated spectral crystal plasticity approach presented in [An efficient spectral crystal plasticity solver for GPU architectures](https://doi.org/10.1007/s00466-018-1565-x). If you use this implementation in your research, please cite that work.

It also contains an optimised implementation and the test cases for the iterative crystal plasticity approach in Kalidindi SR, Bronkhorst CA, Anand L (1992) Crystallographic texture evolution in bulk deformation processing of FCC metals. *J Mech Phys Solids* 40(3):537–569.

For comparisons, it also contains implementations of the test cases from:
- Mihaila B, Knezevic M, Cardenas A (2014) Three orders of magnitude improved efficiency with high-performance spectral crystal plasticity on GPU platforms. *Int J Numer Meth Eng* 97(11):785–798.
- Savage DJ, Knezevic M (2015) Computer implementations of iterative and non-iterative crystal plasticity solvers on high performance graphics hardware. *Comput Mech* 56(4):677–690.

## Installation
- [Installation instructions](doc/install.md)

## Documentation
- [API Documentation](https://mmalahe.com/hpp/doc)
	- If you have Doxygen, you can also generate this yourself by going into the doc folder and running `doxygen`.
- [Examples](doc/examples.md)

## License
- [License](./LICENSE). This library uses the LGPL version 2.1. that requires that derived works be licensed under the same license, but works that only link to this project (as e.g. a shared library) do not fall under that restriction. Roughly-speaking, if you're using it for research purposes all is peachy, and if you're using it for commercial purposes you need to make sure the code in this library stays separated from your core code. This library also contains individual files acquired from other projects with different licenses displayed prominently at the top of those files.

## Continuous integration status
[![Build Status](https://travis-ci.org/mmalahe/hpp.png)](https://travis-ci.org/mmalahe/hpp)
