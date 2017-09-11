PATH=$(pwd):$PATH
doxygen
cd latex
make
cd ..
ln -s html/index.html documentation.html
ln -s latex/refman.pdf documentation.pdf
