.PHONY: all build test clean

all: 
	CC=g++ LDSHARED='$(shell python3 scripts/configure.py)' python3 setup.py build
	python3 setup.py install
	# python3 tests/test_bcc_classifier.py
	# python3 tests/test_vehicle_make.py
	python3 tests/test.py
	# python3 tests/test_vehicle_reidv10.py

build:
	# ar cru preprocess_test/build/libpreprocess.a
	# ranlib preprocess_test/build/libpreprocess.a
	# ranlib preprocess_test/build/libpreprocess.a

test: build
	g++ tests/test.c -L. -lvectoradd -o main -L${CUDA_HOME}/lib64 -lcudart

clean:
	rm -f libvectoradd.a *.o main temp.py
	rm -rf build
