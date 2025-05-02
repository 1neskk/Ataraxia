.PHONY: all
all: test build

.PHONY: build
build:
	mkdir -p build
	echo "Building project in Release mode..."
	cd build && \
	cmake -DCMAKE_BUILD_TYPE=release .. && \
	make

.PHONY: debug
debug:
	mkdir -p build
	echo "Building project in Debug mode..."
	cd build && \
	cmake -DCMAKE_BUILD_TYPE=debug .. && \
	make

.PHONY: relwithdebinfo
relwithdebinfo:
	mkdir -p build
	echo "Building project in RelWithDebInfo mode..."
	cd build && \
	cmake -DCMAKE_BUILD_TYPE=relwithdebinfo .. && \
	make

.PHONY: clean
clean:
	rm -rf build
