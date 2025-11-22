.PHONY: all build test debug relwithdebinfo clean format help
all: test build

build:
	mkdir -p build
	echo "Building project in Release mode..."
	cd build && \
	cmake -DCMAKE_BUILD_TYPE=release .. && \
	make

test:
	mkdir -p build
	echo "Building project in Release mode..."
	cd build && \
	cmake -DCMAKE_BUILD_TYPE=release .. && \
	make EngineTests

debug:
	mkdir -p build
	echo "Building project in Debug mode..."
	cd build && \
	cmake -DCMAKE_BUILD_TYPE=debug .. && \
	make

relwithdebinfo:
	mkdir -p build
	echo "Building project in RelWithDebInfo mode..."
	cd build && \
	cmake -DCMAKE_BUILD_TYPE=relwithdebinfo .. && \
	make

clean:
	rm -rf build

format:
	echo "Formatting code..."
	clang-format Core/include/*.h Core/src/*.cpp Engine/include/*.h Engine/src/*.cpp Engine/tests/*.cpp -i --style=file

help:
	@echo "Available targets:"
	@echo "  all: Build and run tests"
	@echo "  build: Build the project in release mode"
	@echo "  test: Build and run tests"
	@echo "  debug: Build the project in debug mode"
	@echo "  relwithdebinfo: Build the project in release mode with debug info"
	@echo "  clean: Clean the build directory"
	@echo "  format: Format the code"
	@echo "  help: Show this help message"