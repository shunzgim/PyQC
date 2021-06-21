# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/cmake-3.16.8-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /opt/cmake-3.16.8-Linux-x86_64/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/shunzi/qvm/PyQC1.6

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shunzi/qvm/PyQC1.6/build

# Include any dependencies generated for this target.
include pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/depend.make

# Include the progress variables for this target.
include pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/progress.make

# Include the compile flags for this target's objects.
include pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/flags.make

pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/full_amplitude_sim.cpp.o: pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/flags.make
pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/full_amplitude_sim.cpp.o: ../pyqc/backends/simulator/cpp_kernel/src/full_amplitude_sim.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shunzi/qvm/PyQC1.6/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/full_amplitude_sim.cpp.o"
	cd /home/shunzi/qvm/PyQC1.6/build/pyqc/backends/simulator/cpp_kernel/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fas.dir/full_amplitude_sim.cpp.o -c /home/shunzi/qvm/PyQC1.6/pyqc/backends/simulator/cpp_kernel/src/full_amplitude_sim.cpp

pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/full_amplitude_sim.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fas.dir/full_amplitude_sim.cpp.i"
	cd /home/shunzi/qvm/PyQC1.6/build/pyqc/backends/simulator/cpp_kernel/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shunzi/qvm/PyQC1.6/pyqc/backends/simulator/cpp_kernel/src/full_amplitude_sim.cpp > CMakeFiles/fas.dir/full_amplitude_sim.cpp.i

pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/full_amplitude_sim.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fas.dir/full_amplitude_sim.cpp.s"
	cd /home/shunzi/qvm/PyQC1.6/build/pyqc/backends/simulator/cpp_kernel/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shunzi/qvm/PyQC1.6/pyqc/backends/simulator/cpp_kernel/src/full_amplitude_sim.cpp -o CMakeFiles/fas.dir/full_amplitude_sim.cpp.s

# Object files for target fas
fas_OBJECTS = \
"CMakeFiles/fas.dir/full_amplitude_sim.cpp.o"

# External object files for target fas
fas_EXTERNAL_OBJECTS =

pyqc/backends/simulator/cpp_kernel/lib/libfas.so: pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/full_amplitude_sim.cpp.o
pyqc/backends/simulator/cpp_kernel/lib/libfas.so: pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/build.make
pyqc/backends/simulator/cpp_kernel/lib/libfas.so: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
pyqc/backends/simulator/cpp_kernel/lib/libfas.so: /usr/lib/x86_64-linux-gnu/libpthread.so
pyqc/backends/simulator/cpp_kernel/lib/libfas.so: pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shunzi/qvm/PyQC1.6/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library ../lib/libfas.so"
	cd /home/shunzi/qvm/PyQC1.6/build/pyqc/backends/simulator/cpp_kernel/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fas.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/build: pyqc/backends/simulator/cpp_kernel/lib/libfas.so

.PHONY : pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/build

pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/clean:
	cd /home/shunzi/qvm/PyQC1.6/build/pyqc/backends/simulator/cpp_kernel/src && $(CMAKE_COMMAND) -P CMakeFiles/fas.dir/cmake_clean.cmake
.PHONY : pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/clean

pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/depend:
	cd /home/shunzi/qvm/PyQC1.6/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shunzi/qvm/PyQC1.6 /home/shunzi/qvm/PyQC1.6/pyqc/backends/simulator/cpp_kernel/src /home/shunzi/qvm/PyQC1.6/build /home/shunzi/qvm/PyQC1.6/build/pyqc/backends/simulator/cpp_kernel/src /home/shunzi/qvm/PyQC1.6/build/pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : pyqc/backends/simulator/cpp_kernel/src/CMakeFiles/fas.dir/depend

