### Run benchmark_kernel
This is for running cpp kernel under different configs. 

First please do the codegen in `include`. 
```bash
cd include
python codegen.py
```

Then build with cmake.
```bash
cd benchmark_cpp
mkdir build
cd build
cmake ..
make
```

Run the reuslt
```bash
cd build
./benchmark ${.npy file} ${N}
```