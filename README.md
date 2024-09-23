# Some tests for memory access speed for GTX 1050 mobile GPU with HIP

Build:

``` 
cmake -S . -B build -D GPU_RUNTIME=CUDA
cmake --build build && build/hello_world
```

Then run `build/hello_world` to get some numbers from system time on execution of different matrix multiplication kernels.

(Those kernels might not give the right answer by the way.)
