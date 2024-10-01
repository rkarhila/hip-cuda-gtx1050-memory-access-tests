# Some tests for memory access speed for GTX 1050 mobile GPU with HIP

This work is intended to run in a docker container built from the Dockerfile. Vscode configuration for running in the container included. Use `--build-arg USERNAME=... --build-arg PASSWORD=...` flags to set up a user. The ssh server started by the entry `CMD` forces the ssh server to use less secure algorithms to allow old Nvidia debugging tools to access the container via ssh.

Build:

``` 
cmake -S . -B build -D GPU_RUNTIME=CUDA
cmake --build build
```

Then run `build/hello_world` to get some numbers from system time on execution of different matrix multiplication kernels.

(Those kernels might not give the right answer by the way.)
