## linux

check to see if the runtime linker will find an opencl runtime

```bash
ldconfig --print-cache | grep opencl 
```

make sure the path to clFFT is set.  For example on linux get [clFFT](https://github.com/arrayfire/clFFT) then build or install and add to the path.

```
export LD_LIBRARY_PATH="/home/bnorthan/array-fire/clFFT-2.12.2-Linux-x64/lib64/:$LD_LIBRARY_PATH"
```
