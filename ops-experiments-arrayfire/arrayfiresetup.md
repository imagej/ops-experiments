## Linux

[Install Arraryfire](http://arrayfire.org/docs/installing.htm)  

note they recommend using ldconfig to setup the run-time link path.  

```bash
echo /opt/arrayfire/lib64 > /etc/ld.so.conf.d/arrayfire.conf
sudo ldconfig
```

This could cause problems with other applications that use MKL (Anaconda Python, Matlab) so alternatively you can add the array fire path to LD_LIBRARY_PATH.

In case you run into issues  linking at run time some research into how the linux dynamic linker searches paths may be useful.  


[Difference between java.libary.path and LD_LIBRARY_PATH](https://stackoverflow.com/questions/27945268/difference-between-using-java-library-path-and-ld-library-path)  
[Order that Linux dynamic linker searches path](https://unix.stackexchange.com/questions/367600/what-is-the-order-that-linuxs-dynamic-linker-searches-paths-in)  

I almost had everything working but ran into errors linking libmkl_avx2 and libmkl_def.  Which is potentially because arraryfire does not distribute libmkl_sequential... or potentially because of a conflict with another installation of MKL. 

See this issue with Julia (similar issues have happened with R)

[ArrayFire from Julia fails with MKL_FATAL_ERROR](https://github.com/JuliaGPU/ArrayFire.jl/issues/215)

If debugging in eclipse

start a terminal and run 
```bash
export LD_PRELOAD=/opt/arrayfire/lib64/libmkl_avx2.so:/opt/arrayfire/lib64/libmkl_def.so:/opt/intel/mkl/lib/intel64/libmkl_sequential.so:/opt/arrayfire/lib64/libmkl_core.so
```

Then start eclipse from that terminal