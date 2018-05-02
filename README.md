This is an experimental repository for Ops developers
to add algorithms, play with ideas, etc.

Things that work well here may be migrated to the main [ImageJ
Ops](https://github.com/imagej/imagej-ops) project in the future.

Check out the following submodules and ask questions on [ImageJ Forum](http://forum.imagej.net/)

* ops-experiments-cuda: Example showing how to build and wrap native CUDA code (YacuDecu deconvolution) using maven and javacpp.  (Note: The maven build is currently configured for linux only. Cuda 9.1 must be installed in default location.) 

* ops-experiments-imglib2cache: Example showing how to call GPU deconvolution on individual cells of a cached image.  The example is overkill for the small test image, however this technique would be very useful for partitioning big datasets that can not be processed entirely on a GPU.  

* ops-experiments-tensorflow: Example showing how to create an op based on the tensor flow library. 

* ops-experiments-mkl: Example showing how to build and wrap native code which uses the MKL library. (Note: The maven build is currently configured for linux only. A recent version of MKL must be installed in default location). 
