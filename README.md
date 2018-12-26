## Overview

This is an experimental repository for Ops developers
to add algorithms, play with ideas, etc.

Things that work well here may be migrated to the main [ImageJ
Ops](https://github.com/imagej/imagej-ops) project in the future.

Check out the following submodules and ask questions on [ImageJ Forum](http://forum.imagej.net/)

* ops-experiments-cuda: Example showing how to build and wrap native CUDA code (YacuDecu deconvolution) using maven and javacpp.  

* ops-experiments-imglib2cache: Example showing how to call GPU deconvolution on individual cells of a cached image.  The example is overkill for the small test image, however this technique would be very useful for partitioning big datasets that can not be processed entirely on a GPU.  

* ops-experiments-tensorflow: Example showing how to create an op based on the tensor flow library. 

* ops-experiments-mkl: Example showing how to build and wrap native code which uses the MKL library. (Note: The maven build is currently configured for linux only. A recent version of MKL must be installed in default location). 

## How to build ops-experiments-cuda and the YacuDecu+ GPU deconvolution code

Note:  These instructions assume the developer is not building other components of Ops experiments which require installing more 3rd party libraries like Tensorflow and MKL.  

### Prequisites
1.  Maven and Java. 

2. Install [Cuda](https://developer.nvidia.com/cuda-downloads).  The version of Cuda being used is the most recent version that has been tested with a given operating system.  This means the version can be different for different operating systems.  Currently the Linux build is using version 9.0, and the Windows build is using 10.0 

Take a look at the [YacuDecuRichardsonLucyWrapper Properties Annotation](https://github.com/imagej/ops-experiments/blob/master/ops-experiments-cuda/src/main/java/net/imagej/ops/experiments/filter/deconvolve/YacuDecuRichardsonLucyWrapper.java#L9) to check if any changes have been made to the Cuda version.  Note the annotation defines platform specific settings including the location of the NVIDIA GPU COmputing toolkit. 

3.  If using Windows install Visual Studio and Git for Windows 

4.  Build [ops-experiments-common](https://github.com/imagej/ops-experiments/tree/master/ops-experiments-common) using maven. 

### Linux

1.  Build [ops-experiments-cuda](https://github.com/imagej/ops-experiments/tree/master/ops-experiments-cuda) using maven. 

Note:  If the build cannot find the Cuda Toolkit, check to make sure the toolkit was installed in the default location specified in the [YacuDecuRichardsonLucyWrapper Properties Annotation](https://github.com/imagej/ops-experiments/blob/master/ops-experiments-cuda/src/main/java/net/imagej/ops/experiments/filter/deconvolve/YacuDecuRichardsonLucyWrapper.java#L9).  If not you may have to change the location specified in the annotation. 

### Windows (Using Visual Studio)

1.  Start a “x64 native tools VS” terminal, then within the "x64 native tools" VS terminal invoke a mingw64 terminal using “C:\Program Files\Git\bin\sh.exe”.  (This essentially gives you a bash terminal that has a Microsoft C++ build environment set up within it.  Alternatively you could write a bash script that sets up the required paths and system variables. 

2.  Build [ops-experiments-cuda](https://github.com/imagej/ops-experiments/tree/master/ops-experiments-cuda) using maven. 

### Mac Build 

A Mac build would require a bit of hacking.

1.  Write a MacOsx specific Makefile to build YacuDecu (deconv.cu) and place it [here](https://github.com/imagej/ops-experiments/tree/master/ops-experiments-cuda/native/YacuDecu).  Note ops experiments uses a customized version of YacuDecu with a few enhancements and bug fixes, so make sure you build the version of YacuDecu (deconv.cu) that is in the ops-experiments repo [here](https://github.com/imagej/ops-experiments/tree/master/ops-experiments-cuda/native/YacuDecu/src). 

2.  Modify the following lines of [cppbuild.sh](https://github.com/imagej/ops-experiments/blob/master/ops-experiments-cuda/native/YacuDecu/cppbuild.sh#L26) to invoke the MacOxs Makefile (the logic should be very similar to the Linux and Windows sections).  

3.  In [YacuDecuRichardsonLucyWrapper](https://github.com/imagej/ops-experiments/blob/master/ops-experiments-cuda/src/main/java/net/imagej/ops/experiments/filter/deconvolve/YacuDecuRichardsonLucyWrapper.java) add a MacOsx Platform section to the properties annotaiton.  This is where the location of the Cuda Toolkit is specified.  



### Linux
2.  Simply typing 
