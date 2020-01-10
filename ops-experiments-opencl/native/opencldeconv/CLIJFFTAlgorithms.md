# CLIJ FFT Based Algorithms (Deconvolution etc.)

## Complex Kernels

Need to write functions for complex math.  In this case I used interleaved float arrays because that is the format returned by clFFT.  [Complex Kernels](https://github.com/imagej/ops-experiments/blob/master/ops-experiments-opencl/native/opencldeconv/opencldeconv.cpp#L31)

## Get and install clFFT  
 [clFFT binaries](https://github.com/clMathLibraries/clFFT/releases) 

 Unpack to /opt/clFFT/

 TODO: Windows instructions.

## clFFT C Wrapper

It may be feasible to wrap all of clFFT via java, however in many cases algorithm developers will want to work on native c code anyway (to make it easier to also wrap in Python).  As well clFFT may be more obtuse for routine use. 

[an example FFT function which works on long pointers to existing GPU memory, context and queue ](https://github.com/imagej/ops-experiments/blob/master/ops-experiments-opencl/native/opencldeconv/opencldeconv.cpp#L127)

[an example FFT function which works on CPU memory (transfers to GPU then calls FFT)](https://github.com/imagej/ops-experiments/blob/master/ops-experiments-opencl/native/opencldeconv/opencldeconv.cpp#L209)

## Deconvolution (Richardson Lucy) C Wrappers

[clFFT based Richardson Lucy implementations that works on long pointers to existing GPU Memory, context and queue](https://github.com/imagej/ops-experiments/blob/master/ops-experiments-opencl/native/opencldeconv/opencldeconv.cpp#L209)

## JavaCPP Wrappers
Native Builder [here](https://github.com/imagej/ops-experiments/blob/master/ops-experiments-opencl/native/cppbuild.sh) and [here](https://github.com/imagej/ops-experiments/blob/master/ops-experiments-opencl/native/opencldeconv/cppbuild.sh) .
  
[JavaCPP Wrapper](https://github.com/imagej/ops-experiments/blob/master/ops-experiments-opencl/src/main/java/net/imagej/ops/experiments/filter/deconvolve/OpenCLWrapper.java)  

[Native Build](https://github.com/imagej/ops-experiments/blob/master/ops-experiments-opencl/pom.xml#L99) and [Wrapper Build](https://github.com/imagej/ops-experiments/blob/master/ops-experiments-opencl/pom.xml#L153) invoked from the POM. 

## How to call from Java  
[Call FFT](https://github.com/imagej/ops-experiments/blob/master/ops-experiments-opencl/src/main/java/net/imagej/ops/experiments/filter/deconvolve/OpenCLFFTUtility.java#L25)  
Note clFFT does not support all sizes, so need to extend to a smooth number first  
```java
img = (RandomAccessibleInterval<FloatType>) ops.run(
			DefaultPadInputFFT.class, img, img, false);
```  
[call deconvolution passing GPU memory (CLBuffer)](https://github.com/imagej/ops-experiments/blob/master/ops-experiments-opencl/src/main/java/net/imagej/ops/experiments/filter/deconvolve/OpenCLFFTUtility.java#L92)   

Note: the PSF is passed in as a java RAI, because there is additional conditioning that needs to be done.  PSF needs to be extended to the image size and shifted so the center is at 0,0.  

```java
// extend and shift the PSF
RandomAccessibleInterval<FloatType> extendedPSF = Views.zeroMin(ops.filter().padShiftFFTKernel(psf, new FinalDimensions(gpuImg.getDimensions())));

// transfer PSF to the GPU
ClearCLBuffer gpuPSF = clij.push(extendedPSF);
``` 