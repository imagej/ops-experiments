#@ OpService ops
#@ UIService ui
#@ Dataset data
#@ Dataset psf

from net.imglib2.type.numeric.real import FloatType
from net.imglib2.view import Views;
from net.imagej.ops.experiments import ConvertersUtility
from net.imagej.ops.experiments.filter.deconvolve import OpenCLFFTUtility
from net.imglib2.view import Views;
from net.imglib2 import FinalDimensions;

from java.lang import System


# preprocessing image and PSF
psfF=ops.convert().float32(psf);
imgF=ops.convert().float32(data);

extendedPSF = Views.zeroMin(ops.filter().padShiftFFTKernel(psfF, FinalDimensions([imgF.dimension(0), imgF.dimension(1), imgF.dimension(2)])));



# init CLIJ and GPU
from net.haesleinhuepf.clij import CLIJ;
from net.haesleinhuepf.clij2 import CLIJ2;
from net.haesleinhuepf.clijx.plugins import DeconvolveFFT;

# show installed OpenCL devices
print CLIJ.getAvailableDeviceNames();

# initialize a device with a given name
clij2 = CLIJ2.getInstance("RTX");
clij2.clear();

print "Using GPU: " + clij2.getGPUName();

# transfer image to the GPU
gpuImg = clij2.push(imgF);
gpuPSF = clij2.push(extendedPSF);

#debug: show extended psf
#clij2.show(gpuPSF, "ext gpu psf");

# measure start time
start = System.currentTimeMillis();

# create memory for the output image first
gpuEstimate = clij2.create(gpuImg);

# submit deconvolution task
DeconvolveFFT.deconvolveFFT(clij2, gpuImg, gpuPSF, gpuEstimate);

# measure end time
finish = System.currentTimeMillis();

print('CLIJ decon time ', (finish-start));
clij2.show(gpuEstimate, "GPU Decon Result");

# clean up memory
clij2.clear();

