#@ OpService ops
#@ UIService ui
#@ Dataset data
#@ Dataset psf

from net.imglib2.type.numeric.real import FloatType
from net.imglib2.view import Views;
from net.imagej.ops.experiments import ConvertersUtility
from net.imagej.ops.experiments.filter.deconvolve import OpenCLFFTUtility
from net.imglib2 import FinalDimensions;

from java.lang import System

# init CLIJ and GPU
from net.haesleinhuepf.clij import CLIJ;

print CLIJ.getAvailableDeviceNames();

clij = CLIJ.getInstance("RTX");

psfF=ops.convert().float32(psf);
imgF=ops.convert().float32(data);

# transfer image to the GPU
#gpuImg= clij.push(imgF);

# now call the function that pads to a supported size and pushes to the GPU
gpuImg = OpenCLFFTUtility.padInputFFTAndPush(imgF, imgF, ops, clij);

# now call the function that pads to a supported size and pushes to the GPU 
gpuPSF = OpenCLFFTUtility.padKernelFFTAndPush(psfF, FinalDimensions(gpuImg.getDimensions()), ops, clij);

# call decon passing the image as a CLBuffer but the PSF as 
# a java RAI, (there needs to be some preprocessing done on the PSF
# and runDecon does that on the CPU in java)
start = System.currentTimeMillis();
gpuEstimate = OpenCLFFTUtility.runDecon(gpuImg, gpuPSF);
finish = System.currentTimeMillis();

print('CLIJ decon time ', (finish-start));
clij.show(gpuEstimate, "GPU Decon Result");


