#@ OpService ops
#@ UIService ui
#@ Dataset data
#@ Dataset psf

from net.imglib2.type.numeric.real import FloatType
from net.imglib2.view import Views;
from net.imagej.ops.experiments import ConvertersUtility
from net.imagej.ops.experiments.filter.deconvolve import OpenCLFFTUtility

from java.lang import System

# init CLIJ and GPU
from net.haesleinhuepf.clij import CLIJ;

print CLIJ.getAvailableDeviceNames();

clij = CLIJ.getInstance("RTX");

psfF=ops.convert().float32(psf);
imgF=ops.convert().float32(data);

# transfer image to the GPU
gpuImg= clij.push(imgF);

# call decon passing the image as a CLBuffer but the PSF as 
# a java RAI, (there needs to be some preprocessing done on the PSF
# and runDecon does that on the CPU in java)
start = System.currentTimeMillis();
gpuEstimate = OpenCLFFTUtility.runDecon(gpuImg, psfF, ops);
finish = System.currentTimeMillis();

print('CLIJ decon time ', (finish-start));
clij.show(gpuEstimate, "GPU Decon Result");


