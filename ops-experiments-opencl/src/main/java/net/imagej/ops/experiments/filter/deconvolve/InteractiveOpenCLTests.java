package net.imagej.ops.experiments.filter.deconvolve;

import com.sun.jna.Pointer;

import java.io.IOException;

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ops.Ops;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.complex.ComplexFloatType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.jocl.NativePointerObject;
import org.jocl.cl_mem;

import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.clearcl.ClearCLContext;
import net.haesleinhuepf.clij.clearcl.ClearCLPeerPointer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;

public class InteractiveOpenCLTests <T extends RealType<T> & NativeType<T>> {

	final static ImageJ ij = new ImageJ();

	public static <T extends RealType<T> & NativeType<T>> void main(
		final String[] args) throws IOException
	{
		ij.launch(args);
		
		System.out.println(System.getProperty("java.library.path"));

		// load the dataset
		Dataset dataset = (Dataset) ij.io().open("../images/bridge.tif");

		Img<FloatType> img = ij.op().convert().float32((Img)dataset.getImgPlus().getImg());
		
		// show the image
		ij.ui().show(img);

		CLIJ clij = CLIJ.getInstance();
		
		ClearCLBuffer gpuInput = clij.push(img);
		
		long[] fftDim=new long[] {(gpuInput.getWidth()/2+1)*2, gpuInput.getHeight()};
		ClearCLBuffer gpuFFT = clij.create(fftDim, NativeTypeEnum.Float);
		
		NativePointerObject inPointer=(NativePointerObject)(gpuInput.getPeerPointer().getPointer());
		NativePointerObject outPointer=(NativePointerObject)(gpuFFT.getPeerPointer().getPointer());
	
		long l_in=hackPointer(inPointer);
		long l_out=hackPointer(outPointer);
		long l_context= hackPointer((NativePointerObject)(clij.getClearCLContext().getPeerPointer().getPointer()));
		long l_queue= hackPointer((NativePointerObject)(clij.getClearCLContext().getDefaultQueue().getPeerPointer().getPointer()));
	
		System.out.println(inPointer);
		System.out.println(inPointer.getClass());
		
		System.out.println(inPointer.toString());
	
		OpenCLWrapper.fft2d_long((long)(gpuInput.getWidth()),gpuInput.getHeight(), l_in, l_out, l_context, l_queue);
		
		clij.show(gpuFFT, "GPU FFT");
		
		RandomAccessibleInterval<FloatType> test = (RandomAccessibleInterval<FloatType>)clij.pullRAI(gpuFFT);

		Img<ComplexFloatType> test2 = stupidCopy(test);
		
	
		ImageJFunctions.show(test2, "FFT OpenCL");
		
		RandomAccessibleInterval<ComplexFloatType> fftOps = ij.op().filter().fft(img);
		ImageJFunctions.show(fftOps, "FFT Ops");

		gpuInput.close();
		gpuFFT.close();

	}
	
	static long hackPointer(NativePointerObject pointer) {
		
		String splitString = pointer.toString().split("\\[")[1];
		//System.out.println("the split "+splitString);
		String hack=splitString.substring(0, splitString.length()-1);
		//System.out.println("the hack "+hack);
		//System.out.println();
		
		return Long.decode(hack);
	}
	
	static Img<ComplexFloatType> stupidCopy(RandomAccessibleInterval<FloatType> in) {
		float[] dumb=new float[(int)(in.dimension(0)*in.dimension(1))];
		int i=0;
		for (FloatType f:Views.iterable(in)) {
			dumb[i++]=f.getRealFloat();
		}
		
		return ArrayImgs.complexFloats(dumb, new long[] {in.dimension(0)/2, in.dimension(1)});
	}
}
