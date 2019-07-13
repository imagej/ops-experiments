package net.imagej.ops.experiments.deconvolution;

import net.imglib2.Dimensions;

import org.bytedeco.javacpp.FloatPointer;

/**
 * 
 * Interface for a native implementation of Richardson Lucy that uses JavaCPP 
 * 
 * @author bnorthan
 *
 */
public interface NativeRichardsonLucy {
	
	/**
	 * Load the native Library
	 */
	void loadLibrary();
	
	/**
	 * Create the non-circulant normalization factor
	 * 
	 * (see http://bigwww.epfl.ch/deconvolution/challenge/index.html?p=documentation/theory/richardsonlucy)
	 * 
	 * @param paddedDimensions
	 * @param originalDimensions
	 * @param fpPSF
	 * @return
	 */
	FloatPointer createNormal(Dimensions paddedDimensions,
		Dimensions originalDimensions, FloatPointer fpPSF);
	
	/**
	 * call the richardson lucy iterations
	 * 
	 * @param iterations
	 * @param paddedInput
	 * @param fpInput
	 * @param fpPSF
	 * @param fpOutput
	 * @param normalFP
	 */
	void callRichardsonLucy(int iterations, Dimensions paddedInput, FloatPointer fpInput,
		FloatPointer fpPSF, FloatPointer fpOutput, FloatPointer normalFP);

}
