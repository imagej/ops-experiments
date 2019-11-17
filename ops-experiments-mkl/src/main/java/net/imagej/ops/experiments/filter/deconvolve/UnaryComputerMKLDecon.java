
package net.imagej.ops.experiments.filter.deconvolve;

import net.imagej.ops.OpService;
import net.imagej.ops.Ops;
import net.imagej.ops.experiments.deconvolution.NativeRichardsonLucy;
import net.imagej.ops.experiments.deconvolution.UnaryComputerNativeRichardsonLucy;
import net.imagej.ops.special.computer.AbstractUnaryComputerOp;
import net.imagej.ops.special.computer.Computers;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imglib2.Dimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.outofbounds.OutOfBoundsFactory;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.RealType;

import org.bytedeco.javacpp.FloatPointer;
import org.scijava.Priority;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

/**
 * Non-circulant version of Richardson Lucy
 * http://bigwww.epfl.ch/deconvolution/challenge/index.html?p=documentation/theory/richardsonlucy
 * 
 * @author bnorthan
 */
@Plugin(type = Ops.Deconvolve.RichardsonLucy.class,
	priority = Priority.EXTREMELY_HIGH)
public class UnaryComputerMKLDecon<I extends RealType<I>, O extends RealType<O>, K extends RealType<K>, C extends ComplexType<C>>
	extends
	AbstractUnaryComputerOp<RandomAccessibleInterval<I>, RandomAccessibleInterval<O>>
	implements NativeRichardsonLucy
{

	@Parameter
	OpService ops;

	@Parameter
	UIService ui;

	@Parameter
	LogService log;

	@Parameter
	RandomAccessibleInterval<K> psf;

	@Parameter
	int iterations;

	@Parameter(required = false)
	boolean nonCirculant = true;
	
	@Parameter(required = false)
	long[] extendedSize = null;

	OutOfBoundsFactory<I, RandomAccessibleInterval<I>> obfInput;

	@SuppressWarnings("unchecked")
	@Override
	public void compute(final RandomAccessibleInterval<I> input,
		final RandomAccessibleInterval<O> output)
	{

		@SuppressWarnings("unchecked")
		final UnaryComputerOp<RandomAccessibleInterval<I>, RandomAccessibleInterval<O>> deconvolver =
			(UnaryComputerOp) Computers.unary(ops, UnaryComputerNativeRichardsonLucy.class,
				RandomAccessibleInterval.class, input, psf, iterations, nonCirculant, extendedSize, this);

		deconvolver.compute(input, output);

	}

	@Override
	public void loadLibrary() {
		// load the MKL RL library
		MKLRichardsonLucyWrapper.load();

	}

	@Override
	public FloatPointer createNormal(Dimensions paddedDimensions,
		Dimensions originalDimensions, FloatPointer fpPSF)
	{

		final FloatPointer normalFP;

		// create the normalization factor needed for non-circulant mode
		if (nonCirculant == true) {

			normalFP = NativeDeconvolutionUtility.createNormalizationFactor(ops,
				paddedDimensions, originalDimensions, fpPSF);

		}
		else {
			normalFP = null;
		}

		return normalFP;
	}

	@Override
	public int callRichardsonLucy(int numIterations, Dimensions paddedInput,
		FloatPointer fpInput, FloatPointer fpPSF, FloatPointer fpOutput,
		FloatPointer normalFP)
	{
		// Call the MKL wrapper
		MKLRichardsonLucyWrapper.mklRichardsonLucy3D(numIterations, fpInput, fpPSF,
			fpOutput, (int) paddedInput.dimension(2), (int) paddedInput.dimension(1),
			(int) paddedInput.dimension(0), normalFP);
	
		// error handling not supported yet for MKL version
		return 0;
	}

}
