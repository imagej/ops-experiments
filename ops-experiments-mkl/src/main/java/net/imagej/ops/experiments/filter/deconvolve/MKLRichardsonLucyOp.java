
package net.imagej.ops.experiments.filter.deconvolve;

import net.imagej.ops.OpService;
import net.imagej.ops.Ops;
import net.imagej.ops.experiments.filter.AbstractNativeFFTFilterF;
import net.imglib2.Dimensions;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.outofbounds.OutOfBoundsConstantValueFactory;
import net.imglib2.outofbounds.OutOfBoundsMirrorFactory;
import net.imglib2.outofbounds.OutOfBoundsMirrorFactory.Boundary;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Util;

import org.bytedeco.javacpp.FloatPointer;
import org.scijava.Priority;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

/**
 * Implements MKl version of Richardson Lucy deconvolution.
 * 
 * @author bnorthan
 * @param <I>
 * @param <O>
 * @param <K>
 * @param <C>
 */
@Plugin(type = Ops.Deconvolve.RichardsonLucy.class,
	priority = Priority.LOW_PRIORITY)
public class MKLRichardsonLucyOp<I extends RealType<I>, O extends RealType<O> & NativeType<O>, K extends RealType<K>, C extends ComplexType<C> & NativeType<C>>
	extends AbstractNativeFFTFilterF<I, O, K, C>
{

	@Parameter
	UIService ui;

	@Parameter
	OpService ops;

	@Parameter
	int iterations;

	@Parameter(required = false)
	boolean nonCirculant = false;

	@Override
	protected void loadNativeLibraries() {
		MKLRichardsonLucyWrapper.load();
	}

	@Override
	public void initialize() {

		// the out of bounds factory will be different depending on whether we
		// are
		// using circulant or non-circulant
		if (this.getOBFInput() == null) {

			if (!nonCirculant) {
				setOBFInput(new OutOfBoundsMirrorFactory<>(Boundary.SINGLE));
			}
			else if (nonCirculant) {
				setOBFInput(new OutOfBoundsConstantValueFactory<>(Util
					.getTypeFromInterval(in()).createVariable()));
			}
		}

		super.initialize();
	}

	@Override
	protected void runNativeFilter(final Dimensions inputDimensions,
		final Dimensions outputDimensions, final FloatPointer input,
		final FloatPointer kernel, final FloatPointer output)
	{

		final long[] fftSize = new long[] { inputDimensions.dimension(0) / 2 + 1,
			inputDimensions.dimension(1), inputDimensions.dimension(2) };

		final FloatPointer X_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] *
			fftSize[2]));

		final FloatPointer H_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] *
			fftSize[2]));

		final FloatPointer normalFP;

		final int arraySize = (int) (inputDimensions.dimension(0) * inputDimensions
			.dimension(1) * inputDimensions.dimension(2));

		// create the normalization factor needed for non-circulant mode
		if (nonCirculant == true) {

			normalFP = NativeDeconvolutionUtility.createNormalizationFactor(ops,
				inputDimensions, outputDimensions, kernel, X_, H_);

			// ui.show(ConvertersUtility.floatPointerToImg3D(normalFP,
			// inputDimensions));

		}
		else {
			normalFP = null;
		}
		final long startTime = System.currentTimeMillis();

		// Call the MKL wrapper
		MKLRichardsonLucyWrapper.mklRichardsonLucy3D(iterations, input, kernel,
			output, X_, H_, (int) inputDimensions.dimension(2), (int) inputDimensions
				.dimension(1), (int) inputDimensions.dimension(0), normalFP);

		final float[] arrayDecon = new float[arraySize];

		output.get(arrayDecon);
		ArrayImgs.floats(arrayDecon, new long[] { inputDimensions.dimension(0),
			inputDimensions.dimension(1), inputDimensions.dimension(2) });

		// Pointer.free(fpInput);
		// Pointer.free(fpKernel);
		// Pointer.free(fpOutput);

		final long endTime = System.currentTimeMillis();

		log.info("Total execution time (MKL) is: " + (endTime - startTime));
	}

}
