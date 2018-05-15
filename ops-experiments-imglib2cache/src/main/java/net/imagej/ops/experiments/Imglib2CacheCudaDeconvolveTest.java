
package net.imagej.ops.experiments;

import static net.imglib2.cache.img.DiskCachedCellImgOptions.options;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imagej.ops.experiments.testImages.Bars;
import net.imagej.ops.experiments.testImages.DeconvolutionTestData;
import net.imagej.ops.special.computer.Computers;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.cache.img.DiskCachedCellImgFactory;
import net.imglib2.cache.img.DiskCachedCellImgOptions;
import net.imglib2.cache.img.DiskCachedCellImgOptions.CacheType;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class Imglib2CacheCudaDeconvolveTest<T extends RealType<T> & NativeType<T>> {

	final static ImageJ ij = new ImageJ();

	/**
	 * This examples demonstrates calling GPU deconvolution cell by cell on an
	 * image using DiskCachedCellFactory.
	 */
	public static <T extends RealType<T> & NativeType<T>> void main(
		final String[] args) throws IOException
	{

		ij.launch(args);

		DeconvolutionTestData testData = new Bars();
		// DeconvolutionTestData testData = new CElegans();
		// DeconvolutionTestData testData = new HalfBead();

		testData.LoadImages(ij);
		RandomAccessibleInterval<FloatType> imgF = testData.getImg();
		RandomAccessibleInterval<FloatType> psfF = testData.getPSF();


		ImageJFunctions.show(imgF);
		ImageJFunctions.show(psfF);

		final int iterations = 100;
		final int cellBorderXY = (int) psfF.dimension(0);
		final int cellBorderZ = 0;

		final int[] cellDimensions = new int[] { (int) Math.ceil(imgF.dimension(0) /
			2), (int) Math.ceil(imgF.dimension(1) / 2), (int) imgF.dimension(2) };


		@SuppressWarnings("unchecked")
		final UnaryComputerOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> deconvolver =
			(UnaryComputerOp) Computers.unary(ij.op(), UnaryComputerYacuDecu.class,
				RandomAccessibleInterval.class, imgF, psfF, new long[] { cellBorderXY,
					cellBorderXY, cellBorderZ }, iterations);

		final FloatType type = new FloatType();

		final DiskCachedCellImgOptions writeOnlyDirtyOptions = options()
			.cellDimensions(cellDimensions).cacheType(CacheType.BOUNDED).maxCacheSize(
				100);
		final DiskCachedCellImgFactory<FloatType> factory =
			new DiskCachedCellImgFactory<>(writeOnlyDirtyOptions);

		// create a new image using the disk cache factory. As the loader we pass it
		// a lambda that calls the deconvolution op.
		final Img<FloatType> out = factory.create(new long[] { imgF.dimension(0),
			imgF.dimension(1), imgF.dimension(2) }, type, cell -> deconvolver.compute(
				imgF, cell), options().initializeCellsAsDirty(true));

		// show the output (this will invoke deconvolution on each cell lazily).
		ImageJFunctions.show(out, "Output");

	}

}
