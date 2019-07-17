
package net.imagej.ops.experiments;

import static net.imglib2.cache.img.DiskCachedCellImgOptions.options;

import java.io.IOException;

import net.imagej.Dataset;
import net.imagej.ImageJ;
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
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

public class Imglib2CacheCudaDeconvolveTest<T extends RealType<T> & NativeType<T>> {

	final static ImageJ ij  = new ImageJ();

	/**
	 * This examples demonstrates calling GPU deconvolution cell by cell on an
	 * image using DiskCachedCellFactory.
	 */
	public static <T extends RealType<T> & NativeType<T>> void main(
		final String[] args) throws IOException
	{

		ij.launch(args);

		// DeconvolutionTestData testData = new Bars("../images/");
		// DeconvolutionTestData testData = new CElegans();
		// DeconvolutionTestData testData = new HalfBead();
		// testData.LoadImages(ij);

		// RandomAccessibleInterval<FloatType> imgF = testData.getImg();
		// RandomAccessibleInterval<FloatType> psfF = testData.getPSF();

		Dataset dataset = (Dataset) ij.io().open(
			"../../images/Slide_17015-02-cropped2.tif");
		Dataset psf = (Dataset) ij.io().open("../../images/psfsmall.tif");
		/*
				ImgPlus<FloatType> imgF = SimplifiedIO.openImage(
					"../../images/Slide_17015-02-cropped2.tif", new FloatType());
				Img<FloatType> psfF = SimplifiedIO.openImage(
					"../../images/psfsmall.tif", new FloatType()).getImg();
		*/
//		RandomAccessibleInterval<FloatType> imgF = SimplifiedIO.convert(img,
//			new FloatType());
//		RandomAccessibleInterval<FloatType> psfF = SimplifiedIO.convert(img,
//			new FloatType());

		RandomAccessibleInterval<FloatType> img =
			(RandomAccessibleInterval<FloatType>) dataset.getImgPlus().getImg();
		RandomAccessibleInterval<FloatType> psf_ =
			(RandomAccessibleInterval<FloatType>) psf.getImgPlus().getImg();

		RandomAccessibleInterval<FloatType> imgF_ = ij.op().convert().float32(Views
			.iterable(img));

		// test with odd size
		RandomAccessibleInterval<FloatType> imgF = Views.interval(imgF_, Views
			.interval(imgF_, Intervals.createMinMax(0, 0, 0, imgF_.dimension(0) - 1,
				imgF_.dimension(1) - 1, imgF_.dimension(2) - 1)));

		RandomAccessibleInterval<FloatType> psfF = ij.op().convert().float32(Views
			.iterable(psf_));

		ImageJFunctions.show(imgF);
		ImageJFunctions.show(psfF);

		final int iterations = 100;
		final int cellBorderXY = (int) psfF.dimension(0);
		final int cellBorderZ = 0;
		final int cellDivisor = 2;

		final int[] cellDimensions = new int[] { (int) Math.ceil((float) imgF
			.dimension(0) / (float) cellDivisor), (int) (float) Math.ceil(imgF
				.dimension(1) / (float) cellDivisor), (int) imgF.dimension(2) };

		// normalize PSF energy to 1
		float sumPSF = ij.op().stats().sum(Views.iterable(psfF)).getRealFloat();
		FloatType val = new FloatType();
		val.set(sumPSF);
		psfF = (Img<FloatType>) ij.op().math().divide(Views.iterable(psfF), val);

		@SuppressWarnings("unchecked")
		final UnaryComputerOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> deconvolver =
			(UnaryComputerOp) Computers.unary(ij.op(),
				UnaryComputerYacuDecuWrapper.class, RandomAccessibleInterval.class,
				imgF, psfF, new long[] { cellBorderXY, cellBorderXY, cellBorderZ },
				iterations);

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
