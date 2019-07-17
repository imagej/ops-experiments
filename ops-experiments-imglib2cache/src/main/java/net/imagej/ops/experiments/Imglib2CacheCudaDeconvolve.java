
package net.imagej.ops.experiments;

import static net.imglib2.cache.img.DiskCachedCellImgOptions.options;

import java.io.IOException;

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imagej.ops.special.computer.Computers;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.cache.img.DiskCachedCellImg;
import net.imglib2.cache.img.DiskCachedCellImgFactory;
import net.imglib2.cache.img.DiskCachedCellImgOptions;
import net.imglib2.cache.img.DiskCachedCellImgOptions.CacheType;
import net.imglib2.img.Img;
import net.imglib2.img.cell.Cell;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

@Plugin(type = Command.class, headless = true,
	menuPath = "Plugins>OpsExperiments>Imglib2-Cache YacuDecu")
public class Imglib2CacheCudaDeconvolve<T extends RealType<T> & NativeType<T>>
	implements Command
{

	@Parameter
	LogService log;

	@Parameter
	OpService ops;

	@Parameter
	Dataset img;

	@Parameter
	Dataset psf;

	@Parameter
	Integer iterations = 100;

	@Parameter
	Integer numCells = 2;

	@Parameter(type = ItemIO.OUTPUT)
	Img<FloatType> deconvolved;

	final static ImageJ ij = new ImageJ();

	/**
	 * This examples demonstrates calling GPU deconvolution cell by cell on an
	 * image using DiskCachedCellFactory.
	 */
	public static <T extends RealType<T> & NativeType<T>> void main(
		final String[] args) throws IOException
	{

		ij.launch(args);
	}

	@Override
	public void run() {

		log.error("starting log");

		Img<FloatType> imgF = ops.convert().float32((Img<T>) img.getImgPlus()
			.getImg());

		@SuppressWarnings("unchecked")
		Img<FloatType> psfF = ops.convert().float32((Img<T>) psf.getImgPlus()
			.getImg());

		final int cellBorderXY = (int) psfF.dimension(0);
		final int cellBorderZ = 0;

		final int[] cellDimensions = new int[] { (int) Math.ceil((float) imgF
			.dimension(0) / (float) numCells), (int) (float) Math.ceil(imgF
				.dimension(1) / (float) numCells), (int) imgF.dimension(2) };
		
		// normalize PSF energy to 1
		float sumPSF = ij.op().stats().sum(Views.iterable(psfF)).getRealFloat();
		FloatType val = new FloatType();
		val.set(sumPSF);
		psfF = (Img<FloatType>) ij.op().math().divide(Views.iterable(psfF), val);

		@SuppressWarnings("unchecked")
		final UnaryComputerOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> deconvolver =
			(UnaryComputerOp) Computers.unary(ops, UnaryComputerYacuDecuWrapper.class,
				RandomAccessibleInterval.class, imgF, psfF, new long[] { cellBorderXY,
					cellBorderXY, cellBorderZ }, iterations);

		final FloatType type = new FloatType();

		final DiskCachedCellImgOptions writeOnlyDirtyOptions = options()
			.cellDimensions(cellDimensions).cacheType(CacheType.BOUNDED).maxCacheSize(
				100).numIoThreads(1);
		// options().n
		// writeOnlyDirtyOptions.
		// options().
		final DiskCachedCellImgFactory<FloatType> factory =
			new DiskCachedCellImgFactory<>(new FloatType(), writeOnlyDirtyOptions);

		// factory.create
		// create a new image using the disk cache factory. As the loader we pass it
		// a lambda that calls the deconvolution op.
		DiskCachedCellImg<FloatType, RandomAccessibleInterval<FloatType>> test =
			(DiskCachedCellImg) factory.create(new long[] { imgF.dimension(0), imgF
				.dimension(1), imgF.dimension(2) }, type, cell -> deconvolver.compute(
					imgF, cell), options().initializeCellsAsDirty(true));

		test.getCells().forEach(Cell::getData);

		deconvolved = test;
		/*
		RandomAccess<FloatType> ra = deconvolved.randomAccess();
		
		// trigger
		for (int x=0;x<imgF.dimension(0);x+=cellDimensions[0]) {
			for (int y=0;y<imgF.dimension(1);y+=cellDimensions[1]) {
				ra.setPosition(new int[] {x,y,0});
				ra.get().getRealDouble();
			}
		}*/

	}

}
