
package net.imagej.ops.experiments.filter.deconvolve;

import net.imagej.ImageJ;
import net.imagej.ImgPlus;
import net.imagej.ops.OpService;
import net.imagej.ops.experiments.VisualizationUtility;
import net.imagej.ops.special.computer.Computers;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imglib2.FinalDimensions;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.logic.BitType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

import org.scijava.Context;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.table.DefaultGenericTable;
import org.scijava.table.DoubleColumn;
import org.scijava.table.GenericColumn;
import org.scijava.table.Table;
import org.scijava.ui.UIService;

import ij.IJ;

@Plugin(type = Command.class, headless = true,
	menuPath = "Plugins>OpsExperiments>YacuDecu Refractive Index Tester")
public class YacuDecuSpecimenRIOtsuTester<T extends RealType<T> & NativeType<T>>
	implements Command
{

	@Parameter
	ImageJ ij;

	@Parameter
	ImgPlus<T> img;

	@Parameter(type = ItemIO.INPUT)
	Integer iterations = 100;

	@Parameter(type = ItemIO.INPUT)
	Float numericalAperture = 1.4f;

	@Parameter(type = ItemIO.INPUT)
	//Float wavelength = 530f;
	Float wavelength = 500f;

	@Parameter(type = ItemIO.INPUT)
	Float riImmersion = 1.518f;

	@Parameter(type = ItemIO.INPUT)
	Float riSampleStart = 1.0f;

	@Parameter(type = ItemIO.INPUT)
	Float riSampleEnd = 1.75f;

	@Parameter(type = ItemIO.INPUT)
	Float riSampleInc = 0.1f;

	@Parameter(type = ItemIO.INPUT)
	//Float xySpacing = 64.5f;
	Float xySpacing = 103.7f;

	@Parameter(type = ItemIO.INPUT)
	//Float zSpacing = 160f;
	Float zSpacing = 100f;

	@Parameter(type = ItemIO.INPUT)
	Float depth = 5000f;

	@Override
	public void run() {

		OpService ops = ij.op();
		UIService ui = ij.ui();

		Img<FloatType> imgF = ops.convert().float32(img);

		ui.show("Original (YZ)", VisualizationUtility.project(ij, imgF, 0));
		// ui.show("Original (XY)", VisualizationUtility.project(ij, imgF, 2));

		DoubleColumn stdColumn = new DoubleColumn();
		DoubleColumn sumColumn = new DoubleColumn();
		DoubleColumn otsuSumColumn = new DoubleColumn();
		GenericColumn nameColumn = new GenericColumn();

		Img<BitType> thresholdedOrig = (Img)ops.threshold().otsu(imgF);
		ui.show("Thresholded orig (YZ)", VisualizationUtility.project(ij,  thresholdedOrig, 0));

		stdColumn.add(ops.stats().stdDev(imgF).getRealDouble());
		sumColumn.add(ops.stats().sum(imgF).getRealDouble());
		otsuSumColumn.add(ops.stats().sum(thresholdedOrig).getRealDouble());

		nameColumn.add("original");

		
		wavelength = wavelength * 1E-9f;
		xySpacing = xySpacing * 1E-9f;
		zSpacing = zSpacing * 1E-9F;
		depth = depth * 1E-9F;

		if (wavelength < 545E-9) {
			wavelength = 545E-9f;
		}

		FinalDimensions psfDims = new FinalDimensions(imgF.dimension(0), imgF.dimension(1), imgF.dimension(2));

		for (Float riSample = riSampleStart; riSample <= riSampleEnd; riSample =
			 riSample+ riSampleInc)
		{
			// create the diffraction based psf
			Img<FloatType> psf = ops.create().kernelDiffraction(psfDims,
				numericalAperture, wavelength, riSample, riImmersion, xySpacing,
				zSpacing, depth, new FloatType());

			// normalize PSF energy to 1
			float sumPSF = ops.stats().sum(psf).getRealFloat();
			FloatType val = new FloatType();
			val.set(sumPSF);
			psf = (Img<FloatType>) ops.math().divide(psf, val);

			@SuppressWarnings("unchecked")
			final UnaryComputerOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> deconvolver =
				(UnaryComputerOp) Computers.unary(ops, UnaryComputerYacuDecu.class,
					RandomAccessibleInterval.class, imgF, psf, iterations);

			Img<FloatType> deconvolved = ops.create().img(imgF);

			deconvolver.compute(imgF, deconvolved);

			ui.show("Deconvolved (YZ)"+riSample, VisualizationUtility.project(ij, deconvolved,
				0));
			
			Img<BitType> thresholded = (Img)ops.threshold().otsu(deconvolved);
			
			ui.show("Thresholded (YZ)"+riSample, VisualizationUtility.project(ij,  thresholded, 0));
			
			// ui.show("Deconvolved (XY)", VisualizationUtility.project(ij,
			// deconvolved,
			// 2));

			// ui.show("PSF (XY)", VisualizationUtility.project(ij, psf, 2));
			// ui.show("PSF (XZ)", VisualizationUtility.project(ij, psf, 0));

			double std = ops.stats().stdDev(deconvolved).getRealDouble();
			double sum = ops.stats().sum(deconvolved).getRealDouble();
			double otsuSum = ops.stats().sum(thresholded).getRealDouble();

			stdColumn.add(std);
			sumColumn.add(sum);
			nameColumn.add("deconvolved "+riSample);
			otsuSumColumn.add(otsuSum);
		}
		
		Table table = new DefaultGenericTable();
		table.add(nameColumn);
		table.add(sumColumn);
		table.add(stdColumn);
		table.add(otsuSumColumn);
		
		table.setColumnHeader(0, "Name");
		table.setColumnHeader(1, "Sum Deconvolved");
		table.setColumnHeader(2, "Std Deconvolved");
		table.setColumnHeader(3, "Otsu Sum");

		ij.ui().show(table);

		IJ.run("Tile", "");
	}

}
