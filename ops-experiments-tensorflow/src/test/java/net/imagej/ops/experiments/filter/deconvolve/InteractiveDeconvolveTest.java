package net.imagej.ops.experiments.filter.deconvolve;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imagej.ops.Ops;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class InteractiveDeconvolveTest {
	
	@SuppressWarnings({ "unchecked", "rawtypes" })
	static <T> Img<T> project(ImageJ ij, Img<T> img, int dim){
		int d;
		int[ ] projected_dimensions = new int[img.numDimensions()-1];
		int i = 0;
		for (d=0; d < img.numDimensions();d++){
		    if(d != dim) {
		         projected_dimensions[i]= (int) img.dimension(d);
		         i += 1;
		    }
		}
				
		Img<T> proj = (Img<T>) ij.op().create().img(projected_dimensions);

		UnaryComputerOp op = (UnaryComputerOp) ij.op().op(Ops.Stats.Max.NAME, img);
		
		Img<T> projection=(Img<T>)ij.op().transform().project(proj, img, op, dim);
		return projection;
	}
	
	@SuppressWarnings({ "unchecked", "deprecation" })
	public static <T extends RealType<T> & NativeType<T>> void main(final String[] args) throws InterruptedException, IOException {

		// create an instance of imagej
		final ImageJ ij = new ImageJ();

		// launch it
		ij.launch(args);

		String dir = "./images/bars-25pct";
		String imgPath = dir + "/data.tif";
		String psfPath = dir + "/kernel.tif";
		
		final Img<FloatType> img = (Img<FloatType>) ij.dataset()
				.open(imgPath).getImgPlus().getImg();
		
		final Img<FloatType> psf = (Img<FloatType>) ij.dataset()
				.open(psfPath).getImgPlus().getImg();
		
		int iterations = 30;
		int[] pad = new int[] {10, 10, 10};
		Img<FloatType> res = (Img<FloatType>) ij.op().run(
				FlowdecOp.class, img, psf, iterations, pad);

		// Render projections along X and Z axes
		ij.ui().show("Original (YZ)", project(ij, img, 0));
		ij.ui().show("Deconvolved (YZ)", project(ij, res, 0));
		ij.ui().show("Original (XY)", project(ij, img, 2));
		ij.ui().show("Deconvolved (XY)", project(ij, res, 2));

	}

}
