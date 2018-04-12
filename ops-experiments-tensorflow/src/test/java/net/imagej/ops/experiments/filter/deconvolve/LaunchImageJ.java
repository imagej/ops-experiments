package net.imagej.ops.experiments.filter.deconvolve;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class LaunchImageJ {
	
	@SuppressWarnings({ "unchecked", "deprecation" })
	public static <T extends RealType<T> & NativeType<T>> void main(final String[] args) throws InterruptedException, IOException {

		// create an instance of imagej
		final ImageJ ij = new ImageJ();

		// launch it
		ij.launch(args);
		
		String imgPath = "/Users/eczech/repos/hammer/flowdec/python/flowdec/datasets/bars-25pct";
		final Img<FloatType> img = (Img<FloatType>) ij.dataset()
				.open(imgPath + "/data.tif").getImgPlus().getImg();
		
		ij.ui().show("Original", img);

	}

}
