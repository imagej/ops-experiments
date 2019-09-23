package net.imagej.ops.experiments.testImages;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imglib2.FinalDimensions;
import net.imglib2.Point;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.region.hypersphere.HyperSphere;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

public class BeedPhantom extends AbstractDeconvolutionPhantomData {

	Img<FloatType> img;
	Img<FloatType> psf;
	
	@Override
	public void LoadImages(ImageJ ij) throws IOException {
		// hard code size for now
		long[] dims=new long[] {128,128,800};
		
		img=ij.op().create().img(new FinalDimensions(dims), new FloatType());
		
		placeSphereInCenter(img);
		
		RandomAccessibleInterval temp=Views.subsample(img, new long[] {1,1,3});
		
		img=(Img)ij.op().copy().iterableInterval(Views.iterable(temp));
		
		psf = this.createTheoreticalPSF(img, 1.4, 550e-9, 1.4, 1.515, 62.5e-9, 160e-9, 5000e-9, ij);
		
		img = (Img)ij.op().filter().convolve(img, psf);
}

	@Override
	public RandomAccessibleInterval<FloatType> getImg() {
		// TODO Auto-generated method stub
		return img;
	}

	@Override
	public RandomAccessibleInterval<FloatType> getPSF() {
		// TODO Auto-generated method stub
		return psf;
	}

	// utility to place a small sphere at the center of the image
	static private void placeSphereInCenter(Img<FloatType> img) {

		final Point center = new Point(img.numDimensions());

		for (int d = 0; d < img.numDimensions(); d++)
			center.setPosition(img.dimension(d) / 2, d);

		HyperSphere<FloatType> hyperSphere = new HyperSphere<>(img, center, 20);

		for (final FloatType value : hyperSphere) {
			value.setReal(1000);
		}
	}

}
