package net.imagej.ops.experiments;

import net.imagej.ImageJ;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class LaunchImageJ {
	public static <T extends RealType<T> & NativeType<T>> void main(final String[] args) throws InterruptedException {

		// create an instance of imagej
		final ImageJ ij = new ImageJ();

		// launch it
		ij.launch(args);
	}

}
