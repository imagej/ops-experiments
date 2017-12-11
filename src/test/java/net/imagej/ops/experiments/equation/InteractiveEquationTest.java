package net.imagej.ops.experiments.equation;

import net.imagej.ImageJ;
import net.imglib2.FinalDimensions;
import net.imglib2.type.numeric.real.DoubleType;

public class InteractiveEquationTest {

	final static ImageJ ij = new ImageJ();

	public static void main(final String[] args) {

		ij.launch(args);

		long w = 500;
		long h = 500;

		// apply the illusion operator
		ij.ui().show(
				ij.op().image().equation(ij.op().create().img(new FinalDimensions(w, h), new DoubleType()), (x, y) -> {

					double frequency = 2 * Math.PI * .02;
					double amplitude = 5;
					double thickness = 2;
					int waveOffset = 10;
					int waveSpace = 60;

					double gray1 = 175;
					double gray2 = 75;

					double val = x + y < w / 2 ? 255 : 0;
					val = x + y >= w / 2 && x + y < 3. / 2. * w ? 127 : val;

					int i = 0;
					for (int yy = waveOffset; yy < h; yy += waveSpace) {
						double grayShift = (i++) % 2 * Math.PI / frequency / 2;

						double gray = (x + grayShift) % (2 * Math.PI / frequency) < (Math.PI / frequency) ? gray1
								: gray2;

						val = yy + amplitude * Math.cos(x * frequency) - y >= 0
								&& yy + amplitude * Math.cos(x * frequency) - y <= 0 + thickness ? gray : val;
						val = yy + waveOffset + amplitude * Math.cos(x * frequency) - y >= 0
								&& yy + waveOffset + amplitude * Math.cos(x * frequency) - y <= thickness ? gray : val;

					}

					return val;
				}));

	}

}
