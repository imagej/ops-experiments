package net.imagej.ops.experiments;

import static org.bytedeco.javacpp.cuda.cudaDeviceReset;

public class CudaUtility {

	static final int EXIT_FAILURE = 1;
	static final int EXIT_SUCCESS = 0;
	static final int EXIT_WAIVED = 0;

	public static void checkCudaErrors(int status) {
		if (status != 0) {
			FatalError("Cuda failure: " + status);
		}
	}

	public static void FatalError(String s) {
		System.err.println(s);
		Thread.dumpStack();
		System.err.println("Aborting...");
		cudaDeviceReset();
		System.exit(EXIT_FAILURE);
	}
}
