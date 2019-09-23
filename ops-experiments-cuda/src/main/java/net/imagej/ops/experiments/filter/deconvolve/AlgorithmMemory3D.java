package net.imagej.ops.experiments.filter.deconvolve;

public class AlgorithmMemory3D {
	
	final static long KB_GB_DIVISOR=1024*1024*1024;
	
	/**
	 * extended width
	 */
	long extendedWidth;
	
	/**
	 * extended height
	 */
	long extendedHeight;
	
	/**
	 * extended depth
	 */
	long extendedNumSlices;
	
	/**
	 * bytes per pixel
	 */
	int bytesPerPixel;
	
	/**
	 * size per 3d image buffer (G)
	 */
	float imageBufferSize;
	
	/** 
	 * work space size
	 */
	float workSpaceSize;
	
	/**
	 * number buffers
	 */
	int numBuffers;
	
	/**
	 * total memory
	 */
	float memoryNeeded;
	
}
