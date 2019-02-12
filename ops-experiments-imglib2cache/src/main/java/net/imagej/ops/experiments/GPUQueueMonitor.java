package net.imagej.ops.experiments;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.FutureTask;

import net.imagej.ops.experiments.filter.deconvolve.YacuDecuRichardsonLucyWrapper;

public class GPUQueueMonitor extends Thread {
	private boolean stop;
	private BlockingQueue<FutureTask> queue;
	private int deviceNum;

	public GPUQueueMonitor(BlockingQueue<FutureTask> queue, int deviceNum) {
		this.queue = queue;
		this.deviceNum = deviceNum;
	}

	public void run() {
		// set device num for this thread
		YacuDecuRichardsonLucyWrapper.setDevice(deviceNum);

		while (!stop) {

			try {
				FutureTask job = queue.take();
				System.out.println("start decon on GPU " + deviceNum);
				job.run(); // will "complete" the future
				System.out.println("finished decon on GPU " + deviceNum);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public void setStop(boolean stop) {
		this.stop = stop;
	}

}
