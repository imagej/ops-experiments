package net.imagej.ops.experiments.filter.deconvolve;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import org.scijava.command.Command;
import org.scijava.command.CommandModule;
import org.scijava.command.CommandService;
import org.scijava.log.LogService;
import org.scijava.module.Module;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import io.scif.services.DatasetIOService;
import net.imagej.Dataset;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

@Plugin(type = Command.class, headless = true, menuPath = "Plugins>OpsExperiments>YacuDecu Deconvolution Batch")
public class YacuDecuRichardsonLucyBatch<T extends RealType<T> & NativeType<T>> implements Command {

	@Parameter
	LogService log;

	@Parameter
	DatasetIOService data;

	@Parameter
	CommandService cmd;

	@Parameter
	File imgFile;

	@Parameter
	Dataset psf;

	@Parameter
	Integer iterations;

	@Override
	public void run() {

		Dataset img;

		try {
			img = data.open(imgFile.getPath());
		} catch (IOException e) {
			// TODO Auto-generated catch block

			System.out.println("File: " + imgFile.getPath() + " could not be opened");
			e.printStackTrace();
			return;
		}

		Future<CommandModule> instance = cmd.run(YacuDecuRichardsonLucyCommand.class, true, "img", img, "psf", psf,
				"iterations", iterations);
		
		// here we wait for the outputs so we don't return before finishing the command call.  
		try {
			instance.get().getOutputs();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ExecutionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
