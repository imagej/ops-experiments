package net.imagej.ops.experiments.javacpp;

import org.bytedeco.javacpp.FloatPointer;
import org.junit.Test;

import net.imagej.ops.experiments.javacpp.NativeLibrary.NativeClass;

public class NativeLibraryTest {

	@Test
	public void NativeLibraryTest() {
		NativeClass l = new NativeClass();
		l.set_property("Hello World!");
		System.out.println(l.property());

		float[] test = new float[] { 1, 2, 3, 4, 5 };
		FloatPointer p = new FloatPointer(test);

		l.print_pointer(test.length, p);
	}

}
