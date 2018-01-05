package net.imagej.ops.experiments.kernel;

import net.imagej.ops.OpService;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccess;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.special.BesselJ;

//import ij.ImageStack;
//import ij.process.FloatProcessor;

public class GibsonLanni {
	// //////// physical parameters /////////////
	private int nx = 256; // psf size
	private int ny = 256;
	private int nz = 128;
	private double NA = 1.4; // numerical aperture
	private double lambda = 610E-09; // wavelength
	private double ns = 1.33; // specimen refractive index
	private double ng0 = 1.5; // coverslip refractive index, design value
	private double ng = 1.5; // coverslip refractive index, experimental
	private double ni0 = 1.5; // immersion refractive index, design
	private double ni = 1.5; // immersion refractive index, experimental
	private double ti0 = 150E-06; // working distance of the objective,
									// desig

	// a bit precision lost if use 170*1.0E-6
	private double tg0 = 170E-6; // coverslip thickness, design value
	private double tg = 170E-06; // coverslip thickness, experimental value
	private double resLateral = 100E-9; // lateral pixel size
	private double resAxial = 250E-9; // axial pixel size
	private double pZ = 2000E-9D; // position of particle

	// ////////approximation parameters /////////////
	private int numBasis = 100; // number of basis functions
	private int numSamp = 1000; // number of sampling
	private int overSampling = 2; // overSampling

	/**
	 * @return the nx
	 */
	public int getNx() {
		return nx;
	}

	/**
	 * @param nx
	 *            the nx to set
	 */
	public void setNx(int nx) {
		this.nx = nx;
	}

	/**
	 * @return the ny
	 */
	public int getNy() {
		return ny;
	}

	/**
	 * @param ny
	 *            the ny to set
	 */
	public void setNy(int ny) {
		this.ny = ny;
	}

	/**
	 * @return the nz
	 */
	public int getNz() {
		return nz;
	}

	/**
	 * @param nz
	 *            the nz to set
	 */
	public void setNz(int nz) {
		this.nz = nz;
	}

	/**
	 * @return the nA
	 */
	public double getNA() {
		return NA;
	}

	/**
	 * @param nA
	 *            the nA to set
	 */
	public void setNA(double nA) {
		NA = nA;
	}

	/**
	 * @return the lambda
	 */
	public double getLambda() {
		return lambda;
	}

	/**
	 * @param lambda
	 *            the lambda to set
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * @return the ns
	 */
	public double getNs() {
		return ns;
	}

	/**
	 * @param ns
	 *            the ns to set
	 */
	public void setNs(double ns) {
		this.ns = ns;
	}

	/**
	 * @return the ng0
	 */
	public double getNg0() {
		return ng0;
	}

	/**
	 * @param ng0
	 *            the ng0 to set
	 */
	public void setNg0(double ng0) {
		this.ng0 = ng0;
	}

	/**
	 * @return the ng
	 */
	public double getNg() {
		return ng;
	}

	/**
	 * @param ng
	 *            the ng to set
	 */
	public void setNg(double ng) {
		this.ng = ng;
	}

	/**
	 * @return the ni0
	 */
	public double getNi0() {
		return ni0;
	}

	/**
	 * @param ni0
	 *            the ni0 to set
	 */
	public void setNi0(double ni0) {
		this.ni0 = ni0;
	}

	/**
	 * @return the ni
	 */
	public double getNi() {
		return ni;
	}

	/**
	 * @param ni
	 *            the ni to set
	 */
	public void setNi(double ni) {
		this.ni = ni;
	}

	/**
	 * @return the ti0
	 */
	public double getTi0() {
		return ti0;
	}

	/**
	 * @param ti0
	 *            the ti0 to set
	 */
	public void setTi0(double ti0) {
		this.ti0 = ti0;
	}

	/**
	 * @return the tg0
	 */
	public double getTg0() {
		return tg0;
	}

	/**
	 * @param tg0
	 *            the tg0 to set
	 */
	public void setTg0(double tg0) {
		this.tg0 = tg0;
	}

	/**
	 * @return the tg
	 */
	public double getTg() {
		return tg;
	}

	/**
	 * @param tg
	 *            the tg to set
	 */
	public void setTg(double tg) {
		this.tg = tg;
	}

	/**
	 * @return the resLateral
	 */
	public double getResLateral() {
		return resLateral;
	}

	/**
	 * @param resLateral
	 *            the resLateral to set
	 */
	public void setResLateral(double resLateral) {
		this.resLateral = resLateral;
	}

	/**
	 * @return the resAxial
	 */
	public double getResAxial() {
		return resAxial;
	}

	/**
	 * @param resAxial
	 *            the resAxial to set
	 */
	public void setResAxial(double resAxial) {
		this.resAxial = resAxial;
	}

	/**
	 * @return the pZ
	 */
	public double getpZ() {
		return pZ;
	}

	/**
	 * @param pZ
	 *            the pZ to set
	 */
	public void setpZ(double pZ) {
		this.pZ = pZ;
	}

	/**
	 * @return the numBasis
	 */
	public int getNumBasis() {
		return numBasis;
	}

	/**
	 * @param numBasis
	 *            the numBasis to set
	 */
	public void setNumBasis(int numBasis) {
		this.numBasis = numBasis;
	}

	/**
	 * @return the numSamp
	 */
	public int getNumSamp() {
		return numSamp;
	}

	/**
	 * @param numSamp
	 *            the numSamp to set
	 */
	public void setNumSamp(int numSamp) {
		this.numSamp = numSamp;
	}

	/**
	 * @return the overSampling
	 */
	public int getOverSampling() {
		return overSampling;
	}

	/**
	 * @param overSampling
	 *            the overSampling to set
	 */
	public void setOverSampling(int overSampling) {
		this.overSampling = overSampling;
	}

	public Img<FloatType> compute(OpService ops) {
		
		
		int distanceFromCenter=(int)Math.abs(Math.ceil(pZ/resAxial));
		nz=nz+2*distanceFromCenter;
		
		double x0 = (this.nx - 1) / 2.0D;
		double y0 = (this.ny - 1) / 2.0D;

		double xp = x0;
		double yp = y0;

		
		int maxRadius = (int) Math.round(Math.sqrt((this.nx - x0) * (this.nx - x0) + (this.ny - y0) * (this.ny - y0)))
				+ 1;
		double[] r = new double[maxRadius * this.overSampling];
		double[][] h = new double[this.nz][r.length];

		double a = 0.0D;
		double b = Math.min(1.0D, this.ns / this.NA);

		double k0 = 2 * Math.PI / this.lambda;
		double factor1 = 545 * 1.0E-9 / this.lambda;
		double factor = factor1 * this.NA / 1.4;
		double deltaRho = (b - a) / (this.numSamp - 1);

		// basis construction
		double rho = 0.0D;
		double am = 0.0;
		double[][] Basis = new double[this.numSamp][this.numBasis];

		BesselJ bj0 = new BesselJ(0);
		BesselJ bj1 = new BesselJ(1);

		long startTime = 0;
		long endTime = 0;

		for (int m = 0; m < this.numBasis; m++) {
			// am = (3 * m + 1) * factor;
			am = (3 * m + 1);
			for (int rhoi = 0; rhoi < this.numSamp; rhoi++) {
				rho = rhoi * deltaRho;
				Basis[rhoi][m] = bj0.value(am * rho);
			}
		}

		// compute the function to be approximated

		double ti = 0.0D;
		double OPD = 0;
		double W = 0;

		startTime = System.currentTimeMillis();

		double[][] Coef = new double[this.nz][this.numBasis * 2];
		double[][] Ffun = new double[this.numSamp][this.nz * 2];

		double rhoNA2;

		for (int z = 0; z < this.nz; z++) {
			ti = (this.ti0 + this.resAxial * (z - (this.nz - 1.0D) / 2.0D));

			for (int rhoi = 0; rhoi < this.numSamp; rhoi++) {
				rho = rhoi * deltaRho;
				rhoNA2 = rho * rho * this.NA * this.NA;

				OPD = this.pZ * Math.sqrt(this.ns * this.ns - rhoNA2);
				OPD += this.tg * Math.sqrt(this.ng * this.ng - rhoNA2)
						- this.tg0 * Math.sqrt(this.ng0 * this.ng0 - rhoNA2);
				OPD += ti * Math.sqrt(this.ni * this.ni - rhoNA2) - this.ti0 * Math.sqrt(this.ni0 * this.ni0 - rhoNA2);

				W = k0 * OPD;

				Ffun[rhoi][z] = Math.cos(W);
				Ffun[rhoi][z + this.nz] = Math.sin(W);
			}
		}

		// solve the linear system
		// begin....... (Using Common Math)

		RealMatrix coefficients = new Array2DRowRealMatrix(Basis, false);
		RealMatrix rhsFun = new Array2DRowRealMatrix(Ffun, false);
		DecompositionSolver solver = new SingularValueDecomposition(coefficients).getSolver(); // slower
																								// but
																								// more
																								// accurate
		// DecompositionSolver solver = new
		// QRDecomposition(coefficients).getSolver(); // faster, less accurate

		RealMatrix solution = solver.solve(rhsFun);
		Coef = solution.getData();

		// end.......

		double[][] RM = new double[this.numBasis][r.length];
		double beta = 0.0D;

		double rm = 0.0D;
		for (int n = 0; n < r.length; n++) {
			r[n] = (n * 1.0 / this.overSampling);
			beta = k0 * this.NA * r[n] * this.resLateral;

			for (int m = 0; m < this.numBasis; m++) {
				am = (3 * m + 1) * factor;
				rm = am * bj1.value(am * b) * bj0.value(beta * b) * b;
				rm = rm - beta * b * bj0.value(am * b) * bj1.value(beta * b);
				RM[m][n] = rm / (am * am - beta * beta);

			}
		}

		// obtain one component
		double maxValue = 0.0D;
		for (int z = 0; z < this.nz; z++) {
			for (int n = 0; n < r.length; n++) {
				double realh = 0.0D;
				double imgh = 0.0D;
				for (int m = 0; m < this.numBasis; m++) {
					realh = realh + RM[m][n] * Coef[m][z];
					imgh = imgh + RM[m][n] * Coef[m][z + this.nz];

				}
				h[z][n] = realh * realh + imgh * imgh;
			}
		}

		// assign

		double[][] Pixel = new double[this.nz][this.nx * this.ny];

		// startTime = System.currentTimeMillis();

		for (int z = 0; z < this.nz; z++) {

			for (int x = 0; x < this.nx; x++) {
				for (int y = 0; y < this.ny; y++) {
					double rPixel = Math.sqrt((x - xp) * (x - xp) + (y - yp) * (y - yp));
					int index = (int) Math.floor(rPixel * this.overSampling);

					double value = h[z][index]
							+ (h[z][(index + 1)] - h[z][index]) * (rPixel - r[index]) * this.overSampling;
					Pixel[z][(x + this.nx * y)] = value;
					if (value > maxValue) {
						maxValue = value;
					}
				}
			}
			//

		}
		// endTime = System.currentTimeMillis();
		// System.out.println("Assign:" + (endTime - startTime));
		//
		//
		// startTime = System.currentTimeMillis();
		
		Img<FloatType> psf3d = ops.create().img(new FinalDimensions(nx, ny, nz-2*distanceFromCenter), new FloatType());

		RandomAccess<FloatType> ra = psf3d.randomAccess();

		for (int z = 2*distanceFromCenter; z < this.nz; z++) {

			for (int x = 0; x < this.nx; x++) {
				for (int y = 0; y < this.ny; y++) {

					double value = Pixel[z][(x + this.nx * y)] / maxValue;

					ra.setPosition(new int[] { x, y, z-2*distanceFromCenter });
					ra.get().setReal(value);
				}
			}
		}

		// endTime = System.currentTimeMillis();
		// System.out.println("To stack:" + (endTime - startTime));

		return psf3d;

	}

}
