package net.imagej.ops.experiments.kernel;

import net.imagej.ops.OpService;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccess;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.complex.ComplexFloatType;
import net.imglib2.type.numeric.real.FloatType;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_2D;

public class WidefieldKernel {
	private int _w;
	private int _h;
	private int _z;
	private double _indexImmersion;
	private double _na;
	private int _lem;
	private double _indexSpRefr;
	private double _xySampling;
	private double _zSampling;
	private double _depth;
	
	public final static int DEFAULT_W = 128;
	public final static int DEFAULT_H = 128;
	public final static int DEFAULT_Z = 128;
	public final static double DEFAULT_INDEXIMMERSION = 1.515;
	public final static double DEFAULT_NA = 1.4;
	public final static int DEFAULT_LEM = 520;
	public final static double DEFAULT_INDEXSP = 1.33;
	public final static double DEFAULT_XYSAMPLING = 92.00;
	public final static double DEFAULT_ZSAMPLING = 277.00;
	public final static double DEFAULT_DEPTH = 0.0;
	
	public WidefieldKernel() {
		setW(DEFAULT_W);
		setH(DEFAULT_H);
		setZ(DEFAULT_Z);
		setXYSAMPLING(DEFAULT_XYSAMPLING);
		setZSAMPLING(DEFAULT_ZSAMPLING);
		setIndexImmersion(DEFAULT_INDEXIMMERSION);
		setNA(DEFAULT_NA);
		setLEM(DEFAULT_LEM);
		setIndexSp(DEFAULT_INDEXSP);
		
		setDEPTH(DEFAULT_DEPTH);		
	}
	public Img compute(OpService ops){
		
		final DoubleFFT_2D fft = new DoubleFFT_2D(_w, _h);
		//IcyBufferedImage psf3d = new IcyBufferedImage(_w, _h, 1, DataType.DOUBLE);
		
		int hc = _h/2;
		int wc = _w/2;
		int zc = _z/2;
		double kSampling = (2*Math.PI)/(_h*_xySampling); //Fourier space sampling
		double lambdaObj = _lem/_indexImmersion;//Wavelength of light inside the medium
		double lambdaSp = _lem/_indexSpRefr;//Wavelength of light inside the medium
		double k0 = (2*Math.PI)/_lem;//Wave vector
		double kObj = (2*Math.PI)/lambdaObj;//Wave vector in the immersion medium
		double kSp = (2*Math.PI)/lambdaSp;//Wave vector in the medium
		double kMax = (2*Math.PI*_na)/(_lem*kSampling);//Maximum aperture radius
		//double saCoeff = _depth * (_indexSpRefr-_indexImmersion);
		
		// Define the zero defocus pupil function
		//IcyBufferedImage pupil = new IcyBufferedImage(_w, _h, 2, DataType.FLOAT); // channel 1 is real and channel 2 is imaginary
		//float[] pupilRealBuffer = new float[_w*_h];//pupil.getDataXYAsFloat(0);//Real
		//float[] pupilImagBuffer = new float[_w*_h];//pupil.getDataXYAsFloat(1);//imaginary
		float[] pupilComplexBuffer = new float[_w*_h*2];//pupil.getDataXYAsFloat(1);//imaginary
		
		//IcyBufferedImage dpupil = new IcyBufferedImage(_w, _h, 2, DataType.DOUBLE); // channel 1 is real and channel 2 is imaginary
		//double[] dpupilRealBuffer = new double[_w*_h];//dpupil.getDataXYAsDouble(0); //Real
		//double[] dpupilImagBuffer = new double[_w*_h];//dpupil.getDataXYAsDouble(1); //imaginary
		double[] dpupilComplexBuffer = new double[_w*_h*2];//dpupil.getDataXYAsDouble(1); //imaginary
		
		//Calculate the cosine and the sine components
		//IcyBufferedImage ctheta = new IcyBufferedImage(_w, _h, 1, DataType.DOUBLE);
		double[] cthetaBuffer = new double[_w*_h];//ctheta.getDataXYAsDouble(0);
		//IcyBufferedImage cthetaSp = new IcyBufferedImage(_w, _h, 1, DataType.DOUBLE);
		double[] cthetaSpBuffer = new double[_w*_h];//cthetaSp.getDataXYAsDouble(0);
		//IcyBufferedImage stheta = new IcyBufferedImage(_w, _h, 1, DataType.DOUBLE);
		double[] sthetaBuffer = new double[_w*_h];//stheta.getDataXYAsDouble(0);
		//IcyBufferedImage sthetaSp = new IcyBufferedImage(_w, _h, 1, DataType.DOUBLE);
		double[] sthetaSpBuffer = new double[_w*_h];//sthetaSp.getDataXYAsDouble(0);
		
		Img<FloatType> psf3d = ops.create().img(new FinalDimensions(_w,_h,_z), new FloatType());
		//psf3d.setName("WideField PSF");
        
		for (int k =  0 ; k < _z; k++)
    	{// Define the defocus pupils			
			
			double defocus = k-zc;
			defocus = defocus*_zSampling;	
			
			for(int x = 0; x < _w; x++)
			{
				for(int y = 0; y < _h; y++)
				{   
					double kxy = Math.sqrt( Math.pow(x-wc, 2) + Math.pow(y-hc, 2) );
        		
					int index=x + y * _w;
					
					//pupilRealBuffer[x + y * _w] = ((kxy < kMax) ? 1 : 0); //Pupil bandwidth constraints
					//pupilImagBuffer[x + y * _w] = 0; //Zero phase 
					pupilComplexBuffer[2*index] = ((kxy < kMax) ? 1 : 0); //Pupil bandwidth constraints
					pupilComplexBuffer[2*index+1] = 0; //Zero phase 
        		
					sthetaBuffer[x + y * _w] = Math.sin( kxy * kSampling / kObj );
					sthetaBuffer[x + y * _w] = (sthetaBuffer[x + y * _w]< 0) ? 0: sthetaBuffer[x + y * _w];
					sthetaSpBuffer[x + y * _w] = Math.sin( kxy * kSampling / kSp );
					sthetaSpBuffer[x + y * _w] = (sthetaSpBuffer[x + y * _w]< 0) ? 0: sthetaSpBuffer[x + y * _w];
					cthetaBuffer[x + y * _w] = Double.MIN_VALUE + Math.sqrt(1 - Math.pow(sthetaBuffer[x + y * _w], 2));
					cthetaSpBuffer[x + y * _w] = Double.MIN_VALUE + Math.sqrt(1 - Math.pow(sthetaSpBuffer[x + y * _w], 2));
					dpupilComplexBuffer[2*index] = pupilComplexBuffer[2*index] * Math.cos((defocus * k0 * cthetaBuffer[x + y * _w]) + (k0 * _depth * (_indexSpRefr * cthetaSpBuffer[x + y * _w]-_indexImmersion * cthetaBuffer[x + y * _w])));
					dpupilComplexBuffer[2*index+1] = pupilComplexBuffer[2*index] * Math.sin((defocus * k0 * cthetaBuffer[x + y * _w]) + (k0 * _depth * (_indexSpRefr * cthetaSpBuffer[x + y * _w]-_indexImmersion * cthetaBuffer[x + y * _w])));
        		}
			}
			double[] psf2d = dpupilComplexBuffer;//dpupil.getDataCopyCXYAsDouble();
			fft.complexInverse(psf2d, false);
			
			RandomAccess<FloatType> ra=psf3d.randomAccess();
			
			//IcyBufferedImage timg = new IcyBufferedImage(_w, _h, 1, DataType.DOUBLE);
			//timg.beginUpdate();
			try{
				for(int x = 0; x < (wc+1); x++)
				{
					for(int y = 0; y < (hc+1); y++)
					{
						double val=Math.sqrt(Math.pow(psf2d[(((wc-x) + (hc-y) * _w)*2)+0],2)+Math.pow(psf2d[(((wc-x) + (hc-y) * _w)*2)+1], 2));			
						       val=Math.sqrt(Math.pow(psf2d[(((wc-x) + (hc-y) * _h)*2)+0],2)+Math.pow(psf2d[(((wc-x) + (hc-y) * _h)*2)+1], 2));
						ra.setPosition(new int[]{x,y,k});
						ra.get().setReal(val);
						
						//timg.setDataAsDouble(x, y, 0, Math.sqrt(Math.pow(psf2d[(((wc-x) + (hc-y) * _w)*2)+0],2)+Math.pow(psf2d[(((wc-x) + (hc-y) * _w)*2)+1], 2)));
						//timg.setDataAsDouble(x, y, 1, psf2d[(((wc-x) + (hc-y) * _h)*2)+1]);

					}
					for(int y = hc+1; y < _h; y++)
					{
						
						
						double val=Math.sqrt(Math.pow(psf2d[(((wc-x) + (_h+hc-y) * _w)*2)+0], 2)+Math.pow(psf2d[(((wc-x) + (_h+hc-y) * _w)*2)+1], 2));
						ra.setPosition(new int[]{x,y,k});
						ra.get().setReal(val);
						
						//timg.setDataAsDouble(x, y, 0, Math.sqrt(Math.pow(psf2d[(((wc-x) + (_h+hc-y) * _w)*2)+0], 2)+Math.pow(psf2d[(((wc-x) + (_h+hc-y) * _w)*2)+1], 2)));
						//timg.setDataAsDouble(x, y, 1, psf2d[(((wc-x) + (_h+hc-y) * _h)*2)+1]);
					}
					
				}
				for(int x = (wc+1); x < _w; x++)
				{
					for(int y = 0; y < (hc+1); y++)
					{
						double val=Math.sqrt(Math.pow(psf2d[(((_w+wc-x) + (hc-y) * _w)*2)+0], 2)+Math.pow(psf2d[(((_w+wc-x) + (hc-y) * _w)*2)+1], 2));
						ra.setPosition(new int[]{x,y,k});
						ra.get().setReal(val);
						
						//timg.setDataAsDouble(x, y, 0, Math.sqrt(Math.pow(psf2d[(((_w+wc-x) + (hc-y) * _w)*2)+0], 2)+Math.pow(psf2d[(((_w+wc-x) + (hc-y) * _w)*2)+1], 2)));
						//timg.setDataAsDouble(x, y, 1, psf2d[(((_w+wc-x) + (hc-y) * _h)*2)+1]);
					}
					for(int y = hc+1; y < _h; y++)
					{
						double val=Math.sqrt(Math.pow(psf2d[(((_w+wc-x) + (_h+hc-y) * _w)*2)+0], 2)+Math.pow(psf2d[(((_w+wc-x) + (_h+hc-y) * _w)*2)+1],2));
						ra.setPosition(new int[]{x,y,k});
						ra.get().setReal(val);
						
						//timg.setDataAsDouble(x, y, 0, Math.sqrt(Math.pow(psf2d[(((_w+wc-x) + (_h+hc-y) * _w)*2)+0], 2)+Math.pow(psf2d[(((_w+wc-x) + (_h+hc-y) * _w)*2)+1],2)));
						//timg.setDataAsDouble(x, y, 1, psf2d[(((_w+wc-x) + (_h+hc-y) * _h)*2)+1]);
					}
				}

			}finally {
			//timg.endUpdate();
			}
			
           // psf3d.addImage(timg);
            // TO DO: add normalization
    	}
		
	return psf3d;

	}
	public int getW() {
		return _w;
	}
	public int getH() {
		return _h;
	}
	public int getZ() {
		return _z;
	}
	public double getIndexRefr() {
		return _indexImmersion;
	}
	public double getNA() {
		return _na;
	}
	public int getLEM() {
		return _lem;
	}
	public double getSA() {
		return _indexSpRefr;
	}
	public double getXYSAMPLING() {
		return _xySampling;
	}
	public double getZSAMPLING() {
		return _zSampling;
	}
	public double getDEPTH() {
		return _depth;
	}
	
	public void setW(int src) {
		_w = src;
	}
	public void setH(int src) {
		_h = src;
	}
	public void setZ(int src) {
		_z = src;
	}
	public void setIndexImmersion(double src) {
		_indexImmersion = src;
	}
	public void setNA(double src) {
		_na = src;
	}
	public void setLEM(int src) {
		_lem = src;
	}
	public void setIndexSp(double src) {
		_indexSpRefr = src;
	}
	public void setXYSAMPLING(double src) {
		_xySampling = src;
	}
	public void setZSAMPLING(double src) {
		_zSampling = src;
	}
	public void setDEPTH(double src) {
		_depth = src;
	}
}