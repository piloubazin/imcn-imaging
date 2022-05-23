package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.Ngb;
import nl.uva.imcn.utilities.Numerics;
import nl.uva.imcn.utilities.BasicInfo;
import nl.uva.imcn.libraries.ImageFilters;
import nl.uva.imcn.libraries.ImageStatistics;
import nl.uva.imcn.libraries.ObjectExtraction;
import nl.uva.imcn.libraries.ObjectLabeling;
import nl.uva.imcn.structures.BinaryHeapPair;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.commons.math3.special.Erf;
//import org.apache.commons.math3.analysis.*;

import java.util.BitSet;

public class LinearFiberMapping {
	
	// jist containers
	private float[] inputImage;
	private String brightParam = "bright";
	public static final String[] brightTypes = {"bright","dark","both"};
	
	private int minscaleParam = 0;
	private int maxscaleParam = 3;
	
	private float difffactorParam = 1.0f;
	private float simscaleParam = 0.1f;
	private int ngbParam = 4;
	private int iterParam = 100;
	private float maxdiffParam = 0.001f;
	
	private float detectionThreshold = 0.01f;
	private float maxLineDist = 1.0f;
	private boolean extend = true;
	private float stoppingRatio = 0.1f;
	private float extendRatio = 0.5f;
	
	private float[] probaImage;
	private int[] lineImage;
	private float[] lengthImage;
	private float[] aniImage;
	private float[] thetaImage;
	
	// global variables
	private int nx, ny, nz, nc, nxyz;
	private float rx, ry, rz;
	
	// numerical quantities
	private static final	float	INVSQRT2 = (float)(1.0/FastMath.sqrt(2.0));
	private static final	float	INVSQRT3 = (float)(1.0/FastMath.sqrt(3.0));
	private static final	float	SQRT2 = (float)FastMath.sqrt(2.0);
	private static final	float	SQRT3 = (float)FastMath.sqrt(3.0);
	private static final	float	SQRT2PI = (float)FastMath.sqrt(2.0*(float)Math.PI);
	private static final	float	PI2 = (float)(Math.PI/2.0);
	private static final	float   	L2N2=2.0f*(float)(FastMath.sqrt(2.0*(float)(FastMath.log(2.0))));
	
	// direction labeling		
	public	static	final	byte	X = 0;
	public	static	final	byte	Y = 1;
	public	static 	final 	byte 	XpY = 2;
	public	static 	final 	byte 	XmY = 3;
	public	static	final	byte	mX = 4;
	public	static	final	byte	mY = 5;
	public	static 	final 	byte 	mXpY = 6;
	public	static 	final 	byte 	mXmY = 7;
	
	public	static 	final 	byte 	NC = 4;
	public	static 	final 	byte 	NC2 = 8;
	
	private static final boolean debug=true;
	
	//set inputs
	public final void setInputImage(float[] val) { inputImage = val; }
	public final void setRidgeIntensities(String val) { brightParam = val; }

	public final void setMinimumScale(int val) { minscaleParam = val; }
	public final void setMaximumScale(int val) { maxscaleParam = val; }

	public final void setDiffusionFactor(float val) { difffactorParam = val; }
	public final void setSimilarityScale(float val) { simscaleParam = val; }
	public final void setNeighborhoodSize(int val) { ngbParam = val; }
	public final void setMaxIterations(int val) { iterParam = val; }
	public final void setMaxDifference(float val) { maxdiffParam = val; }
		
	public final void setDetectionThreshold(float val) { detectionThreshold = val; }
	public final void setMaxLineDistance(float val) { maxLineDist = val; }
	public final void setExtendResult(boolean val) { extend = val; }
	public final void setInclusionRatio(float val) { stoppingRatio = val; }
	public final void setExtendRatio(float val) { extendRatio = val; }
		
	// set generic inputs	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
			
	// create outputs
	public final float[] getProbabilityResponseImage() { return probaImage;}
	public final int[] getLineImage() { return lineImage;}
	public final float[] getLengthImage() { return lengthImage;}
	public final float[] getAngleImage() { return thetaImage;}
	public final float[] getAnisotropyImage() { return aniImage;}

	public void execute(){
		BasicInfo.displayMessage("linear fiber mapping:\n");
		
		// import the inputImage data into 1D arrays: already done
		BasicInfo.displayMessage("...load data\n");
		
		// find surfaceImage dimension, if multiple surfaces are used
		boolean[] mask = new boolean[nxyz];
		float minI = 1e9f;
		float maxI = -1e9f;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int id = x + nx*y + nx*ny*z;
			// mask
			if (inputImage[id]==0) mask[id] = false;
			else mask[id] = true;
			// remove border from computations
			if (x<=1 || x>=nx-2 || y<=1 || y>=ny-2) mask[id] = false;
			// normalize
			if (inputImage[id]>maxI) maxI = inputImage[id];
			if (inputImage[id]<minI) minI = inputImage[id];
		}
		
		// normalize, invert inputImage if looking for dark features
		BasicInfo.displayMessage("...normalize intensities (detection: "+brightParam+")\n");
		for (int xyz=0;xyz<nxyz;xyz++) {
			if (brightParam.equals("bright"))
				inputImage[xyz] = (inputImage[xyz]-minI)/(maxI-minI);
			else if (brightParam.equals("dark"))
				inputImage[xyz] = (maxI-inputImage[xyz])/(maxI-minI);
			else 
				inputImage[xyz] = (inputImage[xyz]-minI)/(maxI-minI);
		}
		boolean unidirectional = true;
		if (brightParam.equals("both")) unidirectional = false;
		
		// Compute filter at different scales
		// new filter response from raw inputImage		
		float[] maxresponse = new float[nxyz];
		byte[] maxdirection = new byte[nxyz];
		int[] maxscale = new int[nxyz];
		
		if (minscaleParam==0) {
            BasicInfo.displayMessage("...first filter response\n");
            
            directionFromRecursiveRidgeFilter1D(inputImage, mask, maxresponse, maxdirection, unidirectional);
        }
        
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
			if (maxresponse[xyz]>0) {
				maxscale[xyz] = 0;
			} else {
				maxscale[xyz] = -1;
			}
		}
		for (int i=Numerics.max(minscaleParam-1,0);i<maxscaleParam;i++) {
			float scale = 1.0f+i;
			float[] smoothed = new float[nxyz];

			BasicInfo.displayMessage("...filter response at scale "+scale+"\n");
		
			// Gaussian Kernel
			float[][] G = ImageFilters.separableGaussianKernel2D(scale/L2N2,scale/L2N2);
				
			// smoothed inputImage
			smoothed = ImageFilters.separableConvolution2D(inputImage,nx,ny,nz,G); 

			byte[] direction = new byte[nxyz];
			float[] response = new float[nxyz];
			directionFromRecursiveRidgeFilter1D(smoothed, mask, response, direction, unidirectional);
			
			//Combine scales: keep maximum response
			for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
				if (response[xyz]>maxresponse[xyz]) {
					maxresponse[xyz] = response[xyz];
					maxdirection[xyz] = direction[xyz];
					maxscale[xyz] = (1+i);
				}
			}
		}
		
		// Equalize histogram (Exp Median)
		BasicInfo.displayMessage("...normalization into probabilities\n");
		float[] proba = new float[nxyz];
		probabilityFromRecursiveRidgeFilter(maxresponse, proba);	
		
		
		// rescale the probability response
		float pmax = ImageStatistics.robustMaximum(proba, 0.000001f, 6, nx, ny, nz);
		if (pmax>0) for (int xyz=0;xyz<nxyz;xyz++) proba[xyz] = Numerics.min(proba[xyz]/pmax,1.0f);
		
		// generate a direction vector
		float[][] direction = new float[2][nxyz];
		for (int xyz=0;xyz<nxyz;xyz++){
			float[] vec = directionVector(maxdirection[xyz]);
			direction[X][xyz] = vec[X];
			direction[Y][xyz] = vec[Y];
		}
		
		// 3. diffuse the data to neighboring structures
		BasicInfo.displayMessage("...diffusion\n");
		
		float[] propag = new float[nxyz];
		propag = probabilisticDiffusion1D(proba, maxdirection, ngbParam, maxdiffParam, simscaleParam, difffactorParam, iterParam);
		
		// 4. Estimate groupings / linear fits?
		BasicInfo.displayMessage("...line groupings\n");
		// -> grow region as long as it's linear-ish and >0?
		// => estimate length, orientation and anisotropy in one go?
		int[] lines = new int[nxyz];
		float[] theta = new float[nxyz];
		float[] length = new float[nxyz];
		float[] ani = new float[nxyz];
		BinaryHeapPair heap = new BinaryHeapPair(nx+ny, nx+ny, BinaryHeapPair.MAXTREE);
		BinaryHeapPair ordering = new BinaryHeapPair(nx*ny, nx*ny, BinaryHeapPair.MAXTREE);

        // simply order them by size instead?
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x + nx*y + nx*ny*z;
			if (propag[xyz]>detectionThreshold && lines[xyz]==0) {
                ordering.addValue(propag[xyz], x, y);
            }
        }
        
        boolean[] used = new boolean[nx*ny];
        while (ordering.isNotEmpty()) {
            float maxpropag = ordering.getFirst();
            int xM = ordering.getFirstId1();
            int yM = ordering.getFirstId2();
            
            ordering.removeFirst();
            
            if (!used[xM+nx*yM]) {
                // region grow 2: toward lower values, fitting a line
                // use probability to weight the line coeffs, to avoid end curls
                heap.reset();
                int[] lx = new int[nx+ny];
                int[] ly = new int[nx+ny];
                float[] lw = new float[nx+ny];
                int nl = 0;
                
                // start from local maximum
                lx[nl] = xM;
                ly[nl] = yM;
                lw[nl] = maxpropag;
                used[xM+nx*yM] = true;
                nl++;
                
                // assume a relationship between detection scale and thickness?
                // not good: makes aggregates rather than clean lines
                //double maxLineDist2 = Numerics.square(Numerics.max(1.0f,maxscale[xM+nx*yM])*maxLineDist);
                double maxLineDist2 = Numerics.square(maxLineDist);
                // instead, allow thickness increase up to length, but no further
                
                double linedist=0.0;
                float minscore = maxpropag;
                
                for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) {
                    int ngb = xM+dx + nx*(yM+dy);
                    if (mask[ngb] && !used[ngb]) {
                        if (propag[ngb]>detectionThreshold && propag[ngb]>stoppingRatio*maxpropag) {
                            heap.addValue(propag[ngb], xM+dx, yM+dy);
                        }
                    }
                }
                boolean stop = false;
                while (heap.isNotEmpty() && !stop) {
                    // fit a line to the current set of coordinates + new one
                    float score = heap.getFirst();
                    float cx = score*heap.getFirstId1();
                    float cy = score*heap.getFirstId2();
                    float cw = score;
                    
                    for (int n=0;n<nl;n++) {
                        cx += lw[n]*lx[n];
                        cy += lw[n]*ly[n];
                        cw += lw[n];
                    }
                    cx /= cw;
                    cy /= cw;
                    
                    float vxx = score*(heap.getFirstId1()-cx)*(heap.getFirstId1()-cx);
                    float vxy = score*(heap.getFirstId1()-cx)*(heap.getFirstId2()-cy);
                    float vyy = score*(heap.getFirstId2()-cy)*(heap.getFirstId2()-cy);
                    
                    for (int n=0;n<nl;n++) {
                        vxx += lw[n]*(lx[n]-cx)*(lx[n]-cx);
                        vxy += lw[n]*(lx[n]-cx)*(ly[n]-cy);
                        vyy += lw[n]*(ly[n]-cy)*(ly[n]-cy);
                    }
                    vxx /= cw;
                    vyy /= cw;
                    vxy /= cw;
                    
                    if (vxy!=0) {
                        double vx = vyy-vxx + FastMath.sqrt( (vyy-vxx)*(vyy-vxx) + 4.0*vxy*vxy);
                        double vy = -2.0*vxy;
                        double norm = vx*vx+vy*vy;
                        
                        linedist = Numerics.square(vx*(heap.getFirstId1()-cx) + vy*(heap.getFirstId2()-cy))/norm;
                        for (int n=0;n<nl;n++) {
                            double newdist = Numerics.square(vx*(lx[n]-cx) + vy*(ly[n]-cy))/norm;
                            if (newdist>linedist) linedist = newdist;
                        }
                    }
                    if (linedist>maxLineDist2) {
                        // do not stop directly as other voxels with lower proba
                        // might still be fittingthe line
                        //stop = true;
                        // instead, skip so eventually the list is empty
                        // (mark it as used, so it doesn't get picked up again)
                        used[heap.getFirstId1()+nx*heap.getFirstId2()] = true;
                        heap.removeFirst();
                    } else {
                        lx[nl] = heap.getFirstId1();
                        ly[nl] = heap.getFirstId2();
                        lw[nl] = heap.getFirst();
                        minscore = heap.getFirst();
                        used[lx[nl]+nx*ly[nl]] = true;
                        heap.removeFirst();
    
                        for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) {
                            int ngb = lx[nl]+dx + nx*(ly[nl]+dy);
                            if (mask[ngb] && !used[ngb]) {
                                if (propag[ngb]>detectionThreshold && propag[ngb]>stoppingRatio*maxpropag) {
                                    heap.addValue(propag[ngb], lx[nl]+dx, ly[nl]+dy);
                                }
                            }
                        }
                        nl++;
                        if (nl>=nx+ny) stop=true;
                    }
                }
                //System.out.print("("+nl+","+minscore+")");
                double thetaL = 0.0;
                float lengthL = 1.0f;
                float aniL = 1.0f;
                if (nl>1) {
                    // compute line parameters and angle
                    float cx = 0.0f;
                    float cy = 0.0f;
                    float cw = 0.0f;
                    
                    for (int n=0;n<nl;n++) {
                        cx += lw[n]*lx[n];
                        cy += lw[n]*ly[n];
                        cw += lw[n];
                    }
                    cx /= cw;
                    cy /= cw;
                    
                    float vxx = 0.0f;
                    float vxy = 0.0f;
                    float vyy = 0.0f;
                    
                    for (int n=0;n<nl;n++) {
                        vxx += lw[n]*(lx[n]-cx)*(lx[n]-cx);
                        vxy += lw[n]*(lx[n]-cx)*(ly[n]-cy);
                        vyy += lw[n]*(ly[n]-cy)*(ly[n]-cy);
                    }
                    vxx /= cw;
                    vyy /= cw;
                    vxy /= cw;
                    
                    // (vx,vy) is the orthogonal vector, not the direction
                    double vx = 0.0;
                    double vy = 1.0;
                    double norm = 1.0;
                    
                    thetaL = 0.0f;
                    
                    if (vxy!=0) {
                        vx = vyy-vxx + FastMath.sqrt( (vyy-vxx)*(vyy-vxx) + 4.0*vxy*vxy);
                        vy = -2.0*vxy;
                        norm = vx*vx+vy*vy;
                        
                        thetaL = FastMath.atan(vx/vy);
                    } else if (vxx==0) {
                        vx = 1.0;
                        vy = 0.0;
                        norm = 1.0;
                        
                        thetaL = FastMath.PI/2.0;
                    }
                    norm = FastMath.sqrt(norm);
                    // compute length? or simply number of voxels?
                    // (maximum projection onto main vector on both sides)
                    float minL = 0.0f;
                    float maxL = 0.0f;
                    for (int n=0;n<nl;n++) {
                        float dist = (float)( (vy*(lx[n]-cx) - vx*(ly[n]-cy))/norm);
                        if (dist>maxL) maxL = dist;
                        if (dist<minL) minL = dist;
                    }
                    lengthL = 1.0f+maxL-minL;
                                       
                    // grow the region to estimate anisotropy?
                    // use the minscore as detection threshold
                    // but only add voxels that don't increase the length
                    // not great: should happen once all lines have been found
                    // issue: can grow multiple parallel lines at once, increase
                    // shape variations -> better to fit multiple lines to a 
                    // single thick fiber
                    /*
                    heap.reset();
                    for (int n=0;n<nl;n++) {
                        for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) {
                            int ngb = lx[n]+dx + nx*(ly[n]+dy);
                            if (mask[ngb] && !used[ngb]) {
                                float dist = (float)( (vy*(lx[n]+dx-cx) - vx*(ly[n]+dy-cy))/norm);
                                float thck = (float)( (vx*(lx[n]+dx-cx) + vy*(ly[n]+dy-cy))/norm);
                                if (propag[ngb]>=minscore && dist<maxL && dist>minL && thck<maxL && thck>minL) {
                                    heap.addValue(propag[ngb], lx[n]+dx, ly[n]+dy);
                                }
                            }
                        }
                    }
                    stop = false;
                    while (heap.isNotEmpty() && !stop) {
                        lx[nl] = heap.getFirstId1();
                        ly[nl] = heap.getFirstId2();
                        heap.removeFirst();
                        if (!used[lx[nl]+ nx*ly[nl]]) {
                            used[lx[nl]+ nx*ly[nl]] = true;
    
                            for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) {
                                int ngb = lx[nl]+dx + nx*(ly[nl]+dy);
                                if (mask[ngb] && !used[ngb]) {
                                    float dist = (float)( (vy*(lx[nl]+dx-cx) - vx*(ly[nl]+dy-cy))/norm);
                                    float thck = (float)( (vx*(lx[nl]+dx-cx) + vy*(ly[nl]+dy-cy))/norm);
                                    if (propag[ngb]>=minscore && dist<maxL && dist>minL && thck<maxL && thck>minL) {
                                        heap.addValue(propag[ngb], lx[nl]+dx, ly[nl]+dy);
                                    }
                                }
                            }
                            nl++;
                            if (nl>=nx+ny) stop=true;
                        }
                    }
                    //System.out.print("["+nl+"]");
                    */
                    // compute thickness same as before
                    minL = 0.0f;
                    maxL = 0.0f;
                    for (int n=0;n<nl;n++) {
                        float dist = (float)( (vx*(lx[n]-cx) + vy*(ly[n]-cy))/norm);
                        if (dist>maxL) maxL = dist;
                        if (dist<minL) minL = dist;
                    }
                    aniL = 1.0f+maxL-minL;   

                    // compute average probability score for the entire line
                    float meanp = 0.0f;
                    for (int n=0;n<nl;n++) {
                        meanp += propag[lx[n]+nx*ly[n]]/nl;
                    }
                    
                    // Add line to detected ones
                    for (int n=0;n<nl;n++) {
                        // label with starting location id, so each get a different id
                        lines[lx[n]+nx*ly[n]] = xM+nx*yM;
                        theta[lx[n]+nx*ly[n]] = (float)(thetaL*180.0/FastMath.PI);
                        length[lx[n]+nx*ly[n]] = lengthL;
                        ani[lx[n]+nx*ly[n]] = aniL;
                        propag[lx[n]+nx*ly[n]] = meanp;
                    }
                } else {
                    // remove single point detections (artefacts)
                    propag[lx[0]+nx*ly[0]] = 0.0f;
                }
            }
		}
		if (extend) {
            // expansion to neighboring background regions through binary heap
            ordering.reset();
            for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
                int xyz = x + nx*y;
                if (lines[xyz]!=0) ordering.addValue(propag[xyz], x, y);
            }
            while (ordering.isNotEmpty()) {
                float score = ordering.getFirst();
                int x = ordering.getFirstId1();
                int y = ordering.getFirstId2();
                ordering.removeFirst();
                
                int xyz = x + nx*y;
                for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) {
                    int ngb = x+dx + nx*(y+dy);
                    if (extendRatio<0) {
                        if (mask[ngb] && lines[ngb]==0) {
                            lines[ngb] = lines[xyz];
                            theta[ngb] = theta[xyz];
                            length[ngb] = length[xyz];
                            ani[ngb] = ani[xyz];
                            propag[ngb] = score-1.0f;
                            ordering.addValue(propag[ngb], x+dx,y+dy);
                        }
                    } else {
                        if (mask[ngb] && propag[ngb]<extendRatio*score) {
                            lines[ngb] = lines[xyz];
                            theta[ngb] = theta[xyz];
                            length[ngb] = length[xyz];
                            ani[ngb] = ani[xyz];
                            propag[ngb] = extendRatio*score;
                            ordering.addValue(propag[ngb], x+dx,y+dy);
                        }
                    }
                }	
            }
        }
		// Output
		BasicInfo.displayMessage("...output inputImages\n");
		probaImage = propag;
		lineImage = lines;
		thetaImage = theta;
		aniImage = ani;
		lengthImage = length;
		
		return;
	}
	
	private final void directionFromRecursiveRidgeFilter1D(float[] img, boolean[] mask, float[] filter,byte[] direction, boolean unidirectional) {
			
			// get the tubular filter response
			float[][][] linescore = new float[nx][ny][nz];
			byte[][][] linedir = new byte[nx][ny][nz];
			float[][][] inputImage = new float[nx][ny][nz];
			//float[] filter = new float[nx*ny*nz];
			for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
				int xyz = x + nx*y + nx*ny*z;
				inputImage[x][y][z] = img[xyz];
			}
			for (int x=2;x<nx-2;x++) for (int y=2;y<ny-2;y++) for (int z=0;z<nz;z++) {
				int xyz = x + nx*y + nx*ny*z;
				filter[xyz] = 0.0f;
				if (mask[xyz] && !zeroNeighbor(img, mask, x,y,z,2)) {
					// check for zero-valued neighbors as well
					minmaxlineScore(inputImage, linescore, linedir, x,y,z, NC);
					filter[xyz] = linescore[x][y][z];
					direction[xyz] = linedir[x][y][z];
					if (filter[xyz]<0) if (unidirectional) { filter[xyz]=0; direction[xyz] = -1; } else filter[xyz]*=-1.0f;
				}
			}
			linescore = null;
			linedir = null;
			inputImage = null;
			return;
	}
	private final void probabilityFromRecursiveRidgeFilter( float[] filter, float[] shape) {
		// normalization: best is the iterParamative robust exponential (others are removed)
		int nb = 0;
		double min = 1e9;
		double max = 0.0;
		for (int x=2;x<nx-2;x++) for (int y=2;y<ny-2;y++) for (int z=0;z<nz;z++) {
			int xyz = x + nx*y + nx*ny*z;
				// keep only the proper sign
				if (filter[xyz]<=0) filter[xyz] = 0.0f;
				else {
					// fit exp only to non-zero data
					nb++;
					min = Numerics.min(filter[xyz], min);
					max = Numerics.max(filter[xyz], max);
				}
		}
		
		// robust measures? pb is the exponential is not steep enough
		double[] response = new double[nb];
		int n=0;
		for (int x=2;x<nx-2;x++) for (int y=2;y<ny-2;y++) for (int z=0;z<nz;z++) {
			int xyz = x + nx*y + nx*ny*z;
				if (filter[xyz]>0) {
					response[n] = filter[xyz];
					n++;
				}
		}
		Percentile measure = new Percentile();
		double median = measure.evaluate(response, 50.0);
		double beta = median/FastMath.log(2.0);
		
		BasicInfo.displayMessage("exponential parameter estimates: median "+median+", beta "+beta+",\n");
		
		// model the filter response as something more interesting, e.g. log-normal (removing the bg samples)
		double[] weights = new double[nb];
		for (int b=0;b<nb;b++) { 
			weights[b] = (1.0-FastMath.exp( -response[b]/beta));
			response[b] = FastMath.log(response[b]);
		}
		
		double fmean = ImageStatistics.weightedPercentile(response,weights,50.0,nb);
		
		// stdev: 50% +/- 1/2*erf(1/sqrt(2)) (~0.341344746..)
		double dev = 100.0*0.5*Erf.erf(1.0/FastMath.sqrt(2.0));
		double fdev = 0.5*(ImageStatistics.weightedPercentile(response,weights,50.0+dev,nb) - ImageStatistics.weightedPercentile(response,weights,50.0-dev,nb));
		
		BasicInfo.displayMessage("Log-normal parameter estimates: mean = "+FastMath.exp(fmean)+", stdev = "+FastMath.exp(fdev)+",\n");
		
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++){
			int xyz = x + nx*y + nx*ny*z;
			if (filter[xyz]>0) {
				double pe = FastMath.exp( -filter[xyz]/beta)/beta;
				double plg = FastMath.exp(-Numerics.square(FastMath.log(filter[xyz])-fmean)/(2.0*fdev*fdev))/FastMath.sqrt(2.0*FastMath.PI*fdev*fdev);
				shape[xyz] = (float)(plg/(plg+pe));
				//shape[xyz] = (float)(1.0-pe);
			}
		}
		return;
	}
	boolean zeroNeighbor(float[] inputImage, boolean[] mask, int x, int y, int z, int d) {
			for (int i=-d;i<=d;i++) for (int j=-d;j<=d;j++) {
				if (inputImage[x+i+nx*(y+j)+nx*ny*z]!=inputImage[x+nx*y+nx*ny*z] && i*i+j*j<=2*d*d) return false;
			}
			return true;
		}
	void minmaxlineScore(float[][][] inputImage, float[][][] line, byte[][][] dir, int x, int y, int z, int dmax) {
		float maxgrad = 0.0f; 
		float minval = 0.0f; 
		float sign = 0.0f;
		byte direction = -1;
		for (byte d=0;d<dmax;d++) {
			float val1 = 0.0f, val2 = 0.0f;
			if (d==X) {		
				val1=(inputImage[x][y][z]		-inputImage[x-1][y][z]
					 +inputImage[x][y-1][z]		-inputImage[x-1][y-1][z]
					 +inputImage[x][y+1][z]		-inputImage[x-1][y+1][z])/3.0f;
				val2=(inputImage[x][y][z]		-inputImage[x+1][y][z]
					 +inputImage[x][y-1][z]		-inputImage[x+1][y-1][z]
					 +inputImage[x][y+1][z]		-inputImage[x+1][y+1][z])/3.0f;
			} else if (d==Y) {
				val1=(inputImage[x][y][z]		-inputImage[x][y-1][z]
					 +inputImage[x-1][y][z]		-inputImage[x-1][y-1][z]
					 +inputImage[x+1][y][z]		-inputImage[x+1][y-1][z])/3.0f;
				val2=(inputImage[x][y][z]		-inputImage[x][y+1][z]
					 +inputImage[x-1][y][z]		-inputImage[x-1][y+1][z]
					 +inputImage[x+1][y][z]		-inputImage[x+1][y+1][z])/3.0f;
			} else if (d==XpY) { 			
				val1=(inputImage[x][y][z]		-inputImage[x-1][y-1][z]
					 +inputImage[x-1][y+1][z]	-inputImage[x-2][y][z]
					 +inputImage[x+1][y-1][z]	-inputImage[x][y-2][z])/3.0f;
				val2=(inputImage[x][y][z]		-inputImage[x+1][y+1][z]
					 +inputImage[x-1][y+1][z]	-inputImage[x][y+2][z]
					 +inputImage[x+1][y-1][z]	-inputImage[x+2][y][z])/3.0f;
			} else if (d==XmY) { 			
				val1=(inputImage[x][y][z]		-inputImage[x-1][y+1][z]
					 +inputImage[x-1][y-1][z]	-inputImage[x-2][y][z]
					 +inputImage[x+1][y+1][z]	-inputImage[x][y-2][z])/3.0f;
				val2=(inputImage[x][y][z]		-inputImage[x+1][y-1][z]
					 +inputImage[x-1][y-1][z]	-inputImage[x][y-2][z]
					 +inputImage[x+1][y+1][z]	-inputImage[x+2][y][z])/3.0f;
			}
			// find the strongest gradient direction, then estimate the corresponding filter response
			if (val1*val1+val2*val2>maxgrad) {
				maxgrad = val1*val1+val2*val2;
				if (val1*val1<val2*val2) minval = val1;
				else minval = val2;
				sign = val1*val2;
				direction = d;
			}
		}
		if (sign>0) {
			line[x][y][z] = minval;
			dir[x][y][z] = direction;
		} else {
			line[x][y][z] = 0.0f;
			dir[x][y][z] = -1;
		}
		return;
		
	}
	
	private final float[] directionVector(int d) {
		if (d==X) return new float[]{1.0f, 0.0f, 0.0f};
		else if (d==Y) return new float[]{0.0f, 1.0f, 0.0f};
		else if (d==XpY) return new float[]{INVSQRT2, INVSQRT2, 0.0f};
		else if (d==XmY) return new float[]{INVSQRT2, -INVSQRT2, 0.0f};
		else return new float[]{0.0f, 0.0f, 0.0f};
	}
	private final byte[] directionNeighbor(int d) {
		if (d==X) return new byte[]{1, 0, 0};
		else if (d==Y) return new byte[]{0, 1, 0};
		else if (d==XpY) return new byte[]{1, 1, 0};
		else if (d==XmY) return new byte[]{1, -1, 0};
		else return new byte[]{0, 0, 0};
	}
	
	
	private final int neighborIndex(byte d, int id) {
		int idn=id;
		
			 if (d==X) 	idn+=1; 		
		else if (d==mX)	idn-=1;
		else if (d==Y) 	idn+=nx;
		else if (d==mY)	idn-=nx;
		else if (d==XpY) 	idn+=1+nx;
		else if (d==mXpY) 	idn-=1+nx;
		else if (d==XmY) 	idn+=1-nx;	
		else if (d==mXmY)	idn-=1-nx;
		
		return idn;
	}
	private final float directionProduct(int dir, int id, float[][] imdir) {
		float dv=0.0f;
			
			 if (dir==X) dv = imdir[0][id];
		else if (dir==Y) dv = imdir[1][id];
		else if (dir==XpY) dv = imdir[0][id]/SQRT2+imdir[1][id]/SQRT2;
		else if (dir==XmY) dv = imdir[0][id]/SQRT2-imdir[1][id]/SQRT2;
		else if (dir==mX) dv = -imdir[0][id];
		else if (dir==mY) dv = -imdir[1][id];
		else if (dir==mXpY) dv = -(imdir[0][id]/SQRT2+imdir[1][id]/SQRT2);
		else if (dir==mXmY) dv = -(imdir[0][id]/SQRT2-imdir[1][id]/SQRT2);
		
		return dv;
	}	
	
	
	private final float[] probabilisticDiffusion1D(float[] proba, byte[] dir, int ngbParam, float maxdiffParam, float angle, float factor, int iterParam) {
		// mask out inputImage boundaries
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			if (x<1 || y<1 || x>=nx-1 || y>=ny-1) proba[xyz] = 0.0f;
		}
		
		// build a similarity function from aligned neighbors
		float[][] similarity = new float[ngbParam][nxyz];
		byte[][] neighbor = new byte[ngbParam][nxyz];
    	estimateSimpleDiffusionSimilarity1D(dir, proba, ngbParam, neighbor, similarity, angle);
		
		// run the diffusion process
		float[] diffused = new float[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) {
			diffused[xyz] = (float)FastMath.log(1.0f + proba[xyz]);
		}

		factor /= (float)ngbParam;
		for (int t=0;t<iterParam;t++) {
			BasicInfo.displayMessage("iterParamation "+(t+1)+": ");
			float diff = 0.0f;
			for (int xyz=0;xyz<nxyz;xyz++) if (proba[xyz]>0) {
				float prev = diffused[xyz];
				diffused[xyz] = proba[xyz];
				// weight with distance to 0 or 1
				for (int n=0;n<ngbParam;n++) {
					float ngb = diffused[neighborIndex(neighbor[n][xyz],xyz)];
					// remap neighbors?
					ngb = (float)FastMath.exp(ngb)-1.0f;
				
					// integration over the whole vessel (log version is more stable (??) )
					diffused[xyz] += factor*similarity[n][xyz]*ngb;
				}
				diffused[xyz] = (float)FastMath.log(1.0f + diffused[xyz]);
				
				if (diffused[xyz]+prev>0) {
					if (Numerics.abs(prev-diffused[xyz])/(prev+diffused[xyz])>diff) diff = Numerics.abs(prev-diffused[xyz])/(prev+diffused[xyz]);
				}
			}
			
			BasicInfo.displayMessage("diff "+diff+"\n");
			if (diff<maxdiffParam) t=iterParam;
		}
		
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int id = x + nx*y + nx*ny*z;
			if (iterParam<2) diffused[id] = Numerics.bounded((float)(FastMath.exp(diffused[id])-1.0), 0.0f, 1.0f);
			else diffused[id] = Numerics.bounded((float)(FastMath.exp(diffused[id])-1.0)/(1.0f+2.0f*factor), 0.0f, 1.0f);
		}
		return diffused;
	}

    private final void estimateSimpleDiffusionSimilarity1D(byte[] dir, float[] proba, int ngbParam, byte[][] neighbor, float[][] similarity, float factor) {
    	
    	float[][] parallelweight = new float[NC2][NC2];
		for (int d1=0;d1<NC2;d1++) for (int d2=0;d2<NC2;d2++) {
			float[] dir1 = directionVector(d1);
			float[] dir2 = directionVector(d2);
			parallelweight[d1][d2] = (float)FastMath.pow(2.0f*FastMath.asin(Numerics.abs(dir1[X]*dir2[X] + dir1[Y]*dir2[Y]))/FastMath.PI,factor);
		}
		float[] weight = new float[NC2];
		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=0;z<nz;z++) {
			int id = x+nx*y+nx*ny*z;
			if (proba[id]>0) {
				// find the N best aligned discrete directions
				for (byte d=0;d<NC2;d++) {
					int idn = neighborIndex(d,id);
					
					if (proba[idn]>0) {
						if (ngbParam==2) weight[d] = parallelweight[d][dir[id]];
						else weight[d] = parallelweight[d][dir[id]]*proba[idn];
					} else {
						weight[d] = 0.0f;
					}
				}
				byte[] ngb = Numerics.argmax(weight, ngbParam);
				for (int n=0;n<ngbParam;n++) {
					neighbor[n][id] = ngb[n];
					int idn = neighborIndex(ngb[n],id);
					if (proba[idn]>0) {
						similarity[n][id] = parallelweight[dir[id]][dir[idn]];
					} else {
						similarity[n][id] = 0.0f;
					}
				}
			}
		}
		
		return;
    }

}
