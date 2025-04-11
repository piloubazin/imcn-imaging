package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.Ngb;
import nl.uva.imcn.utilities.Numerics;
import nl.uva.imcn.utilities.BasicInfo;
import nl.uva.imcn.libraries.ImageFilters;
import nl.uva.imcn.libraries.ImageStatistics;
import nl.uva.imcn.libraries.ObjectExtraction;
import nl.uva.imcn.libraries.ObjectLabeling;
import nl.uva.imcn.structures.BinaryHeap3D;
import nl.uva.imcn.structures.BinaryHeapPair;
import nl.uva.imcn.methods.VesselDiameterCostFunction;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.commons.math3.special.Erf;
//import org.apache.commons.math3.analysis.*;

import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.*;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.*;

import java.util.BitSet;

public class LinearFiberMapping3D {
	
	// jist containers
	private float[] inputImage;
	private String brightParam = "bright";
	public static final String[] brightTypes = {"bright","dark","both"};
	
	private int minscaleParam = 0;
	private int maxscaleParam = 3;
	
	private boolean maskBg = true;
	
	private float difffactorParam = 1.0f;
	private float simscaleParam = 0.1f;
	private int ngbParam = 4;
	private int iterParam = 100;
	private float maxdiffParam = 0.001f;
	private boolean skipDetect = false;
	
	private float detectionThreshold = 0.01f;
	private float maxLineDist = 1.0f;
	private boolean extend = true;
	private float stoppingRatio = 0.1f;
	private float extendRatio = 0.5f;
	
	private boolean estimateDiameter = false;
	
	private float[] probaImage;
	private int[] lineImage;
	private float[] lengthImage;
	private float[] aniImage;
	private float[] thetaImage;
	private float[] pvImage;
	private float[] diameterImage;
	
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
	public	static	final	byte	Z = 2;
	public	static 	final 	byte 	XpY = 3;
	public	static 	final 	byte 	YpZ = 4;
	public	static 	final 	byte 	ZpX = 5;
	public	static 	final 	byte 	XmY = 6;
	public	static 	final 	byte 	YmZ = 7;
	public	static 	final 	byte 	ZmX = 8;
	public	static 	final 	byte 	XpYpZ = 9;
	public	static 	final 	byte 	XmYmZ = 10;
	public	static 	final 	byte 	XmYpZ = 11;
	public	static 	final 	byte 	XpYmZ = 12;
	public	static	final	byte	mX = 13;
	public	static	final	byte	mY = 14;
	public	static	final	byte	mZ = 15;
	public	static 	final 	byte 	mXpY = 16;
	public	static 	final 	byte 	mYpZ = 17;
	public	static 	final 	byte 	mZpX = 18;
	public	static 	final 	byte 	mXmY = 19;
	public	static 	final 	byte 	mYmZ = 20;
	public	static 	final 	byte 	mZmX = 21;
	public	static 	final 	byte 	mXpYpZ = 22;
	public	static 	final 	byte 	mXmYmZ = 23;
	public	static 	final 	byte 	mXmYpZ = 24;
	public	static 	final 	byte 	mXpYmZ = 25;
	
	public	static 	final 	byte 	NC = 13;
	public	static 	final 	byte 	NC2 = 26;

	private static final boolean debug=true;
	
	//set inputs
	public final void setInputImage(float[] val) { inputImage = val; }
	public final void setRidgeIntensities(String val) { brightParam = val; }

	public final void setMaskBackground(boolean val) { maskBg = val; }
	
	public final void setMinimumScale(int val) { minscaleParam = val; }
	public final void setMaximumScale(int val) { maxscaleParam = val; }

	public final void setDiffusionFactor(float val) { difffactorParam = val; }
	public final void setSimilarityScale(float val) { simscaleParam = val; }
	public final void setNeighborhoodSize(int val) { ngbParam = val; }
	public final void setMaxIterations(int val) { iterParam = val; }
	public final void setMaxDifference(float val) { maxdiffParam = val; }
	public final void setSkipDetection(boolean val) { skipDetect = val; }
		
	public final void setDetectionThreshold(float val) { detectionThreshold = val; }
	public final void setMaxLineDistance(float val) { maxLineDist = val; }
	public final void setExtendResult(boolean val) { extend = val; }
	public final void setInclusionRatio(float val) { stoppingRatio = val; }
	public final void setExtendRatio(float val) { extendRatio = val; }
	
	public final void setEstimateDiameter(boolean val) { estimateDiameter = val; }
		
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
	public final float[] getDiameterImage() { return diameterImage;}
	public final float[] getPartialVolumeImage() { return pvImage;}

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
			if (maskBg && inputImage[id]==0) mask[id] = false;
			else mask[id] = true;
			// remove border from computations
			if (x<=1 || x>=nx-2 || y<=1 || y>=ny-2 || z<=1 || z>=nz-2) mask[id] = false;
			// normalize
			if (inputImage[id]>maxI) maxI = inputImage[id];
			if (inputImage[id]<minI) minI = inputImage[id];
		}
		
		// skip the detection step in case we already have a probability map as input
		float[] propag;
		int[] maxscale=null;
		byte[] maxdirection=null;
		if (!skipDetect) {
		    
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
            maxdirection = new byte[nxyz];
            maxscale = new int[nxyz];
            
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
                float[][] G = ImageFilters.separableGaussianKernel(scale/L2N2,scale/L2N2,scale/L2N2);
                    
                // smoothed inputImage
                smoothed = ImageFilters.separableConvolution(inputImage,nx,ny,nz,G); 
    
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
            
            // 3. diffuse the data to neighboring structures
            BasicInfo.displayMessage("...diffusion\n");
            
            propag = probabilisticDiffusion1D(proba, maxdirection, ngbParam, maxdiffParam, simscaleParam, difffactorParam, iterParam);
		} else{
		    BasicInfo.displayMessage("skip ridge detection step\n");
		    propag = inputImage;
		}

		// 4. Estimate groupings / linear fits?
		BasicInfo.displayMessage("...line groupings\n");
		// -> grow region as long as it's linear-ish and >0?
		// => estimate length, orientation and anisotropy in one go?
		int[] lines = new int[nxyz];
		float[] theta = new float[3*nxyz];
		float[] length = new float[nxyz];
		float[] ani = new float[nxyz];
		BinaryHeap3D heap = new BinaryHeap3D(nx+ny+nz, BinaryHeap3D.MAXTREE);
		BinaryHeap3D ordering = new BinaryHeap3D(nx+ny+nz, BinaryHeap3D.MAXTREE);

        // simply order them by size instead?
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x + nx*y + nx*ny*z;
			if (propag[xyz]>detectionThreshold && lines[xyz]==0) {
                ordering.addValue(propag[xyz], x, y, z);
            }
        }
        
        boolean[] used = new boolean[nx*ny*nz];
        float[] proba = new float[nx*ny*nz];
        while (ordering.isNotEmpty()) {
            float maxpropag = ordering.getFirst();
            int xM = ordering.getFirstX();
            int yM = ordering.getFirstY();
            int zM = ordering.getFirstZ();
            
            ordering.removeFirst();
            
            if (!used[xM+nx*yM+nx*ny*zM]) {
                // region grow 2: toward lower values, fitting a line
                // use probability to weight the line coeffs, to avoid end curls
                heap.reset();
                int[] lx = new int[nx+ny+nz];
                int[] ly = new int[nx+ny+nz];
                int[] lz = new int[nx+ny+nz];
                float[] lw = new float[nx+ny+nz];
                int nl = 0;
                
                // start from local maximum
                lx[nl] = xM;
                ly[nl] = yM;
                lz[nl] = zM;
                lw[nl] = maxpropag;
                used[xM+nx*yM+nx*ny*zM] = true;
                nl++;
                
                // assume a relationship between detection scale and thickness?
                // not good: makes aggregates rather than clean lines
                //double maxLineDist2 = Numerics.square(Numerics.max(1.0f,maxscale[xM+nx*yM])*maxLineDist);
                double maxLineDist2 = Numerics.square(maxLineDist);
                // instead, allow thickness increase up to length, but no further
                
                double lvx=0.0;
                double lvy=0.0;
                double lvz=0.0;
                double linedist=0.0;
                float minscore = maxpropag;
                
                for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) for (int dz=-1;dz<=1;dz++) {
                    if (xM+dx>=0 && xM+dx<nx && yM+dy>=0 && yM+dy<ny && zM+dz>=0 && zM+dz<nz) { 
                        int ngb = xM+dx + nx*(yM+dy) + nx*ny*(zM+dz);
                        if (mask[ngb] && !used[ngb]) {
                            if (propag[ngb]>detectionThreshold || propag[ngb]>stoppingRatio*maxpropag) {
                                heap.addValue(propag[ngb], xM+dx, yM+dy, zM+dz);
                            }
                        }
                    }
                }
                boolean stop = false;
                while (heap.isNotEmpty() && !stop) {
                    // fit a line to the current set of coordinates + new one
                    float score = heap.getFirst();
                    float cx = score*heap.getFirstX();
                    float cy = score*heap.getFirstY();
                    float cz = score*heap.getFirstZ();
                    float cw = score;
                    
                    for (int n=0;n<nl;n++) {
                        cx += lw[n]*lx[n];
                        cy += lw[n]*ly[n];
                        cz += lw[n]*lz[n];
                        cw += lw[n];
                    }
                    cx /= cw;
                    cy /= cw;
                    cz /= cw;
                    
                    // recurrent system to add points
                    if (nl==1) {
                        // init
                        lvx = (heap.getFirstX()-lx[0]);
                        lvy = (heap.getFirstY()-ly[0]);
                        lvz = (heap.getFirstZ()-lz[0]);
                        double lnorm = FastMath.sqrt(lvx*lvx+lvy*lvy+lvz*lvz);
                        lvx /= lnorm;
                        lvy /= lnorm;
                        lvz /= lnorm;
                    } else {
                        for (int t=0;t<10;t++) {
                            double dot = score*( (heap.getFirstX()-cx)*lvx
                                                +(heap.getFirstY()-cy)*lvy
                                                +(heap.getFirstZ()-cz)*lvz );
                            double vx = dot*(heap.getFirstX()-cx);
                            double vy = dot*(heap.getFirstY()-cy);
                            double vz = dot*(heap.getFirstZ()-cz);
                            for (int n=0;n<nl;n++) {
                                dot = lw[n]*( (lx[n]-cx)*lvx+(ly[n]-cy)*lvy+(lz[n]-cz)*lvz);
                                vx += dot*(lx[n]-cx);
                                vy += dot*(ly[n]-cy);
                                vz += dot*(lz[n]-cz);
                            }
                            double norm = FastMath.sqrt(vx*vx+vy*vy+vz*vz);
                            lvx = vx/norm;
                            lvy = vy/norm;
                            lvz = vz/norm;
                        }
                    }
                    // compute distance
                    linedist = (heap.getFirstX()-cx)*(heap.getFirstX()-cx)*(1.0-lvx*lvx)
                              +(heap.getFirstY()-cy)*(heap.getFirstY()-cx)*(1.0-lvy*lvy)
                              +(heap.getFirstZ()-cz)*(heap.getFirstZ()-cx)*(1.0-lvz*lvz);
                                 
                    for (int n=0;n<nl;n++) {
                        double dist = (lx[n]-cx)*(lx[n]-cx)*(1.0-lvx*lvx)
                                     +(ly[n]-cy)*(ly[n]-cy)*(1.0-lvy*lvy)
                                     +(lz[n]-cz)*(lz[n]-cz)*(1.0-lvz*lvz);
                        if (dist>linedist) linedist = dist;
                    }
                    if (linedist>maxLineDist2) {
                        // do not stop directly as other voxels with lower proba
                        // might still be fittingthe line
                        //stop = true;
                        // instead, skip so eventually the list is empty
                        // (mark it as used, so it doesn't get picked up again)
                        used[heap.getFirstX()+nx*heap.getFirstY()+nx*ny*heap.getFirstZ()] = true;
                        heap.removeFirst();
                    } else {
                        lx[nl] = heap.getFirstX();
                        ly[nl] = heap.getFirstY();
                        lz[nl] = heap.getFirstZ();
                        lw[nl] = heap.getFirst();
                        minscore = heap.getFirst();
                        used[lx[nl] + nx*ly[nl] + nx*ny*lz[nl]] = true;
                        heap.removeFirst();
    
                        for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) for (int dz=-1;dz<=1;dz++) {
                            if (lx[nl]+dx>=0 && lx[nl]+dx<nx && ly[nl]+dy>=0 && ly[nl]+dy<ny && lz[nl]+dz>=0 && lz[nl]+dz<nz) { 
                                int ngb = lx[nl]+dx + nx*(ly[nl]+dy) + nx*ny*(lz[nl]+dz);
                                if (mask[ngb] && !used[ngb]) {
                                    if (propag[ngb]>detectionThreshold || propag[ngb]>stoppingRatio*maxpropag) {
                                        heap.addValue(propag[ngb], lx[nl]+dx, ly[nl]+dy, lz[nl]+dz);
                                    }
                                }
                            }
                        }
                        nl++;
                        if (nl>=nx+ny+nz) stop=true;
                    }
                }
                //System.out.print("("+nl+","+minscore+")");
                if (nl>1) {
                    // re-compute line parameters
                    float cx = 0.0f;
                    float cy = 0.0f;
                    float cz = 0.0f;
                    float cw = 0.0f;
                    
                    for (int n=0;n<nl;n++) {
                        cx += lw[n]*lx[n];
                        cy += lw[n]*ly[n];
                        cz += lw[n]*lz[n];
                        cw += lw[n];
                    }
                    cx /= cw;
                    cy /= cw;
                    cz /= cw;
                    
                    // restart from last vector (still close to the good one, hopefully)
                    for (int t=0;t<10;t++) {
                        double vx = 0.0;
                        double vy = 0.0;
                        double vz = 0.0;
                        for (int n=0;n<nl;n++) {
                            double dot = lw[n]*( (lx[n]-cx)*lvx+(ly[n]-cy)*lvy+(lz[n]-cz)*lvz);
                            vx += dot*(lx[n]-cx);
                            vy += dot*(ly[n]-cy);
                            vz += dot*(lz[n]-cz);
                        }
                        double norm = FastMath.sqrt(vx*vx+vy*vy+vz*vz);
                        lvx = vx/norm;
                        lvy = vy/norm;
                        lvz = vz/norm;
                    }
                    
                    // compute length? or simply number of voxels?
                    // (maximum projection onto main vector on both sides)
                    float minL = 0.0f;
                    float maxL = 0.0f;
                    for (int n=0;n<nl;n++) {
                        float dist = (float)( lvx*(lx[n]-cx) + lvy*(ly[n]-cy) + lvz*(lz[n]-cz) );
                        if (dist>maxL) maxL = dist;
                        if (dist<minL) minL = dist;
                    }
                    float lengthL = 1.0f+maxL-minL;

                    // compute thickness same as before
                    double maxL2 = 0.0;
                    for (int n=0;n<nl;n++) {
                        double dist = (lx[n]-cx)*(lx[n]-cx)*(1.0-lvx*lvx)
                                     +(ly[n]-cy)*(ly[n]-cy)*(1.0-lvy*lvy)
                                     +(lz[n]-cz)*(lz[n]-cz)*(1.0-lvz*lvz);
                        if (dist>maxL2) maxL2 = dist;
                    }
                    float thickL = (float)FastMath.sqrt(maxL2);
                    
                    // compute average probability score for the entire line
                    float meanp = 0.0f;
                    for (int n=0;n<nl;n++) {
                        meanp += propag[lx[n]+nx*ly[n]+nx*ny*lz[n]]/nl;
                    }
                    
                    // Add line to detected ones
                    for (int n=0;n<nl;n++) {
                        int xyz = lx[n]+nx*ly[n]+nx*ny*lz[n];
                        // label with starting location id, so each get a different id
                        lines[xyz] = xM+nx*yM+nx*ny*zM;
                        theta[xyz+nx*ny*nz*X] = (float)lvx;
                        theta[xyz+nx*ny*nz*Y] = (float)lvy;
                        theta[xyz+nx*ny*nz*Z] = (float)lvz;
                        length[xyz] = lengthL;
                        ani[xyz] = 1.0f-thickL/lengthL;
                        proba[xyz] = meanp;
                    }
                } else {
                    // remove single point detections (artefacts)
                    propag[lx[0]+nx*ly[0]+nx*ny*lz[0]] = 0.0f;
                }
            }
		}
		if (estimateDiameter) {
		    // only look at points that are keptlines
		    //boolean[] obj = ObjectExtraction.objectFromImage(proba, nx,ny,nz, 0.0f, ObjectExtraction.SUPERIOR);
		
		    //estimateDiameter(inputImage, obj, maxscale, maxdirection, mask);    
		    probaImage = propag;
		    growPartialVolume(inputImage, lines, mask, detectionThreshold);
		}
		
		if (extend) {
            BasicInfo.displayMessage("...spatial extension\n");
		    // expansion to neighboring background regions through binary heap
            ordering.reset();
            for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                int xyz = x + nx*y + nx*ny*z;
                if (lines[xyz]!=0) ordering.addValue(proba[xyz], x, y, z);
            }
            while (ordering.isNotEmpty()) {
                float score = ordering.getFirst();
                int x = ordering.getFirstX();
                int y = ordering.getFirstY();
                int z = ordering.getFirstZ();
                ordering.removeFirst();
                
                int xyz = x + nx*y + nx*ny*z;
                for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) for (int dz=-1;dz<=1;dz++) {
                    if (x+dx>=0 && x+dx<nx && y+dy>=0 && y+dy<ny && z+dz>=0 && z+dz<nz) { 
                        int ngb = x+dx + nx*(y+dy) + nx*ny*(z+dz);
                        if (extendRatio<=0) {
                            if (mask[ngb] && lines[ngb]==0) {
                                lines[ngb] = lines[xyz];
                                theta[ngb+nx*ny*nz*X] = theta[xyz+nx*ny*nz*X];
                                theta[ngb+nx*ny*nz*Y] = theta[xyz+nx*ny*nz*Y];
                                theta[ngb+nx*ny*nz*Z] = theta[xyz+nx*ny*nz*Z];
                                length[ngb] = length[xyz];
                                ani[ngb] = ani[xyz];
                                proba[ngb] = score-1.0f;
                                ordering.addValue(proba[ngb], x+dx,y+dy,z+dz);
                            }
                        } else {
                            if (mask[ngb] && lines[ngb]==0 && proba[ngb]>extendRatio*score) {
                                lines[ngb] = lines[xyz];
                                theta[ngb+nx*ny*nz*X] = theta[xyz+nx*ny*nz*X];
                                theta[ngb+nx*ny*nz*Y] = theta[xyz+nx*ny*nz*Y];
                                theta[ngb+nx*ny*nz*Z] = theta[xyz+nx*ny*nz*Z];
                                length[ngb] = length[xyz];
                                ani[ngb] = ani[xyz];
                                proba[ngb] = extendRatio*score;
                                ordering.addValue(proba[ngb], x+dx,y+dy,z+dz);
                            }
                        }
                    }
                }	
            }
        }
		// Output
		BasicInfo.displayMessage("...output inputImages\n");
		probaImage = proba;
		lineImage = lines;
		thetaImage = theta;
		aniImage = ani;
		lengthImage = length;
		
		return;
	}
	
	private final void directionFromRecursiveRidgeFilter1D(float[] img, boolean[] mask, float[] filter, byte[] direction, boolean unidirectional) {
			
			// get the tubular filter response
			float[][][] planescore = new float[nx][ny][nz];
			byte[][][] planedir = new byte[nx][ny][nz];
			byte[][][] linedir = new byte[nx][ny][nz];
			float[][][] inputImage = new float[nx][ny][nz];
			//float[] filter = new float[nx*ny*nz];
			for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
				int xyz = x + nx*y + nx*ny*z;
				inputImage[x][y][z] = img[xyz];
			}
			for (int x=2;x<nx-2;x++) for (int y=2;y<ny-2;y++) for (int z=2;z<nz-2;z++) {
				int xyz = x + nx*y + nx*ny*z;
				filter[xyz] = 0.0f;
				if (mask[xyz] && !zeroNeighbor(img, mask, x,y,z,2)) {
					// check for zero-valued neighbors as well
					minmaxplaneScore(inputImage, planescore, planedir, x,y,z, 13);
					// sign issue: remove all that have different sign, keep global sign
					float linescore = minmaxlineScore(inputImage, planedir, linedir, x,y,z, 4);
					if (planescore[x][y][z]*linescore>0) {
						filter[xyz] = Numerics.sign(linescore)*Numerics.sqrt(planescore[x][y][z]*linescore);
						direction[xyz] = linedir[x][y][z];
						if(filter[xyz]<0) if (unidirectional) { filter[xyz]=0; direction[xyz] = -1; } else filter[xyz]*=-1.0f;
					} else {
						filter[xyz] = 0.0f;
						direction[xyz] = -1;
					}
				}
			}
			planescore = null;
			planedir = null;
			linedir = null;
			inputImage = null;
			return;
	}
	private final void probabilityFromRecursiveRidgeFilter( float[] filter, float[] shape) {
		// normalization: best is the iterParamative robust exponential (others are removed)
		int nb = 0;
		double min = 1e9;
		double max = 0.0;
		for (int x=2;x<nx-2;x++) for (int y=2;y<ny-2;y++) for (int z=2;z<nz-2;z++) {
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
		for (int x=2;x<nx-2;x++) for (int y=2;y<ny-2;y++) for (int z=2;z<nz-2;z++) {
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
			for (int i=-d;i<=d;i++) for (int j=-d;j<=d;j++) for (int l=-d;l<=d;l++) {
				if (inputImage[x+i+nx*(y+j)+nx*ny*(z+l)]!=inputImage[x+nx*y+nx*ny*z] && i*i+j*j+l*l<=2*d*d) return false;
			}
			return true;
		}
	void minmaxplaneScore(float[][][] inputImage, float[][][] plane, byte[][][] dir, int x, int y, int z, int dmax) {
		float maxgrad = 0.0f; 
		float minval = 0.0f; 
		float sign = 0.0f;
		byte direction = -1;
		for (byte d=0;d<dmax;d++) {
			float val1 = 0.0f, val2 = 0.0f;
			if (d==0) {		
				val1=(inputImage[x][y][z]		-inputImage[x-1][y][z]
					 +inputImage[x][y-1][z]		-inputImage[x-1][y-1][z]
					 +inputImage[x][y+1][z]		-inputImage[x-1][y+1][z]
					 +inputImage[x][y][z-1]		-inputImage[x-1][y][z-1]
					 +inputImage[x][y][z+1]		-inputImage[x-1][y][z+1]
					 +inputImage[x][y-1][z-1]	-inputImage[x-1][y-1][z-1]
					 +inputImage[x][y-1][z+1]	-inputImage[x-1][y-1][z+1]
					 +inputImage[x][y+1][z-1]	-inputImage[x-1][y+1][z-1]
					 +inputImage[x][y+1][z+1]	-inputImage[x-1][y+1][z+1])/9.0f;
				val2=(inputImage[x][y][z]		-inputImage[x+1][y][z]
					 +inputImage[x][y-1][z]		-inputImage[x+1][y-1][z]
					 +inputImage[x][y+1][z]		-inputImage[x+1][y+1][z]
					 +inputImage[x][y][z-1]		-inputImage[x+1][y][z-1]
					 +inputImage[x][y][z+1]		-inputImage[x+1][y][z+1]
					 +inputImage[x][y-1][z-1]	-inputImage[x+1][y-1][z-1]
					 +inputImage[x][y-1][z+1]	-inputImage[x+1][y-1][z+1]
					 +inputImage[x][y+1][z-1]	-inputImage[x+1][y+1][z-1]
					 +inputImage[x][y+1][z+1]	-inputImage[x+1][y+1][z+1])/9.0f;
			} else if (d==1) {
				val1=(inputImage[x][y][z]		-inputImage[x][y-1][z]
					 +inputImage[x-1][y][z]		-inputImage[x-1][y-1][z]
					 +inputImage[x+1][y][z]		-inputImage[x+1][y-1][z]	
					 +inputImage[x][y][z-1]		-inputImage[x][y-1][z-1]	
					 +inputImage[x][y][z+1]		-inputImage[x][y-1][z+1]
					 +inputImage[x-1][y][z-1]	-inputImage[x-1][y-1][z-1]
					 +inputImage[x-1][y][z+1]	-inputImage[x-1][y-1][z+1]
					 +inputImage[x+1][y][z-1]	-inputImage[x+1][y-1][z-1]
					 +inputImage[x+1][y][z+1]	-inputImage[x+1][y-1][z+1])/9.0f;
				val2=(inputImage[x][y][z]		-inputImage[x][y+1][z]
					 +inputImage[x-1][y][z]		-inputImage[x-1][y+1][z]
					 +inputImage[x+1][y][z]		-inputImage[x+1][y+1][z]
					 +inputImage[x][y][z-1]		-inputImage[x][y+1][z-1]
					 +inputImage[x][y][z+1]		-inputImage[x][y+1][z+1]
					 +inputImage[x-1][y][z-1]	-inputImage[x-1][y+1][z-1]
					 +inputImage[x-1][y][z+1]	-inputImage[x-1][y+1][z+1]
					 +inputImage[x+1][y][z-1]	-inputImage[x+1][y+1][z-1]
					 +inputImage[x+1][y][z+1]	-inputImage[x+1][y+1][z+1])/9.0f;
			} else if (d==2) { 			
				val1=(inputImage[x][y][z]		-inputImage[x][y][z-1]
					 +inputImage[x-1][y][z]		-inputImage[x-1][y][z-1]
					 +inputImage[x+1][y][z]		-inputImage[x+1][y][z-1]
					 +inputImage[x][y-1][z]		-inputImage[x][y-1][z-1]
					 +inputImage[x][y+1][z]		-inputImage[x][y+1][z-1]	
					 +inputImage[x-1][y-1][z]	-inputImage[x-1][y-1][z-1]
					 +inputImage[x-1][y+1][z]	-inputImage[x-1][y+1][z-1]
					 +inputImage[x+1][y-1][z]	-inputImage[x+1][y-1][z-1]
					 +inputImage[x+1][y+1][z]	-inputImage[x+1][y+1][z-1])/9.0f;
				val2=(inputImage[x][y][z]		-inputImage[x][y][z+1]
					 +inputImage[x-1][y][z]		-inputImage[x-1][y][z+1]
					 +inputImage[x+1][y][z]		-inputImage[x+1][y][z+1]
					 +inputImage[x][y-1][z]		-inputImage[x][y-1][z+1]
					 +inputImage[x][y+1][z]		-inputImage[x][y+1][z+1]
					 +inputImage[x-1][y-1][z]	-inputImage[x-1][y-1][z+1]
					 +inputImage[x-1][y+1][z]	-inputImage[x-1][y+1][z+1]
					 +inputImage[x+1][y-1][z]	-inputImage[x+1][y-1][z+1]
					 +inputImage[x+1][y+1][z]	-inputImage[x+1][y+1][z+1])/9.0f;
			} else if (d==3) { 			
				val1=(inputImage[x][y][z]		-inputImage[x-1][y-1][z]
					 +inputImage[x-1][y+1][z]	-inputImage[x-2][y][z]
					 +inputImage[x+1][y-1][z]	-inputImage[x][y-2][z]
					 +inputImage[x][y][z-1]		-inputImage[x-1][y-1][z-1]
					 +inputImage[x-1][y+1][z-1]	-inputImage[x-2][y][z-1]
					 +inputImage[x+1][y-1][z-1]	-inputImage[x][y-2][z-1]
					 +inputImage[x][y][z+1]		-inputImage[x-1][y-1][z+1]
					 +inputImage[x-1][y+1][z+1]	-inputImage[x-2][y][z+1]
					 +inputImage[x+1][y-1][z+1]	-inputImage[x][y-2][z+1])/9.0f;
				val2=(inputImage[x][y][z]		-inputImage[x+1][y+1][z]
					 +inputImage[x-1][y+1][z]	-inputImage[x][y+2][z]
					 +inputImage[x+1][y-1][z]	-inputImage[x+2][y][z]
					 +inputImage[x][y][z-1]		-inputImage[x+1][y+1][z-1]
					 +inputImage[x-1][y+1][z-1]	-inputImage[x][y+2][z-1]
					 +inputImage[x+1][y-1][z-1]	-inputImage[x+2][y][z-1]
					 +inputImage[x][y][z+1]		-inputImage[x+1][y+1][z+1]
					 +inputImage[x-1][y+1][z+1]	-inputImage[x][y+2][z+1]
					 +inputImage[x+1][y-1][z+1]	-inputImage[x+2][y][z+1])/9.0f;
			} else if (d==4) { 			
				val1=(inputImage[x][y][z]		-inputImage[x][y-1][z-1]
					 +inputImage[x][y+1][z-1]	-inputImage[x][y][z-2]
					 +inputImage[x][y-1][z+1]	-inputImage[x][y-2][z]
					 +inputImage[x-1][y][z]		-inputImage[x-1][y-1][z-1]
					 +inputImage[x-1][y+1][z-1]-inputImage[x-1][y][z-2]
					 +inputImage[x-1][y-1][z+1]-inputImage[x-1][y-2][z]
					 +inputImage[x+1][y][z]		-inputImage[x+1][y-1][z-1]
					 +inputImage[x+1][y+1][z-1]-inputImage[x+1][y][z-2]
					 +inputImage[x+1][y-1][z+1]-inputImage[x+1][y-2][z])/9.0f;
				val2=(inputImage[x][y][z]		-inputImage[x][y+1][z+1]
					 +inputImage[x][y+1][z-1]	-inputImage[x][y+2][z]
					 +inputImage[x][y-1][z+1]	-inputImage[x][y][z+2]
					 +inputImage[x-1][y][z]		-inputImage[x-1][y+1][z+1]
					 +inputImage[x-1][y+1][z-1]	-inputImage[x-1][y+2][z]
					 +inputImage[x-1][y-1][z+1]	-inputImage[x-1][y][z+2]
					 +inputImage[x+1][y][z]		-inputImage[x+1][y+1][z+1]
					 +inputImage[x+1][y+1][z-1]	-inputImage[x+1][y+2][z]
					 +inputImage[x+1][y-1][z+1]	-inputImage[x+1][y][z+2])/9.0f;
			} else if (d==5) { 			
				val1=(inputImage[x][y][z]		-inputImage[x-1][y][z-1]
					 +inputImage[x+1][y][z-1]	-inputImage[x][y][z-2]
					 +inputImage[x-1][y][z+1]	-inputImage[x-2][y][z]
					 +inputImage[x][y-1][z]		-inputImage[x-1][y-1][z-1]
					 +inputImage[x+1][y-1][z-1]-inputImage[x][y-1][z-2]
					 +inputImage[x-1][y-1][z+1]-inputImage[x-2][y-1][z]
					 +inputImage[x][y+1][z]		-inputImage[x-1][y+1][z-1]
					 +inputImage[x+1][y+1][z-1]-inputImage[x][y+1][z-2]
					 +inputImage[x-1][y+1][z+1]-inputImage[x-2][y+1][z])/9.0f;
				val2=(inputImage[x][y][z]		-inputImage[x+1][y][z+1]
					 +inputImage[x+1][y][z-1]	-inputImage[x+2][y][z]
					 +inputImage[x-1][y][z+1]	-inputImage[x][y][z+2]
					 +inputImage[x][y-1][z]		-inputImage[x+1][y-1][z+1]
					 +inputImage[x+1][y-1][z-1]	-inputImage[x+2][y-1][z]
					 +inputImage[x-1][y-1][z+1]	-inputImage[x][y-1][z+2]
					 +inputImage[x][y+1][z]		-inputImage[x+1][y+1][z+1]
					 +inputImage[x+1][y+1][z-1]	-inputImage[x+2][y+1][z]
					 +inputImage[x-1][y+1][z+1]	-inputImage[x][y+1][z+2])/9.0f;
			} else if (d==6) { 			
				val1=(inputImage[x][y][z]		-inputImage[x-1][y+1][z]
					 +inputImage[x-1][y-1][z]	-inputImage[x-2][y][z]
					 +inputImage[x+1][y+1][z]	-inputImage[x][y-2][z]
					 +inputImage[x][y][z-1]		-inputImage[x-1][y+1][z-1]
					 +inputImage[x-1][y-1][z-1]-inputImage[x-2][y][z-1]
					 +inputImage[x+1][y+1][z-1]-inputImage[x][y-2][z-1]
					 +inputImage[x][y][z+1]		-inputImage[x-1][y+1][z+1]
					 +inputImage[x-1][y-1][z+1]-inputImage[x-2][y][z+1]
					 +inputImage[x+1][y+1][z+1]-inputImage[x][y-2][z+1])/9.0f;
				val2=(inputImage[x][y][z]		-inputImage[x+1][y-1][z]
					 +inputImage[x-1][y-1][z]	-inputImage[x][y-2][z]
					 +inputImage[x+1][y+1][z]	-inputImage[x+2][y][z]
					 +inputImage[x][y][z-1]		-inputImage[x+1][y-1][z-1]
					 +inputImage[x-1][y-1][z-1]	-inputImage[x][y-2][z-1]
					 +inputImage[x+1][y+1][z-1]	-inputImage[x+2][y][z-1]
					 +inputImage[x][y][z+1]		-inputImage[x+1][y-1][z+1]
					 +inputImage[x-1][y-1][z+1]	-inputImage[x][y-2][z+1]
					 +inputImage[x+1][y+1][z+1]	-inputImage[x+2][y][z+1])/9.0f;
			} else if (d==7) { 			
				val1=(inputImage[x][y][z]		-inputImage[x][y-1][z+1]
					 +inputImage[x][y-1][z-1]	-inputImage[x][y-2][z]
					 +inputImage[x][y+1][z+1]	-inputImage[x][y][z+2]
					 +inputImage[x-1][y][z]		-inputImage[x-1][y-1][z+1]
					 +inputImage[x-1][y-1][z-1]	-inputImage[x-1][y-2][z]
					 +inputImage[x-1][y+1][z+1]	-inputImage[x-1][y][z+2]
					 +inputImage[x+1][y][z]		-inputImage[x+1][y-1][z+1]
					 +inputImage[x+1][y-1][z-1]	-inputImage[x+1][y-2][z]
					 +inputImage[x+1][y+1][z+1]	-inputImage[x+1][y][z+2])/9.0f;
				val2=(inputImage[x][y][z]		-inputImage[x][y+1][z-1]
					 +inputImage[x][y-1][z-1]	-inputImage[x][y][z-2]
					 +inputImage[x][y+1][z+1]	-inputImage[x][y+2][z]
					 +inputImage[x-1][y][z]		-inputImage[x-1][y+1][z-1]
					 +inputImage[x-1][y-1][z-1]	-inputImage[x-1][y][z-2]
					 +inputImage[x-1][y+1][z+1]	-inputImage[x-1][y+2][z]
					 +inputImage[x+1][y][z]		-inputImage[x+1][y+1][z-1]
					 +inputImage[x+1][y-1][z-1]	-inputImage[x+1][y][z-2]
					 +inputImage[x+1][y+1][z+1]	-inputImage[x+1][y+2][z])/9.0f;
			} else if (d==8) { 			
				val1=(inputImage[x][y][z]		-inputImage[x-1][y][z+1]
					 +inputImage[x-1][y][z-1]	-inputImage[x-2][y][z]
					 +inputImage[x+1][y][z+1]	-inputImage[x][y][z+2]
					 +inputImage[x][y-1][z]		-inputImage[x-1][y-1][z+1]
					 +inputImage[x-1][y-1][z-1]-inputImage[x-2][y-1][z]
					 +inputImage[x+1][y-1][z+1]-inputImage[x][y-1][z+2]
					 +inputImage[x][y+1][z]		-inputImage[x-1][y+1][z+1]
					 +inputImage[x-1][y+1][z-1]-inputImage[x-2][y+1][z]
					 +inputImage[x+1][y+1][z+1]-inputImage[x][y+1][z+2])/9.0f;
				val2=(inputImage[x][y][z]		-inputImage[x+1][y][z-1]
					 +inputImage[x-1][y][z-1]	-inputImage[x][y][z-2]
					 +inputImage[x+1][y][z+1]	-inputImage[x+2][y][z]
					 +inputImage[x][y-1][z]		-inputImage[x+1][y-1][z-1]
					 +inputImage[x-1][y-1][z-1]	-inputImage[x][y-1][z-2]
					 +inputImage[x+1][y-1][z+1]	-inputImage[x+2][y-1][z]
					 +inputImage[x][y+1][z]		-inputImage[x+1][y+1][z-1]
					 +inputImage[x-1][y+1][z-1]	-inputImage[x][y+1][z-2]
					 +inputImage[x+1][y+1][z+1]	-inputImage[x+2][y+1][z])/9.0f;
			} else if (d==9) { 			
				val1=(inputImage[x][y][z]		-inputImage[x-1][y-1][z-1]
					 +inputImage[x-1][y-1][z+1]	-inputImage[x-2][y-2][z]
					 +inputImage[x-1][y+1][z-1]	-inputImage[x-2][y][z-2]
					 +inputImage[x+1][y-1][z-1]	-inputImage[x][y-2][z-2]
					 +inputImage[x+1][y+1][z-1]	-inputImage[x][y][z-2]
					 +inputImage[x-1][y+1][z+1]	-inputImage[x-2][y][z]
					 +inputImage[x+1][y-1][z+1]	-inputImage[x][y-2][z])/7.0f;
				val2=(inputImage[x][y][z]		-inputImage[x+1][y+1][z+1]
					 +inputImage[x-1][y-1][z+1]	-inputImage[x][y][z+2]
					 +inputImage[x-1][y+1][z-1]	-inputImage[x][y+2][z]
					 +inputImage[x+1][y-1][z-1]	-inputImage[x+2][y][z]
					 +inputImage[x+1][y+1][z-1]	-inputImage[x+2][y+2][z]
					 +inputImage[x-1][y+1][z+1]	-inputImage[x][y+2][z+2]
					 +inputImage[x+1][y-1][z+1]	-inputImage[x+2][y][z+2])/7.0f;
			} else if (d==10) { 			
				val1=(inputImage[x][y][z]		-inputImage[x+1][y-1][z-1]
					 +inputImage[x+1][y-1][z+1]	-inputImage[x+2][y-2][z]
					 +inputImage[x+1][y+1][z-1]	-inputImage[x+2][y][z-2]
					 +inputImage[x-1][y-1][z-1]	-inputImage[x][y-2][z-2]
					 +inputImage[x-1][y+1][z-1]	-inputImage[x][y][z-2]
					 +inputImage[x-1][y-1][z+1]	-inputImage[x][y-2][z]
					 +inputImage[x+1][y+1][z+1]	-inputImage[x+2][y][z])/7.0f;
				val2=(inputImage[x][y][z]		-inputImage[x-1][y+1][z+1]
					 +inputImage[x+1][y-1][z+1]	-inputImage[x][y][z+2]
					 +inputImage[x+1][y+1][z-1]	-inputImage[x][y+2][z]
					 +inputImage[x-1][y-1][z-1]	-inputImage[x-2][y][z]
					 +inputImage[x-1][y+1][z-1]	-inputImage[x-2][y+2][z]
					 +inputImage[x-1][y-1][z+1]	-inputImage[x-2][y][z+2]
					 +inputImage[x+1][y+1][z+1]	-inputImage[x][y+2][z+2])/7.0f;
			} else if (d==11) { 			
				val1=(inputImage[x][y][z]		-inputImage[x-1][y+1][z-1]
					 +inputImage[x-1][y+1][z+1]	-inputImage[x-2][y+2][z]
					 +inputImage[x+1][y+1][z-1]	-inputImage[x][y+2][z-2]
					 +inputImage[x-1][y-1][z-1]	-inputImage[x-2][y][z-2]
					 +inputImage[x+1][y-1][z-1]	-inputImage[x][y][z-2]
					 +inputImage[x-1][y-1][z+1]	-inputImage[x-2][y][z]
					 +inputImage[x+1][y+1][z+1]	-inputImage[x][y+2][z])/7.0f;
				val2=(inputImage[x][y][z]		-inputImage[x+1][y-1][z+1]
					 +inputImage[x-1][y+1][z+1]	-inputImage[x][y][z+2]
					 +inputImage[x+1][y+1][z-1]	-inputImage[x+2][y][z]
					 +inputImage[x-1][y-1][z-1]	-inputImage[x][y-2][z]
					 +inputImage[x+1][y-1][z-1]	-inputImage[x+2][y-2][z]
					 +inputImage[x-1][y-1][z+1]	-inputImage[x][y-2][z+2]
					 +inputImage[x+1][y+1][z+1]	-inputImage[x+2][y][z+2])/7.0f;
			} else if (d==12) { 			
				val1=(inputImage[x][y][z]		-inputImage[x-1][y-1][z+1]
					 +inputImage[x-1][y+1][z+1]	-inputImage[x-2][y][z+2]
					 +inputImage[x+1][y-1][z+1]	-inputImage[x][y-2][z+2]
					 +inputImage[x-1][y-1][z-1]	-inputImage[x-2][y-2][z]
					 +inputImage[x+1][y-1][z-1]	-inputImage[x][y-2][z]
					 +inputImage[x-1][y+1][z-1]	-inputImage[x-2][y][z]
					 +inputImage[x+1][y+1][z+1]	-inputImage[x][y][z+2])/7.0f;
				val2=(inputImage[x][y][z]		-inputImage[x+1][y+1][z-1]
					 +inputImage[x-1][y+1][z+1]	-inputImage[x][y+2][z]
					 +inputImage[x+1][y-1][z+1]	-inputImage[x+2][y][z]
					 +inputImage[x-1][y-1][z-1]	-inputImage[x][y][z-2]
					 +inputImage[x+1][y-1][z-1]	-inputImage[x+2][y][z-2]
					 +inputImage[x-1][y+1][z-1]	-inputImage[x][y+2][z-2]
					 +inputImage[x+1][y+1][z+1]	-inputImage[x+2][y+2][z])/7.0f;
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
			plane[x][y][z] = minval;
			dir[x][y][z] = direction;
		} else {
			plane[x][y][z] = 0.0f;
			dir[x][y][z] = -1;
		}
		return;
		
	}
	
	float minmaxlineScore(float[][][] inputImage, byte[][][] planedir, byte[][][] linedir, int x, int y, int z, int dmax) {
		float maxgrad = 0.0f; 
		float minval = 0.0f; 
		float sign = 0.0f;
		byte direction = -1;
		
		float val1 = 0.0f, val2 = 0.0f;
		if (planedir[x][y][z]==0) {
			for (byte d=0;d<dmax;d++) {
				if (d==0) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x][y-1][z]	+inputImage[x][y+1][z]
							-inputImage[x][y][z-1]	-inputImage[x][y-1][z-1]	-inputImage[x][y+1][z-1];
					val2 = 	 inputImage[x][y][z]		+inputImage[x][y-1][z]	+inputImage[x][y+1][z]
							-inputImage[x][y][z+1]	-inputImage[x][y-1][z+1]	-inputImage[x][y+1][z+1];
					direction = Y;
				} else if (d==1) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x][y][z-1]	+inputImage[x][y][z+1]
							-inputImage[x][y-1][z]	-inputImage[x][y-1][z-1]	-inputImage[x][y-1][z+1];
					val2 = 	 inputImage[x][y][z]		+inputImage[x][y][z-1]	+inputImage[x][y][z+1]
							-inputImage[x][y+1][z]	-inputImage[x][y+1][z-1]	-inputImage[x][y+1][z+1];
					direction = Z;		
				} else if (d==2) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x][y-1][z-1]	+inputImage[x][y+1][z+1]
							-inputImage[x][y-1][z+1]	-inputImage[x][y-2][z]	-inputImage[x][y][z+2];
					val2 = 	 inputImage[x][y][z]		+inputImage[x][y-1][z-1]	+inputImage[x][y+1][z+1]
							-inputImage[x][y+1][z-1]	-inputImage[x][y][z-2]	-inputImage[x][y+2][z];
					direction = YpZ;		
				} else if (d==3) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x][y-1][z+1]	+inputImage[x][y+1][z-1]
							-inputImage[x][y-1][z-1]	-inputImage[x][y-2][z]	-inputImage[x][y][z-2];
					val2 = 	 inputImage[x][y][z]		+inputImage[x][y+1][z-1]	+inputImage[x][y-1][z+1]
							-inputImage[x][y+1][z+1]	-inputImage[x][y+2][z]	-inputImage[x][y][z+2];
					direction = YmZ;		
				}
				// find the strongest gradient direction, then estimate the corresponding filter response
				if (val1*val1+val2*val2>maxgrad) {
					maxgrad = val1*val1+val2*val2;
					if (val1*val1<val2*val2) minval = val1;
					else minval = val2;
					sign = val1*val2;
					linedir[x][y][z] = direction;
				}
			}
		} else if (planedir[x][y][z]==1) {
			for (byte d=0;d<dmax;d++) {
				if (d==0) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x][y][z-1]	+inputImage[x][y][z+1]
							-inputImage[x-1][y][z]	-inputImage[x-1][y][z-1]	-inputImage[x-1][y][z+1];
					val2 = 	 inputImage[x][y][z]		+inputImage[x][y][z-1]	+inputImage[x][y][z+1]
							-inputImage[x+1][y][z]	-inputImage[x+1][y][z-1]	-inputImage[x+1][y][z+1];
					direction = Z;		
				} else if (d==1) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x-1][y][z]	+inputImage[x+1][y][z]
							-inputImage[x][y][z-1]	-inputImage[x-1][y][z-1]	-inputImage[x+1][y][z-1];
					val2 = 	 inputImage[x][y][z]		+inputImage[x-1][y][z]	+inputImage[x+1][y][z]
							-inputImage[x][y][z+1]	-inputImage[x-1][y][z+1]	-inputImage[x+1][y][z+1];
					direction = X;		
				} else if (d==2) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x-1][y][z-1]	+inputImage[x+1][y][z+1]
							-inputImage[x-1][y][z+1]	-inputImage[x-2][y][z]	-inputImage[x][y][z+2];
					val2 = 	 inputImage[x][y][z]		+inputImage[x-1][y][z-1]	+inputImage[x+1][y][z+1]
							-inputImage[x+1][y][z-1]	-inputImage[x][y][z-2]	-inputImage[x+2][y][z];
					direction = ZpX;		
				} else if (d==3) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x-1][y][z+1]	+inputImage[x+1][y][z-1]
							-inputImage[x-1][y][z-1]	-inputImage[x-2][y][z]	-inputImage[x][y][z-2];
					val2 = 	 inputImage[x][y][z]		+inputImage[x-1][y][z+1]	+inputImage[x+1][y][z-1]
							-inputImage[x+1][y][z+1]	-inputImage[x][y][z+2]	-inputImage[x+2][y][z];
					direction = ZmX;		
				}	
				// find the strongest gradient direction, then estimate the corresponding filter response
				if (val1*val1+val2*val2>maxgrad) {
					maxgrad = val1*val1+val2*val2;
					if (val1*val1<val2*val2) minval = val1;
					else minval = val2;
					sign = val1*val2;
					linedir[x][y][z] = direction;
				}
			}
		} else if (planedir[x][y][z]==2) {
			for (byte d=0;d<dmax;d++) {
				if (d==0) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x-1][y][z]	+inputImage[x+1][y][z]
							-inputImage[x][y-1][z]	-inputImage[x-1][y-1][z]	-inputImage[x+1][y-1][z];
					val2 = 	 inputImage[x][y][z]		+inputImage[x-1][y][z]	+inputImage[x+1][y][z]
							-inputImage[x][y+1][z]	-inputImage[x-1][y+1][z]	-inputImage[x+1][y+1][z];
					direction = X;		
				} else if (d==1) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x][y-1][z]	+inputImage[x][y+1][z]
							-inputImage[x-1][y][z]	-inputImage[x-1][y-1][z]	-inputImage[x-1][y+1][z];
					val2 = 	 inputImage[x][y][z]		+inputImage[x][y-1][z]	+inputImage[x][y+1][z]
							-inputImage[x+1][y][z]	-inputImage[x+1][y-1][z]	-inputImage[x+1][y+1][z];
					direction = Y;		
				} else if (d==2) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x-1][y-1][z]	+inputImage[x+1][y+1][z]
							-inputImage[x-1][y+1][z]	-inputImage[x-2][y][z]	-inputImage[x][y+2][z];
					val2 = 	 inputImage[x][y][z]		+inputImage[x-1][y-1][z]	+inputImage[x+1][y+1][z]
							-inputImage[x+1][y-1][z]	-inputImage[x][y-2][z]	-inputImage[x+2][y][z];
					direction = XpY;		
				} else if (d==3) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x-1][y+1][z]	+inputImage[x+1][y-1][z]
							-inputImage[x-1][y-1][z]	-inputImage[x-2][y][z]	-inputImage[x][y-2][z];
					val2 = 	 inputImage[x][y][z]		+inputImage[x-1][y+1][z]	+inputImage[x+1][y-1][z]
							-inputImage[x+1][y+1][z]	-inputImage[x][y+2][z]	-inputImage[x+2][y][z];
					direction = XmY;		
				}	
				// find the strongest gradient direction, then estimate the corresponding filter response
				if (val1*val1+val2*val2>maxgrad) {
					maxgrad = val1*val1+val2*val2;
					if (val1*val1<val2*val2) minval = val1;
					else minval = val2;
					sign = val1*val2;
					linedir[x][y][z] = direction;
				}
			}
		} else if (planedir[x][y][z]==3) {
			for (byte d=0;d<dmax;d++) {
				if (d==0) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x-1][y+1][z]		+inputImage[x+1][y-1][z]
							-inputImage[x][y][z-1]	-inputImage[x-1][y+1][z-1]	-inputImage[x+1][y-1][z-1];
					val2 = 	 inputImage[x][y][z]		+inputImage[x-1][y+1][z]		+inputImage[x+1][y-1][z]
							-inputImage[x][y][z+1]	-inputImage[x-1][y+1][z+1]	-inputImage[x+1][y-1][z+1];
					direction = XmY;		
				} else if (d==1) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x][y][z-1]		+inputImage[x][y][z+1]
							-inputImage[x-1][y+1][z]	-inputImage[x-1][y+1][z-1]	-inputImage[x-1][y+1][z+1];
					val2 = 	 inputImage[x][y][z]		+inputImage[x][y][z-1]		+inputImage[x][y][z+1]
							-inputImage[x+1][y-1][z]	-inputImage[x+1][y-1][z-1]	-inputImage[x+1][y-1][z+1];
					direction = Z;		
				} else if (d==2) {
					val1 = 	 inputImage[x][y][z]			+inputImage[x-1][y+1][z-1]	+inputImage[x+1][y-1][z+1]
							-inputImage[x-1][y+1][z+1]	-inputImage[x-2][y+2][z]		-inputImage[x][y][z+2];
					val2 = 	 inputImage[x][y][z]			+inputImage[x-1][y+1][z-1]	+inputImage[x+1][y-1][z+1]
							-inputImage[x+1][y-1][z-1]	-inputImage[x][y][z-2]		-inputImage[x+2][y-2][z];
					direction = XmYpZ;		
				} else if (d==3) {
					val1 = 	 inputImage[x][y][z]			+inputImage[x-1][y+1][z+1]	+inputImage[x+1][y-1][z-1]
							-inputImage[x-1][y+1][z-1]	-inputImage[x-2][y+2][z]		-inputImage[x][y][z-2];
					val2 = 	 inputImage[x][y][z]			+inputImage[x-1][y+1][z+1]	+inputImage[x+1][y-1][z-1]
							-inputImage[x+1][y-1][z+1]	-inputImage[x][y][z+2]		-inputImage[x+2][y-2][z];
					direction = XmYmZ;		
				}
				// find the strongest gradient direction, then estimate the corresponding filter response
				if (val1*val1+val2*val2>maxgrad) {
					maxgrad = val1*val1+val2*val2;
					if (val1*val1<val2*val2) minval = val1;
					else minval = val2;
					sign = val1*val2;
					linedir[x][y][z] = direction;
				}
			}
		} else if (planedir[x][y][z]==4) {
			for (byte d=0;d<dmax;d++) {
				if (d==0) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x][y+1][z-1]		+inputImage[x][y-1][z+1]
							-inputImage[x-1][y][z]	-inputImage[x-1][y+1][z-1]	-inputImage[x-1][y-1][z+1];
					val2 = 	 inputImage[x][y][z]		+inputImage[x][y+1][z-1]		+inputImage[x][y-1][z+1]
							-inputImage[x+1][y][z]	-inputImage[x+1][y+1][z-1]	-inputImage[x+1][y-1][z+1];
					direction = YmZ;		
				} else if (d==1) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x-1][y][z]		+inputImage[x+1][y][z]
							-inputImage[x][y+1][z-1]	-inputImage[x-1][y+1][z-1]	-inputImage[x+1][y+1][z-1];
					val2 = 	 inputImage[x][y][z]		+inputImage[x-1][y][z]		+inputImage[x+1][y][z]
							-inputImage[x][y-1][z+1]	-inputImage[x-1][y-1][z+1]	-inputImage[x+1][y-1][z+1];
					direction = X;		
				} else if (d==2) {
					val1 = 	 inputImage[x][y][z]			+inputImage[x-1][y+1][z-1]	+inputImage[x+1][y-1][z+1]
							-inputImage[x-1][y-1][z+1]	-inputImage[x-2][y][z]		-inputImage[x][y-2][z+2];
					val2 = 	 inputImage[x][y][z]			+inputImage[x-1][y+1][z-1]	+inputImage[x+1][y-1][z+1]
							-inputImage[x+1][y+1][z-1]	-inputImage[x][y+2][z-2]		-inputImage[x+2][y][z];
					direction = XmYpZ;		
				} else if (d==3) {
					val1 = 	 inputImage[x][y][z]			+inputImage[x+1][y+1][z-1]	+inputImage[x-1][y-1][z+1]
							-inputImage[x+1][y-1][z+1]	-inputImage[x+2][y][z]		-inputImage[x][y-2][z+2];
					val2 = 	 inputImage[x][y][z]			+inputImage[x+1][y+1][z-1]	+inputImage[x-1][y-1][z+1]
							-inputImage[x-1][y+1][z-1]	-inputImage[x][y+2][z-2]		-inputImage[x-2][y][z];
					direction = XpYmZ;		
				}
				// find the strongest gradient direction, then estimate the corresponding filter response
				if (val1*val1+val2*val2>maxgrad) {
					maxgrad = val1*val1+val2*val2;
					if (val1*val1<val2*val2) minval = val1;
					else minval = val2;
					sign = val1*val2;
					linedir[x][y][z] = direction;
				}
			}
		} else if (planedir[x][y][z]==5) {
			for (byte d=0;d<dmax;d++) {
				if (d==0) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x+1][y][z-1]		+inputImage[x-1][y][z+1]
							-inputImage[x][y-1][z]	-inputImage[x+1][y-1][z-1]	-inputImage[x-1][y-1][z+1];
					val2 = 	 inputImage[x][y][z]		+inputImage[x+1][y][z-1]		+inputImage[x-1][y][z+1]
							-inputImage[x][y+1][z]	-inputImage[x+1][y+1][z-1]	-inputImage[x-1][y+1][z+1];
					direction = ZmX;			
				} else if (d==1) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x][y-1][z]		+inputImage[x][y+1][z]
							-inputImage[x+1][y][z-1]	-inputImage[x+1][y-1][z-1]	-inputImage[x+1][y+1][z-1];
					val2 = 	 inputImage[x][y][z]		+inputImage[x][y-1][z]		+inputImage[x][y+1][z]
							-inputImage[x-1][y][z+1]	-inputImage[x-1][y-1][z+1]	-inputImage[x-1][y+1][z+1];
					direction = Y;		
				} else if (d==2) {
					val1 = 	 inputImage[x][y][z]			+inputImage[x+1][y-1][z-1]	+inputImage[x-1][y+1][z+1]
							-inputImage[x-1][y-1][z+1]	-inputImage[x][y-2][z]		-inputImage[x-2][y][z+2];
					val2 = 	 inputImage[x][y][z]			+inputImage[x+1][y-1][z-1]	+inputImage[x-1][y+1][z+1]
							-inputImage[x+1][y+1][z-1]	-inputImage[x+2][y][z-2]		-inputImage[x][y+2][z];
					direction = XmYmZ;		
				} else if (d==3) {
					val1 = 	 inputImage[x][y][z]			+inputImage[x+1][y+1][z-1]	+inputImage[x-1][y-1][z+1]
							-inputImage[x-1][y+1][z+1]	-inputImage[x][y+2][z]		-inputImage[x-2][y][z+2];
					val2 = 	 inputImage[x][y][z]			+inputImage[x+1][y+1][z-1]	+inputImage[x-1][y-1][z+1]
							-inputImage[x+1][y-1][z-1]	-inputImage[x+2][y][z-2]		-inputImage[x][y-2][z];
					direction = XpYmZ;		
				}
				// find the strongest gradient direction, then estimate the corresponding filter response
				if (val1*val1+val2*val2>maxgrad) {
					maxgrad = val1*val1+val2*val2;
					if (val1*val1<val2*val2) minval = val1;
					else minval = val2;
					sign = val1*val2;
					linedir[x][y][z] = direction;
				}
			}
		} else if (planedir[x][y][z]==6) {
			for (byte d=0;d<dmax;d++) {
				if (d==0) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x-1][y-1][z]		+inputImage[x+1][y+1][z]
							-inputImage[x][y][z-1]	-inputImage[x-1][y-1][z-1]	-inputImage[x+1][y+1][z-1];
					val2 = 	 inputImage[x][y][z]		+inputImage[x-1][y-1][z]		+inputImage[x+1][y+1][z]
							-inputImage[x][y][z+1]	-inputImage[x-1][y-1][z+1]	-inputImage[x+1][y+1][z+1];
					direction = XpY;		
				} else if (d==1) {
					val1 = 	 inputImage[x][y][z]		+inputImage[x][y][z-1]		+inputImage[x][y][z+1]
							-inputImage[x-1][y-1][z]	-inputImage[x-1][y-1][z-1]	-inputImage[x-1][y-1][z+1];
					val2 = 	 inputImage[x][y][z]		+inputImage[x][y][z-1]		+inputImage[x][y][z+1]
							-inputImage[x+1][y+1][z]	-inputImage[x+1][y+1][z-1]	-inputImage[x+1][y+1][z+1];
					direction = Z;		
				} else if (d==2) {
					val1 = 	 inputImage[x][y][z]			+inputImage[x-1][y-1][z-1]	+inputImage[x+1][y+1][z+1]
							-inputImage[x-1][y-1][z+1]	-inputImage[x-2][y-2][z]		-inputImage[x][y][z+2];
					val2 = 	 inputImage[x][y][z]			+inputImage[x-1][y-1][z-1]	+inputImage[x+1][y+1][z+1]
							-inputImage[x+1][y+1][z-1]	-inputImage[x][y][z-2]		-inputImage[x+2][y+2][z];
					direction = XpYpZ;		
				} else if (d==3) {
					val1 = 	 inputImage[x][y][z]			+inputImage[x-1][y-1][z+1]	+inputImage[x+1][y+1][z-1]
							-inputImage[x-1][y-1][z-1]	-inputImage[x-2][y-2][z]		-inputImage[x][y][z-2];
					val2 = 	 inputImage[x][y][z]			+inputImage[x-1][y-1][z+1]	+inputImage[x+1][y+1][z-1]
							-inputImage[x+1][y+1][z+1]	-inputImage[x][y][z+2]		-inputImage[x+2][y+2][z];
					direction = XpYmZ;		
				}
				// find the strongest gradient direction, then estimate the corresponding filter response
				if (val1*val1+val2*val2>maxgrad) {
					maxgrad = val1*val1+val2*val2;
					if (val1*val1<val2*val2) minval = val1;
					else minval = val2;
					sign = val1*val2;
					linedir[x][y][z] = direction;
				}
			}
		} else if (planedir[x][y][z]==7) {
			for (byte d=0;d<dmax;d++) {
				if (d==0) {
					val1 =	 inputImage[x][y][z]		+inputImage[x][y-1][z-1]		+inputImage[x][y+1][z+1]
							-inputImage[x-1][y][z]	-inputImage[x-1][y-1][z-1]	-inputImage[x-1][y+1][z+1];
					val2 =	 inputImage[x][y][z]		+inputImage[x][y-1][z-1]		+inputImage[x][y+1][z+1]
							-inputImage[x+1][y][z]	-inputImage[x+1][y-1][z-1]	-inputImage[x+1][y+1][z+1];
					direction = YpZ;		
				} else if (d==1) {
					val1 =	 inputImage[x][y][z]		+inputImage[x-1][y][z]		+inputImage[x+1][y][z]
							-inputImage[x][y-1][z-1]	-inputImage[x-1][y-1][z-1]	-inputImage[x+1][y-1][z-1];
					val2 =	 inputImage[x][y][z]		+inputImage[x-1][y][z]		+inputImage[x+1][y][z]
							-inputImage[x][y+1][z+1]	-inputImage[x-1][y+1][z+1]	-inputImage[x+1][y+1][z+1];
					direction = X;		
				} else if (d==2) {
					val1 =   inputImage[x][y][z]			+inputImage[x-1][y-1][z-1]	+inputImage[x+1][y+1][z+1]
							-inputImage[x+1][y-1][z-1]	-inputImage[x][y-2][z-2]		-inputImage[x+2][y][z];
					val2 =	 inputImage[x][y][z]			+inputImage[x-1][y-1][z-1]	+inputImage[x+1][y+1][z+1]
							-inputImage[x-1][y+1][z+1]	-inputImage[x-2][y][z]		-inputImage[x][y+2][z+2];
					direction = XpYpZ;		
				} else if (d==3) {
					val1 =   inputImage[x][y][z]			+inputImage[x+1][y-1][z-1]	+inputImage[x-1][y+1][z+1]
							-inputImage[x-1][y-1][z-1]	-inputImage[x][y-2][z-2]		-inputImage[x-2][y][z];
					val2 =	 inputImage[x][y][z]			+inputImage[x+1][y-1][z-1]	+inputImage[x-1][y+1][z+1]
							-inputImage[x+1][y+1][z+1]	-inputImage[x+2][y][z]		-inputImage[x][y+2][z+2];
					direction = XmYmZ;		
				}
				// find the strongest gradient direction, then estimate the corresponding filter response
				if (val1*val1+val2*val2>maxgrad) {
					maxgrad = val1*val1+val2*val2;
					if (val1*val1<val2*val2) minval = val1;
					else minval = val2;
					sign = val1*val2;
					linedir[x][y][z] = direction;
				}
			}
		} else if (planedir[x][y][z]==8) {
			for (byte d=0;d<dmax;d++) {
				if (d==0) {
					val1 =	 inputImage[x][y][z]		+inputImage[x-1][y][z-1]		+inputImage[x+1][y][z+1]
							-inputImage[x][y-1][z]	-inputImage[x-1][y-1][z-1]	-inputImage[x+1][y-1][z+1];
					val2 =	 inputImage[x][y][z]		+inputImage[x-1][y][z-1]		+inputImage[x+1][y][z+1]
							-inputImage[x][y+1][z]	-inputImage[x-1][y+1][z-1]	-inputImage[x+1][y+1][z+1];
					direction = ZpX;		
				} else if (d==1) {
					val1 =	 inputImage[x][y][z]		+inputImage[x][y-1][z]		+inputImage[x][y+1][z]
							-inputImage[x-1][y][z-1]	-inputImage[x-1][y-1][z-1]	-inputImage[x-1][y+1][z-1];
					val2 =	 inputImage[x][y][z]		+inputImage[x][y-1][z]		+inputImage[x][y+1][z]
							-inputImage[x+1][y][z+1]	-inputImage[x+1][y-1][z+1]	-inputImage[x+1][y+1][z+1];
					direction = Y;		
				} else if (d==2) {
					val1 = 	 inputImage[x][y][z]			+inputImage[x-1][y-1][z-1]	+inputImage[x+1][y+1][z+1]
							-inputImage[x-1][y+1][z-1]	-inputImage[x-2][y][z-2]		-inputImage[x][y+2][z];
					val2 = 	 inputImage[x][y][z]			+inputImage[x-1][y-1][z-1]	+inputImage[x+1][y+1][z+1]
							-inputImage[x+1][y-1][z+1]	-inputImage[x][y-2][z]		-inputImage[x+2][y][z+2];
					direction = XpYpZ;		
				} else if (d==3) {
					val1 = 	 inputImage[x][y][z]			+inputImage[x-1][y+1][z-1]	+inputImage[x+1][y-1][z+1]
							-inputImage[x-1][y-1][z-1]	-inputImage[x-2][y][z-2]		-inputImage[x][y-2][z];
					val2 = 	 inputImage[x][y][z]			+inputImage[x-1][y+1][z-1]	+inputImage[x+1][y-1][z+1]
							-inputImage[x+1][y+1][z+1]	-inputImage[x][y+2][z]		-inputImage[x+2][y][z+2];
					direction = XmYpZ;		
				}					
				// find the strongest gradient direction, then estimate the corresponding filter response
				if (val1*val1+val2*val2>maxgrad) {
					maxgrad = val1*val1+val2*val2;
					if (val1*val1<val2*val2) minval = val1;
					else minval = val2;
					sign = val1*val2;
					linedir[x][y][z] = direction;
				}
			}
		} else if (planedir[x][y][z]==9) {
			for (byte d=0;d<dmax;d++) {
				if (d==0) {
					val1 =	 inputImage[x][y][z]			+inputImage[x-1][y-1][z+1]	+inputImage[x+1][y+1][z-1]
											  -1.5f*(inputImage[x-1][y+1][z-1]	+inputImage[x-1][y+1][z+1]);
					val2 =	 inputImage[x][y][z]			+inputImage[x-1][y-1][z+1]	+inputImage[x+1][y+1][z-1]
											  -1.5f*(inputImage[x+1][y-1][z+1]	+inputImage[x+1][y-1][z-1]);
					direction = XpYmZ;						  
				} else if (d==1) {
					val1 =	 inputImage[x][y][z]			+inputImage[x-1][y+1][z-1]	+inputImage[x+1][y-1][z+1]
											  -1.5f*(inputImage[x-1][y+1][z+1]	+inputImage[x-1][y-1][z+1]);
					val2 =	 inputImage[x][y][z]			+inputImage[x-1][y+1][z-1]	+inputImage[x+1][y-1][z+1]
											  -1.5f*(inputImage[x+1][y-1][z-1]	+inputImage[x+1][y+1][z-1]);
					direction = XmYpZ;						  
				} else if (d==2) {
					val1 = 	 inputImage[x][y][z]			+inputImage[x+1][y-1][z-1]	+inputImage[x-1][y+1][z+1]
											  -1.5f*(inputImage[x-1][y-1][z+1]	+inputImage[x+1][y-1][z+1]);
					val2 = 	 inputImage[x][y][z]			+inputImage[x+1][y-1][z-1]	+inputImage[x-1][y+1][z+1]
											  -1.5f*(inputImage[x+1][y+1][z-1]	+inputImage[x-1][y+1][z-1]);
					direction = XmYmZ;						  
				}
				// find the strongest gradient direction, then estimate the corresponding filter response
				if (val1*val1+val2*val2>maxgrad) {
					maxgrad = val1*val1+val2*val2;
					if (val1*val1<val2*val2) minval = val1;
					else minval = val2;
					sign = val1*val2;
					linedir[x][y][z] = direction;
				}
			}
		} else if (planedir[x][y][z]==10) {
			for (byte d=0;d<dmax;d++) {
				if (d==0) {
					val1 =	 inputImage[x][y][z]			+inputImage[x+1][y-1][z+1]	+inputImage[x-1][y+1][z-1]
											  -1.5f*(inputImage[x+1][y+1][z-1]	+inputImage[x+1][y+1][z+1]);
					val2 =	 inputImage[x][y][z]			+inputImage[x+1][y-1][z+1]	+inputImage[x-1][y+1][z-1]
											  -1.5f*(inputImage[x-1][y-1][z+1]	+inputImage[x-1][y-1][z-1]);
					direction = XmYpZ;						  
				} else 	if (d==1) {
					val1 =	 inputImage[x][y][z]			+inputImage[x+1][y+1][z-1]	+inputImage[x-1][y-1][z+1]
											  -1.5f*(inputImage[x-1][y-1][z-1]	+inputImage[x-1][y+1][z-1]);
					val2 =	 inputImage[x][y][z]			+inputImage[x+1][y+1][z-1]	+inputImage[x-1][y-1][z+1]
											  -1.5f*(inputImage[x+1][y+1][z+1]	+inputImage[x+1][y-1][z+1]);
					direction = XpYmZ;						  
				} else 	if (d==2) {
					val1 =	 inputImage[x][y][z]			+inputImage[x+1][y+1][z+1]	+inputImage[x-1][y-1][z-1]
											  -1.5f*(inputImage[x+1][y-1][z+1]	+inputImage[x-1][y-1][z+1]);
					val2 =	 inputImage[x][y][z]			+inputImage[x+1][y+1][z+1]	+inputImage[x-1][y-1][z-1]
											  -1.5f*(inputImage[x-1][y+1][z-1]	+inputImage[x+1][y+1][z-1]);
					direction = XpYpZ;						  
				}		
				// find the strongest gradient direction, then estimate the corresponding filter response
				if (val1*val1+val2*val2>maxgrad) {
					maxgrad = val1*val1+val2*val2;
					if (val1*val1<val2*val2) minval = val1;
					else minval = val2;
					sign = val1*val2;
					linedir[x][y][z] = direction;
				}
			}			
		} else if (planedir[x][y][z]==11) {
			for (byte d=0;d<dmax;d++) {
				if (d==0) {
					val1 =	 inputImage[x][y][z]			+inputImage[x-1][y+1][z+1]	+inputImage[x+1][y-1][z-1]
											  -1.5f*(inputImage[x+1][y+1][z-1]	+inputImage[x+1][y+1][z+1]);
					val2 =	 inputImage[x][y][z]			+inputImage[x-1][y+1][z+1]	+inputImage[x+1][y-1][z-1]
											  -1.5f*(inputImage[x-1][y-1][z+1]	+inputImage[x-1][y-1][z-1]);
					direction = XmYmZ;						  
				} else if (d==1) {
					val1 =	 inputImage[x][y][z]			+inputImage[x+1][y+1][z-1]	+inputImage[x-1][y-1][z+1]
											  -1.5f*(inputImage[x-1][y-1][z-1]	+inputImage[x+1][y-1][z-1]);
					val2 =	 inputImage[x][y][z]			+inputImage[x+1][y+1][z-1]	+inputImage[x-1][y-1][z+1]
											  -1.5f*(inputImage[x+1][y+1][z+1]	+inputImage[x-1][y+1][z+1]);
					direction = XpYmZ;						  
				} else 	if (d==2) {
					val1 =	 inputImage[x][y][z]			+inputImage[x+1][y+1][z+1]	+inputImage[x-1][y-1][z-1]
											  -1.5f*(inputImage[x-1][y+1][z+1]	+inputImage[x-1][y-1][z+1]);
					val2 =	 inputImage[x][y][z]			+inputImage[x+1][y+1][z+1]	+inputImage[x-1][y-1][z-1]
											  -1.5f*(inputImage[x+1][y-1][z-1]	+inputImage[x+1][y+1][z-1]);
					direction = XpYpZ;						  
				}
				// find the strongest gradient direction, then estimate the corresponding filter response
				if (val1*val1+val2*val2>maxgrad) {
					maxgrad = val1*val1+val2*val2;
					if (val1*val1<val2*val2) minval = val1;
					else minval = val2;
					sign = val1*val2;
					linedir[x][y][z] = direction;
				}
			}
		} else if (planedir[x][y][z]==12) {
			for (byte d=0;d<dmax;d++) {
				if (d==0) {
					val1 =	 inputImage[x][y][z]			+inputImage[x-1][y+1][z+1]	+inputImage[x+1][y-1][z-1]
											  -1.5f*(inputImage[x+1][y-1][z+1]	+inputImage[x+1][y+1][z+1]);
					val2 =	 inputImage[x][y][z]			+inputImage[x-1][y+1][z+1]	+inputImage[x+1][y-1][z-1]
											  -1.5f*(inputImage[x-1][y+1][z-1]	+inputImage[x-1][y-1][z-1]);
					direction = XmYmZ;						  
				} else if (d==1) {
					val1 =	 inputImage[x][y][z]			+inputImage[x+1][y-1][z+1]	+inputImage[x-1][y+1][z-1]
											  -1.5f*(inputImage[x-1][y-1][z-1]	+inputImage[x+1][y-1][z-1]);
					val2 =	 inputImage[x][y][z]			+inputImage[x+1][y-1][z+1]	+inputImage[x-1][y+1][z-1]
											  -1.5f*(inputImage[x+1][y+1][z+1]	+inputImage[x-1][y+1][z+1]);
					direction = XmYpZ;						  
				} else if (d==2) {
					val1 =	 inputImage[x][y][z]			+inputImage[x-1][y-1][z-1]	+inputImage[x+1][y+1][z+1]
											  -1.5f*(inputImage[x-1][y+1][z+1]	+inputImage[x-1][y+1][z-1]);
					val2 =	 inputImage[x][y][z]			+inputImage[x-1][y-1][z-1]	+inputImage[x+1][y+1][z+1]
											  -1.5f*(inputImage[x-1][y+1][z+1]	+inputImage[x-1][y+1][z-1]);
					direction = XpYpZ;						  
				}
				// find the strongest gradient direction, then estimate the corresponding filter response
				if (val1*val1+val2*val2>maxgrad) {
					maxgrad = val1*val1+val2*val2;
					if (val1*val1<val2*val2) minval = val1;
					else minval = val2;
					sign = val1*val2;
					linedir[x][y][z] = direction;
				}
			}
		}
		if (sign>0) return minval;
		else return 0.0f;
	}
	void minmaxplaneRatio(float[][][] inputImage, float[][][] plane, byte[][][] dir, int x, int y, int z, int dmax) {
		float maxgrad = 0.0f; 
		float minval = 0.0f; 
		float sign = 0.0f;
		byte direction = -1;
		for (byte d=0;d<dmax;d++) {
			float val0 = 0.0f, val1 = 0.0f, val2 = 0.0f;
			if (d==0) {		
				val0=(inputImage[x][y][z]
					 +inputImage[x][y-1][z]
					 +inputImage[x][y+1][z]
					 +inputImage[x][y][z-1]
					 +inputImage[x][y][z+1]
					 +inputImage[x][y-1][z-1]
					 +inputImage[x][y-1][z+1]
					 +inputImage[x][y+1][z-1]
					 +inputImage[x][y+1][z+1])/9.0f;
				val1=(inputImage[x-1][y][z]
					 +inputImage[x-1][y-1][z]
					 +inputImage[x-1][y+1][z]
					 +inputImage[x-1][y][z-1]
					 +inputImage[x-1][y][z+1]
					 +inputImage[x-1][y-1][z-1]
					 +inputImage[x-1][y-1][z+1]
					 +inputImage[x-1][y+1][z-1]
					 +inputImage[x-1][y+1][z+1])/9.0f;
				val2=(inputImage[x+1][y][z]
					 +inputImage[x+1][y-1][z]
					 +inputImage[x+1][y+1][z]
					 +inputImage[x+1][y][z-1]
					 +inputImage[x+1][y][z+1]
					 +inputImage[x+1][y-1][z-1]
					 +inputImage[x+1][y-1][z+1]
					 +inputImage[x+1][y+1][z-1]
					 +inputImage[x+1][y+1][z+1])/9.0f;
			} else if (d==1) {
				val0=(inputImage[x][y][z]
					 +inputImage[x-1][y][z]
					 +inputImage[x+1][y][z]
					 +inputImage[x][y][z-1]
					 +inputImage[x][y][z+1]
					 +inputImage[x-1][y][z-1]
					 +inputImage[x-1][y][z+1]
					 +inputImage[x+1][y][z-1]
					 +inputImage[x+1][y][z+1])/9.0f;
				val1=(inputImage[x][y-1][z]
					 +inputImage[x-1][y-1][z]
					 +inputImage[x+1][y-1][z]	
					 +inputImage[x][y-1][z-1]	
					 +inputImage[x][y-1][z+1]
					 +inputImage[x-1][y-1][z-1]
					 +inputImage[x-1][y-1][z+1]
					 +inputImage[x+1][y-1][z-1]
					 +inputImage[x+1][y-1][z+1])/9.0f;
				val2=(inputImage[x][y+1][z]
					 +inputImage[x-1][y+1][z]
					 +inputImage[x+1][y+1][z]
					 +inputImage[x][y+1][z-1]
					 +inputImage[x][y+1][z+1]
					 +inputImage[x-1][y+1][z-1]
					 +inputImage[x-1][y+1][z+1]
					 +inputImage[x+1][y+1][z-1]
					 +inputImage[x+1][y+1][z+1])/9.0f;
			} else if (d==2) { 			
				val0=(inputImage[x][y][z]
					 +inputImage[x-1][y][z]
					 +inputImage[x+1][y][z]	
					 +inputImage[x][y-1][z]
					 +inputImage[x][y+1][z]
					 +inputImage[x-1][y-1][z]
					 +inputImage[x-1][y+1][z]
					 +inputImage[x+1][y-1][z]
					 +inputImage[x+1][y+1][z])/9.0f;
				val1=(inputImage[x][y][z-1]
					 +inputImage[x-1][y][z-1]
					 +inputImage[x+1][y][z-1]
					 +inputImage[x][y-1][z-1]
					 +inputImage[x][y+1][z-1]	
					 +inputImage[x-1][y-1][z-1]
					 +inputImage[x-1][y+1][z-1]
					 +inputImage[x+1][y-1][z-1]
					 +inputImage[x+1][y+1][z-1])/9.0f;
				val2=(inputImage[x][y][z+1]
					 +inputImage[x-1][y][z+1]
					 +inputImage[x+1][y][z+1]
					 +inputImage[x][y-1][z+1]
					 +inputImage[x][y+1][z+1]
					 +inputImage[x-1][y-1][z+1]
					 +inputImage[x-1][y+1][z+1]
					 +inputImage[x+1][y-1][z+1]
					 +inputImage[x+1][y+1][z+1])/9.0f;
			} else if (d==3) { 			
				val0=(inputImage[x][y][z]
					 +inputImage[x-1][y+1][z]
					 +inputImage[x+1][y-1][z]
					 +inputImage[x][y][z-1]
					 +inputImage[x-1][y+1][z-1]
					 +inputImage[x+1][y-1][z-1]
					 +inputImage[x][y][z+1]	
					 +inputImage[x-1][y+1][z+1]
					 +inputImage[x+1][y-1][z+1])/9.0f;
				val1=(inputImage[x-1][y-1][z]
					 +inputImage[x-2][y][z]
					 +inputImage[x][y-2][z]
					 +inputImage[x-1][y-1][z-1]
					 +inputImage[x-2][y][z-1]
					 +inputImage[x][y-2][z-1]
					 +inputImage[x-1][y-1][z+1]
					 +inputImage[x-2][y][z+1]
					 +inputImage[x][y-2][z+1])/9.0f;
				val2=(inputImage[x+1][y+1][z]
					 +inputImage[x][y+2][z]
					 +inputImage[x+2][y][z]
					 +inputImage[x+1][y+1][z-1]
					 +inputImage[x][y+2][z-1]
					 +inputImage[x+2][y][z-1]
					 +inputImage[x+1][y+1][z+1]
					 +inputImage[x][y+2][z+1]
					 +inputImage[x+2][y][z+1])/9.0f;
			} else if (d==4) { 			
				val0=(inputImage[x][y][z]
					 +inputImage[x][y+1][z-1]
					 +inputImage[x][y-1][z+1]
					 +inputImage[x-1][y][z]	
					 +inputImage[x-1][y+1][z-1]
					 +inputImage[x-1][y-1][z+1]
					 +inputImage[x+1][y][z]		
					 +inputImage[x+1][y+1][z-1]
					 +inputImage[x+1][y-1][z+1])/9.0f;
				val1=(inputImage[x][y-1][z-1]
					 +inputImage[x][y][z-2]
					 +inputImage[x][y-2][z]
					 +inputImage[x-1][y-1][z-1]
					 +inputImage[x-1][y][z-2]
					 +inputImage[x-1][y-2][z]
					 +inputImage[x+1][y-1][z-1]
					 +inputImage[x+1][y][z-2]
					 +inputImage[x+1][y-2][z])/9.0f;
				val2=(inputImage[x][y+1][z+1]
					 +inputImage[x][y+2][z]
					 +inputImage[x][y][z+2]
					 +inputImage[x-1][y+1][z+1]
					 +inputImage[x-1][y+2][z]
					 +inputImage[x-1][y][z+2]
					 +inputImage[x+1][y+1][z+1]
					 +inputImage[x+1][y+2][z]
					 +inputImage[x+1][y][z+2])/9.0f;
			} else if (d==5) { 			
				val0=(inputImage[x][y][z]
					 +inputImage[x+1][y][z-1]
					 +inputImage[x-1][y][z+1]
					 +inputImage[x][y-1][z]
					 +inputImage[x+1][y-1][z-1]
					 +inputImage[x-1][y-1][z+1]
					 +inputImage[x][y+1][z]
					 +inputImage[x+1][y+1][z-1]
					 +inputImage[x-1][y+1][z+1])/9.0f;
				val1=(inputImage[x-1][y][z-1]
					 +inputImage[x][y][z-2]
					 +inputImage[x-2][y][z]
					 +inputImage[x-1][y-1][z-1]
					 +inputImage[x][y-1][z-2]
					 +inputImage[x-2][y-1][z]
					 +inputImage[x-1][y+1][z-1]
					 +inputImage[x][y+1][z-2]
					 +inputImage[x-2][y+1][z])/9.0f;
				val2=(inputImage[x+1][y][z+1]
					 +inputImage[x+2][y][z]
					 +inputImage[x][y][z+2]
					 +inputImage[x+1][y-1][z+1]
					 +inputImage[x+2][y-1][z]
					 +inputImage[x][y-1][z+2]
					 +inputImage[x+1][y+1][z+1]
					 +inputImage[x+2][y+1][z]
					 +inputImage[x][y+1][z+2])/9.0f;
			} else if (d==6) { 			
				val0=(inputImage[x][y][z]
					 +inputImage[x-1][y-1][z]
					 +inputImage[x+1][y+1][z]
					 +inputImage[x][y][z-1]
					 +inputImage[x-1][y-1][z-1]
					 +inputImage[x+1][y+1][z-1]
					 +inputImage[x][y][z+1]
					 +inputImage[x-1][y-1][z+1]
					 +inputImage[x+1][y+1][z+1])/9.0f;
				val1=(inputImage[x-1][y+1][z]
					 +inputImage[x-2][y][z]
					 +inputImage[x][y-2][z]
					 +inputImage[x-1][y+1][z-1]
					 +inputImage[x-2][y][z-1]
					 +inputImage[x][y-2][z-1]
					 +inputImage[x-1][y+1][z+1]
					 +inputImage[x-2][y][z+1]
					 +inputImage[x][y-2][z+1])/9.0f;
				val2=(inputImage[x+1][y-1][z]
					 +inputImage[x][y-2][z]
					 +inputImage[x+2][y][z]
					 +inputImage[x+1][y-1][z-1]
					 +inputImage[x][y-2][z-1]
					 +inputImage[x+2][y][z-1]
					 +inputImage[x+1][y-1][z+1]
					 +inputImage[x][y-2][z+1]
					 +inputImage[x+2][y][z+1])/9.0f;
			} else if (d==7) { 			
				val0=(inputImage[x][y][z]
					 +inputImage[x][y-1][z-1]
					 +inputImage[x][y+1][z+1]
					 +inputImage[x-1][y][z]	
					 +inputImage[x-1][y-1][z-1]
					 +inputImage[x-1][y+1][z+1]
					 +inputImage[x+1][y][z]
					 +inputImage[x+1][y-1][z-1]
					 +inputImage[x+1][y+1][z+1])/9.0f;
				val1=(inputImage[x][y-1][z+1]
					 +inputImage[x][y-2][z]
					 +inputImage[x][y][z+2]
					 +inputImage[x-1][y-1][z+1]
					 +inputImage[x-1][y-2][z]
					 +inputImage[x-1][y][z+2]
					 +inputImage[x+1][y-1][z+1]
					 +inputImage[x+1][y-2][z]
					 +inputImage[x+1][y][z+2])/9.0f;
				val2=(inputImage[x][y+1][z-1]
					 +inputImage[x][y][z-2]
					 +inputImage[x][y+2][z]
					 +inputImage[x-1][y+1][z-1]
					 +inputImage[x-1][y][z-2]
					 +inputImage[x-1][y+2][z]
					 +inputImage[x+1][y+1][z-1]
					 +inputImage[x+1][y][z-2]
					 +inputImage[x+1][y+2][z])/9.0f;
			} else if (d==8) { 			
				val0=(inputImage[x][y][z]
					 +inputImage[x-1][y][z-1]
					 +inputImage[x+1][y][z+1]
					 +inputImage[x][y-1][z]
					 +inputImage[x-1][y-1][z-1]
					 +inputImage[x+1][y-1][z+1]
					 +inputImage[x][y+1][z]	
					 +inputImage[x-1][y+1][z-1]
					 +inputImage[x+1][y+1][z+1])/9.0f;
				val1=(inputImage[x-1][y][z+1]
					 +inputImage[x-2][y][z]
					 +inputImage[x][y][z+2]
					 +inputImage[x-1][y-1][z+1]
					 +inputImage[x-2][y-1][z]
					 +inputImage[x][y-1][z+2]
					 +inputImage[x-1][y+1][z+1]
					 +inputImage[x-2][y+1][z]
					 +inputImage[x][y+1][z+2])/9.0f;
				val2=(inputImage[x+1][y][z-1]
					 +inputImage[x][y][z-2]
					 +inputImage[x+2][y][z]
					 +inputImage[x+1][y-1][z-1]
					 +inputImage[x][y-1][z-2]
					 +inputImage[x+2][y-1][z]
					 +inputImage[x+1][y+1][z-1]
					 +inputImage[x][y+1][z-2]
					 +inputImage[x+2][y+1][z])/9.0f;
			} else if (d==9) { 			
				val0=(inputImage[x][y][z]
					 +inputImage[x-1][y-1][z+1]
					 +inputImage[x-1][y+1][z-1]
					 +inputImage[x+1][y-1][z-1]
					 +inputImage[x+1][y+1][z-1]
					 +inputImage[x-1][y+1][z+1]
					 +inputImage[x+1][y-1][z+1])/7.0f;
				val1=(inputImage[x-1][y-1][z-1]
					 +inputImage[x-2][y-2][z]
					 +inputImage[x-2][y][z-2]
					 +inputImage[x][y-2][z-2]
					 +inputImage[x][y][z-2]
					 +inputImage[x-2][y][z]
					 +inputImage[x][y-2][z])/7.0f;
				val2=(inputImage[x+1][y+1][z+1]
					 +inputImage[x][y][z+2]
					 +inputImage[x][y+2][z]
					 +inputImage[x+2][y][z]
					 +inputImage[x+2][y+2][z]
					 +inputImage[x][y+2][z+2]
					 +inputImage[x+2][y][z+2])/7.0f;
			} else if (d==10) { 			
				val0=(inputImage[x][y][z]
					 +inputImage[x+1][y-1][z+1]
					 +inputImage[x+1][y+1][z-1]
					 +inputImage[x-1][y-1][z-1]
					 +inputImage[x-1][y+1][z-1]
					 +inputImage[x-1][y-1][z+1]
					 +inputImage[x+1][y+1][z+1])/7.0f;
				val1=(inputImage[x+1][y-1][z-1]
					 +inputImage[x+2][y-2][z]
					 +inputImage[x+2][y][z-2]
					 +inputImage[x][y-2][z-2]
					 +inputImage[x][y][z-2]
					 +inputImage[x][y-2][z]
					 +inputImage[x+2][y][z])/7.0f;
				val2=(inputImage[x-1][y+1][z+1]
					 +inputImage[x][y][z+2]
					 +inputImage[x][y+2][z]
					 +inputImage[x-2][y][z]
					 +inputImage[x-2][y+2][z]
					 +inputImage[x-2][y][z+2]
					 +inputImage[x][y+2][z+2])/7.0f;
			} else if (d==11) { 			
				val0=(inputImage[x][y][z]
					 +inputImage[x-1][y+1][z+1]
					 +inputImage[x+1][y+1][z-1]
					 +inputImage[x-1][y-1][z-1]
					 +inputImage[x+1][y-1][z-1]
					 +inputImage[x-1][y-1][z+1]
					 +inputImage[x+1][y+1][z+1])/7.0f;
				val1=(inputImage[x-1][y+1][z-1]
					 +inputImage[x-2][y+2][z]
					 +inputImage[x][y+2][z-2]
					 +inputImage[x-2][y][z-2]
					 +inputImage[x][y][z-2]
					 +inputImage[x-2][y][z]
					 +inputImage[x][y+2][z])/7.0f;
				val2=(inputImage[x+1][y-1][z+1]
					 +inputImage[x][y][z+2]
					 +inputImage[x+2][y][z]
					 +inputImage[x][y-2][z]
					 +inputImage[x+2][y-2][z]
					 +inputImage[x][y-2][z+2]
					 +inputImage[x+2][y][z+2])/7.0f;
			} else if (d==12) { 			
				val0=(inputImage[x][y][z]
					 +inputImage[x-1][y+1][z+1]
					 +inputImage[x+1][y-1][z+1]
					 +inputImage[x-1][y-1][z-1]
					 +inputImage[x+1][y-1][z-1]
					 +inputImage[x-1][y+1][z-1]
					 +inputImage[x+1][y+1][z+1])/7.0f;
				val1=(inputImage[x-1][y-1][z+1]
					 +inputImage[x-2][y][z+2]
					 +inputImage[x][y-2][z+2]
					 +inputImage[x-2][y-2][z]
					 +inputImage[x][y-2][z]
					 +inputImage[x-2][y][z]
					 +inputImage[x][y][z+2])/7.0f;
				val2=(inputImage[x+1][y+1][z-1]
					 +inputImage[x][y+2][z]
					 +inputImage[x+2][y][z]
					 +inputImage[x][y][z-2]
					 +inputImage[x+2][y][z-2]
					 +inputImage[x][y+2][z-2]
					 +inputImage[x+2][y+2][z])/7.0f;
			}
			// find the strongest gradient direction, then estimate the corresponding filter response
			if ((val0-val1)*(val0-val1)+(val0-val2)*(val0-val2)>maxgrad) {
				maxgrad = (val0-val1)*(val0-val1)+(val0-val2)*(val0-val2);
				if ((val0-val1)*(val0-val1)<(val0-val2)*(val0-val2)) 
				    //minval = Numerics.sign(val0-val1)*Numerics.max(0.0f, Numerics.abs(val0-val1)-Numerics.abs(val1-val2))/(Numerics.abs(val0-val2)+Numerics.abs(val1-val2));
				    minval = Numerics.sign(val0-val1)*Numerics.max(0.0f, Numerics.abs(val0-val1)-Numerics.abs(val1-val2));
				else
				    //minval = Numerics.sign(val0-val2)*Numerics.max(0.0f, Numerics.abs(val0-val2)-Numerics.abs(val1-val2))/(Numerics.abs(val0-val1)+Numerics.abs(val1-val2));
				    minval = Numerics.sign(val0-val2)*Numerics.max(0.0f, Numerics.abs(val0-val2)-Numerics.abs(val1-val2));
				sign = (val0-val1)*(val0-val2);
				direction = d;
			}
		}
		if (sign>0) {
			plane[x][y][z] = minval;
			dir[x][y][z] = direction;
		} else {
			plane[x][y][z] = 0.0f;
			dir[x][y][z] = -1;
		}
		return;
		
	}
	
	private final float[] directionVector(int d) {
		if (d==X) return new float[]{1.0f, 0.0f, 0.0f};
		else if (d==Y) return new float[]{0.0f, 1.0f, 0.0f};
		else if (d==Z) return new float[]{0.0f, 0.0f, 1.0f};
		else if (d==XpY) return new float[]{INVSQRT2, INVSQRT2, 0.0f};
		else if (d==YpZ) return new float[]{0.0f, INVSQRT2, INVSQRT2};
		else if (d==ZpX) return new float[]{INVSQRT2, 0.0f, INVSQRT2};
		else if (d==XmY) return new float[]{INVSQRT2, -INVSQRT2, 0.0f};
		else if (d==YmZ) return new float[]{0.0f, INVSQRT2, -INVSQRT2};
		else if (d==ZmX) return new float[]{-INVSQRT2, 0.0f, INVSQRT2};
		else if (d==XpYpZ) return new float[]{INVSQRT3, INVSQRT3, INVSQRT3};
		else if (d==XmYmZ) return new float[]{INVSQRT3, -INVSQRT3, -INVSQRT3};
		else if (d==XmYpZ) return new float[]{INVSQRT3, -INVSQRT3, INVSQRT3};
		else if (d==XpYmZ) return new float[]{INVSQRT3, INVSQRT3, -INVSQRT3};
		else return new float[]{0.0f, 0.0f, 0.0f};
	}
	private final byte[] directionNeighbor(int d) {
		if (d==X) return new byte[]{1, 0, 0};
		else if (d==Y) return new byte[]{0, 1, 0};
		else if (d==Z) return new byte[]{0, 0, 1};
		else if (d==XpY) return new byte[]{1, 1, 0};
		else if (d==YpZ) return new byte[]{0, 1, 1};
		else if (d==ZpX) return new byte[]{1, 0, 1};
		else if (d==XmY) return new byte[]{1, -1, 0};
		else if (d==YmZ) return new byte[]{0, 1, -1};
		else if (d==ZmX) return new byte[]{-1, 0, 1};
		else if (d==XpYpZ) return new byte[]{1, 1, 1};
		else if (d==XmYmZ) return new byte[]{1, -1, -1};
		else if (d==XmYpZ) return new byte[]{1, -1, 1};
		else if (d==XpYmZ) return new byte[]{1, 1, -1};
		else return new byte[]{0, 0, 0};
	}
	
	
	private final int neighborIndex(byte d, int id) {
		int idn=id;
		
			 if (d==X) 	idn+=1; 		
		else if (d==mX)	idn-=1;
		else if (d==Y) 	idn+=nx;
		else if (d==mY)	idn-=nx;
		else if (d==Z) 	idn+=nx*ny;
		else if (d==mZ)	idn-=nx*ny;
		else if (d==XpY) 	idn+=1+nx;
		else if (d==mXpY) 	idn-=1+nx;
		else if (d==YpZ) 	idn+=nx+nx*nx;
		else if (d==mYpZ)	idn-=nx+nx*ny;
		else if (d==ZpX) 	idn+=1+nx*ny;	
		else if (d==mZpX)	idn-=1+nx*ny;
		else if (d==XmY) 	idn+=1-nx;	
		else if (d==mXmY)	idn-=1-nx;
		else if (d==YmZ) 	idn+=nx-nx*ny;
		else if (d==mYmZ)	idn-=nx-nx*ny;
		else if (d==ZmX) 	idn+=1-nx*ny;
		else if (d==mZmX)	idn-=1-nx*ny;
		else if (d==XpYpZ) 		idn+=1+nx+nx*ny;
		else if (d==mXpYpZ)		idn-=1+nx+nx*ny;
		else if (d==XmYmZ) 		idn+=1-nx-nx*ny; 
		else if (d==mXmYmZ)		idn-=1-nx-nx*ny;
		else if (d==XmYpZ) 		idn+=1-nx+nx*ny;
		else if (d==mXmYpZ)		idn-=1-nx+nx*ny;
		else if (d==XpYmZ) 		idn+=1+nx-nx*ny; 
		else if (d==mXpYmZ)		idn-=1+nx-nx*ny;

		return idn;
	}
	private final float directionProduct(int dir, int id, float[][] imdir) {
		float dv=0.0f;
			
			 if (dir==X) dv = imdir[0][id];
		else if (dir==Y) dv = imdir[1][id];
		else if (dir==Z) dv = imdir[2][id];
		else if (dir==XpY) dv = imdir[0][id]/SQRT2+imdir[1][id]/SQRT2;
		else if (dir==YpZ) dv = imdir[1][id]/SQRT2+imdir[2][id]/SQRT2;
		else if (dir==ZpX) dv = imdir[2][id]/SQRT2+imdir[0][id]/SQRT2;
		else if (dir==XmY) dv = imdir[0][id]/SQRT2-imdir[1][id]/SQRT2;
		else if (dir==YmZ) dv = imdir[1][id]/SQRT2-imdir[2][id]/SQRT2;
		else if (dir==ZmX) dv = imdir[2][id]/SQRT2-imdir[0][id]/SQRT2;
		else if (dir==XpYpZ) dv = imdir[0][id]/SQRT3+imdir[1][id]/SQRT3+imdir[2][id]/SQRT3;
		else if (dir==XmYpZ) dv = imdir[0][id]/SQRT3-imdir[1][id]/SQRT3+imdir[2][id]/SQRT3;
		else if (dir==XpYmZ) dv = imdir[0][id]/SQRT3+imdir[1][id]/SQRT3-imdir[2][id]/SQRT3;
		else if (dir==XmYmZ) dv = imdir[0][id]/SQRT3-imdir[1][id]/SQRT3-imdir[2][id]/SQRT3;
		else if (dir==mX) dv = -imdir[0][id];
		else if (dir==mY) dv = -imdir[1][id];
		else if (dir==mZ) dv = -imdir[2][id];
		else if (dir==mXpY) dv = -(imdir[0][id]/SQRT2+imdir[1][id]/SQRT2);
		else if (dir==mYpZ) dv = -(imdir[1][id]/SQRT2+imdir[2][id]/SQRT2);
		else if (dir==mZpX) dv = -(imdir[2][id]/SQRT2+imdir[0][id]/SQRT2);
		else if (dir==mXmY) dv = -(imdir[0][id]/SQRT2-imdir[1][id]/SQRT2);
		else if (dir==mYmZ) dv = -(imdir[1][id]/SQRT2-imdir[2][id]/SQRT2);
		else if (dir==mZmX) dv = -(imdir[2][id]/SQRT2-imdir[0][id]/SQRT2);
		else if (dir==mXpYpZ) dv = -(imdir[0][id]/SQRT3+imdir[1][id]/SQRT3+imdir[2][id]/SQRT3);
		else if (dir==mXmYpZ) dv = -(imdir[0][id]/SQRT3-imdir[1][id]/SQRT3+imdir[2][id]/SQRT3);
		else if (dir==mXpYmZ) dv = -(imdir[0][id]/SQRT3+imdir[1][id]/SQRT3-imdir[2][id]/SQRT3);
		else if (dir==mXmYmZ) dv = -(imdir[0][id]/SQRT3-imdir[1][id]/SQRT3-imdir[2][id]/SQRT3);

		return dv;
	}	
	
	
	private final float diffusionWeightFunction(float[] proba, byte[] dir, int xyz, int ngb, byte dngb, float[][] corr, float scale) {
		float diff = Numerics.abs((proba[xyz] - proba[ngb])/scale);
		float weight = 1.0f;
		if (dir[xyz]>-1 && dngb>-1) weight = corr[dir[xyz]][dngb];
		return weight/(1.0f+Numerics.square(diff/scale));
	}
	
	
	private final float[] probabilisticDiffusion1D(float[] proba, byte[] dir, int ngbParam, float maxdiffParam, float angle, float factor, int iterParam) {
		// mask out inputImage boundaries
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			if (x<1 || y<1 || z<1 || x>=nx-1 || y>=ny-1 || z>=nz-1) proba[xyz] = 0.0f;
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
			BasicInfo.displayMessage("iteration "+(t+1)+": ");
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
    	
    	float[][] parallelweight = new float[26][26];
		for (int d1=0;d1<26;d1++) for (int d2=0;d2<26;d2++) {
			float[] dir1 = directionVector(d1);
			float[] dir2 = directionVector(d2);
			parallelweight[d1][d2] = (float)FastMath.pow(2.0f*FastMath.asin(Numerics.abs(dir1[X]*dir2[X] + dir1[Y]*dir2[Y] + dir1[Z]*dir2[Z]))/FastMath.PI,factor);
		}
		float[] weight = new float[26];
		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
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
    
	private final void estimateSimpleDiffusionSimilarity2D(byte[] dir, float[] proba, int ngbParam, byte[][] neighbor, float[][] similarity, float factor) {
    	
    	float[][] parallelweight = new float[26][26];
		float[][] orthogonalweight = new float[26][26];
		for (int d1=0;d1<26;d1++) for (int d2=0;d2<26;d2++) {
			float[] dir1 = directionVector(d1);
			float[] dir2 = directionVector(d2);
			parallelweight[d1][d2] = (float)FastMath.pow(2.0f*FastMath.asin(Numerics.abs(dir1[X]*dir2[X] + dir1[Y]*dir2[Y] + dir1[Z]*dir2[Z]))/FastMath.PI,factor);
			orthogonalweight[d1][d2] = (float)FastMath.pow(2.0f*FastMath.acos(Numerics.abs(dir1[X]*dir2[X] + dir1[Y]*dir2[Y] + dir1[Z]*dir2[Z]))/FastMath.PI,factor);
		}
		float[] weight = new float[26];
		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
			int id = x+nx*y+nx*ny*z;
			if (proba[id]>0) {
				// find the N best planar discrete directions
				for (byte d=0;d<NC2;d++) {
					int idn = neighborIndex(d,id);
					
					if (proba[idn]>0) {
						if (ngbParam==2) weight[d] = orthogonalweight[d][dir[id]];
						else weight[d] = orthogonalweight[d][dir[id]]*proba[idn];
					} else {
						weight[d] = 0.0f;
					}
				}
				byte[] ngb = Numerics.argmax(weight, ngbParam);
				for (int n=0;n<ngbParam;n++) {
					neighbor[n][id] = ngb[n];
					int idn = neighborIndex(ngb[n],id);
					if (proba[idn]>0) {
						// similarity comes from the normal direction
						similarity[n][id] = parallelweight[dir[id]][dir[idn]];
					} else {
						similarity[n][id] = 0.0f;
					}
				}
			}
		}
		
		return;
    }
	

    private final void estimateDiameter(float[] image, boolean[] object, int[] scaleMax, byte[] finalDirection, boolean[] maskCort) {
		// 4. estimate diameters
		System.out.println("Vessel Intensity \n");
		// in and out vessels intensity initialisation
		float Iin=0.0f;
		int nIn=0;
		float[][][] inVal=new float[nx][ny][nz];	
		float minIn = 1e9f;
		float maxIn = -1e9f;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int id = x + nx*y + nx*ny*z;
			inVal[x][y][z]=0.0f;
			if(object[id] && scaleMax[id]>1){
				Iin+=image[id];
				inVal[x][y][z]=image[id];
				nIn++;
				if (image[id] > maxIn) maxIn = image[id];
				if (image[id] < minIn) minIn = image[id];
			}
		}
		Iin=Iin/(float)nIn;
		
		System.out.println("Ring \n");
		boolean[] maskOut=new boolean[nxyz];
		boolean[] maskVote=new boolean[nxyz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int id = x + nx*y + nx*ny*z;
			maskOut[id]=false;
			if(object[id]) maskVote[id]=true;
			else maskVote[id]=false;
		}		
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x + nx*y + nx*ny*z;
			if(object[xyz]){
				float[] finalDir=directionVector(finalDirection[xyz]);
				for(int d=0;d<13;d++){
					int rIn=3;
					int rOut =5;
					byte[] orthVector = directionNeighbor(d);					
					float orthTest = finalDir[X]*orthVector[X]+finalDir[Y]*orthVector[Y]+finalDir[Z]*orthVector[Z];
					if(orthTest*orthTest<1.0e-9f){
						float dist= (float)FastMath.sqrt( orthVector[X]*orthVector[X]+orthVector[Y]*orthVector[Y]+orthVector[Z]*orthVector[Z]);
						int s=1;
						if(scaleMax[xyz]>1){
							rIn*=2;
							rOut*=2;
						}
						while(dist<=rOut){	
							if(x+s*orthVector[X]>=0 && x+s*orthVector[X]<nx && (y+s*orthVector[Y])>=0 && (y+s*orthVector[Y])<ny && (z+s*orthVector[Z])>=0 && (z+s*orthVector[Z])<nz){
								if(dist>rIn){
									if(!object[x+s*orthVector[X]+nx*(y+s*orthVector[Y])+nx*ny*(z+s*orthVector[Z])])	maskOut[x+s*orthVector[X]+nx*(y+s*orthVector[Y])+nx*ny*(z+s*orthVector[Z])]=true;
								}
								else {
									if(!object[x+s*orthVector[X]+nx*(y+s*orthVector[Y])+nx*ny*(z+s*orthVector[Z])])	maskVote[x+s*orthVector[X]+nx*(y+s*orthVector[Y])+nx*ny*(z+s*orthVector[Z])]=true;
								}
							}
							if(x-s*orthVector[X]>=0 && x+s*orthVector[X]<nx && (y-s*orthVector[Y])>=0 && (y-s*orthVector[Y])<ny && (z-s*orthVector[Z])>=0 && (z-s*orthVector[Z])<nz){
								if(dist>rIn){
									if(!object[x-s*orthVector[X]+nx*(y-s*orthVector[Y])+nx*ny*(z-s*orthVector[Z])])	maskOut[x-s*orthVector[X]+nx*(y-s*orthVector[Y])+nx*ny*(z-s*orthVector[Z])]=true;
								}
								else {
									if(!object[x-s*orthVector[X]+nx*(y-s*orthVector[Y])+nx*ny*(z-s*orthVector[Z])])	maskVote[x-s*orthVector[X]+nx*(y-s*orthVector[Y])+nx*ny*(z-s*orthVector[Z])]=true;
								}
							}
							s++;
							dist= (float)FastMath.sqrt( orthVector[X]*orthVector[X]+orthVector[Y]*orthVector[Y]+orthVector[Z]*orthVector[Z])*(float)s;
						}
					}
				}
			}
		}
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x + nx*y + nx*ny*z;
			if(maskVote[xyz]){
				maskOut[xyz]=false;
			}
		}
				
		System.out.println("Background Intensity \n");
		float Iout=0.0f;
		int nOut=0;
		float minOut = 1e9f;
		float maxOut = -1e9f;
		float[][][] outVal=new float[nx][ny][nz];		
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int id = x + nx*y + nx*ny*z;
			outVal[x][y][z]=0.0f;
			if(maskOut[id]){
				Iout+=image[id];
				outVal[x][y][z]=image[id];
				nOut++;
				if (image[id] > maxOut) maxOut = image[id];
				if (image[id] < minOut) minOut = image[id];
			}
		}
		Iout=Iout/(float)nOut;
		
				
		// in probability map
		float[][][] probaInImg=new float[nx][ny][nz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x + nx*y + nx*ny*z;
			if(maskCort[xyz]){
				//}
				if(image[xyz]>Iin){
					probaInImg[x][y][z]=1.0f;
				}
				else if (image[xyz]<Iout){
					probaInImg[x][y][z]=0.0f;
				}
				else {
					probaInImg[x][y][z]=(image[xyz]-Iout)/(Iin-Iout);
				}
			}
			else {
				probaInImg[x][y][z]=0.0f;
			}
			
		}

		
		// Estime diameter 
			// Estime window size
		System.out.println("Diameter profile\n");
		float[][][] diamEst= new float[nx][ny][nz];
		int[][][] voisNbr= new int[nx][ny][nz];
		boolean[][] obj2=new boolean[nxyz][maxscaleParam]; 
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int id = x + nx*y + nx*ny*z;
			obj2[id][0]=false;
			if(object[id]){
				float[] finalDir=directionVector(finalDirection[id]);
				diamEst[x][y][z]=0.0f;
				int rIn=1;
				int vois=0;
				for (int i=-rIn;i<=rIn;i++) for (int j=-rIn;j<=rIn;j++) for (int k=-rIn;k<=rIn;k++){
					float dist= (float)FastMath.sqrt((float)i*i +(float)j*j +(float)k*k);
					if(dist<=rIn){
						if(x+i<nx &&  x+i>=0 && y+j<ny &&  y+j>=0 && z+k<nz &&  z+k>=0 ){
							float orthTest=(float)i*finalDir[X]+(float)j*finalDir[Y]+(float)k*finalDir[Z];
							orthTest=orthTest*orthTest;
							if(orthTest<1.0e-9f){
								vois++;
								diamEst[x][y][z]+=probaInImg[x+i][y+j][z+k]; 	
							}
						}
					}
				}
				diamEst[x][y][z]/=(float)vois; 
				voisNbr[x][y][z]=vois;
				if(diamEst[x][y][z]>0.5f) obj2[id][0]=true;
			}		          
		}
		
		float[][][][] diamEst2= new float[nx][ny][nz][maxscaleParam-1];
		for(int s=1;s<maxscaleParam;s++){
		System.out.println("Initial search "+s+"\n");
		int rIn=1+s;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int id = x + nx*y + nx*ny*z;
			obj2[id][s]=false;
			if(obj2[id][s-1]){
				float[] finalDir=directionVector(finalDirection[id]);
				diamEst2[x][y][z][s-1]=0.0f;

				int vois=0;
				for (int i=-rIn;i<=rIn;i++) for (int j=-rIn;j<=rIn;j++) for (int k=-rIn;k<=rIn;k++){
					float dist= (float)FastMath.sqrt((float)i*i +(float)j*j +(float)k*k);
					if(dist<=rIn){
						if(x+i<nx &&  x+i>=0 && y+j<ny &&  y+j>=0 && z+k<nz &&  z+k>=0 ){
							float orthTest=(float)i*finalDir[X]+(float)j*finalDir[Y]+(float)k*finalDir[Z];
							orthTest=orthTest*orthTest;
							if(orthTest<1.0e-9f){
								vois++;
								diamEst2[x][y][z][s-1]+=probaInImg[x+i][y+j][z+k]; 	
							}
						}
					}
				}
				diamEst2[x][y][z][s-1]/=(float)vois; 
				voisNbr[x][y][z]=vois;
				if(diamEst2[x][y][z][s-1]>0.5f) obj2[id][s]=true;
			}		          
		}
		}
			// Fit to model	
		double[][][] firstEst=new double[nx][ny][nz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			firstEst[x][y][z]=0.0f;
		}
		double[][][] pv=new double[nx][ny][nz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			pv[x][y][z]=0.0f;	
		}			
		System.out.println("Optimization\n");
		int wtOptim=0;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int id=x+nx*y+nx*ny*z;
			if(object[id]){
				int rIn=1;
				for(int s=1;s<maxscaleParam;s++){
					if(obj2[id][s-1] ){			
						rIn++;	
					}			
				}
				
				float[] finalDir=directionVector(finalDirection[id]);
				float[] tan1=new float[3];
			
				boolean btan=true;
				for(int d=0;d<13;d++){
					if(btan){
						tan1=directionVector(d);
						float orthTest=(float)tan1[X]*finalDir[X]+(float)tan1[Y]*finalDir[Y]+(float)tan1[Z]*finalDir[Z];
						orthTest=orthTest*orthTest;
						if(orthTest<1.0e-9f){
							btan=false;
						}
					}
				}
				float[] tan2=new float[3];
				tan2[X]=tan1[Y]*finalDir[Z]-tan1[Z]*finalDir[Y];
				tan2[Y]=tan1[Z]*finalDir[X]-tan1[X]*finalDir[Z];
				tan2[Z]=tan1[X]*finalDir[Y]-tan1[Y]*finalDir[X];
				
				//System.out.print(".");
		
				VesselDiameterCostFunction jfct= new VesselDiameterCostFunction();
				jfct.setImagesData(probaInImg,nx,ny,nz);
				jfct.setPointData(x,y,z, rIn,finalDir,  tan1, tan2);
				
				double[] init={0.0f,0.0f,(double)rIn} ;
				SimplexOptimizer optimizer= new SimplexOptimizer(1e-5,1e-10);
				MaxEval max=new MaxEval(10000);
				ObjectiveFunction g=new ObjectiveFunction(jfct);
				
				InitialGuess start=new InitialGuess(init);
				MaxIter mIt = new MaxIter(200); 
				double[] step={0.01f,0.01f,0.01f};
				NelderMeadSimplex simplex= new NelderMeadSimplex(step);
				PointValuePair resultPair=optimizer.optimize(max,g, GoalType.MINIMIZE,start,simplex);
				double[] result= resultPair.getPoint();
				
				firstEst[x][y][z]=FastMath.abs(result[2]);	
				float maxDist= (float)rIn+0.5f;
				if (firstEst[x][y][z]>maxDist) {
					firstEst[x][y][z]=maxDist;	
					wtOptim++;
				} else {				
					float xc=FastMath.round(result[0]*tan1[X]+result[1]*tan2[X]);
					float yc=FastMath.round(result[0]*tan1[Y]+result[1]*tan2[Y]);
					float zc=FastMath.round(result[0]*tan1[Z]+result[1]*tan2[Z]);
					for(int i=-rIn;i<=rIn;i++)for(int j=-rIn;j<=rIn;j++)for(int k=-rIn;k<=rIn;k++){	
						if(x+i>=0 && x+i<nx && (y+j)>=0 && (y+j)<ny && (z+k)>=0 && (z+k)<nz){
								float orthTest = finalDir[X]*(float)i+finalDir[Y]*(float)j+finalDir[Z]*(float)k;
								if(orthTest*orthTest<=1e-6){
								float dist= (float)FastMath.sqrt( ((float)i-xc)*((float)i-xc) +((float)j-yc)*((float)j-yc) +((float)k-zc)*((float)k-zc)  );	
								if(dist<=firstEst[x][y][z]-0.5f){
									pv[x+i][y+j][z+k]=1;
								}							
								else if(dist<=firstEst[x][y][z]+0.5f){
									pv[x+i][y+j][z+k]=FastMath.max(0.5f+firstEst[x][y][z]-dist,pv[x+i][y+j][z+k]);
								}
							}
						}					
					}					
				}											
			}
		}

		
		// 4. grow the region around the vessel centers to fit the scale (pv modeling)
		float[] pvol = new float[nxyz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x + nx*y + nx*ny*z;
			if (object[xyz]) {
				// when the diameter is smaller than the voxel, linear approx
				pvol[xyz] = Numerics.bounded((float)firstEst[x][y][z], 0.0f, 1.0f);
				int nsc = Numerics.ceil(scaleMax[xyz]/2.0f); // scale is diameter
				for (int i=-nsc;i<=nsc;i++) for (int j=-nsc;j<=nsc;j++) for (int k=-nsc;k<=nsc;k++) {
					if (x+i>=0 && x+i<nx && y+j>=0 && y+j<ny && z+k>=0 && z+k<nz) {
						float dist = (float)FastMath.sqrt(i*i+j*j+k*k);
						// update the neighbor value for diameters above the voxel size
						pvol[xyz+i+j*nx+k*nx*ny] = Numerics.max(pvol[xyz+i+j*nx+k*nx*ny], Numerics.bounded((float)firstEst[x][y][z]/2.0f+0.5f-dist, 0.0f, 1.0f));
					}
				}
			}
		}
		
		//PV map
		pvImage = new float[nx*ny*nz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int id = x + nx*y + nx*ny*z;
			pvImage[id] = pvol[id];
		}					

		//Diameter map
		diameterImage = new float[nx*ny*nz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int id = x + nx*y + nx*ny*z;
			if (object[id]) diameterImage[id] = (float)firstEst[x][y][z];
		}	
		
        return;       
    }
    
   private final void growPartialVolume(float[] image, int[] labels, boolean[] mask, float threshold) {
        
        
		// mean,stdev inside each vessel
		System.out.println("Vessel Intensity \n");
		float[] avg = new float[nx*ny*nz];
		float[] sum = new float[nx*ny*nz];
		for (int id=0;id<nx*ny*nz;id++) if (labels[id]>0) {
			int lb = labels[id];
			avg[lb] += image[id];
			sum[lb] += 1.0f;
		}
		for (int id=0;id<nx*ny*nz;id++) if (sum[id]>0) {
		    avg[id] /= sum[id];
		}
		float[] var = new float[nx*ny*nz];
		for (int id=0;id<nx*ny*nz;id++) if (labels[id]>0) {
			int lb = labels[id];
			var[lb] += (image[id]-avg[lb])*(image[id]-avg[lb]);
		}
		for (int id=0;id<nx*ny*nz;id++) if (sum[id]>1.0f) {
		    var[id] /= (sum[id]-1.0f);
		}
		
		// grow region with p(mu,sigma)>0.5
		BinaryHeapPair heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, nx*ny+ny*nz+nz*nx, BinaryHeapPair.MAXTREE);
		
        // simply order them by size instead?
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int id = x + nx*y + nx*ny*z;
			if (labels[id]>0) {
			    for (byte k = 0; k<6; k++) {
			        int ngb = fastMarchingNeighborIndex(k, id, nx, ny, nz);
			        if (mask[ngb] && labels[ngb]==0) {
                        int lb = labels[id];
                        float pv = (float)FastMath.exp(-0.5*(image[ngb]-avg[lb])*(image[ngb]-avg[lb])/var[lb]);
                        if (pv>=0.5f) heap.addValue(pv, ngb, lb);
                    }
                }
            }
        }
        
        float[] pvmap = new float[nx*ny*nz];
        for (int id=0;id<nx*ny*nz;id++) if (labels[id]>0) {
            pvmap[id] = 1.0f;
        }
        while (heap.isNotEmpty()) {
            float pv = heap.getFirst();
            int id = heap.getFirstId1();
            int lb = heap.getFirstId2();
            
            heap.removeFirst();
            
            if (pvmap[id]==0) {
                // add to current pv
                pvmap[id] = pv;
                labels[id] = lb;
                
                for (byte k = 0; k<6; k++) {
			        int ngb = fastMarchingNeighborIndex(k, id, nx, ny, nz);
			        if (mask[ngb] && labels[ngb]==0) {
                        float newpv = (float)FastMath.exp(-0.5*(image[ngb]-avg[lb])*(image[ngb]-avg[lb])/var[lb]);
                        if (newpv>=0.5f) heap.addValue(newpv, ngb, lb);
                    }
                }
            }
        }
		
		//debug: compute average probability instead
		float[] pavg = new float[nx*ny*nz];
		float[] psum = new float[nx*ny*nz];
		for (int id=0;id<nx*ny*nz;id++) if (labels[id]>0) {
		    int lb = labels[id];
		    pavg[lb] += probaImage[id];
		    psum[lb] ++;
		}
		for (int id=0;id<nx*ny*nz;id++) if (labels[id]>0) {
		    int lb = labels[id];
		    if (psum[lb]>0) {
                probaImage[id] = pavg[lb]/psum[lb];
            }
        }
		
		// Diameter from skeleton
		float[] nbdist = new float[6];
		boolean[] nbflag = new boolean[6];
		
		heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, nx*ny+ny*nz+nz*nx, BinaryHeapPair.MINTREE);
		heap.reset();
		
		// initialize the heap from boundaries
		float[] distance = new float[nx*ny*nz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int id = x + nx*y + nx*ny*z;
			if (labels[id]>0) {
			    boolean boundary=false;
			    for (byte k = 0; k<6 && !boundary; k++) {
			        int ngb = fastMarchingNeighborIndex(k, id, nx, ny, nz);
			        if (labels[ngb]==0) {
                        boundary=true;
                    }
                }
                if (boundary) heap.addValue(pvmap[id], id, labels[id]);
            }
        }

		while (heap.isNotEmpty()) {
        	// extract point with minimum distance
        	float dist = heap.getFirst();
        	int id = heap.getFirstId1();
        	int lb = heap.getFirstId2();
        	
        	heap.removeFirst();

        	// if more than nmgdm labels have been found already, this is done
			if (distance[id]==0) {
			    distance[id] = dist;
			    
			    // find new neighbors
			    for (byte k = 0; k<6; k++) {
			        int ngb = fastMarchingNeighborIndex(k, id, nx, ny, nz);
				
			        // must be in outside the object or its processed neighborhood
			        if (labels[ngb]==lb && distance[ngb]==0) {
			            // compute new distance based on processed neighbors for the same object
			            for (byte l=0; l<6; l++) {
			                nbdist[l] = -1.0f;
			                nbflag[l] = false;
			                int ngb2 = fastMarchingNeighborIndex(l, ngb, nx, ny, nz);
			                // note that there is at most one value used here
			                if (labels[ngb2]==lb && distance[ngb2]>0) {
			                    nbdist[l] = distance[ngb2];
			                    nbflag[l] = true;
			                }			
			            }
			            float newdist = minimumMarchingDistance(nbdist, nbflag);
					
			            // add to the heap
			            heap.addValue(newdist,ngb,lb);
			        }
				}
			}			
		}
		
		// find regions with low gradient as center?
		boolean[] keep = new boolean[nx*ny*nz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int id = x + nx*y + nx*ny*z;
			if (distance[id]>0) {
			    int lb = labels[id];
			    float gradx = 0.0f;
			    if (labels[id+1]==lb) gradx += 0.5f*distance[id+1];
			    if (labels[id-1]==lb) gradx -= 0.5f*distance[id-1];
			    float grady = 0.0f;
			    if (labels[id+nx]==lb) grady += 0.5f*distance[id+nx];
			    if (labels[id-nx]==lb) grady -= 0.5f*distance[id-nx];
			    float gradz = 0.0f;
			    if (labels[id+nx*ny]==lb) gradz += 0.5f*distance[id+nx*ny];
			    if (labels[id-nx*ny]==lb) gradz -= 0.5f*distance[id-nx*ny];
			    float gradxy = 0.0f;
			    if (labels[id+1+nx]==lb) gradxy += 0.5f*distance[id+1+nx];
			    if (labels[id-1-nx]==lb) gradxy -= 0.5f*distance[id-1-nx];
			    float gradyz = 0.0f;
			    if (labels[id+nx+nx*ny]==lb) gradyz += 0.5f*distance[id+nx+nx*ny];
			    if (labels[id-nx-nx*ny]==lb) gradyz -= 0.5f*distance[id-nx-nx*ny];
			    float gradzx = 0.0f;
			    if (labels[id+nx*ny+1]==lb) gradzx += 0.5f*distance[id+nx*ny+1];
			    if (labels[id-nx*ny-1]==lb) gradzx -= 0.5f*distance[id-nx*ny-1];
			    float gradyx = 0.0f;
			    if (labels[id+1-nx]==lb) gradyx += 0.5f*distance[id+1-nx];
			    if (labels[id-1+nx]==lb) gradyx -= 0.5f*distance[id-1+nx];
			    float gradzy = 0.0f;
			    if (labels[id+nx-nx*ny]==lb) gradzy += 0.5f*distance[id+nx-nx*ny];
			    if (labels[id-nx+nx*ny]==lb) gradzy -= 0.5f*distance[id-nx+nx*ny];
			    float gradxz = 0.0f;
			    if (labels[id+nx*ny-1]==lb) gradxz += 0.5f*distance[id+nx*ny-1];
			    if (labels[id-nx*ny+1]==lb) gradxz -= 0.5f*distance[id-nx*ny+1];
			    
			    // remove everything with high gradient, see what's left?
			    if (Numerics.max(gradx*gradx,grady*grady,gradxy*gradxy,gradyz*gradyz,gradzx*gradzx,gradyx*gradyx,gradzy*gradzy,gradxz*gradxz)<=0.25f) keep[id] = true;
			 }
		}
		// grow inregion back from skeleton points
		float[] radius = new float[nx*ny*nz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int id = x + nx*y + nx*ny*z;
			if (keep[id]) {
			    radius[id] = distance[id];
			    for (byte k = 0; k<6; k++) {
			        int ngb = fastMarchingNeighborIndex(k, id, nx, ny, nz);
			        if (labels[ngb]==labels[id] && !keep[ngb]) {
                        heap.addValue(Numerics.abs(distance[ngb]-distance[id]), ngb, id);
                    }
                }
            }
        }

		while (heap.isNotEmpty()) {
        	// extract point with minimum distance
        	float dist = heap.getFirst();
        	int id = heap.getFirstId1();
        	int prev = heap.getFirstId2();
        	
        	heap.removeFirst();

        	// if more than nmgdm labels have been found already, this is done
			if (!keep[id]) {
			    radius[id] = radius[prev];
			    keep[id]=true;
			    
			    // find new neighbors
			    for (byte k = 0; k<6; k++) {
			        int ngb = fastMarchingNeighborIndex(k, id, nx, ny, nz);
			        if (labels[ngb]==labels[id] && !keep[ngb]) {
                        heap.addValue(dist+Numerics.abs(distance[ngb]-distance[id]), ngb, id);
                    }
				}
			}			
		}
		
		// correct for starting point of distances, turn into diameter
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int id = x + nx*y + nx*ny*z;
		    if (pvmap[id]>0) {
		        radius[id] = 2.0f*Numerics.max(radius[id]-0.5f,0.5f);
		    }
		}
		// correct for background stuff, based on simplistic model of 3D vessel as a line
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int id = x + nx*y + nx*ny*z;
		    if (probaImage[id]<5.0f*threshold/9.0f) {
		        radius[id] = 0.0f;
		        pvmap[id] = 0.0f;
		    }
		}
		//PV map
		pvImage = pvmap;
		
		//Diameter map
		diameterImage = radius;
		
        return;       
    }

    public static final int fastMarchingNeighborIndex(byte d, int id, int nx, int ny, int nz) {
		switch (d) {
			case 0		: 	return id+1; 		
			case 1		:	return id-1;
			case 2		:	return id+nx;
			case 3		:	return id-nx;
			case 4		:	return id+nx*ny;
			case 5		:	return id-nx*ny;
			default		:	return id;
		}
	}

    private static final float minimumMarchingDistance(float[] val, boolean[] flag) {

        float s, s2; // s = a + b +c; s2 = a*a + b*b +c*c
        float tmp;
        int count;
        s = 0;
        s2 = 0;
        
        count = 0;

        for (int n=0; n<6; n+=2) {
			if (flag[n] && flag[n+1]) {
				tmp = Numerics.min(val[n], val[n+1]); // Take the smaller one if both are processed
				s += tmp;
				s2 += tmp*tmp;
				count++;
			} else if (flag[n]) {
				s += val[n]; // Else, take the processed one
				s2 += val[n]*val[n];
				count++;
			} else if (flag[n+1]) {
				s += val[n+1];
				s2 += val[n+1]*val[n+1];
				count++;
			}
		}
         // count must be greater than zero since there must be at least one processed pt in the neighbors
        if (count==0) System.err.print("!");
        if (s*s-count*(s2-1.0)<0) {
        	System.err.print(":");
        	tmp = 0;
        	for (int n=0;n<6;n++) if (flag[n]) tmp = Numerics.max(tmp,val[n]);
        	for (int n=0;n<6;n++) if (flag[n]) tmp = Numerics.min(tmp,val[n]);
        } else {
			tmp = (s + (float)FastMath.sqrt((double) (s*s-count*(s2-1.0))))/count;
		}
        // The larger root
        return tmp;
    }

}
