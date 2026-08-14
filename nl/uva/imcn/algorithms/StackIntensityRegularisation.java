package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import Jama.*;
import nl.uva.imcn.libraries.*;
import nl.uva.imcn.structures.*;

/*
 * @author Pierre-Louis bazin (pilou.bazin@uva.nl)
 *
 */
public class StackIntensityRegularisation {
	float[] image = null;
	float[] foreground = null;
	
	int nx, ny, nz, nxyz;
	float rx, ry, rz;
	
	float cutoff = 50.0f;
	float rmax = 95.0f;
	int mem=1;
	int shift=0;
	
	float[] regularised;
	
	public	static	final	byte	X = 0;
	public	static	final	byte	Y = 1;
	public	static	final	byte	Z = 2;

	private static final boolean debug = true;
	
	// set inputs
	public final void setInputImage(float[] val) { image = val; }
	public final void setForegroundImage(float[] val) { foreground = val; }
	
	public final void setVariationRatio(float val) { cutoff = val; }
	public final void setIntensityRatio(float val) { rmax = val; }
	//public final void setMaxDifference(float val) { cutoff = val; }
	public final void setMemory(int val) { mem = val; }
	public final void setShift(int val) { shift = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }

	// outputs
	public final float[] getRegularisedImage() { return regularised; }

	public void execute() {
	    
	    // mask zero values or estimate background
	    boolean[] mask = new boolean[nxyz];
	    if (foreground==null) {
            for (int xyz=0;xyz<nxyz;xyz++) 
                if (image[xyz]!=0) mask[xyz] = true;
                else mask[xyz] = false;
        } else {
            int nmask=0;
            for (int xyz=0;xyz<nxyz;xyz++) {
                if (foreground[xyz]>0.5) {
                    mask[xyz] = true;
                    nmask++;
                }
                else mask[xyz] = false;
            }
            System.out.print("mask size: "+nmask);
        }     
        
        // remove outlier values (high) from computation?
        if (rmax>0) {
            for (int z=0;z<nz;z++) {
                double[] intens = new double[nx*ny];
                int ni = 0;
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
                    int xyz = x+nx*y+nx*ny*z;
                    intens[ni] = image[xyz];
                    ni++;
                }
                Percentile measure = new Percentile();
                double imax = measure.evaluate(intens, 0, ni, rmax);
                
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
                    int xyz = x+nx*y+nx*ny*z;
                    if (image[xyz]>imax) mask[xyz] = false;
                }
            }
        }
	    
	    // per slice:
	    double[] differences = new double [mem*nx*ny];
	    int ndiff = 0;
	    double minbias = 0;
	    double maxbias = 0;
	    double minfactor = 1;
	    double maxfactor = 1;
	    double minfit = 1;
	    int minbiasid = -1;
	    int maxbiasid = -1;
	    int minfactorid = -1;
	    int maxfactorid = -1;
	    int minfitid = -1;
	    
	    int mid = Numerics.round(nz/2.0f);
	    for (int z=mid+1;z<nz;z++) {
	        System.out.print("Processing slice "+z);
	        ndiff = 0;
	        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
	            int xyz = x+nx*y+nx*ny*z;
	            int ngb1 = xyz-nx*ny;
	            if (mask[xyz] && mask[ngb1]) {
	                differences[ndiff] = image[xyz]-image[ngb1];
	                ndiff++;
	            }
	            for (int m=1;m<mem;m++) {
                    if (z>mid+m) {
                        int ngb2 = xyz-(m+1)*nx*ny;
                        if (mask[xyz] && mask[ngb2]) {
                            differences[ndiff] = image[xyz]-image[ngb2];
                            ndiff++;
                        }
                    }
                }
	        }
	        if (ndiff>0) {
                // find the distribution excluding edges: only use the 50% central differences
                Percentile measure = new Percentile();
                double min = measure.evaluate(differences, 0, ndiff, 50-cutoff/2);
                double max = measure.evaluate(differences, 0, ndiff, 50+cutoff/2);
                //double max = measure.evaluate(differences, 0, ndiff, cutoff);
            
                // estimate the scaling factor (or curve)
                double[] curr = new double[ndiff];
                double[] prev = new double[ndiff];
                double mean = 0;
                int nkept=0;
                ndiff = 0;
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
                    int xyz = x+nx*y+nx*ny*z;
                    int ngb1 = xyz-nx*ny;
                    if (mask[xyz] && mask[ngb1]) {
                        if (differences[ndiff]>=min && differences[ndiff]<=max) {
                        //if (differences[ndiff]<=max) {
                            curr[nkept] = image[xyz];
                            prev[nkept] = image[ngb1];
                            mean += image[xyz];
                            nkept++;
                        }
                        ndiff++;
                    }
                    for (int m=1;m<mem;m++) {
                        if (z>mid+m) {
                            int ngb2 = xyz-(m+1)*nx*ny;
                            if (mask[xyz] && mask[ngb2]) {
                                if (differences[ndiff]>=min && differences[ndiff]<=max) {
                                //if (differences[ndiff]<=max) {
                                    curr[nkept] = image[xyz];
                                    prev[nkept] = image[ngb2];
                                    mean += image[xyz];
                                    nkept++;
                                }
                                ndiff++;
                            }
                        }
                    }
                }
                if (nkept>0) {
                    mean /= (double)nkept;
                        
                    // linear least squares
                    double[][] fit = new double[nkept][1];
                    double[][] poly = new double[nkept][2];
                    for (int n=0;n<nkept;n++) {
                        fit[n][0] = curr[n];
                        poly[n][0] = 1.0;
                        poly[n][1] = prev[n];
                    }
                    // invert the linear model
                    Matrix mtx = new Matrix(poly);
                    Matrix smp = new Matrix(fit);
                    Matrix val = mtx.solve(smp);
                        
                    // compute the new values and residuals
                    double residual = 0;
                    double variance = 0;
                    nkept=0;
                    ndiff = 0;
                    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
                        int xyz = x+nx*y+nx*ny*z;
                        if (mask[xyz]) {
                            // replace the image values directly -> possible drift? (shouldn't be the case)
                            int ngb1 = xyz-nx*ny;
                            double expected1 = val.get(0,0) + image[ngb1]*val.get(1,0);
                            if (mask[ngb1]) { 
                                if (differences[ndiff]>=min && differences[ndiff]<=max) {
                                //if (differences[ndiff]<=max) {
                                    // compute residuals only where relevant
                                    variance += (image[xyz]-mean)*(image[xyz]-mean);
                                    residual += (image[xyz]-expected1)*(image[xyz]-expected1);
                                    nkept++;
                                }
                                ndiff++;
                            }
                            for (int m=1;m<mem;m++) {
                                if (z>mid+m) {
                                    int ngb2 = xyz-(m+1)*nx*ny;
                                    double expected2 = val.get(0,0) + image[ngb2]*val.get(1,0);
                                    if (mask[ngb2]) { 
                                        if (differences[ndiff]>=min && differences[ndiff]<=max) {
                                        //if (differences[ndiff]<=max) {
                                            // compute residuals only where relevant
                                            variance += (image[xyz]-mean)*(image[xyz]-mean);
                                            residual += (image[xyz]-expected2)*(image[xyz]-expected2);
                                            nkept++;
                                        }
                                        ndiff++;
                                    }
                                }
                            }
                            // change values
                            image[xyz] = (float)((image[xyz]-val.get(0,0))/val.get(1,0)); 
                        } else if (image[xyz]>rmax) {
                            // change values in masked intensity regions too
                            image[xyz] = (float)((image[xyz]-val.get(0,0))/val.get(1,0)); 
                        }
                    }
                    double rsquare = 1.0;
                    if (variance>0) rsquare = Numerics.max(1.0 - (residual/variance), 0.0);
                    System.out.print(" bias: "+val.get(0,0));
                    System.out.println(" scaling: "+val.get(1,0));
                    //System.out.println("residuals R^2: "+rsquare);
                    if (val.get(0,0)>maxbias) { maxbias = val.get(0,0); maxbiasid = z; }
                    if (val.get(0,0)<minbias) { minbias = val.get(0,0); minbiasid = z; }
                    if (val.get(1,0)>maxfactor) { maxfactor = val.get(1,0); maxfactorid = z; }
                    if (val.get(1,0)<minfactor) { minfactor = val.get(1,0); minfactorid = z; }
                    if (rsquare<minfit) { minfit = rsquare; minfitid = z; }
                } else {
                    System.out.println("no good data: skip");
                }
            } else {
                System.out.println("empty mask overlap: skip");
            }
        }
	    for (int z=mid-1;z>=0;z--) {
	        System.out.print("Processing slice "+z);
	        ndiff = 0;
	        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
	            int xyz = x+nx*y+nx*ny*z;
	            int ngb1 = xyz+nx*ny;
	            if (mask[xyz] && mask[ngb1]) {
	                differences[ndiff] = image[xyz]-image[ngb1];
	                ndiff++;
	            }
	            for (int m=1;m<mem;m++) {
                    if (z<mid-m) {
                        int ngb2 = xyz+(m+1)*nx*ny;
                        if (mask[xyz] && mask[ngb2]) {
                            differences[ndiff] = image[xyz]-image[ngb2];
                            ndiff++;
                        }
                    }
                }
	        }
	        if (ndiff>0) {
                // find the distribution excluding edges: only use the 50% central differences
                Percentile measure = new Percentile();
                double min = measure.evaluate(differences, 0, ndiff, 50-cutoff/2);
                double max = measure.evaluate(differences, 0, ndiff, 50+cutoff/2);
                //double max = measure.evaluate(differences, 0, ndiff, cutoff);
            
                // estimate the scaling factor (or curve)
                double[] curr = new double[ndiff];
                double[] prev = new double[ndiff];
                double mean = 0;
                int nkept=0;
                ndiff = 0;
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
                    int xyz = x+nx*y+nx*ny*z;
                    int ngb1 = xyz+nx*ny;
                    if (mask[xyz] && mask[ngb1]) {
                        if (differences[ndiff]>=min && differences[ndiff]<=max) {
                        //if (differences[ndiff]<=max) {
                            curr[nkept] = image[xyz];
                            prev[nkept] = image[ngb1];
                            mean += image[xyz];
                            nkept++;
                        }
                        ndiff++;
                    }
                    for (int m=1;m<mem;m++) {
                        if (z<mid-m) {
                            int ngb2 = xyz+(m+1)*nx*ny;
                            if (mask[xyz] && mask[ngb2]) {
                                if (differences[ndiff]>=min && differences[ndiff]<=max) {
                                //if (differences[ndiff]<=max) {
                                    curr[nkept] = image[xyz];
                                    prev[nkept] = image[ngb2];
                                    mean += image[xyz];
                                    nkept++;
                                }
                                ndiff++;
                            }
                        }
                    }
                }
                if (nkept>0) {
                    mean /= (double)nkept;
                        
                    // linear least squares
                    double[][] fit = new double[nkept][1];
                    double[][] poly = new double[nkept][2];
                    for (int n=0;n<nkept;n++) {
                        fit[n][0] = curr[n];
                        poly[n][0] = 1.0;
                        poly[n][1] = prev[n];
                    }
                    // invert the linear model
                    Matrix mtx = new Matrix(poly);
                    Matrix smp = new Matrix(fit);
                    Matrix val = mtx.solve(smp);
                        
                    // compute the new values and residuals
                    double residual = 0;
                    double variance = 0;
                    nkept=0;
                    ndiff = 0;
                    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
                        int xyz = x+nx*y+nx*ny*z;
                        if (mask[xyz]) {
                            // replace the image values directly -> possible drift? (shouldn't be the case)
                            int ngb1 = xyz+nx*ny;
                            double expected1 = val.get(0,0) + image[ngb1]*val.get(1,0);
                            if (mask[ngb1]) { 
                                if (differences[ndiff]>=min && differences[ndiff]<=max) {
                                //if (differences[ndiff]<=max) {
                                    // compute residuals only where relevant
                                    variance += (image[xyz]-mean)*(image[xyz]-mean);
                                    residual += (image[xyz]-expected1)*(image[xyz]-expected1);
                                    nkept++;
                                }
                                ndiff++;
                            }
                            for (int m=1;m<mem;m++) {
                                if (z<mid-m) {
                                    int ngb2 = xyz+(m+1)*nx*ny;
                                    double expected2 = val.get(0,0) + image[ngb2]*val.get(1,0);
                                    if (mask[ngb2]) { 
                                        if (differences[ndiff]>=min && differences[ndiff]<=max) {
                                        //if (differences[ndiff]<=max) {
                                            // compute residuals only where relevant
                                            variance += (image[xyz]-mean)*(image[xyz]-mean);
                                            residual += (image[xyz]-expected2)*(image[xyz]-expected2);
                                            nkept++;
                                        }
                                        ndiff++;
                                    }
                                }
                            }
                            // change values
                            image[xyz] = (float)((image[xyz]-val.get(0,0))/val.get(1,0)); 
                        } else if (image[xyz]>rmax) {
                            // change values in masked intensity regions too
                            image[xyz] = (float)((image[xyz]-val.get(0,0))/val.get(1,0)); 
                        }
                    }
                    double rsquare = 1.0;
                    if (variance>0) rsquare = Numerics.max(1.0 - (residual/variance), 0.0);
                    System.out.print(" bias: "+val.get(0,0));
                    System.out.println(" scaling: "+val.get(1,0));
                    //System.out.println("residuals R^2: "+rsquare);
                    if (val.get(0,0)>maxbias) { maxbias = val.get(0,0); maxbiasid = z; }
                    if (val.get(0,0)<minbias) { minbias = val.get(0,0); minbiasid = z; }
                    if (val.get(1,0)>maxfactor) { maxfactor = val.get(1,0); maxfactorid = z; }
                    if (val.get(1,0)<minfactor) { minfactor = val.get(1,0); minfactorid = z; }
                    if (rsquare<minfit) { minfit = rsquare; minfitid = z; }
                } else {
                    System.out.println("no good data: skip");
                }
            } else {
                System.out.println("empty mask overlap: skip");
            }
        }
        System.out.println("bias: ["+minbias+" ("+minbiasid+"), "+maxbias+" ("+maxbiasid+"]");
        System.out.println("scaling: ["+minfactor+" ("+minfactorid+"), "+maxfactor+" ("+maxfactorid+"]");
        System.out.println("min residuals R^2: "+minfit+" ("+minfitid+")");
	    // provide a global stabilisation? e.g. do the same process from the other direction?
	    // shouldn't be needed, hopefully..
	    
	    // shift for positive values
	    float min = 1e9f;
	    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz])
	        if (image[xyz]<min) min = image[xyz];
	    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz])
	        image[xyz] -= min;

	    regularised = image;
	    
		System.out.print("Done\n");
	}
	
	public void executeSubdivide(int sub) {
	    
	    // mask zero values or estimate background
	    boolean[] mask = new boolean[nxyz];
	    if (foreground==null) {
            for (int xyz=0;xyz<nxyz;xyz++) 
                if (image[xyz]!=0) mask[xyz] = true;
                else mask[xyz] = false;
        } else {
            int nmask=0;
            for (int xyz=0;xyz<nxyz;xyz++) {
                if (foreground[xyz]>0.5) {
                    mask[xyz] = true;
                    nmask++;
                }
                else mask[xyz] = false;
            }
            System.out.print("mask size: "+nmask);
        }     
        
        // remove outlier values (high) from computation?
	    boolean[] outlier = new boolean[nxyz];
	    if (rmax>0) {
            for (int z=0;z<nz;z++) {
                double[] intens = new double[nx*ny];
                int ni = 0;
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
                    int xyz = x+nx*y+nx*ny*z;
                    intens[ni] = image[xyz];
                    ni++;
                }
                Percentile measure = new Percentile();
                double imax = measure.evaluate(intens, 0, ni, rmax);
                
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
                    int xyz = x+nx*y+nx*ny*z;
                    if (image[xyz]>imax) outlier[xyz] = true;
                }
            }
        }
	    
	    // per neighborhood??:
	    double[] differences = new double [mem*nx*ny*(2*shift+1)*(2*shift+1)];
	    int ndiff = 0;
	    
	    int mid = Numerics.round(nz/2.0f);
	    for (int z=mid+1;z<nz;z++) {
	        System.out.println("Processing slice "+z);
	        int sx = Numerics.floor(nx/sub);
	        int sy = Numerics.floor(ny/sub);
	        
	        float[][] offset = new float[sub][sub];
	        for (int dx=0;dx<sub;dx++) for (int dy=0;dy<sub;dy++) {
                ndiff = 0;
                for (int x=dx*sx-sx/2;x<(dx+1)*sx+sx/2;x++) for (int y=dy*sy-sy/2;y<(dy+1)*sy+sy/2;y++) {
                    if (x>=0 && x<nx && y>=0 && y<ny) {
                        int xyz = x+nx*y+nx*ny*z;
                        for (int xn=-shift;xn<=shift;xn++) for (int yn=-shift;yn<=shift;yn++) {
                            if (x+xn>=0 && x+xn<nx && y+yn>=0 && y+yn<ny) {
                                int ngb1 = xyz + xn + yn*nx - nx*ny;
                                if (mask[xyz] && mask[ngb1] && !outlier[xyz] && !outlier[ngb1]) {
                                    differences[ndiff] = image[xyz]-image[ngb1];
                                    ndiff++;
                                }
                                for (int m=1;m<mem;m++) {
                                    if (z>mid+m) {
                                        int ngb2 = xyz + xn + yn*nx - (m+1)*nx*ny;
                                        if (mask[xyz] && mask[ngb2] && !outlier[xyz] && !outlier[ngb2]) {
                                            differences[ndiff] = image[xyz]-image[ngb2];
                                            ndiff++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (ndiff>0) {
                    // find the distribution excluding edges: only use the 50% central differences
                    Percentile measure = new Percentile();
                    double min = measure.evaluate(differences, 0, ndiff, 50-cutoff/2);
                    double max = measure.evaluate(differences, 0, ndiff, 50+cutoff/2);
                    //double max = measure.evaluate(differences, 0, ndiff, cutoff);
                
                    // estimate the scaling factor (or curve)
                    double[] curr = new double[ndiff];
                    double[] prev = new double[ndiff];
                    int nkept=0;
                    ndiff = 0;
                    for (int x=dx*sx-sx/2;x<(dx+1)*sx+sx/2;x++) for (int y=dy*sy-sy/2;y<(dy+1)*sy+sy/2;y++) {
                        if (x>=0 && x<nx && y>=0 && y<ny) {
                            int xyz = x+nx*y+nx*ny*z;
                            for (int xn=-shift;xn<=shift;xn++) for (int yn=-shift;yn<=shift;yn++) {
                                if (x+xn>=0 && x+xn<nx && y+yn>=0 && y+yn<ny) {
                                    int ngb1 = xyz + xn + yn*nx - nx*ny;
                                    if (mask[xyz] && mask[ngb1] && !outlier[xyz] && !outlier[ngb1]) {
                                        if (differences[ndiff]>=min && differences[ndiff]<=max) {
                                            curr[nkept] = image[xyz];
                                            prev[nkept] = image[ngb1];
                                            nkept++;
                                        }
                                        ndiff++;
                                    }
                                    for (int m=1;m<mem;m++) {
                                        if (z>mid+m) {
                                            int ngb2 = xyz + xn + yn*nx - (m+1)*nx*ny;
                                            if (mask[xyz] && mask[ngb2] && !outlier[xyz] && !outlier[ngb2]) {
                                                if (differences[ndiff]>=min && differences[ndiff]<=max) {
                                                    curr[nkept] = image[xyz];
                                                    prev[nkept] = image[ngb2];
                                                    nkept++;
                                                }
                                                ndiff++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (nkept>0) {
                        for (int n=0;n<nkept;n++) {
                            offset[dx][dy] += (float)((curr[n] - prev[n])/nkept);
                        }
                    }
                }
            }
            // compute the new values and residuals
            for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
                int xyz = x+nx*y+nx*ny*z;
                if (mask[xyz]) {
                    // linear interpolate the offset
                    float xsub = (float)x/(float)sx - 0.5f;
                    float ysub = (float)y/(float)sy - 0.5f;
                
                    float off = ImageInterpolation.linearClosestInterpolation(offset, xsub,ysub, sub, sub);
                    //int xsub = Numerics.min(Numerics.floor(x/sx),sub-1);
                    //int ysub = Numerics.min(Numerics.floor(y/sy),sub-1);
                    
                    //float off = offset[xsub][ysub];
                
                    // change values
                    image[xyz] = (float)(image[xyz]-off); 
                }
            }
        }
	    for (int z=mid-1;z>=0;z--) {
	        System.out.println("Processing slice "+z);
	        int sx = Numerics.floor(nx/sub);
	        int sy = Numerics.floor(ny/sub);
	        
	        float[][] offset = new float[sub][sub];
	        for (int dx=0;dx<sub;dx++) for (int dy=0;dy<sub;dy++) {
                ndiff = 0;
                for (int x=dx*sx-sx/2;x<(dx+1)*sx+sx/2;x++) for (int y=dy*sy-sy/2;y<(dy+1)*sy+sy/2;y++) {
                    if (x>=0 && x<nx && y>=0 && y<ny) {
                        int xyz = x+nx*y+nx*ny*z;
                        for (int xn=-shift;xn<=shift;xn++) for (int yn=-shift;yn<=shift;yn++) {
                            if (x+xn>=0 && x+xn<nx && y+yn>=0 && y+yn<ny) {
                                int ngb1 = xyz + xn + yn*nx + nx*ny;
                                if (mask[xyz] && mask[ngb1] && !outlier[xyz] && !outlier[ngb1]) {
                                    differences[ndiff] = image[xyz]-image[ngb1];
                                    ndiff++;
                                }
                                for (int m=1;m<mem;m++) {
                                    if (z<mid-m) {
                                        int ngb2 = xyz +xn + yn*nx + (m+1)*nx*ny;
                                        if (mask[xyz] && mask[ngb2] && !outlier[xyz] && !outlier[ngb2]) {
                                            differences[ndiff] = image[xyz]-image[ngb2];
                                            ndiff++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (ndiff>0) {
                    // find the distribution excluding edges: only use the 50% central differences
                    Percentile measure = new Percentile();
                    double min = measure.evaluate(differences, 0, ndiff, 50-cutoff/2);
                    double max = measure.evaluate(differences, 0, ndiff, 50+cutoff/2);
                    //double max = measure.evaluate(differences, 0, ndiff, cutoff);
                
                    // estimate the scaling factor (or curve)
                    double[] curr = new double[ndiff];
                    double[] prev = new double[ndiff];
                    int nkept=0;
                    ndiff = 0;
                    for (int x=dx*sx-sx/2;x<(dx+1)*sx+sx/2;x++) for (int y=dy*sy-sy/2;y<(dy+1)*sy+sy/2;y++) {
                        if (x>=0 && x<nx && y>=0 && y<ny) {
                            int xyz = x+nx*y+nx*ny*z;
                            for (int xn=-shift;xn<=shift;xn++) for (int yn=-shift;yn<=shift;yn++) {
                                if (x+xn>=0 && x+xn<nx && y+yn>=0 && y+yn<ny) {
                                    int ngb1 = xyz + xn + yn*nx + nx*ny;
                                    if (mask[xyz] && mask[ngb1] && !outlier[xyz] && !outlier[ngb1]) {
                                        if (differences[ndiff]>=min && differences[ndiff]<=max) {
                                            curr[nkept] = image[xyz];
                                            prev[nkept] = image[ngb1];
                                            nkept++;
                                        }
                                        ndiff++;
                                    }
                                    for (int m=1;m<mem;m++) {
                                        if (z<mid-m) {
                                            int ngb2 = xyz + xn + yn*nx + (m+1)*nx*ny;
                                            if (mask[xyz] && mask[ngb2] && !outlier[xyz] && !outlier[ngb2]) {
                                                if (differences[ndiff]>=min && differences[ndiff]<=max) {
                                                    curr[nkept] = image[xyz];
                                                    prev[nkept] = image[ngb2];
                                                    nkept++;
                                                }
                                                ndiff++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (nkept>0) {
                        for (int n=0;n<nkept;n++) {
                            offset[dx][dy] += (float)((curr[n] - prev[n])/nkept);
                        }
                    }
                }
            }
            // compute the new values and residuals
            double residual = 0;
            int nkept=0;
            ndiff = 0;
            for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
                int xyz = x+nx*y+nx*ny*z;
                if (mask[xyz]) {
                    // linear interpolate the offset
                    float xsub = (float)x/(float)sx - 0.5f;
                    float ysub = (float)y/(float)sy - 0.5f;
                
                    float off = ImageInterpolation.linearClosestInterpolation(offset, xsub,ysub, sub, sub);
                    //int xsub = Numerics.min(Numerics.floor(x/sx),sub-1);
                    //int ysub = Numerics.min(Numerics.floor(y/sy),sub-1);
                    
                    //float off = offset[xsub][ysub];
                    
                    // change values
                    image[xyz] = (float)(image[xyz]-off); 
                }
            }
        }
	    
	    // shift for positive values
	    float min = 1e9f;
	    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz])
	        if (image[xyz]<min) min = image[xyz];
	    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz])
	        image[xyz] -= min;

	    regularised = image;
	    
		System.out.print("Done\n");
	}
	
	public void executeSupervoxel(float scaling, float noise, int sub) {

	    // mask zero values or estimate background
	    boolean[] mask = new boolean[nxyz];
	    if (foreground==null) {
            for (int xyz=0;xyz<nxyz;xyz++) 
                if (image[xyz]!=0) mask[xyz] = true;
                else mask[xyz] = false;
        } else {
            int nmask=0;
            for (int xyz=0;xyz<nxyz;xyz++) {
                if (foreground[xyz]>0.5) {
                    mask[xyz] = true;
                    nmask++;
                }
                else mask[xyz] = false;
            }
            System.out.print("mask size: "+nmask);
        }     

	    float xscaling = scaling;
	    float yscaling = scaling; 
	    float zscaling = 1.0f;
        
 	    // Compute the supervoxel grid
	    System.out.println("original dimensions: ("+nx+", "+ny+")");
	    int nsx = Numerics.floor(nx/xscaling);
	    int nsy = Numerics.floor(ny/yscaling);
	    int nsz = Numerics.floor(nz/zscaling);
	    int nsxyz = nsx*nsy*nsz;
	    System.out.println("rescaled dimensions: ("+nsx+", "+nsy+")");
	    
	    // init downscaled images
	    int[] parcel = new int[nxyz];
	    float[] rescaled = new float[nsxyz];
	    
	    int[] count = new int[nsxyz];
	    
	    // init supervoxel centroids
	    // include all supervoxels with non-zero values inside
		float[][] centroid = new float[3][nsxyz];
	    for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) for (int zs=0;zs<nsz;zs++) {
	        int xyzs = xs+nsx*ys+nsx*nsy*zs;
	        centroid[X][xyzs] = 0.0f;
	        centroid[Y][xyzs] = 0.0f;
	        centroid[Z][xyzs] = 0.0f;
	        count[xyzs] = 0;
	        for (int dx=0;dx<xscaling;dx++) for (int dy=0;dy<yscaling;dy++) for (int dz=0;dz<zscaling;dz++) {
	            int xyz = Numerics.floor(xs*xscaling)+dx+nx*(Numerics.floor(ys*yscaling)+dy)+nx*ny*(Numerics.floor(zs*zscaling)+dz);
	            if (mask[xyz]) {
                    centroid[X][xyzs] += Numerics.floor(xs*xscaling) + dx;
                    centroid[Y][xyzs] += Numerics.floor(ys*yscaling) + dy;
                    centroid[Z][xyzs] += Numerics.floor(zs*zscaling) + dz;
                    count[xyzs]++;
                }
            }
            if (count[xyzs]>0) {
                centroid[X][xyzs] /= count[xyzs];
                centroid[Y][xyzs] /= count[xyzs];
                centroid[Z][xyzs] /= count[xyzs];
	        }
	    }
	    
	    // init: search for voxel with lowest gradient within the region instead? (TODO)
	    // OR voxel most representative?
	    double[] selection = new double[27];
	    Percentile median = new Percentile();
	    for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) for (int zs=0;zs<nsz;zs++) {
	        int xyzs = xs+nsx*ys+nsx*nsy*zs;
	        int x0 = Numerics.bounded(Numerics.floor(centroid[X][xyzs]),1,nx-2);
	        int y0 = Numerics.bounded(Numerics.floor(centroid[Y][xyzs]),1,ny-2);
	        int z0 = Numerics.bounded(Numerics.floor(centroid[Z][xyzs]),1,nz-2);
	        int xyz0 = x0+nx*y0+nx*ny*z0;
	        
	        int s=0;
	        for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) {
	            int dz=0;
	            //for (int dz=-1;dz<=1;dz++) {
	            selection[s] = image[xyz0+dx+nx*dy+nx*ny*dz];
	            s++;
	        }
	        double med = median.evaluate(selection, 50.0);    
	        for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) {
	            int dz=0;
	            //for (int dz=-1;dz<=1;dz++) {
	            if (image[xyz0+dx+nx*dy+nx*ny*dz]==med) {
                    centroid[X][xyzs] = x0+dx;
                    centroid[Y][xyzs] = y0+dy;
                    centroid[Z][xyzs] = z0+dz;
                    dx=2;dy=2;dz=2;
	            }
	        }
	    }
	    // Estimate approximate min,max from sampled grid values
	    float Imin = 1e9f;
	    float Imax = -1e9f;
        for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) for (int zs=0;zs<nsz;zs++) {
		    int xyzs = xs+nsx*ys+nsx*nsy*zs;
	        
		    int x = Numerics.bounded(Numerics.floor(centroid[X][xyzs]),1,nx-2);
	        int y = Numerics.bounded(Numerics.floor(centroid[Y][xyzs]),1,ny-2);
	        int z = Numerics.bounded(Numerics.floor(centroid[Z][xyzs]),1,nz-2);
	        int xyz = x+nx*y+nx*ny*z;	        
	        if (mask[xyz]) {
	            if (image[xyz]<Imin) Imin = image[xyz];
	            if (image[xyz]>Imax) Imax = image[xyz];
	        }
	    }
	    // normalize the noise parameter by intensity, but not by distance (-> same speed indep of scale)
	    System.out.println("intensity scale: ["+Imin+", "+Imax+"]");
	    if (Imax>Imin) {
	        noise = noise*noise*(Imax-Imin)*(Imax-Imin);
	    }
	    
	    // start a voxel heap at each center
	    BinaryHeap4D heap = new BinaryHeap4D(nx*ny+ny*nz+nz*nx, BinaryHeap4D.MINTREE);
		boolean[] processed = new boolean[nx*ny*nz];
		for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) for (int zs=0;zs<nsz;zs++) {
		    int xyzs = xs+nsx*ys+nsx*nsy*zs;
	        count[xyzs]=0;
	        
	        int x = Numerics.bounded(Numerics.floor(centroid[X][xyzs]),1,nx-2);
	        int y = Numerics.bounded(Numerics.floor(centroid[Y][xyzs]),1,ny-2);
	        int z = Numerics.bounded(Numerics.floor(centroid[Z][xyzs]),1,nz-2);
	        int xyz = x+nx*y+nx*ny*z;	        
	        if (mask[xyz]) {
	            // set as starting point
	            parcel[xyz] = xyzs+1;
	            rescaled[xyzs] = image[xyz];
	            count[xyzs] = 1;
	            processed[xyz] = true;
	            
	            // add neighbors to the tree
	            for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) {
	                int dz=0;//for (int dz=-1;dz<=1;dz++) {
	                if (dx*dx+dy*dy+dz*dz==1 && x+dx>=0 && y+dy>=0 && z+dz>=0 && x+dx<nx && y+dy<ny && z+dz<nz) {
                        int xyznb = x+dx+nx*(y+dy)+nx*ny*(z+dz);
                         // exclude zero as mask
                        if (mask[xyznb]) {
                        
                            // distance function
                            float dist = (x+dx-centroid[X][xyzs])*(x+dx-centroid[X][xyzs])
                                        +(y+dy-centroid[Y][xyzs])*(y+dy-centroid[Y][xyzs])
                                        +(z+dz-centroid[Z][xyzs])*(z+dz-centroid[Z][xyzs]);
                                    
                            float contrast = (image[xyznb]-rescaled[xyzs])
                                            *(image[xyznb]-rescaled[xyzs]);
                                        
                            heap.addValue(noise*dist+contrast, x+dx,y+dy,z+dz, xyzs+1);
                        }
                    }                    
	            }
	        }
	    }
	    // grow to 
        while (heap.isNotEmpty()) {
        	// extract point with minimum distance
        	float curr = heap.getFirst();
        	int x = heap.getFirstX();
        	int y = heap.getFirstY();
        	int z = heap.getFirstZ();
        	int xyzs = heap.getFirstK()-1;
        	heap.removeFirst();
        	int xyz = x+nx*y+nx*ny*z;
        	
			if (processed[xyz])  continue;
			
        	// update the cluster
			parcel[xyz] = xyzs+1;
            rescaled[xyzs] = count[xyzs]*rescaled[xyzs] + image[xyz];
            
            centroid[X][xyzs] = count[xyzs]*centroid[X][xyzs] + x;
	        centroid[Y][xyzs] = count[xyzs]*centroid[Y][xyzs] + y;
	        centroid[Z][xyzs] = count[xyzs]*centroid[Z][xyzs] + z;
	        
	        count[xyzs] += 1;
	        rescaled[xyzs] /= count[xyzs];
	        centroid[X][xyzs] /= count[xyzs];
	        centroid[Y][xyzs] /= count[xyzs];
	        centroid[Z][xyzs] /= count[xyzs];
	        
	        processed[xyz]=true;
			
            // add neighbors to the tree
            for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) {
                int dz=0;//for (int dz=-1;dz<=1;dz++) {
	            if (dx*dx+dy*dy+dz*dz==1 && x+dx>=0 && y+dy>=0 && z+dz>=0 && x+dx<nx && y+dy<ny && z+dz<nz) {
                    int xyznb = x+dx+nx*(y+dy)+nx*ny*(z+dz);

                    // exclude zero as mask
                    if (mask[xyznb] && !processed[xyznb]) {
                    
                        // distance function
                        float dist = (x+dx-centroid[X][xyzs])*(x+dx-centroid[X][xyzs])
                                    +(y+dy-centroid[Y][xyzs])*(y+dy-centroid[Y][xyzs])
                                    +(z+dz-centroid[Z][xyzs])*(z+dz-centroid[Z][xyzs]);
                                
                        float contrast = (image[xyznb]-rescaled[xyzs])
                                        *(image[xyznb]-rescaled[xyzs]);
                                        
                        heap.addValue(noise*dist+contrast, x+dx,y+dy,z+dz, xyzs+1);
                    }
                }
            }
		}

		/*
        // debug
        for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) if (parcel[xyz]>0) {
	        image[xyz] = rescaled[parcel[xyz]-1];    
	    }    
        regularised = image;
        */
        
        // 2. local intensity correction on the rescaled data
        
       // remove outlier values (high) from computation?
	    boolean[] outlier = new boolean[nsxyz];
	    if (rmax>0) {
            for (int z=0;z<nz;z++) {
                double[] intens = new double[nsx*nsy];
                int ni = 0;
                for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) {
                    int xyzs = xs+nsx*ys+nsx*nsy*z;
                    intens[ni] = rescaled[xyzs];
                    ni++;
                }
                Percentile measure = new Percentile();
                double imax = measure.evaluate(intens, 0, ni, rmax);
                
                for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) {
                    int xyzs = xs+nsx*ys+nsx*nsy*z;
                    if (rescaled[xyzs]>imax) outlier[xyzs] = true;
                }
            }
        }
	    
	    // per neighborhood??:
	    double[] differences = new double [mem*nsx*nsy*(2*shift+1)*(2*shift+1)];
	    int ndiff = 0;
	    
	    float[] orig = new float[nsx*nsy*nz];
	    for (int xyzs=0;xyzs<nsxyz;xyzs++) {
	        orig[xyzs] = rescaled[xyzs];
	    }
	    
	    int mid = Numerics.round(nz/2.0f);
	    for (int z=mid+1;z<nz;z++) {
	        System.out.println("Processing slice "+z);
	        int sx = Numerics.floor(nsx/sub);
	        int sy = Numerics.floor(nsy/sub);
	        
	        float[][] offset = new float[sub][sub];
	        for (int dx=0;dx<sub;dx++) for (int dy=0;dy<sub;dy++) {
                ndiff = 0;
                for (int xs=dx*sx-sx/2;xs<(dx+1)*sx+sx/2;xs++) for (int ys=dy*sy-sy/2;ys<(dy+1)*sy+sy/2;ys++) {
                    if (xs>=0 && xs<nsx && ys>=0 && ys<nsy) {
                        int xyzs = xs+nsx*ys+nsx*nsy*z;
                        for (int xn=-shift;xn<=shift;xn++) for (int yn=-shift;yn<=shift;yn++) {
                            if (xs+xn>=0 && xs+xn<nsx && ys+yn>=0 && ys+yn<nsy) {
                                int ngb1 = xyzs + xn + yn*nsx - nsx*nsy;
                                if (rescaled[xyzs]!=0 && rescaled[ngb1]!=0 && !outlier[xyzs] && !outlier[ngb1]) {
                                    differences[ndiff] = rescaled[xyzs]-rescaled[ngb1];
                                    ndiff++;
                                }
                                for (int m=1;m<mem;m++) {
                                    if (z>mid+m) {
                                        int ngb2 = xyzs + xn + yn*nsx - (m+1)*nsx*nsy;
                                        if (rescaled[xyzs]!=0 && rescaled[ngb2]!=0 && !outlier[xyzs] && !outlier[ngb2]) {
                                            differences[ndiff] = rescaled[xyzs]-rescaled[ngb2];
                                            ndiff++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (ndiff>0) {
                    // find the distribution excluding edges: only use the 50% central differences
                    Percentile measure = new Percentile();
                    double min = measure.evaluate(differences, 0, ndiff, 50-cutoff/2);
                    double max = measure.evaluate(differences, 0, ndiff, 50+cutoff/2);
                    //double max = measure.evaluate(differences, 0, ndiff, cutoff);
                
                    // estimate the scaling factor (or curve)
                    double[] curr = new double[ndiff];
                    double[] prev = new double[ndiff];
                    int nkept=0;
                    ndiff = 0;
                    for (int xs=dx*sx-sx/2;xs<(dx+1)*sx+sx/2;xs++) for (int ys=dy*sy-sy/2;ys<(dy+1)*sy+sy/2;ys++) {
                        if (xs>=0 && xs<nsx && ys>=0 && ys<nsy) {
                            int xyzs = xs+nsx*ys+nsx*nsy*z;
                            for (int xn=-shift;xn<=shift;xn++) for (int yn=-shift;yn<=shift;yn++) {
                                if (xs+xn>=0 && xs+xn<nsx && ys+yn>=0 && ys+yn<nsy) {
                                    int ngb1 = xyzs + xn + yn*nsx - nsx*nsy;
                                    if (rescaled[xyzs]!=0 && rescaled[ngb1]!=0 && !outlier[xyzs] && !outlier[ngb1]) {
                                        if (differences[ndiff]>=min && differences[ndiff]<=max) {
                                            curr[nkept] = rescaled[xyzs];
                                            prev[nkept] = rescaled[ngb1];
                                            nkept++;
                                        }
                                        ndiff++;
                                    }
                                    for (int m=1;m<mem;m++) {
                                        if (z>mid+m) {
                                            int ngb2 = xyzs + xn + yn*nsx - (m+1)*nsx*nsy;
                                            if (rescaled[xyzs]!=0 && rescaled[ngb2]!=0 && !outlier[xyzs] && !outlier[ngb2]) {
                                                if (differences[ndiff]>=min && differences[ndiff]<=max) {
                                                    curr[nkept] = rescaled[xyzs];
                                                    prev[nkept] = rescaled[ngb2];
                                                    nkept++;
                                                }
                                                ndiff++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (nkept>0) {
                        for (int n=0;n<nkept;n++) {
                            offset[dx][dy] += (float)((curr[n] - prev[n])/nkept);
                        }
                    }
                }
            }
            // compute the new values and residuals
            for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) {
                int xyzs = xs+nsx*ys+nsx*nsy*z;
                if (rescaled[xyzs]!=0) {
                    // linear interpolate the offset
                    float xsub = (float)xs/(float)sx - 0.5f;
                    float ysub = (float)ys/(float)sy - 0.5f;
                
                    float off = ImageInterpolation.linearClosestInterpolation(offset, xsub,ysub, sub, sub);
                    //int xsub = Numerics.min(Numerics.floor(x/sx),sub-1);
                    //int ysub = Numerics.min(Numerics.floor(y/sy),sub-1);
                    
                    //float off = offset[xsub][ysub];
                
                    // change values
                    rescaled[xyzs] = (float)(rescaled[xyzs]-off); 
                }
            }
        }
	    for (int z=mid-1;z>=0;z--) {
	        System.out.println("Processing slice "+z);
	        int sx = Numerics.floor(nsx/sub);
	        int sy = Numerics.floor(nsy/sub);
	        
	        float[][] offset = new float[sub][sub];
	        for (int dx=0;dx<sub;dx++) for (int dy=0;dy<sub;dy++) {
                ndiff = 0;
                for (int xs=dx*sx-sx/2;xs<(dx+1)*sx+sx/2;xs++) for (int ys=dy*sy-sy/2;ys<(dy+1)*sy+sy/2;ys++) {
                    if (xs>=0 && xs<nsx && ys>=0 && ys<nsy) {
                        int xyzs = xs+nsx*ys+nsx*nsy*z;
                        for (int xn=-shift;xn<=shift;xn++) for (int yn=-shift;yn<=shift;yn++) {
                            if (xs+xn>=0 && xs+xn<nsx && ys+yn>=0 && ys+yn<nsy) {
                                int ngb1 = xyzs + xn + yn*nsx + nsx*nsy;
                                if (rescaled[xyzs]!=0 && rescaled[ngb1]!=0 && !outlier[xyzs] && !outlier[ngb1]) {
                                    differences[ndiff] = rescaled[xyzs]-rescaled[ngb1];
                                    ndiff++;
                                }
                                for (int m=1;m<mem;m++) {
                                    if (z<mid-m) {
                                        int ngb2 = xyzs +xn + yn*nsx + (m+1)*nsx*nsy;
                                        if (rescaled[xyzs]!=0 && rescaled[ngb2]!=0 && !outlier[xyzs] && !outlier[ngb2]) {
                                            differences[ndiff] = rescaled[xyzs]-rescaled[ngb2];
                                            ndiff++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (ndiff>0) {
                    // find the distribution excluding edges: only use the 50% central differences
                    Percentile measure = new Percentile();
                    double min = measure.evaluate(differences, 0, ndiff, 50-cutoff/2);
                    double max = measure.evaluate(differences, 0, ndiff, 50+cutoff/2);
                    //double max = measure.evaluate(differences, 0, ndiff, cutoff);
                
                    // estimate the scaling factor (or curve)
                    double[] curr = new double[ndiff];
                    double[] prev = new double[ndiff];
                    int nkept=0;
                    ndiff = 0;
                    for (int xs=dx*sx-sx/2;xs<(dx+1)*sx+sx/2;xs++) for (int ys=dy*sy-sy/2;ys<(dy+1)*sy+sy/2;ys++) {
                        if (xs>=0 && xs<nsx && ys>=0 && ys<nsy) {
                            int xyzs = xs+nsx*ys+nsx*nsy*z;
                            for (int xn=-shift;xn<=shift;xn++) for (int yn=-shift;yn<=shift;yn++) {
                                if (xs+xn>=0 && xs+xn<nsx && ys+yn>=0 && ys+yn<nsy) {
                                    int ngb1 = xyzs + xn + yn*nsx + nsx*nsy;
                                    if (rescaled[xyzs]!=0 && rescaled[ngb1]!=0 && !outlier[xyzs] && !outlier[ngb1]) {
                                        if (differences[ndiff]>=min && differences[ndiff]<=max) {
                                            curr[nkept] = rescaled[xyzs];
                                            prev[nkept] = rescaled[ngb1];
                                            nkept++;
                                        }
                                        ndiff++;
                                    }
                                    for (int m=1;m<mem;m++) {
                                        if (z<mid-m) {
                                            int ngb2 = xyzs + xn + yn*nsx + (m+1)*nsx*nsy;
                                            if (rescaled[xyzs]!=0 && rescaled[ngb2]!=0 && !outlier[xyzs] && !outlier[ngb2]) {
                                                if (differences[ndiff]>=min && differences[ndiff]<=max) {
                                                    curr[nkept] = rescaled[xyzs];
                                                    prev[nkept] = rescaled[ngb2];
                                                    nkept++;
                                                }
                                                ndiff++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (nkept>0) {
                        for (int n=0;n<nkept;n++) {
                            offset[dx][dy] += (float)((curr[n] - prev[n])/nkept);
                        }
                    }
                }
            }
            // compute the new values and residuals
            double residual = 0;
            int nkept=0;
            ndiff = 0;
            for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) {
                int xyzs = xs+nsx*ys+nsx*nsy*z;
                if (rescaled[xyzs]!=0) {
                    // linear interpolate the offset
                    float xsub = (float)xs/(float)sx - 0.5f;
                    float ysub = (float)ys/(float)sy - 0.5f;
                
                    float off = ImageInterpolation.linearClosestInterpolation(offset, xsub,ysub, sub, sub);
                    //int xsub = Numerics.min(Numerics.floor(x/sx),sub-1);
                    //int ysub = Numerics.min(Numerics.floor(y/sy),sub-1);
                    
                    //float off = offset[xsub][ysub];
                    
                    // change values
                    rescaled[xyzs] = (float)(rescaled[xyzs]-off); 
                }
            }
        }
        
        // adapt the result to full resolution image 
        for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz] && parcel[xyz]>0) {
            //image[xyz] = image[xyz] - offimg[parcel[xyz]-1];
	        //image[xyz] = rescaled[parcel[xyz]-1];
	        image[xyz] = image[xyz] - orig[parcel[xyz]-1] + rescaled[parcel[xyz]-1];
	    }
	    /*
	    // shift for positive values
	    float min = 1e9f;
	    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz])
	        if (image[xyz]<min) min = image[xyz];
	    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz])
	        image[xyz] -= min;
	        */
	    regularised = image;
	    
        
		System.out.print("Done\n");
        
	}

}
