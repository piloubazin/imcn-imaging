package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import Jama.*;

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
	
	float[] regularised;
	
	// set inputs
	public final void setInputImage(float[] val) { image = val; }
	public final void setForegroundImage(float[] val) { foreground = val; }
	
	public final void setVariationRatio(float val) { cutoff = val; }
	public final void setIntensityRatio(float val) { rmax = val; }
	//public final void setMaxDifference(float val) { cutoff = val; }
	
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
	    double[] differences = new double [2*nx*ny];
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
	            if (z>mid+1) {
                    int ngb2 = xyz-2*nx*ny;
                    if (mask[xyz] && mask[ngb2]) {
                        differences[ndiff] = image[xyz]-image[ngb2];
                        ndiff++;
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
                    if (z>mid+1) {
                        int ngb2 = xyz-2*nx*ny;
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
                            if (z>mid+1) {
                                int ngb2 = xyz-2*nx*ny;
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
	            if (z<mid-1) {
                    int ngb2 = xyz+2*nx*ny;
                    if (mask[xyz] && mask[ngb2]) {
                        differences[ndiff] = image[xyz]-image[ngb2];
                        ndiff++;
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
                    if (z<mid-1) {
                        int ngb2 = xyz+2*nx*ny;
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
                            if (z<mid-1) {
                                int ngb2 = xyz+2*nx*ny;
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
}
