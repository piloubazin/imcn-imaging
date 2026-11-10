package nl.uva.imcn.algorithms;

import java.io.*;
import java.util.*;


import nl.uva.imcn.structures.*;
import nl.uva.imcn.libraries.*;
import nl.uva.imcn.utilities.*;
import nl.uva.imcn.methods.*;

/**
 *
 *  This algorithm handles Fuzzy C-Means operations
 *  for 3D data as in  in FANTASM (membership, centroid computations)
 *
 *	@version    November 2019
 *	@author     Pierre-Louis Bazin
 *		
 *
 */
 
public class FuzzyCmeansMulti {
		
	// numerical quantities
	private static final	float   INF=1e30f;
	private static final	float   ZERO=1e-30f;
	
	// data buffers
	private 	float[][]		images;  			// original images
	private 	float[][]		mems;				// membership function
	private 	float[][]	    centroids=null;			// cluster centroids
	private 	boolean[]		mask;   			// image mask: true for data points
	private     int[]           id;                 // re-ordering of data by increasing centroid values
	private     int[]           classification;     // hard classification
	private static	int 		nx,ny,nz, nc, nxyz; // image dimensions
	private static	float 		rx,ry,rz;   		// image resolutions
	private    float[]          Imin, Imax;         // image range 
	private    float[]          Iscale;             // image scaling factor for normalization 
	
	// parameters
	private 	int 		clusters;    // number of clusters in original membership: > clusters if outliers
	private 	float 		smoothing;	// MRF smoothing
 	private		float		fuzziness = 2.0f;	// fuzziness factor
	private		PowerTable	power, invpower;
	private		float		imgvar = 1.0f;		// image variance
	private     float       maxDist = 0.01f;
	private     int         maxIter = 50;
	
			
	// computation variables
	private		float[]		prev;	// previous membership values
	
	// for debug and display
	static final boolean		debug=true;
	static final boolean		verbose=true;
	
	public final void setImageNumber(int val) { 
	    nc=val;
	    images = new float[nc][];
	    centroids = new float[nc][];
	    Imin = new float[nc];
	    Imax = new float[nc];
	    Iscale = new float[nc];
	}
	public final void setImageAt(int c, float[] val) { images[c] = val; }
	public final void setMaskImage(int[] val) { 
	    mask = new boolean[nxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) mask[xyz] = (val[xyz]>0); 
	}
	public final void initZeroMaskImage() { 
	    mask = new boolean[nxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        mask[xyz] = false;
	        for (int c=0;c<nc;c++) {
	            if (images[c][xyz]!=0) mask[xyz]=true;
	        }
	    }
	}
	public final void initBasicMaskImage() { 
	    mask = new boolean[nxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) mask[xyz] = true;
	}
	public final void maskImageNaNs() { 
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        for (int c=0;c<nc;c++) {
                if (Float.isNaN(images[c][xyz])) {
                    mask[xyz] = false; 
                    images[c][xyz] = 0.0f;
                }
            }
	    }
	}
	
	public final void setClusterNumber(int val) { clusters = val; }
	public final void setSmoothing(float val) { smoothing = val; }
	public final void setFuzziness(float val) { fuzziness = val; }
	public final void setMaxDist(float val) { maxDist = val; }
	public final void setMaxIter(int val) { maxIter = val; }
	
	public final void setInitCentroidsAt(int c, float[] val) { centroids[c] = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
		
	// to be used for JIST definitions, generic info / help
	public final String getPackage() { return "IMCN Toolkit"; }
	public final String getCategory() { return "Clustering"; }
	public final String getLabel() { return "Fuzzy C-means"; }
	public final String getName() { return "FuzzyCmeans"; }

	public final String[] getAlgorithmAuthors() { return new String[]{"Pierre-Louis Bazin"}; }
	public final String getAffiliation() { return "Integrated Model-based Cognitive Neuroscience Reseaerch Unit, Universiteit van Amsterdam"; }
	public final String getDescription() { return "Fuzzy C-means algorithm adapted from (Pham, 2001)"; }
	public final String getLongDescription() { return getDescription(); }
		
	public final String getVersion() { return "1.0"; };

	// create outputs
	public final float[] getMembership(int n) { return mems[id[n+1]-1]; }
   
    public final int[] getClassification() { return classification; }
   
    public final float[] getClassImageAt(int c) { return images[c]; }
   
    public final void execute() {		
		
        // image ranges
        for (int c=0;c<nc;c++) {
            /*
            Imin = INF; 
            Imax = -INF;
            for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                if (images[c][xyz]<Imin[c]) Imin[c] = images[c][xyz];
                if (images[c][xyz]>Imax[c]) Imax[c] = images[c] [xyz];
            }*/
            // use robust estimates?
            Imin[c] = ImageStatistics.robustMinimum(images[c], mask, 0.0001f, 5, nx, ny, nz);
            Imax[c] = ImageStatistics.robustMaximum(images[c], mask, 0.0001f, 5, nx, ny, nz);
            Iscale[c] = ImageStatistics.robustMinimum(images[c], mask, 0.25f, 5, nx, ny, nz)
                        - ImageStatistics.robustMaximum(images[c], mask, 0.25f, 5, nx, ny, nz);
            if (verbose) System.out.print("Image "+c+" range ["+Imin[c]+", "+Imax[c]+"], scale "+Iscale[c]+"\n");
        }
		
        // init all the new arrays
        mems = new float[clusters][nxyz];
        prev = new float[clusters];
        if (fuzziness!=2) {
            power = new PowerTable(0.0f , 1.0f , 0.000001f , fuzziness );
            //invpower = new PowerTable(0, (Imax-Imin)*(Imax-Imin), 0.000001f*(Imax-Imin)*(Imax-Imin), 1.0f/(1.0f-fuzziness) );
            invpower = new PowerTable(0, 1.0f, 0.000001f, 1.0f/(1.0f-fuzziness) );
        } else {
            power = null;
            invpower = null;
        }

		// init values
		for (int k=0;k<clusters;k++) for (int xyz=0;xyz<nxyz;xyz++) {
            mems[k][xyz] = 0.0f;
		}
		for (int c=0;c<nc;c++) {
            if (centroids[c]==null) {
                centroids[c] = new float[clusters];
                for (int k=0;k<clusters;k++) {
                    centroids[c][k] = 0.0f;
                }
            
                // init centroids from data range
                centroids[c][0] = Imin[c] + 0.5f*(Imax[c]-Imin[c])/(float)clusters;
                for (int k=1;k<clusters;k++)
                    centroids[c][k] = centroids[c][k-1] + (Imax[c]-Imin[c])/(float)clusters;
            }
        }        
        if (verbose) {
            for (int c=0;c<nc;c++) {
                System.out.print("Initialization "+c+": centroids: ("+centroids[c][0]);
                for (int k=1;k<clusters;k++) System.out.print(", "+centroids[c][k]);
                System.out.print(")\n");
            }
		}   
        
        // initialize the memberships
        computeMemberships();
		
		// main iterations: compute the classes on the image
		
		// with inhomogeneity correction, two steps: first iterates until mild convergence without inhomogeneity correction
		float distance = 0.0f;
		int Niterations = 1;
		boolean stop = false;
		if (Niterations >= maxIter) stop = true;
		while (!stop) {
			if (verbose) System.out.print("iteration " + Niterations + " (max: " + distance + ") \n");
			
			// update centroids
			computeCentroids();
			
			// update membership
			distance = computeMemberships();
			
			// check for segmentation convergence 
			Niterations++;
			if (Niterations > maxIter) stop = true;
			if (distance < maxDist) stop = true;            
		}

		// order the classes in increasing order of the first image
		id = computeCentroidOrder(0);
        
		// generate classification map
		classification = new int[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) {
		    if (mask[xyz]) {
		        int best=0;
		        for (int k=1;k<clusters;k++) {
		            if (mems[k][xyz]>mems[best][xyz]) best = k;
		        }
		        classification[xyz] = id[best+1];
		    } else {
		        classification[xyz] = id[0];
		    }
		}
		return;
    }
    
    /** 
	 *  compute the FCM membership functions given the centroids
	 *	with the different options (outliers, field, edges, MRF)
	 */
    final public float computeMemberships() {
        
        if (fuzziness!=2) return computeGeneralMemberships();
		
        float distance,dist;
        float den,num;
        float neighbors, ngb;
        
        distance = 0.0f;

        if (nz>1) {
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if ( mask[xyz] ) {
                    den = 0;
                    // remember the previous values
                    for (int k=0;k<clusters;k++) 
                        prev[k] = mems[k][xyz];
    
                    for (int k=0;k<clusters;k++) {
                        
                        // data term
                        num = 0.0f;
                        for (int c=0;c<nc;c++) {
                            num += (images[c][xyz]-centroids[c][k])*(images[c][xyz]-centroids[c][k])/(Iscale[c]*Iscale[c]);
                        }
                        
                        // spatial smoothing
                        if (smoothing > 0.0f) { 
                            ngb = 0.0f;  
                            neighbors = 0.0f;
                            // case by case	: X+
                            if (mask[xyz+1]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += mems[m][xyz+1]*mems[m][xyz+1];
                                neighbors ++;
                            }
                            // case by case	: X-
                            if (mask[xyz-1]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += mems[m][xyz-1]*mems[m][xyz-1];
                                neighbors ++;
                            }
                            // case by case	: Y+
                            if (mask[xyz+nx]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += mems[m][xyz+nx]*mems[m][xyz+nx];
                                neighbors ++;
                            }
                            // case by case	: Y-
                            if (mask[xyz-nx]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += mems[m][xyz-nx]*mems[m][xyz-nx];
                                neighbors ++;
                            }
                            // case by case	: Z+
                            if (mask[xyz+nx*ny]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += mems[m][xyz+nx*ny]*mems[m][xyz+nx*ny];
                                neighbors ++;
                            }
                            // case by case	: Z-
                            if (mask[xyz-nx*ny]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += mems[m][xyz-nx*ny]*mems[m][xyz-nx*ny];
                                neighbors ++;
                            }
                            if (neighbors>0.0) num = num + smoothing*ngb/neighbors;
                        }
                        // invert the result
                        if (num>ZERO) num = 1.0f/num;
                        else num = INF;
    
                        //mems[k][x][y][z] = num;
                        mems[k][xyz] = num;
                        den += num;
                    }
    
                    // normalization
                    for (int k=0;k<clusters;k++) {
                        mems[k][xyz] = mems[k][xyz]/den;
    
                        // compute the maximum distance
                        dist = Math.abs(mems[k][xyz]-prev[k]);
                        if (dist > distance) distance = dist;
                    }
                } else {
                    for (int k=0;k<clusters;k++) 
                        mems[k][xyz] = 0.0f;
                }
            }
        } else {
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) {
                int xyz = x+nx*y;
                if ( mask[xyz] ) {
                    den = 0;
                    // remember the previous values
                    for (int k=0;k<clusters;k++) 
                        prev[k] = mems[k][xyz];
    
                    for (int k=0;k<clusters;k++) {
                        
                        // data term
                        num = 0.0f;
                        for (int c=0;c<nc;c++) {
                            num += (images[c][xyz]-centroids[c][k])*(images[c][xyz]-centroids[c][k])/(Iscale[c]*Iscale[c]);
                        }    
                        
                        // spatial smoothing
                        if (smoothing > 0.0f) { 
                            ngb = 0.0f;  
                            neighbors = 0.0f;
                            // case by case	: X+
                            if (mask[xyz+1]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += mems[m][xyz+1]*mems[m][xyz+1];
                                neighbors ++;
                            }
                            // case by case	: X-
                            if (mask[xyz-1]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += mems[m][xyz-1]*mems[m][xyz-1];
                                neighbors ++;
                            }
                            // case by case	: Y+
                            if (mask[xyz+nx]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += mems[m][xyz+nx]*mems[m][xyz+nx];
                                neighbors ++;
                            }
                            // case by case	: Y-
                            if (mask[xyz-nx]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += mems[m][xyz-nx]*mems[m][xyz-nx];
                                neighbors ++;
                            }
                            if (neighbors>0.0) num = num + smoothing*ngb/neighbors;
                        }
                        // invert the result
                        if (num>ZERO) num = 1.0f/num;
                        else num = INF;
    
                        //mems[k][x][y][z] = num;
                        mems[k][xyz] = num;
                        den += num;
                    }
    
                    // normalization
                    for (int k=0;k<clusters;k++) {
                        mems[k][xyz] = mems[k][xyz]/den;
    
                        // compute the maximum distance
                        dist = Math.abs(mems[k][xyz]-prev[k]);
                        if (dist > distance) distance = dist;
                    }
                } else {
                    for (int k=0;k<clusters;k++) 
                        mems[k][xyz] = 0.0f;
                }
            }
        }
        return distance;
    } // computeMemberships
    
    /** 
	 *  compute the FCM membership functions given the centroids
	 *	with more complex fuzziness values
	 */
    final public float computeGeneralMemberships() {
        float distance,dist;
        float den,num;
        float neighbors, ngb;
        
        distance = 0.0f;
        if (nz>1) {
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if ( mask[xyz] ) {
                    den = 0;
                    // remember the previous values
                    for (int k=0;k<clusters;k++) 
                        prev[k] = mems[k][xyz];
    
                    for (int k=0;k<clusters;k++) {
                        
                        // data term
                        num = 0.0f;
                        for (int c=0;c<nc;c++) {
                            num += (images[c][xyz]-centroids[c][k])*(images[c][xyz]-centroids[c][k])/(Iscale[c]*Iscale[c]);
                        }
                        
                        // spatial smoothing
                        if (smoothing > 0.0f) { 
                            ngb = 0.0f;  
                            neighbors = 0.0f;
                            // case by case	: X+
                            if (mask[xyz+1]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += power.lookup(mems[m][xyz+1],fuzziness);
                                neighbors ++;
                            }
                            // case by case	: X-
                            if (mask[xyz-1]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += power.lookup(mems[m][xyz-1],fuzziness);
                                neighbors ++;
                            }
                            // case by case	: Y+
                            if (mask[xyz+nx]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += power.lookup(mems[m][xyz+nx],fuzziness);
                                neighbors ++;
                            }
                            // case by case	: Y-
                            if (mask[xyz-nx]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += power.lookup(mems[m][xyz-nx],fuzziness);
                                neighbors ++;
                            }
                            // case by case	: Z+
                            if (mask[xyz+nx*ny]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += power.lookup(mems[m][xyz+nx*ny],fuzziness);
                                neighbors ++;
                            }
                            // case by case	: Z-
                            if (mask[xyz-nx*ny]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += power.lookup(mems[m][xyz-nx*ny],fuzziness);
                                neighbors ++;
                            }
                            if (neighbors>0.0) num = num + smoothing*ngb/neighbors;
                        }
                        // invert the result
                        if (num>ZERO) num = (float)invpower.lookup(num,1.0f/(1.0f-fuzziness) );
                        else num = INF;
    
                        mems[k][xyz] = num;
                        den += num;
                    }
    
                    // normalization
                    for (int k=0;k<clusters;k++) {
                        mems[k][xyz] = mems[k][xyz]/den;
    
                        // compute the maximum distance
                        dist = Math.abs(mems[k][xyz]-prev[k]);
                        if (dist > distance) distance = dist;
                    }
                } else {
                    for (int k=0;k<clusters;k++) 
                        mems[k][xyz] = 0.0f;
                }
            }
        } else {
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) {
                int xyz = x+nx*y;
                if ( mask[xyz] ) {
                    den = 0;
                    // remember the previous values
                    for (int k=0;k<clusters;k++) 
                        prev[k] = mems[k][xyz];
    
                    for (int k=0;k<clusters;k++) {
                        
                        // data term
                        num = 0.0f;
                        for (int c=0;c<nc;c++) {
                            num += (images[c][xyz]-centroids[c][k])*(images[c][xyz]-centroids[c][k])/(Iscale[c]*Iscale[c]);
                        }
                        
                        // spatial smoothing
                        if (smoothing > 0.0f) { 
                            ngb = 0.0f;  
                            neighbors = 0.0f;
                            // case by case	: X+
                            if (mask[xyz+1]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += power.lookup(mems[m][xyz+1],fuzziness);
                                neighbors ++;
                            }
                            // case by case	: X-
                            if (mask[xyz-1]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += power.lookup(mems[m][xyz-1],fuzziness);
                                neighbors ++;
                            }
                            // case by case	: Y+
                            if (mask[xyz+nx]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += power.lookup(mems[m][xyz+nx],fuzziness);
                                neighbors ++;
                            }
                            // case by case	: Y-
                            if (mask[xyz-nx]) for (int m=0;m<clusters;m++) if (m!=k) {
                                ngb += power.lookup(mems[m][xyz-nx],fuzziness);
                                neighbors ++;
                            }
                            if (neighbors>0.0) num = num + smoothing*ngb/neighbors;
                        }
                        // invert the result
                        if (num>ZERO) num = (float)invpower.lookup(num,1.0f/(1.0f-fuzziness) );
                        else num = INF;
    
                        mems[k][xyz] = num;
                        den += num;
                    }
    
                    // normalization
                    for (int k=0;k<clusters;k++) {
                        mems[k][xyz] = mems[k][xyz]/den;
    
                        // compute the maximum distance
                        dist = Math.abs(mems[k][xyz]-prev[k]);
                        if (dist > distance) distance = dist;
                    }
                } else {
                    for (int k=0;k<clusters;k++) 
                        mems[k][xyz] = 0.0f;
                }
            }
        }
        return distance;
    } // computeGeneralMemberships
    
    /**
	 * compute the centroids given the membership functions
	 */
    final public void computeCentroids() {
        float num,den;
		
		if (fuzziness!=2) {
			computeGeneralCentroids();
			return;
		}
        for (int c=0;c<nc;c++) {
            for (int k=0;k<clusters;k++) {
                num = 0;
                den = 0;
                for (int xyz=0;xyz<nxyz;xyz++) {
                    if (mask[xyz]) {
                        num += mems[k][xyz]*mems[k][xyz]*images[c][xyz];
                        den += mems[k][xyz]*mems[k][xyz];
                    }
                }
                if (den>0.0) {
                    centroids[c][k] = num/den;
                } else {
                    centroids[c][k] = 0.0f;
                }				
            }
            if (verbose) {
                System.out.print("centroids "+c+": ("+centroids[c][0]);
                for (int k=1;k<clusters;k++) System.out.print(", "+centroids[c][k]);
                System.out.print(")\n");
            }  
        } 
        return;
    } // computeCentroids
       
    /**
	 * compute the centroids given the membership functions
	 * for any value of the fuzziness
	 */
    final public void computeGeneralCentroids() {
        float num,den;
		float val;
		
		for (int c=0;c<nc;c++) {
            for (int k=0;k<clusters;k++) {
                num = 0;
                den = 0;
                for (int xyz=0;xyz<nxyz;xyz++) {
                    if (mask[xyz]) {
                        val = (float)power.lookup(mems[k][xyz],fuzziness);
                        num += val*images[c][xyz];
                        den += val;
                    }
                }
                if (den>0.0) {
                    centroids[c][k] = num/den;
                } else {
                    centroids[c][k] = 0.0f;
                }
            }
            if (verbose) {
                System.out.print("centroids "+c+": ("+centroids[c][0]);
                for (int k=1;k<clusters;k++) System.out.print(", "+centroids[c][k]);
                System.out.print(")\n");
            }   
        }
        return;
    } // computeCentroids
    
	
	/** 
	 *	create ids for the centroids based on ordering
	 */
	public final int[] computeCentroidOrder(int c) {
		int[]	id = new int[clusters+1];
		int		lowest;
		float[] cent = new float[clusters];
		
		// copy the centroids
        for (int k=0;k<clusters;k++) 
			cent[k] = centroids[c][k];
		
		// add the zero
		id[0] = 0;

		// order them from smallest to largest
		for (int k=0;k<clusters;k++) {
			lowest = 0;
			for (int l=1;l<clusters;l++) {
				if (cent[l] < cent[lowest]) {
					lowest = l;
				}
			}
			id[k+1] = lowest+1;
			cent[lowest] = INF;
		}
        if (debug) {
			System.out.print("ordering: ("+id[0]);
			for (int k=0;k<clusters;k++)System.out.print(", "+id[k+1]);
			System.out.print(")\n");
		}   
 		return id;
	} // computeCentroidOrder

	/** 
	 *	build a piecewise-linear intensity map based on centroids
	 */
	public final void interpolateCentroidIntensities() {
	    for (int c=0;c<nc;c++) {
            int[] order = computeCentroidOrder(c);
            for (int xyz=0;xyz<nxyz;xyz++) {
                if (mask[xyz]) {
                    float intens = 0.0f;
                    if (images[c][xyz]<centroids[c][order[1]-1]) {
                        //intens = 2.0f - (centroids[id[2]-1]-image[xyz])/(centroids[id[2]-1]-centroids[id[1]-1]);
                        intens = 0.0f + Numerics.max(0.001f,(images[c][xyz] - Imin[c])/(centroids[c][order[1]-1]-Imin[c]));
                    } else if (images[c][xyz]>=centroids[c][order[clusters]-1]) {
                        //intens = clusters-1.0f + (image[xyz]-centroids[id[clusters-1]-1])/(centroids[id[clusters]-1]-centroids[id[clusters-1]-1]);
                        intens = clusters + Numerics.min(0.999f,(images[c][xyz]-centroids[c][order[clusters]-1])/(Imax[c]-centroids[c][order[clusters]-1]));
                    } else {
                        for (int k=0;k<clusters-1;k++) {
                            if (images[c][xyz]>=centroids[c][order[k+1]-1] && images[c][xyz]<centroids[c][order[k+2]-1]) {
                                intens = (k+1.0f) + (images[c][xyz]-centroids[c][order[k+1]-1])/(centroids[c][order[k+2]-1]-centroids[c][order[k+1]-1]);
                            }
                        }
                    }
                    images[c][xyz] = intens;
                }
            }
        }
        return;
	}

}
