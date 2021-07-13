package nl.uva.imcn.algorithms;

import nl.uva.imcn.libraries.*;
import nl.uva.imcn.structures.*;
import nl.uva.imcn.utilities.*;
import org.apache.commons.math3.util.FastMath;

/*
 * @author Pierre-Louis bazin (pilou.bazin@uva.nl)
 *
 */
public class DirectionalLineClustering {
	private 	int[][] lines = null;
	private 	float[][] dirs = null;
	private 	float[][] scales = null;
	
	private 	float thickness = 20.0f;
	private 	float vardist = 20.0f;
	private 	float vartheta = 15.0f;
	private 	float threshold = 0.5f;
	private     float anisotropy = 0.0f;
	
	private     boolean recompute = true;
	private     int mip=0;
	
	private 	int nx, ny, nz, nxyz;
	private 	float rx, ry, rz;
	private     int nimg;
	
	private 	float[] dir3d;
	private 	int[] groups;
	
	// set inputs
	public final void setImageNumber(int n) { 
	    nimg = n;
	    lines = new int[nimg][];
	    dirs = new float[nimg][];
	    scales = new float[nimg][];
	}
	public final void setLineImageAt(int n, int[] val) { lines[n] = val; }
	public final void setDirImageAt(int n, float[] val) { dirs[n] = val; }
	public final void setScaleImageAt(int n, float[] val) { scales[n] = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }

	public final void setSliceThickness(float val) { thickness = val; }
	public final void setExpectedDistance(float val) { vardist = val; }
	public final void setExpectedAngle(float val) { vartheta = val; }
	public final void setProbabilityThreshold(float val) { threshold = val; }
	public final void setDistanceAnisotropy(float val) { anisotropy = val; }
	public final void setRecomputeLabels(boolean val) { recompute = val; }
	public final void setMaxIntensityProjection(int val) { mip = val; }
	
	// outputs
	public final float[] getDirectionImage() { return dir3d; }
	public final int[] getGroupImage() { return groups; }

	public void execute2D() {

	    // get list of labels from line segmentation: 
	    // assume we have already separate labels or compute them
	    if (recompute) {
	        for (int n=0;n<nimg;n++) {
	            lines[n] = ObjectLabeling.connected26Object3D(lines[n], nx,ny,nz);
	        }
	    }
	    int[] nl = new int[nimg];
	    int nlt = 0;
	    for (int n=0;n<nimg;n++) {
	        nl[n] = ObjectLabeling.countLabels(lines[n], nx, ny, nz);
	        nlt += nl[n];
	    }
	    System.out.println(" found "+nlt+" lines");
	
	    double[] avgscale = new double[nlt];
	    double[] avgsize = new double[nlt];
	    double[][] avgdir = new double[nlt][3];
	    double[][] avgloc = new double[nlt][2];

		for (int n=0;n<nimg;n++) for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
			int xyz = x + nx*y;
			if (lines[n][xyz]>0) {
                int offset = 0;
                for (int n0=0;n0<n;n0++) offset += nl[n0];
                int idx = offset + lines[n][xyz]-1;
                
                avgsize[idx] += 1.0;
                avgscale[idx] += (1.0+scales[n][xyz]);
                
                // in the 2D filter the direction is orthogonal to the detected line!
                if (avgdir[idx][0]*dirs[n][xyz+0*nxyz]-avgdir[idx][1]*dirs[n][xyz+1*nxyz]>=0) {
                    avgdir[idx][0] += dirs[n][xyz+0*nxyz];
                    avgdir[idx][1] += -dirs[n][xyz+1*nxyz];
                } else {
                    avgdir[idx][0] -= dirs[n][xyz+0*nxyz];
                    avgdir[idx][1] -= -dirs[n][xyz+1*nxyz];
                }
                
                avgloc[idx][0] += x;
                avgloc[idx][1] += y;
            }
        }
		
		for (int idx=0;idx<nlt;idx++) {
		    avgscale[idx] /= avgsize[idx];
		    avgdir[idx][0] /= avgsize[idx];
		    avgdir[idx][1] /= avgsize[idx];
		    avgloc[idx][0] /= avgsize[idx];
		    avgloc[idx][1] /= avgsize[idx];
		}
		
		// find endpoints for better distance measure ordered [-1;0;+1]
		int[][] endpoints = new int[nlt][6];
		double[][] endptdist = new double[nlt][3];
		for (int idx=0;idx<nlt;idx++) {
		    endptdist[idx][0] = +1e9;
		    endptdist[idx][1] = +1e9;
		    endptdist[idx][2] = -1e9;
		}
		for (int n=0;n<nimg;n++) for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
			int xyz = x + nx*y;
			if (lines[n][xyz]>0) {
                int offset = 0;
                for (int n0=0;n0<n;n0++) offset += nl[n0];
                int idx = offset + lines[n][xyz]-1;
                
                // projection onto main direction
                double newdist = (x-avgloc[idx][0])*avgdir[idx][0] + (y-avgloc[idx][1])*avgdir[idx][1];
                
                if (newdist<endptdist[idx][0]) {
                    endpoints[idx][0] = x;
                    endpoints[idx][1] = y;
                    
                    endptdist[idx][0] = newdist;
                }
                if (Numerics.abs(newdist)<endptdist[idx][1]) {
                    endpoints[idx][2] = x;
                    endpoints[idx][3] = y;
                    
                    endptdist[idx][1] = Numerics.abs(newdist);
                }
                if (newdist>endptdist[idx][2]) {
                    endpoints[idx][4] = x;
                    endpoints[idx][5] = y;
                    
                    endptdist[idx][2] = newdist;
                }
            }
        }
        // better length: computed from endpoints
		double[] ptlength = new double[nlt];
        for (int idx=0;idx<nlt;idx++) {
            ptlength[idx] = FastMath.sqrt( Numerics.square(endpoints[idx][0]-endpoints[idx][2])
		                                  +Numerics.square(endpoints[idx][1]-endpoints[idx][3]) )
		                   +FastMath.sqrt( Numerics.square(endpoints[idx][2]-endpoints[idx][4])
		                                  +Numerics.square(endpoints[idx][3]-endpoints[idx][5]) );
		}
		
		// compute elevation
		for (int idx=0;idx<nlt;idx++) {
		    double norm = FastMath.sqrt(avgdir[idx][0]*avgdir[idx][0]+avgdir[idx][1]*avgdir[idx][1]);
		    //double length = avgsize[idx]/avgscale[idx];
		    double length = ptlength[idx];
		    if (norm==0 || length==0) {
		        //System.err.print("!");
		        avgdir[idx][0] = 0.0;
                avgdir[idx][1] = 0.0;
                avgdir[idx][2] = 1.0;               
		    } else { 
                avgdir[idx][0] = length*avgdir[idx][0]/norm;
                avgdir[idx][1] = length*avgdir[idx][1]/norm;
                avgdir[idx][2] = thickness;
                norm = FastMath.sqrt(avgdir[idx][0]*avgdir[idx][0]+avgdir[idx][1]*avgdir[idx][1]+avgdir[idx][2]*avgdir[idx][2]);
                avgdir[idx][0] /= norm;
                avgdir[idx][1] /= norm;
                avgdir[idx][2] /= norm;
            }
		}
		
		// build pairwise distances from location and angles
		vardist *= vardist;
		vartheta = (float)(vartheta/180.0*FastMath.PI);
		vartheta *= vartheta;
		
		// sanity check
		double[] mintheta = new double[nlt];
		double[] distance = new double[nlt];
		double[] maxproba = new double[nlt];
		for (int idx=0;idx<nlt;idx++) {
		    mintheta[idx] = 4.0*vartheta;
		    distance[idx] = 4.0*vardist;
		    maxproba[idx] = 0.0;
		}
		
		BinaryHeapPair heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeapPair.MAXTREE);
		int ntree = 0;
		for (int idx1=0;idx1<nlt;idx1++) {
		    for (int idx2=idx1+1;idx2<nlt;idx2++) {
		        //double dist = (avgloc[idx1][0]-avgloc[idx2][0])
		        //              +(avgloc[idx1][1]-avgloc[idx2][1]);
		        // basic distance: to exclude far locations quickly
		        /* not working?
		        double dist0 = Numerics.square(avgloc[idx1][0]-avgloc[idx2][0])
		                      +Numerics.square(avgloc[idx1][1]-avgloc[idx2][1]);
		        if (dist0<4.0*vardist+ptlength[idx1]*ptlength[idx1]+ptlength[idx2]*ptlength[idx2]) {
		        */
                // compute angles
                double theta = FastMath.acos(Numerics.abs(avgdir[idx1][0]*avgdir[idx2][0]
                                                         +avgdir[idx1][1]*avgdir[idx2][1]
                                                         +avgdir[idx1][2]*avgdir[idx2][2]));
                theta *= theta;
                
                //if (theta<4.0*vartheta) {
                // compute distances
                double mindist;
                if (anisotropy>0) {
                    double dx = avgsize[idx1]*avgdir[idx1][0]+avgsize[idx2]*avgdir[idx2][0];
                    double dy = avgsize[idx1]*avgdir[idx1][1]+avgsize[idx2]*avgdir[idx2][1];
                    double norm = FastMath.sqrt(dx*dx+dy*dy);
                    
                    double d12x = endpoints[idx1][0]-endpoints[idx2][0];
                    double d12y = endpoints[idx1][1]-endpoints[idx2][1];
                    mindist = d12x*d12x+d12y*d12y;
                    double newdist = Numerics.square(endpoints[idx1][0]-endpoints[idx2][4])
                                    +Numerics.square(endpoints[idx1][1]-endpoints[idx2][5]);
                    if (newdist<mindist) {
                        mindist = newdist;
                        d12x = endpoints[idx1][0]-endpoints[idx2][4];
                        d12y = endpoints[idx1][1]-endpoints[idx2][5];
                    }
                    newdist = Numerics.square(endpoints[idx1][4]-endpoints[idx2][4])
                             +Numerics.square(endpoints[idx1][5]-endpoints[idx2][5]);
                    if (newdist<mindist) {
                        mindist = newdist;
                        d12x = endpoints[idx1][4]-endpoints[idx2][4];
                        d12y = endpoints[idx1][5]-endpoints[idx2][5];
                    }
                    newdist = Numerics.square(endpoints[idx1][4]-endpoints[idx2][0])
                             +Numerics.square(endpoints[idx1][5]-endpoints[idx2][1]);
                    if (newdist<mindist) {
                        mindist = newdist;
                        d12x = endpoints[idx1][4]-endpoints[idx2][0];
                        d12y = endpoints[idx1][5]-endpoints[idx2][1];
                    }
                    
                    mindist = 1.0/(1.0-anisotropy)*(mindist-Numerics.square(d12x*dx/norm+d12y*dy/norm))
                              +Numerics.square(d12x*dx/norm+d12y*dy/norm);
                    
                } else {
                    mindist = Numerics.square(endpoints[idx1][0]-endpoints[idx2][0])
                             +Numerics.square(endpoints[idx1][1]-endpoints[idx2][1]);
                    mindist = Numerics.min(mindist, Numerics.square(endpoints[idx1][0]-endpoints[idx2][4])
                                                   +Numerics.square(endpoints[idx1][1]-endpoints[idx2][5]));                
                    mindist = Numerics.min(mindist, Numerics.square(endpoints[idx1][4]-endpoints[idx2][4])
                                                   +Numerics.square(endpoints[idx1][5]-endpoints[idx2][5]));                
                    mindist = Numerics.min(mindist, Numerics.square(endpoints[idx1][4]-endpoints[idx2][0])
                                                   +Numerics.square(endpoints[idx1][5]-endpoints[idx2][1]));                
                }
                
                double proba = FastMath.sqrt(FastMath.exp(-0.5*mindist/vardist)
                                            *FastMath.exp(-0.5*theta/vartheta));
                
                if (proba>threshold) {
                    heap.addValue((float)proba,idx1,idx2);
                    ntree++;
                }
                
                if (mindist<distance[idx1]) distance[idx1] = mindist;
                if (mindist<distance[idx2]) distance[idx2] = mindist;
                
                if (theta<mintheta[idx1]) mintheta[idx1] = theta;
                if (theta<mintheta[idx2]) mintheta[idx2] = theta;
                
                if (proba>maxproba[idx1]) maxproba[idx1] = proba;
                if (proba>maxproba[idx2]) maxproba[idx2] = proba;
                //}
		        //}
		    }
		}
		System.out.println(" computing "+ntree+" correlations");
		
		// travel through the tree to relabel groups
		int[] relabel = new int[nlt];
		for (int idx=0;idx<nlt;idx++) {
		    relabel[idx] = idx;
		}
		while (heap.isNotEmpty()) {
        	// extract point with minimum distance
        	float dist = heap.getFirst();
        	int idx1 = heap.getFirstId1();
        	int idx2 = heap.getFirstId2();
			heap.removeFirst();
			
			if (idx1<idx2) {
			    for (int idx=0;idx<nlt;idx++) {
			        if (relabel[idx]==idx2) {
                        relabel[idx] = idx1;
                    }
                }
			} else {
			    for (int idx=0;idx<nlt;idx++) {
			        if (relabel[idx]==idx1) {
                        relabel[idx] = idx2;
                    }
                }
			}
		}
		// renumber to keep small values
		int[] labellist = ObjectLabeling.listOrderedLabels(relabel, nlt, 1, 1);
		int[] newlabel = new int[nlt];
		for (int idx=0;idx<nlt;idx++) {
		    for (int lb=0;lb<labellist.length;lb++) {
		        if (relabel[idx]==labellist[lb]) {
                    newlabel[idx] = lb;
		        }
		    }
		}
				
		// map the results back to images
		dir3d = new float[3*nimg*nxyz];
		groups = new int[nimg*nxyz];
		for (int n=0;n<nimg;n++) for (int xyz=0;xyz<nxyz;xyz++) if (lines[n][xyz]>0) {
            int offset = 0;
            for (int n0=0;n0<n;n0++) offset += nl[n0];
            int idx = offset + lines[n][xyz]-1;
                
		    dir3d[xyz+0*nxyz+3*n*nxyz] = (float)avgdir[idx][0];
		    dir3d[xyz+1*nxyz+3*n*nxyz] = (float)avgdir[idx][1];
		    dir3d[xyz+2*nxyz+3*n*nxyz] = (float)avgdir[idx][2];
		    
		    // for debug
		    dir3d[xyz+0*nxyz+3*n*nxyz] = (float)FastMath.sqrt(distance[idx]/vardist);
		    dir3d[xyz+1*nxyz+3*n*nxyz] = (float)FastMath.sqrt(mintheta[idx]/vartheta);
		    dir3d[xyz+2*nxyz+3*n*nxyz] = (float)maxproba[idx];
		    
			groups[xyz+n*nxyz] = 1+newlabel[idx];
		}
		/* not needed, working properly
		// add the endpoints for sanity check
		for (int idx=0;idx<nl;idx++) {
		    int xyz = (int)endpoints[idx][0] + nx*(int)endpoints[idx][1];
		    dir3d[xyz+0*nxyz] = 1.0f;
		    
		    xyz = (int)endpoints[idx][2] + nx*(int)endpoints[idx][3];
		    dir3d[xyz+1*nxyz] = 1.0f;
		    
		    xyz = (int)endpoints[idx][4] + nx*(int)endpoints[idx][5];
		    dir3d[xyz+2*nxyz] = 1.0f;
		}*/    
		return;
	}

	public void execute3D() {

	    // get list of labels from line segmentation: 
	    // assume we have already separate labels or compute them
	    if (recompute) {
	        for (int n=0;n<nimg;n++) {
	            lines[n] = ObjectLabeling.connected26Object3D(lines[n], nx,ny,nz);
	        }
	    } 
	    int[] nl = new int[nimg];
	    int nlt = 0;
	    for (int n=0;n<nimg;n++) {
	        nl[n] = ObjectLabeling.countLabels(lines[n], nx, ny, nz);
	        nlt += nl[n];
	    }
	    
	    System.out.println(" found "+nlt+" lines");
	
	    double[] avgscale = new double[nlt];
	    double[] avgsize = new double[nlt];
	    double[][] avgdir = new double[nlt][3];
	    double[][] avgloc = new double[nlt][3];

		for (int n=0;n<nimg;n++) for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x + nx*y + nx*ny*z;
			if (lines[n][xyz]>0) {
                int offset = 0;
                for (int n0=0;n0<n;n0++) offset += nl[n0];
                int idx = offset + lines[n][xyz]-1;
                
                avgsize[idx] += 1.0;
                avgscale[idx] += (1.0+scales[n][xyz]);
                
                // in 3D the direction follows the detected line
                if ( avgdir[idx][0]*dirs[n][xyz+0*nxyz]
                    +avgdir[idx][1]*dirs[n][xyz+1*nxyz]
                    +avgdir[idx][2]*dirs[n][xyz+2*nxyz]>=0 ) {
                
                    avgdir[idx][0] += dirs[n][xyz+0*nxyz];
                    avgdir[idx][1] += dirs[n][xyz+1*nxyz];
                    avgdir[idx][2] += dirs[n][xyz+2*nxyz];
                } else {
                    avgdir[idx][0] -= dirs[n][xyz+0*nxyz];
                    avgdir[idx][1] -= dirs[n][xyz+1*nxyz];
                    avgdir[idx][2] -= dirs[n][xyz+2*nxyz];
                }
                
                avgloc[idx][0] += x;
                avgloc[idx][1] += y;
                avgloc[idx][2] += z;
            }
        }
		
		for (int idx=0;idx<nlt;idx++) {
		    avgscale[idx] /= avgsize[idx];
		    
		    avgdir[idx][0] /= avgsize[idx];
		    avgdir[idx][1] /= avgsize[idx];
		    avgdir[idx][2] /= avgsize[idx];
		    
		    avgloc[idx][0] /= avgsize[idx];
		    avgloc[idx][1] /= avgsize[idx];
		    avgloc[idx][2] /= avgsize[idx];
		}
		
		// find endpoints for better distance measure ordered [-1;0;+1]
		int[][] endpoints = new int[nlt][9];
		double[][] endptdist = new double[nlt][3];
		for (int idx=0;idx<nlt;idx++) {
		    endptdist[idx][0] = +1e9;
		    endptdist[idx][1] = +1e9;
		    endptdist[idx][2] = -1e9;
		}
		for (int n=0;n<nimg;n++) for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x + nx*y + nx*ny*z;
			if (lines[n][xyz]>0) {
			    int offset = 0;
                for (int n0=0;n0<n;n0++) offset += nl[n0];
                int idx = offset + lines[n][xyz]-1;
                                
                // projection onto main direction
                double newdist = (x-avgloc[idx][0])*avgdir[idx][0]
                               + (y-avgloc[idx][1])*avgdir[idx][1]
                               + (z-avgloc[idx][2])*avgdir[idx][2];
                
                if (newdist<endptdist[idx][0]) {
                    endpoints[idx][0] = x;
                    endpoints[idx][1] = y;
                    endpoints[idx][2] = z;
                    
                    endptdist[idx][0] = newdist;
                }
                if (Numerics.abs(newdist)<endptdist[idx][1]) {
                    endpoints[idx][3] = x;
                    endpoints[idx][4] = y;
                    endpoints[idx][5] = z;
                    
                    endptdist[idx][1] = Numerics.abs(newdist);
                }
                if (newdist>endptdist[idx][2]) {
                    endpoints[idx][6] = x;
                    endpoints[idx][7] = y;
                    endpoints[idx][8] = z;
                    
                    endptdist[idx][2] = newdist;
                }
            }
        }
        // better length: computed from endpoints
		double[] ptlength = new double[nlt];
        for (int idx=0;idx<nlt;idx++) {
            ptlength[idx] = FastMath.sqrt( Numerics.square(endpoints[idx][0]-endpoints[idx][3])
		                                  +Numerics.square(endpoints[idx][1]-endpoints[idx][4])
		                                  +Numerics.square(endpoints[idx][2]-endpoints[idx][5]) )
		                   +FastMath.sqrt( Numerics.square(endpoints[idx][3]-endpoints[idx][6])
		                                  +Numerics.square(endpoints[idx][4]-endpoints[idx][7])
		                                  +Numerics.square(endpoints[idx][5]-endpoints[idx][8]) );
		}
		
		// no need to compute elevation
		
		// build pairwise distances from location and angles
		vardist *= vardist;
		vartheta = (float)(vartheta/180.0*FastMath.PI);
		vartheta *= vartheta;
		
		// sanity check
		double[] mintheta = new double[nlt];
		double[] distance = new double[nlt];
		double[] maxproba = new double[nlt];
		for (int idx=0;idx<nlt;idx++) {
		    mintheta[idx] = 4.0*vartheta;
		    distance[idx] = 4.0*vardist;
		    maxproba[idx] = 0.0;
		}
		
		BinaryHeapPair heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeapPair.MAXTREE);
		int ntree = 0;
		for (int idx1=0;idx1<nlt;idx1++) {
		    for (int idx2=idx1+1;idx2<nlt;idx2++) {
		        //double dist = (avgloc[idx1][0]-avgloc[idx2][0])
		        //              +(avgloc[idx1][1]-avgloc[idx2][1]);
		        // basic distance: to exclude far locations quickly
		        /* not working?
		        double dist0 = Numerics.square(avgloc[idx1][0]-avgloc[idx2][0])
		                      +Numerics.square(avgloc[idx1][1]-avgloc[idx2][1]);
		        if (dist0<4.0*vardist+ptlength[idx1]*ptlength[idx1]+ptlength[idx2]*ptlength[idx2]) {
		        */
                // compute angles
                double theta = FastMath.acos(Numerics.abs(avgdir[idx1][0]*avgdir[idx2][0]
                                                         +avgdir[idx1][1]*avgdir[idx2][1]
                                                         +avgdir[idx1][2]*avgdir[idx2][2]));
                theta *= theta;
                
                //if (theta<4.0*vartheta) {
                // compute distances
                double mindist;
                if (anisotropy>0) {
                    double dx = avgsize[idx1]*avgdir[idx1][0]+avgsize[idx2]*avgdir[idx2][0];
                    double dy = avgsize[idx1]*avgdir[idx1][1]+avgsize[idx2]*avgdir[idx2][1];
                    double dz = avgsize[idx1]*avgdir[idx1][2]+avgsize[idx2]*avgdir[idx2][2];
                    double norm = FastMath.sqrt(dx*dx+dy*dy+dz*dz);
                    
                    double d12x = endpoints[idx1][0]-endpoints[idx2][0];
                    double d12y = endpoints[idx1][1]-endpoints[idx2][1];
                    double d12z = endpoints[idx1][2]-endpoints[idx2][2];
                    mindist = d12x*d12x+d12y*d12y+d12z*d12z;
                    
                    double newdist = Numerics.square(endpoints[idx1][0]-endpoints[idx2][6])
                                    +Numerics.square(endpoints[idx1][1]-endpoints[idx2][7])
                                    +Numerics.square(endpoints[idx1][2]-endpoints[idx2][8]);
                    if (newdist<mindist) {
                        mindist = newdist;
                        d12x = endpoints[idx1][0]-endpoints[idx2][6];
                        d12y = endpoints[idx1][1]-endpoints[idx2][7];
                        d12z = endpoints[idx1][2]-endpoints[idx2][8];
                    }
                    
                    newdist = Numerics.square(endpoints[idx1][6]-endpoints[idx2][6])
                             +Numerics.square(endpoints[idx1][7]-endpoints[idx2][7])
                             +Numerics.square(endpoints[idx1][8]-endpoints[idx2][8]);
                    if (newdist<mindist) {
                        mindist = newdist;
                        d12x = endpoints[idx1][6]-endpoints[idx2][6];
                        d12y = endpoints[idx1][7]-endpoints[idx2][7];
                        d12z = endpoints[idx1][8]-endpoints[idx2][8];
                    }
                    
                    newdist = Numerics.square(endpoints[idx1][6]-endpoints[idx2][0])
                             +Numerics.square(endpoints[idx1][7]-endpoints[idx2][1])
                             +Numerics.square(endpoints[idx1][8]-endpoints[idx2][2]);
                    if (newdist<mindist) {
                        mindist = newdist;
                        d12x = endpoints[idx1][6]-endpoints[idx2][0];
                        d12y = endpoints[idx1][7]-endpoints[idx2][1];
                        d12z = endpoints[idx1][8]-endpoints[idx2][2];
                    }
                    
                    mindist = 1.0/(1.0-anisotropy)*(mindist-Numerics.square(d12x*dx/norm+d12y*dy/norm+d12z*dz/norm))
                              +Numerics.square(d12x*dx/norm+d12y*dy/norm+d12z*dz/norm);
                    
                } else {
                    mindist = Numerics.square(endpoints[idx1][0]-endpoints[idx2][0])
                             +Numerics.square(endpoints[idx1][1]-endpoints[idx2][1])
                             +Numerics.square(endpoints[idx1][2]-endpoints[idx2][2]);
                             
                    mindist = Numerics.min(mindist, Numerics.square(endpoints[idx1][0]-endpoints[idx2][6])
                                                   +Numerics.square(endpoints[idx1][1]-endpoints[idx2][7])
                                                   +Numerics.square(endpoints[idx1][2]-endpoints[idx2][8]));  
                    
                    mindist = Numerics.min(mindist, Numerics.square(endpoints[idx1][6]-endpoints[idx2][6])
                                                   +Numerics.square(endpoints[idx1][7]-endpoints[idx2][7])
                                                   +Numerics.square(endpoints[idx1][8]-endpoints[idx2][8]));  
                    
                    mindist = Numerics.min(mindist, Numerics.square(endpoints[idx1][6]-endpoints[idx2][0])
                                                   +Numerics.square(endpoints[idx1][7]-endpoints[idx2][1])
                                                   +Numerics.square(endpoints[idx1][8]-endpoints[idx2][2]));                
                }
                
                double proba = FastMath.sqrt(FastMath.exp(-0.5*mindist/vardist)
                                            *FastMath.exp(-0.5*theta/vartheta));
                
                if (proba>threshold) {
                    heap.addValue((float)proba,idx1,idx2);
                    ntree++;
                }
                
                if (mindist<distance[idx1]) distance[idx1] = mindist;
                if (mindist<distance[idx2]) distance[idx2] = mindist;
                
                if (theta<mintheta[idx1]) mintheta[idx1] = theta;
                if (theta<mintheta[idx2]) mintheta[idx2] = theta;
                
                if (proba>maxproba[idx1]) maxproba[idx1] = proba;
                if (proba>maxproba[idx2]) maxproba[idx2] = proba;
                //}
		        //}
		    }
		}
		System.out.println(" computing "+ntree+" correlations");
		
		// travel through the tree to relabel groups
		int[] relabel = new int[nlt];
		for (int idx=0;idx<nlt;idx++) {
		    relabel[idx] = idx;
		}
		while (heap.isNotEmpty()) {
        	// extract point with minimum distance
        	float dist = heap.getFirst();
        	int idx1 = heap.getFirstId1();
        	int idx2 = heap.getFirstId2();
			heap.removeFirst();
			
			if (idx1<idx2) {
			    for (int idx=0;idx<nlt;idx++) {
			        if (relabel[idx]==idx2) {
                        relabel[idx] = idx1;
                    }
                }
			} else {
			    for (int idx=0;idx<nlt;idx++) {
			        if (relabel[idx]==idx1) {
                        relabel[idx] = idx2;
                    }
                }
			}
		}
		// renumber to keep small values
		int[] labellist = ObjectLabeling.listOrderedLabels(relabel, nlt, 1, 1);
		int[] newlabel = new int[nlt];
		for (int idx=0;idx<nlt;idx++) {
		    for (int lb=0;lb<labellist.length;lb++) {
		        if (relabel[idx]==labellist[lb]) {
                    newlabel[idx] = lb;
		        }
		    }
		}
				
		// map the results back to images
		dir3d = new float[3*nimg*nxyz];
		groups = new int[nimg*nxyz];
		for (int n=0;n<nimg;n++) for (int xyz=0;xyz<nxyz;xyz++) if (lines[n][xyz]>0) {
            int offset = 0;
            for (int n0=0;n0<n;n0++) offset += nl[n0];
            int idx = offset + lines[n][xyz]-1;
            
		    dir3d[xyz+0*nxyz+3*n*nxyz] = (float)avgdir[idx][0];
		    dir3d[xyz+1*nxyz+3*n*nxyz] = (float)avgdir[idx][1];
		    dir3d[xyz+2*nxyz+3*n*nxyz] = (float)avgdir[idx][2];
		    
		    // for debug
		    dir3d[xyz+0*nxyz+3*n*nxyz] = (float)FastMath.sqrt(distance[idx]/vardist);
		    dir3d[xyz+1*nxyz+3*n*nxyz] = (float)FastMath.sqrt(mintheta[idx]/vartheta);
		    dir3d[xyz+2*nxyz+3*n*nxyz] = (float)maxproba[idx];
		    
			groups[xyz+n*nxyz] = 1+newlabel[idx];
		}
		/* not needed, working properly
		// add the endpoints for sanity check
		for (int idx=0;idx<nl;idx++) {
		    int xyz = (int)endpoints[idx][0] + nx*(int)endpoints[idx][1];
		    dir3d[xyz+0*nxyz] = 1.0f;
		    
		    xyz = (int)endpoints[idx][2] + nx*(int)endpoints[idx][3];
		    dir3d[xyz+1*nxyz] = 1.0f;
		    
		    xyz = (int)endpoints[idx][4] + nx*(int)endpoints[idx][5];
		    dir3d[xyz+2*nxyz] = 1.0f;
		}*/    
		if (mip>0) {
		    for (int n=0;n<nimg;n++) for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz-mip;z++) {
		        int xyz = x + nx*y + nx*ny*z;
		        for (int d=0;d<mip;d++) {
		            groups[xyz+n*nxyz] = Numerics.max(groups[xyz+n*nxyz],groups[xyz+d*nx*ny+n*nxyz]);
		        }
		    }
		}
		return;
	}
	
	public void buildLines3D() {

	    // build lists of detected voxels
	    int[] nl = new int[nimg];
	    int nlt = 0;
	    for (int n=0;n<nimg;n++) {
	        for (int xyz=0;xyz<nxyz;xyz++) {
	            if (lines[n][xyz]>0) {
	                lines[n][xyz]=nlt+1;
	                nlt=nlt+1;
	            }
	        }
	        nl[n] = nlt;
	        nlt = 0;
	    }
	    for (int n=0;n<nimg;n++) nlt += nl[n];
	    
	    System.out.println(" found "+nlt+" voxels");
	
	    double[] avgscale = new double[nlt];
	    double[] avgsize = new double[nlt];
	    double[][] avgdir = new double[nlt][3];
	    double[][] avgloc = new double[nlt][3];

		for (int n=0;n<nimg;n++) for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x + nx*y + nx*ny*z;
			if (lines[n][xyz]>0) {
                int offset = 0;
                for (int n0=0;n0<n;n0++) offset += nl[n0];
                int idx = offset + lines[n][xyz]-1;
                
                avgsize[idx] += 1.0;
                avgscale[idx] += (1.0+scales[n][xyz]);
                
                // in 3D the direction follows the detected line
                if ( avgdir[idx][0]*dirs[n][xyz+0*nxyz]
                    +avgdir[idx][1]*dirs[n][xyz+1*nxyz]
                    +avgdir[idx][2]*dirs[n][xyz+2*nxyz]>=0 ) {
                
                    avgdir[idx][0] += dirs[n][xyz+0*nxyz];
                    avgdir[idx][1] += dirs[n][xyz+1*nxyz];
                    avgdir[idx][2] += dirs[n][xyz+2*nxyz];
                } else {
                    avgdir[idx][0] -= dirs[n][xyz+0*nxyz];
                    avgdir[idx][1] -= dirs[n][xyz+1*nxyz];
                    avgdir[idx][2] -= dirs[n][xyz+2*nxyz];
                }
                
                avgloc[idx][0] += x;
                avgloc[idx][1] += y;
                avgloc[idx][2] += z;
            }
        }
		
		for (int idx=0;idx<nlt;idx++) {
		    avgscale[idx] /= avgsize[idx];
		    
		    avgdir[idx][0] /= avgsize[idx];
		    avgdir[idx][1] /= avgsize[idx];
		    avgdir[idx][2] /= avgsize[idx];
		    
		    avgloc[idx][0] /= avgsize[idx];
		    avgloc[idx][1] /= avgsize[idx];
		    avgloc[idx][2] /= avgsize[idx];
		}
			
		// no need to compute endpoints, length or elevation
		
		// build pairwise distances from location and angles
		vardist *= vardist;
		vartheta = (float)(vartheta/180.0*FastMath.PI);
		vartheta *= vartheta;
		
		// sanity check
		double[] mintheta = new double[nlt];
		double[] distance = new double[nlt];
		double[] maxproba = new double[nlt];
		for (int idx=0;idx<nlt;idx++) {
		    mintheta[idx] = 4.0*vartheta;
		    distance[idx] = 4.0*vardist;
		    maxproba[idx] = 0.0;
		}
		
		BinaryHeapPair heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeapPair.MAXTREE);
		int ntree = 0;
		for (int idx1=0;idx1<nlt;idx1++) {
		    for (int idx2=idx1+1;idx2<nlt;idx2++) {
		        //double dist = (avgloc[idx1][0]-avgloc[idx2][0])
		        //              +(avgloc[idx1][1]-avgloc[idx2][1]);
		        // basic distance: to exclude far locations quickly
		        /* not working?
		        double dist0 = Numerics.square(avgloc[idx1][0]-avgloc[idx2][0])
		                      +Numerics.square(avgloc[idx1][1]-avgloc[idx2][1]);
		        if (dist0<4.0*vardist+ptlength[idx1]*ptlength[idx1]+ptlength[idx2]*ptlength[idx2]) {
		        */
                // compute angles
                double theta = FastMath.acos(Numerics.abs(avgdir[idx1][0]*avgdir[idx2][0]
                                                         +avgdir[idx1][1]*avgdir[idx2][1]
                                                         +avgdir[idx1][2]*avgdir[idx2][2]));
                theta *= theta;
                
                //if (theta<4.0*vartheta) {
                // compute distances
                double mindist;
                if (anisotropy>0) {
                    double dx = avgsize[idx1]*avgdir[idx1][0]+avgsize[idx2]*avgdir[idx2][0];
                    double dy = avgsize[idx1]*avgdir[idx1][1]+avgsize[idx2]*avgdir[idx2][1];
                    double dz = avgsize[idx1]*avgdir[idx1][2]+avgsize[idx2]*avgdir[idx2][2];
                    double norm = FastMath.sqrt(dx*dx+dy*dy+dz*dz);
                    
                    double d12x = avgloc[idx1][0]-avgloc[idx2][0];
                    double d12y = avgloc[idx1][1]-avgloc[idx2][1];
                    double d12z = avgloc[idx1][2]-avgloc[idx2][2];
                    mindist = d12x*d12x+d12y*d12y+d12z*d12z;
                    
                    mindist = 1.0/(1.0-anisotropy)*(mindist-Numerics.square(d12x*dx/norm+d12y*dy/norm+d12z*dz/norm))
                              +Numerics.square(d12x*dx/norm+d12y*dy/norm+d12z*dz/norm);
                    
                } else {
                    mindist = Numerics.square(avgloc[idx1][0]-avgloc[idx2][0])
                             +Numerics.square(avgloc[idx1][1]-avgloc[idx2][1])
                             +Numerics.square(avgloc[idx1][2]-avgloc[idx2][2]);
                }
                
                double proba = FastMath.sqrt(FastMath.exp(-0.5*mindist/vardist)
                                            *FastMath.exp(-0.5*theta/vartheta));
                
                if (proba>threshold) {
                    heap.addValue((float)proba,idx1,idx2);
                    ntree++;
                }
                
                if (mindist<distance[idx1]) distance[idx1] = mindist;
                if (mindist<distance[idx2]) distance[idx2] = mindist;
                
                if (theta<mintheta[idx1]) mintheta[idx1] = theta;
                if (theta<mintheta[idx2]) mintheta[idx2] = theta;
                
                if (proba>maxproba[idx1]) maxproba[idx1] = proba;
                if (proba>maxproba[idx2]) maxproba[idx2] = proba;
                //}
		        //}
		    }
		}
		System.out.println(" computing "+ntree+" correlations");
		
		// travel through the tree to relabel groups
		int[] relabel = new int[nlt];
		for (int idx=0;idx<nlt;idx++) {
		    relabel[idx] = idx;
		}
		while (heap.isNotEmpty()) {
        	// extract point with minimum distance
        	float dist = heap.getFirst();
        	int idx1 = heap.getFirstId1();
        	int idx2 = heap.getFirstId2();
			heap.removeFirst();
			
			if (idx1<idx2) {
			    for (int idx=0;idx<nlt;idx++) {
			        if (relabel[idx]==idx2) {
                        relabel[idx] = idx1;
                    }
                }
			} else {
			    for (int idx=0;idx<nlt;idx++) {
			        if (relabel[idx]==idx1) {
                        relabel[idx] = idx2;
                    }
                }
			}
		}
		// renumber to keep small values
		int[] labellist = ObjectLabeling.listOrderedLabels(relabel, nlt, 1, 1);
		int[] newlabel = new int[nlt];
		for (int idx=0;idx<nlt;idx++) {
		    for (int lb=0;lb<labellist.length;lb++) {
		        if (relabel[idx]==labellist[lb]) {
                    newlabel[idx] = lb;
		        }
		    }
		}
				
		// map the results back to images
		dir3d = new float[3*nimg*nxyz];
		groups = new int[nimg*nxyz];
		for (int n=0;n<nimg;n++) for (int xyz=0;xyz<nxyz;xyz++) if (lines[n][xyz]>0) {
		    int offset = 0;
            for (int n0=0;n0<n;n0++) offset += nl[n0];
            int idx = offset + lines[n][xyz]-1;
                            
		    dir3d[xyz+0*nxyz+3*n*nxyz] = (float)avgdir[idx][0];
		    dir3d[xyz+1*nxyz+3*n*nxyz] = (float)avgdir[idx][1];
		    dir3d[xyz+2*nxyz+3*n*nxyz] = (float)avgdir[idx][2];
		    
		    // for debug
		    dir3d[xyz+0*nxyz+3*n*nxyz] = (float)FastMath.sqrt(distance[idx]/vardist);
		    dir3d[xyz+1*nxyz+3*n*nxyz] = (float)FastMath.sqrt(mintheta[idx]/vartheta);
		    dir3d[xyz+2*nxyz+3*n*nxyz] = (float)maxproba[idx];
		    
			groups[xyz+n*nxyz] = 1+newlabel[idx];
		}
		/* not needed, working properly
		// add the endpoints for sanity check
		for (int idx=0;idx<nl;idx++) {
		    int xyz = (int)endpoints[idx][0] + nx*(int)endpoints[idx][1];
		    dir3d[xyz+0*nxyz] = 1.0f;
		    
		    xyz = (int)endpoints[idx][2] + nx*(int)endpoints[idx][3];
		    dir3d[xyz+1*nxyz] = 1.0f;
		    
		    xyz = (int)endpoints[idx][4] + nx*(int)endpoints[idx][5];
		    dir3d[xyz+2*nxyz] = 1.0f;
		}*/    
		if (mip>0) {
		    for (int n=0;n<nimg;n++) for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz-mip;z++) {
		        int xyz = x + nx*y + nx*ny*z;
		        for (int d=0;d<mip;d++) {
		            groups[xyz+n*nxyz] = Numerics.max(groups[xyz+n*nxyz],groups[xyz+d*nx*ny+n*nxyz]);
		        }
		    }
		}
		return;
	}
	
    public void combineAcrossDimensions3D() {

	    // get list of labels from line segmentation: 
	    // assume we have already separate labels or compute them
	    if (recompute) {
	        for (int n=0;n<nimg;n++) {
	            lines[n] = ObjectLabeling.connected26Object3D(lines[n], nx,ny,nz);
	        }
	    } 
	    int[] nl = new int[nimg];
	    int nlt = 0;
	    for (int n=0;n<nimg;n++) {
	        nl[n] = ObjectLabeling.countLabels(lines[n], nx, ny, nz);
	        nlt += nl[n];
	    }
	    
	    System.out.println(" found "+nlt+" lines");
	
	    double[] avgscale = new double[nlt];
	    double[] avgsize = new double[nlt];
	    double[][] avgdir = new double[nlt][3];
	    double[][] avgloc = new double[nlt][3];

		for (int n=0;n<nimg;n++) for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x + nx*y + nx*ny*z;
			if (lines[n][xyz]>0) {
                int offset = 0;
                for (int n0=0;n0<n;n0++) offset += nl[n0];
                int idx = offset + lines[n][xyz]-1;
                
                avgsize[idx] += 1.0;
                avgscale[idx] += (1.0+scales[n][xyz]);
                
                // in 3D the direction follows the detected line
                if ( avgdir[idx][0]*dirs[n][xyz+0*nxyz]
                    +avgdir[idx][1]*dirs[n][xyz+1*nxyz]
                    +avgdir[idx][2]*dirs[n][xyz+2*nxyz]>=0 ) {
                
                    avgdir[idx][0] += dirs[n][xyz+0*nxyz];
                    avgdir[idx][1] += dirs[n][xyz+1*nxyz];
                    avgdir[idx][2] += dirs[n][xyz+2*nxyz];
                } else {
                    avgdir[idx][0] -= dirs[n][xyz+0*nxyz];
                    avgdir[idx][1] -= dirs[n][xyz+1*nxyz];
                    avgdir[idx][2] -= dirs[n][xyz+2*nxyz];
                }
                
                avgloc[idx][0] += x;
                avgloc[idx][1] += y;
                avgloc[idx][2] += z;
            }
        }
		
		for (int idx=0;idx<nlt;idx++) {
		    avgscale[idx] /= avgsize[idx];
		    
		    avgdir[idx][0] /= avgsize[idx];
		    avgdir[idx][1] /= avgsize[idx];
		    avgdir[idx][2] /= avgsize[idx];
		    
		    avgloc[idx][0] /= avgsize[idx];
		    avgloc[idx][1] /= avgsize[idx];
		    avgloc[idx][2] /= avgsize[idx];
		}
		
		// find endpoints for better distance measure ordered [-1;0;+1]
		int[][] endpoints = new int[nlt][9];
		double[][] endptdist = new double[nlt][3];
		for (int idx=0;idx<nlt;idx++) {
		    endptdist[idx][0] = +1e9;
		    endptdist[idx][1] = +1e9;
		    endptdist[idx][2] = -1e9;
		}
		for (int n=0;n<nimg;n++) for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x + nx*y + nx*ny*z;
			if (lines[n][xyz]>0) {
                int offset = 0;
                for (int n0=0;n0<n;n0++) offset += nl[n0];
                int idx = offset + lines[n][xyz]-1;
                
                // projection onto main direction
                double newdist = (x-avgloc[idx][0])*avgdir[idx][0]
                               + (y-avgloc[idx][1])*avgdir[idx][1]
                               + (z-avgloc[idx][2])*avgdir[idx][2];
                
                if (newdist<endptdist[idx][0]) {
                    endpoints[idx][0] = x;
                    endpoints[idx][1] = y;
                    endpoints[idx][2] = z;
                    
                    endptdist[idx][0] = newdist;
                }
                if (Numerics.abs(newdist)<endptdist[idx][1]) {
                    endpoints[idx][3] = x;
                    endpoints[idx][4] = y;
                    endpoints[idx][5] = z;
                    
                    endptdist[idx][1] = Numerics.abs(newdist);
                }
                if (newdist>endptdist[idx][2]) {
                    endpoints[idx][6] = x;
                    endpoints[idx][7] = y;
                    endpoints[idx][8] = z;
                    
                    endptdist[idx][2] = newdist;
                }
            }
        }
        // better length: computed from endpoints
		double[] ptlength = new double[nlt];
        for (int idx=0;idx<nlt;idx++) {
            ptlength[idx] = FastMath.sqrt( Numerics.square(endpoints[idx][0]-endpoints[idx][3])
		                                  +Numerics.square(endpoints[idx][1]-endpoints[idx][4])
		                                  +Numerics.square(endpoints[idx][2]-endpoints[idx][5]) )
		                   +FastMath.sqrt( Numerics.square(endpoints[idx][3]-endpoints[idx][6])
		                                  +Numerics.square(endpoints[idx][4]-endpoints[idx][7])
		                                  +Numerics.square(endpoints[idx][5]-endpoints[idx][8]) );
		}
		
		// no need to compute elevation
		
		// build pairwise distances from location and angles
		vardist *= vardist;
		vartheta = (float)(vartheta/180.0*FastMath.PI);
		vartheta *= vartheta;
		
		// sanity check
		double[] mintheta = new double[nlt];
		double[] distance = new double[nlt];
		double[] maxproba = new double[nlt];
		for (int idx=0;idx<nlt;idx++) {
		    mintheta[idx] = 4.0*vartheta;
		    distance[idx] = 4.0*vardist;
		    maxproba[idx] = 0.0;
		}
		
		BinaryHeapPair heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeapPair.MAXTREE);
		int ntree = 0;
		// here we only look for common labellings across the dimensions
		for (int n1=0;n1<nimg;n1++) {
		    int offset1=0;
		    for (int n0=0;n0<n1;n0++) offset1 += nl[n0];
		    
            for (int idx1=offset1;idx1<offset1+nl[n1];idx1++) {
                for (int n2=n1+1;n2<nimg;n2++) {
                    int offset2=0;
                    for (int n0=0;n0<n2;n0++) offset2 += nl[n0];
                    
                    for (int idx2=offset2;idx2<offset2+nl[n2];idx2++) {
                        //double dist = (avgloc[idx1][0]-avgloc[idx2][0])
                        //              +(avgloc[idx1][1]-avgloc[idx2][1]);
                        // basic distance: to exclude far locations quickly
                        /* not working?
                        double dist0 = Numerics.square(avgloc[idx1][0]-avgloc[idx2][0])
                                      +Numerics.square(avgloc[idx1][1]-avgloc[idx2][1]);
                        if (dist0<4.0*vardist+ptlength[idx1]*ptlength[idx1]+ptlength[idx2]*ptlength[idx2]) {
                        */
                        // compute angles
                        double theta = FastMath.acos(Numerics.abs(avgdir[idx1][0]*avgdir[idx2][0]
                                                                 +avgdir[idx1][1]*avgdir[idx2][1]
                                                                 +avgdir[idx1][2]*avgdir[idx2][2]));
                        theta *= theta;
                        
                        //if (theta<4.0*vartheta) {
                        // compute distances
                        double mindist;
                        if (anisotropy>0) {
                            double dx = avgsize[idx1]*avgdir[idx1][0]+avgsize[idx2]*avgdir[idx2][0];
                            double dy = avgsize[idx1]*avgdir[idx1][1]+avgsize[idx2]*avgdir[idx2][1];
                            double dz = avgsize[idx1]*avgdir[idx1][2]+avgsize[idx2]*avgdir[idx2][2];
                            double norm = FastMath.sqrt(dx*dx+dy*dy+dz*dz);
                            
                            double d12x = endpoints[idx1][0]-endpoints[idx2][0];
                            double d12y = endpoints[idx1][1]-endpoints[idx2][1];
                            double d12z = endpoints[idx1][2]-endpoints[idx2][2];
                            mindist = d12x*d12x+d12y*d12y+d12z*d12z;
                            
                            double newdist = Numerics.square(endpoints[idx1][0]-endpoints[idx2][6])
                                            +Numerics.square(endpoints[idx1][1]-endpoints[idx2][7])
                                            +Numerics.square(endpoints[idx1][2]-endpoints[idx2][8]);
                            if (newdist<mindist) {
                                mindist = newdist;
                                d12x = endpoints[idx1][0]-endpoints[idx2][6];
                                d12y = endpoints[idx1][1]-endpoints[idx2][7];
                                d12z = endpoints[idx1][2]-endpoints[idx2][8];
                            }
                            
                            newdist = Numerics.square(endpoints[idx1][6]-endpoints[idx2][6])
                                     +Numerics.square(endpoints[idx1][7]-endpoints[idx2][7])
                                     +Numerics.square(endpoints[idx1][8]-endpoints[idx2][8]);
                            if (newdist<mindist) {
                                mindist = newdist;
                                d12x = endpoints[idx1][6]-endpoints[idx2][6];
                                d12y = endpoints[idx1][7]-endpoints[idx2][7];
                                d12z = endpoints[idx1][8]-endpoints[idx2][8];
                            }
                            
                            newdist = Numerics.square(endpoints[idx1][6]-endpoints[idx2][0])
                                     +Numerics.square(endpoints[idx1][7]-endpoints[idx2][1])
                                     +Numerics.square(endpoints[idx1][8]-endpoints[idx2][2]);
                            if (newdist<mindist) {
                                mindist = newdist;
                                d12x = endpoints[idx1][6]-endpoints[idx2][0];
                                d12y = endpoints[idx1][7]-endpoints[idx2][1];
                                d12z = endpoints[idx1][8]-endpoints[idx2][2];
                            }
                            
                            mindist = 1.0/(1.0-anisotropy)*(mindist-Numerics.square(d12x*dx/norm+d12y*dy/norm+d12z*dz/norm))
                                      +Numerics.square(d12x*dx/norm+d12y*dy/norm+d12z*dz/norm);
                            
                        } else {
                            mindist = Numerics.square(endpoints[idx1][0]-endpoints[idx2][0])
                                     +Numerics.square(endpoints[idx1][1]-endpoints[idx2][1])
                                     +Numerics.square(endpoints[idx1][2]-endpoints[idx2][2]);
                                     
                            mindist = Numerics.min(mindist, Numerics.square(endpoints[idx1][0]-endpoints[idx2][6])
                                                           +Numerics.square(endpoints[idx1][1]-endpoints[idx2][7])
                                                           +Numerics.square(endpoints[idx1][2]-endpoints[idx2][8]));  
                            
                            mindist = Numerics.min(mindist, Numerics.square(endpoints[idx1][6]-endpoints[idx2][6])
                                                           +Numerics.square(endpoints[idx1][7]-endpoints[idx2][7])
                                                           +Numerics.square(endpoints[idx1][8]-endpoints[idx2][8]));  
                            
                            mindist = Numerics.min(mindist, Numerics.square(endpoints[idx1][6]-endpoints[idx2][0])
                                                           +Numerics.square(endpoints[idx1][7]-endpoints[idx2][1])
                                                           +Numerics.square(endpoints[idx1][8]-endpoints[idx2][2]));                
                        }
                        
                        double proba = FastMath.sqrt(FastMath.exp(-0.5*mindist/vardist)
                                                    *FastMath.exp(-0.5*theta/vartheta));
                        
                        if (proba>threshold) {
                            heap.addValue((float)proba,idx1,idx2);
                            ntree++;
                        }
                        
                        if (mindist<distance[idx1]) distance[idx1] = mindist;
                        if (mindist<distance[idx2]) distance[idx2] = mindist;
                        
                        if (theta<mintheta[idx1]) mintheta[idx1] = theta;
                        if (theta<mintheta[idx2]) mintheta[idx2] = theta;
                        
                        if (proba>maxproba[idx1]) maxproba[idx1] = proba;
                        if (proba>maxproba[idx2]) maxproba[idx2] = proba;
                    }
                }
		    }
		}
		System.out.println(" computing "+ntree+" correlations");
		
		// travel through the tree to relabel groups
		int[] relabel = new int[nlt];
		for (int idx=0;idx<nlt;idx++) {
		    relabel[idx] = idx;
		}
		while (heap.isNotEmpty()) {
        	// extract point with minimum distance
        	float dist = heap.getFirst();
        	int idx1 = heap.getFirstId1();
        	int idx2 = heap.getFirstId2();
			heap.removeFirst();
			
			if (idx1<idx2) {
			    for (int idx=0;idx<nlt;idx++) {
			        if (relabel[idx]==idx2) {
                        relabel[idx] = idx1;
                    }
                }
			} else if (idx2<idx1) {
			    for (int idx=0;idx<nlt;idx++) {
			        if (relabel[idx]==idx1) {
                        relabel[idx] = idx2;
                    }
                }
			}
		}
		// renumber to keep small values
		int[] labellist = ObjectLabeling.listOrderedLabels(relabel, nlt, 1, 1);
		int[] newlabel = new int[nlt];
		for (int idx=0;idx<nlt;idx++) {
		    for (int lb=0;lb<labellist.length;lb++) {
		        if (relabel[idx]==labellist[lb]) {
                    newlabel[idx] = lb;
		        }
		    }
		}
		// collapse the labels into types
		int[] labeltype = new int[labellist.length];
		for (int n=0;n<nimg;n++) {
		    int flag = (int)FastMath.pow(10,n);
		    int offset = 0;
		    for (int n0=0;n0<n;n0++) offset += nl[n0];
		    
		    for (int idx=offset;idx<offset+nl[n];idx++) {
		        if (labeltype[newlabel[idx]]<flag) {
                    labeltype[newlabel[idx]] += flag;
                }
		    }
		}
				
		// map the results back to images
		dir3d = new float[3*nimg*nxyz];
		groups = new int[nimg*nxyz];
		for (int n=0;n<nimg;n++) for (int xyz=0;xyz<nxyz;xyz++) if (lines[n][xyz]>0) {
            int offset = 0;
            for (int n0=0;n0<n;n0++) offset += nl[n0];
            int idx = offset + lines[n][xyz]-1;
            
		    dir3d[xyz+0*nxyz+3*n*nxyz] = (float)avgdir[idx][0];
		    dir3d[xyz+1*nxyz+3*n*nxyz] = (float)avgdir[idx][1];
		    dir3d[xyz+2*nxyz+3*n*nxyz] = (float)avgdir[idx][2];
		    
		    // for debug
		    dir3d[xyz+0*nxyz+3*n*nxyz] = (float)FastMath.sqrt(distance[idx]/vardist);
		    dir3d[xyz+1*nxyz+3*n*nxyz] = (float)FastMath.sqrt(mintheta[idx]/vartheta);
		    dir3d[xyz+2*nxyz+3*n*nxyz] = (float)maxproba[idx];
		    
			//groups[xyz+n*nxyz] = 1+newlabel[idx];
			groups[xyz+n*nxyz] = labeltype[newlabel[idx]];
		}
		/* not needed, working properly
		// add the endpoints for sanity check
		for (int idx=0;idx<nl;idx++) {
		    int xyz = (int)endpoints[idx][0] + nx*(int)endpoints[idx][1];
		    dir3d[xyz+0*nxyz] = 1.0f;
		    
		    xyz = (int)endpoints[idx][2] + nx*(int)endpoints[idx][3];
		    dir3d[xyz+1*nxyz] = 1.0f;
		    
		    xyz = (int)endpoints[idx][4] + nx*(int)endpoints[idx][5];
		    dir3d[xyz+2*nxyz] = 1.0f;
		}*/    
		if (mip>0) {
		    for (int n=0;n<nimg;n++) for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz-mip;z++) {
		        int xyz = x + nx*y + nx*ny*z;
		        for (int d=0;d<mip;d++) {
		            groups[xyz+n*nxyz] = Numerics.max(groups[xyz+n*nxyz],groups[xyz+d*nx*ny+n*nxyz]);
		        }
		    }
		}
		return;
	}
	
}
