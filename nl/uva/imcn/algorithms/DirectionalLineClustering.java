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
	int[] lines = null;
	float[] dirs = null;
	float[] scales = null;
	
	float thickness = 20.0f;
	float vardist = 20.0f;
	float vartheta = 15.0f;
	float threshold = 0.5f;
	
	int nx, ny, nz, nxyz;
	float rx, ry, rz;
	
	float[] dir3d;
	int[] groups;
	
	// set inputs
	public final void setLineImage(int[] val) { lines = val; }
	public final void setDirImage(float[] val) { dirs = val; }
	public final void setScaleImage(float[] val) { scales = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }

	public final void setSliceThickness(float val) { thickness = val; }
	public final void setExpectedDistance(float val) { vardist = val; }
	public final void setExpectedAngle(float val) { vartheta = val; }
	public final void setProbabilityThreshold(float val) { threshold = val; }
	
	// outputs
	public final float[] getDirectionImage() { return dir3d; }
	public final int[] getGroupImage() { return groups; }

	public void execute() {

	    // get list of labels from line segmentation: assume we have already separate labels
	    int nl = ObjectLabeling.countLabels(lines, nx, ny, nz);
	    System.out.println(" found "+nl+" lines");
	
	    double[] avgscale = new double[nl];
	    double[] avgsize = new double[nl];
	    double[][] avgdir = new double[nl][3];
	    double[][] avgloc = new double[nl][2];

		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
			int xyz = x + nx*y;
			if (lines[xyz]>0) {
                int idx = lines[xyz]-1;
                avgsize[idx] += 1.0;
                avgscale[idx] += scales[xyz];
                
                if (avgdir[idx][0]*dirs[xyz+0*nxyz]+avgdir[idx][1]*dirs[xyz+1*nxyz]>=0) {
                    avgdir[idx][0] += dirs[xyz+0*nxyz];
                    avgdir[idx][1] += dirs[xyz+1*nxyz];
                } else {
                    avgdir[idx][0] -= dirs[xyz+0*nxyz];
                    avgdir[idx][1] -= dirs[xyz+1*nxyz];
                }
                
                avgloc[idx][0] += x;
                avgloc[idx][1] += y;
            }
        }
		
		for (int idx=0;idx<nl;idx++) {
		    avgscale[idx] /= avgsize[idx];
		    avgdir[idx][0] /= avgsize[idx];
		    avgdir[idx][1] /= avgsize[idx];
		    avgloc[idx][0] /= avgsize[idx];
		    avgloc[idx][1] /= avgsize[idx];
		    avgloc[idx][2] /= avgsize[idx];
		}
		
		// compute elevation
		for (int idx=0;idx<nl;idx++) {
		    double norm = FastMath.sqrt(avgdir[idx][0]*avgdir[idx][0]+avgdir[idx][1]*avgdir[idx][1]);
		    double length = avgsize[idx]/avgscale[idx];
		    avgdir[idx][0] = length*avgdir[idx][0]/norm;
		    avgdir[idx][1] = length*avgdir[idx][1]/norm;
		    avgdir[idx][2] = thickness;
		    norm = FastMath.sqrt(avgdir[idx][0]*avgdir[idx][0]+avgdir[idx][1]*avgdir[idx][1]+avgdir[idx][2]*avgdir[idx][2]);
		    avgdir[idx][0] /= norm;
		    avgdir[idx][1] /= norm;
		    avgdir[idx][2] /= norm;
		}
		
		// build pairwise distances from location and angles
		vardist *= vardist;
		vartheta = (float)(vartheta/180.0*FastMath.PI);
		vartheta *= vartheta;
		
		BinaryHeapPair heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeapPair.MAXTREE);
		int ntree = 0;
		for (int idx1=0;idx1<nl;idx1++) {
		    for (int idx2=0;idx2<nl;idx2++) {
		        double dist = (avgloc[idx1][0]-avgloc[idx2][0])
		                     +(avgloc[idx1][1]-avgloc[idx2][1]);
		        if (dist<4.0*vardist) {
		            double theta = FastMath.acos(Numerics.abs(avgdir[idx1][0]*avgdir[idx2][0]
		                                                     +avgdir[idx1][1]*avgdir[idx2][1]
		                                                     +avgdir[idx1][2]*avgdir[idx2][2]));
		            theta *= theta;
		            
		            if (theta<4.0*vartheta) {
		                double proba = FastMath.sqrt(FastMath.exp(-0.5*dist/vardist)
		                                            *FastMath.exp(-0.5*theta/vartheta));
		                
		                if (proba>threshold) {
		                    heap.addValue((float)proba,idx1,idx2);
		                    ntree++;
		                }
		            }
		        }
		    }
		}
		System.out.println(" computing "+ntree+" correlations");
		
		// travel through the tree to relabel groups
		int[] relabel = new int[nl];
		for (int idx=0;idx<nl;idx++) {
		    relabel[idx] = idx;
		}
		while (heap.isNotEmpty()) {
        	// extract point with minimum distance
        	float dist = heap.getFirst();
        	int idx1 = heap.getFirstId1();
        	int idx2 = heap.getFirstId2();
			heap.removeFirst();
			
			if (idx1<idx2) {
			    relabel[idx2] = idx1;
			} else {
			    relabel[idx1] = idx2;
			}
		}
		
		// map the results back to images
		dir3d = new float[3*nxyz];
		groups = new int[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (lines[xyz]>0) {
		    int idx = lines[xyz]-1;
		    dir3d[xyz+0*nxyz] = (float)avgdir[idx][0];
		    dir3d[xyz+1*nxyz] = (float)avgdir[idx][1];
		    dir3d[xyz+2*nxyz] = (float)avgdir[idx][2];
		    
			groups[xyz] = relabel[idx];
		}
		return;
	}
}
