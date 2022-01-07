package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import nl.uva.imcn.structures.*;
import nl.uva.imcn.libraries.*;
import nl.uva.imcn.methods.*;

import org.apache.commons.math3.util.FastMath;


/*
 * @author Pierre-Louis Bazin
 */
public class RestrictedRegionConnectivity {

	// jist containers
	private int[] regionImage=null;
	private int[] sourceLabelImage=null;
	private int[] targetLabelImage=null;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;

	private int[] pathImage;
	private float[] distImage;
	
	// numerical quantities
	private static final	float	INVSQRT2 = (float)(1.0/FastMath.sqrt(2.0));
	private static final	float	INVSQRT3 = (float)(1.0/FastMath.sqrt(3.0));
	private static final	float	SQRT2 = (float)FastMath.sqrt(2.0);
	private static final	float	SQRT3 = (float)FastMath.sqrt(3.0);

	// direction labeling		
	public	static	final	byte	X = 0;
	public	static	final	byte	Y = 1;
	public	static	final	byte	Z = 2;

	// feature choice labeling		
	public	static	final	byte	DIST = 100;
	public	static	final	byte	PROBA = 101;
	public	static	final	byte	MASK = 102;
	
	// computation variables
	private boolean[][][] obj = new boolean[3][3][3];
	private CriticalPointLUT lut;
	private BinaryHeap3D	heap;
	private String	lutdir = null;
	
	// for debug and display
	private static final boolean		debug=true;
	private static final boolean		verbose=true;

	// create inputs
	public final void setRegionImage(int[] val) { regionImage = val; }
	public final void setSourceLabelImage(int[] val) { sourceLabelImage = val; }
	public final void setTargetLabelImage(int[] val) { targetLabelImage = val; }

	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
				
	// create outputs
	public final int[] getPathImage() { return pathImage; }
	public final float[] getDistanceImage() { return distImage; }

	public void execute(){
		
	    // if needed, convert region to levelset
		float rmax = Numerics.max(rx,ry,rz);
	    
		float size = 0.0f;
		
		boolean[] domain = new boolean[nx*ny*nz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (x==0 || x==nx-1 || y==0 || y==ny-1 || z==0 || z==nz-1) {
                domain[xyz] = false;
            } else if (targetLabelImage[xyz]>0) {
                domain[xyz] = true;
            } else if (regionImage[xyz]>0) {
                domain[xyz] = true;
            } else {
                domain[xyz] = false;
            }
        }
		
		// find all labels and run restricted region growing
        int nlb = ObjectLabeling.countLabels(sourceLabelImage, nx, ny, nz);
	    int[] lbl = ObjectLabeling.listLabels(sourceLabelImage, nx, ny, nz);
		
	    float[][] lbldist = new float[nlb][];
	    for (int n=0;n<nlb;n++) if (lbl[n]>0) {
	        lbldist[n] = fastMarchingRestrictedDistanceToTarget(lbl[n], sourceLabelImage, targetLabelImage, domain, nx, ny, nz, rx, ry, rz);
	    }
	    
	    // combine distances as minimum? or rescale by maximum distance to target?
	    float[] maxdist = new float[nlb];
	    for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (targetLabelImage[xyz]>0) {
                for (int n=0;n<nlb;n++) if (lbl[n]==targetLabelImage[xyz]) {
                    maxdist[n] = Numerics.max(maxdist[n], lbldist[n][xyz]);
                }
            }
        }
        float[] combined = new float[nx*ny*nz];
	    for (int xyz=0;xyz<nx*ny*nz;xyz++) {
            for (int n=0;n<nlb;n++) if (lbl[n]>0) {
                if (lbldist[n][xyz]>0.0f) {
                    if (combined[xyz]==0.0f) combined[xyz] = lbldist[n][xyz]/maxdist[n];
                    else combined[xyz] = Numerics.min(combined[xyz], lbldist[n][xyz]/maxdist[n]);
                }
            }
        }
		
       // joint fast marching of all the labels
		BinaryHeapPair heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeapPair.MINTREE);
		boolean[] processed = new boolean[nx*ny*nz];
		float[] path = new float[nx*ny*nz];
		int[] labeling = new int[nx*ny*nz];
		
		// count the number of target locations
		int[] ntargets = new int[nlb];
		for (int xyz=0;xyz<nx*ny*nz; xyz++) if (targetLabelImage[xyz]>0) {
		    for (int n=0;n<nlb;n++) if (lbl[n]==targetLabelImage[xyz]) {
		        ntargets[n]++;
		    }
		}
		
        heap.reset();
		// initialize the heap from boundaries
		for (int xyz=0;xyz<nx*ny*nz; xyz++) if (sourceLabelImage[xyz]>0) {
		    int label = sourceLabelImage[xyz];
        	// search for boundaries
        	for (byte k = 0; k<6; k++) {
				int xyzn = ObjectTransforms.fastMarchingNeighborIndex(k, xyz, nx, ny, nz);
				if (sourceLabelImage[xyzn]==0 && domain[xyz]) {
					// we assume the levelset value is correct at the boundary
					
					// add to the heap with previous value
					heap.addValue(combined[xyzn],xyzn,label);
                }
            }
        }
        
        // grow until target is reached? grow in both directions?
        // or use a gradient integral path, rather?
        boolean allreached=false;
        while (heap.isNotEmpty() && !allreached) {
        	// extract point with minimum distance
        	float dist = heap.getFirst();
        	int xyz = heap.getFirstId1();
        	int label = heap.getFirstId2();
        	heap.removeFirst();
        	
			// if more than nmgdm labels have been found already, this is done
			if (processed[xyz])  continue;
			
        	// update the distance functions at the current label
			path[xyz] = dist;
			labeling[xyz] = label;
			processed[xyz]=true;
			
			if (targetLabelImage[xyz]>0) {
                for (int n=0;n<nlb;n++) if (lbl[n]==targetLabelImage[xyz]) {
                    ntargets[n]--;
                }
                boolean reached=true;
                for (int n=0;n<nlb;n++) if (ntargets[n]>0) reached  = false;
                if (reached) allreached=true;
			}
			
			// find new neighbors
			for (byte k = 0; k<6; k++) {
				int xyzn = ObjectTransforms.fastMarchingNeighborIndex(k, xyz, nx, ny, nz);
				
				// must be in outside the object or its processed neighborhood
				if (!processed[xyzn] && domain[xyzn]) {
					float newdist = dist+combined[xyzn];
					
					// add to the heap
					heap.addValue(newdist,xyzn,label);
				}
			}			
		}
	}

	private static final float[] fastMarchingRestrictedDistanceToTarget(int label, int[] source, int[] target, boolean[] mask, int nx, int ny, int nz, float rx, float ry, float rz)  {
        // computation variables
        float[] levelset = new float[nx*ny*nz]; // note: using a byte instead of boolean for the second pass
		boolean[] processed = new boolean[nx*ny*nz]; // note: using a byte instead of boolean for the second pass
		float[] nbdist = new float[6];
		boolean[] nbflag = new boolean[6];
		BinaryHeapPair heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeapPair.MINTREE);
				        		
		float rmax = Numerics.max(rx,ry,rz);
		float[] r2 = new float[6];
		r2[0] = rx/rmax;
		r2[1] = rx/rmax;
		r2[2] = ry/rmax;
		r2[3] = ry/rmax;
		r2[4] = rz/rmax;
		r2[5] = rz/rmax;
		for (int l=0;l<6;l++) r2[l] *= r2[l];

		// compute the neighboring labels and corresponding distance functions (! not the MGDM functions !)
        //if (debug) System.out.print("fast marching\n");		
        heap.reset();
        // initialize mask and processing domain
		float maxlvl = Numerics.max(nx/2.0f,ny/2.0f,nz/2.0f);
		for (int x=0; x<nx; x++) for (int y=0; y<ny; y++) for (int z = 0; z<nz; z++) {
			int xyz = x+nx*y+nx*ny*z;
        	if (source[xyz]==label) levelset[xyz]=-0.5f;
        	else levelset[xyz] = 0.5f;
			if (!mask[xyz]) { // inside the masked region: either fully inside or fully outside
				if (source[xyz]==label) levelset[xyz] = -maxlvl;
				else levelset[xyz] = maxlvl;
			}
		}
		// initialize the heap from boundaries
		for (int xyz=0;xyz<nx*ny*nz; xyz++) if (source[xyz]==label) {
        	// search for boundaries
        	for (byte k = 0; k<6; k++) {
				int xyzn = ObjectTransforms.fastMarchingNeighborIndex(k, xyz, nx, ny, nz);
				if (source[xyzn]!=label && mask[xyz]) {
					// we assume the levelset value is correct at the boundary
					
					// add to the heap with previous value
					heap.addValue(Numerics.abs(levelset[xyzn]),xyzn,label);
                }
            }
        }
		//if (debug) System.out.print("init\n");		

		// count the number of target locations
		int ntarget=0;
		for (int xyz=0;xyz<nx*ny*nz; xyz++) if (target[xyz]==label) {
		    ntarget++;
		}
		
        // grow the labels and functions
        int reached=0;
        while (heap.isNotEmpty() && reached<ntarget) {
        	// extract point with minimum distance
        	float dist = heap.getFirst();
        	int xyz = heap.getFirstId1();
        	heap.removeFirst();
        	
			// if more than nmgdm labels have been found already, this is done
			if (processed[xyz])  continue;
			
        	if (target[xyz]==label) reached++;

			// update the distance functions at the current level
			//if (lb==1) levelset[xyz] = -dist;
			//else levelset[xyz] = dist;
			levelset[xyz] = -dist;
			processed[xyz]=true; // update the current level
 			
			// find new neighbors
			for (byte k = 0; k<6; k++) {
				int xyzn = ObjectTransforms.fastMarchingNeighborIndex(k, xyz, nx, ny, nz);
				
				// must be in outside the object or its processed neighborhood
				if (!processed[xyzn] && mask[xyzn]) if (source[xyzn]==source[xyz]) {
					// compute new distance based on processed neighbors for the same object
					for (byte l=0; l<6; l++) {
						nbdist[l] = -1.0f;
						nbflag[l] = false;
						int xyznb = ObjectTransforms.fastMarchingNeighborIndex(l, xyzn, nx, ny, nz);
						// note that there is at most one value used here
						if (processed[xyznb] && mask[xyznb]) if (source[xyznb]==source[xyz]) {
							nbdist[l] = Numerics.abs(levelset[xyznb]);
							nbflag[l] = true;
						}			
					}
					float newdist = ObjectTransforms.minimumMarchingDistance(nbdist, nbflag, r2);
					
					// add to the heap
					heap.addValue(newdist,xyzn,label);
				}
			}			
		}
		//if (debug) System.out.print("done\n");		
		
       return levelset;
    }


}
