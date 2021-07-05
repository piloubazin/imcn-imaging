package nl.uva.imcn.algorithms;

import nl.uva.imcn.libraries.*;
import nl.uva.imcn.structures.*;
import nl.uva.imcn.utilities.*;
import org.apache.commons.math3.util.FastMath;

/*
 * @author Pierre-Louis bazin (pilou.bazin@uva.nl)
 *
 */
public class CompetingProbabilityDiffusion {
	private 	float[] prior = null;
	private 	float[][] proba = null;
	
	private 	float ratio = 0.5f;
	private 	int maxiter = 100;
	private 	float maxdiff = 0.001f;
	private     int ngb = 4;
	
	private 	int nx, ny, nz, nxyz;
	private 	float rx, ry, rz;
	private     int np;
	
	private     float[][] posterior = null;
	private 	int[] clustering;
	
	// set inputs
	public final void setImageNumber(int n) { 
	    np = n;
	    proba = new float[np][];
	}
	public final void setProbaImageAt(int n, float[] val) { proba[n] = val; }
	public final void setPriorImage(float[] val) { prior = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }

	public final void setDiffusionRatio(float val) { ratio = val; }
	public final void setNeighborhoodSize(int val) { ngb = val; }
	public final void setMaxIterations(int val) { maxiter = val; }
	public final void setMaxDifference(float val) { maxdiff = val; }
	
	// outputs
	public final float[] getPosteriorImageAt(int n) { return posterior[n]; }
	public final float[] getPosteriorImages() { 
	    float[] val = new float[np*nxyz];
	    for (int p=0;p<np;p++) for (int xyz=0;xyz<nxyz;xyz++) val[xyz+p*nxyz] = posterior[p][xyz];
	    return val; 
	}
	public final int[] getClusteringImage() { return clustering; }

	public void execute() {
	    
	    // for speed and memory: shrink to acceptable values first
	    int nmask = 0;
	    boolean[] mask = new boolean[nxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        boolean found=false;
	        if (prior[xyz]>0) found=true;
	        else {
	            for (int p=0;p<np && !found;p++)
	                if (proba[p][xyz]>0) found=true;
	        }
	        mask[xyz] = found;
	        if (found) nmask++;
	    }
	    int[] idmap = new int[nmask];
	    int id=0;
	    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
	        idmap[id] = xyz;
	        id++;
	    }
	    
	    // init posterior and linking function
	    posterior = new float[np][nxyz];
	    for (int p=0;p<np;p++) for (int xyz=0;xyz<nxyz;xyz++) {
	        posterior[p][xyz] = proba[p][xyz];
	    }
	    // save the original proba for later
	    float[][] priors = new float[np][nmask];
	    for (int msk=0;msk<nmask;msk++) {
		    int xyz = idmap[msk];
		    for (int p=0;p<np;p++) {
		        priors[p][msk] = proba[p][xyz];
		    }
		}
	    // replace proba by linking function to save space
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        if (mask[xyz]) {
	            for (int p=0;p<np;p++) {
                    float maxp = 0.0f;
                    for (int q=0;q<np;q++) if (q!=p) {
                        maxp = Numerics.max(maxp, posterior[q][xyz]);
                    }
                    proba[p][xyz] = (float)FastMath.sqrt((1.0f - maxp)*Numerics.max(prior[xyz],posterior[p][xyz]));
                }
            }
	    }
	    // for checking
	    if (maxiter==0) {
            for (int p=0;p<np;p++) for (int xyz=0;xyz<nxyz;xyz++) {
                posterior[p][xyz] = proba[p][xyz];
            }	        
	    }
	    // pre-compute neighborhood ids
	    int[][][] ngbid = new int[np][nmask][ngb];
		float[] ngbval = new float[26];
		for (int msk=0;msk<nmask;msk++) {
		    int xyz = idmap[msk];
		    for (int p=0;p<np;p++) {
                for (byte d=0;d<26;d++) {
                    int idx = Ngb.neighborIndex(d, xyz, nx,ny,nz);
                    ngbval[d] = proba[p][idx];
                }
		        // find best N from neighborhood
		        for (int b=0;b<ngb;b++) {
		            byte nbest = -1;
		            float best = 0.0f;
		             for (byte d=0;d<26;d++) {
		                 if (ngbval[d]>best) {
                            nbest = d;
                            best = ngbval[d];
                         }
                     }
                     if (nbest>-1) {
                         ngbid[p][msk][b] = Ngb.neighborIndex(nbest, xyz, nx,ny,nz);
                         ngbval[nbest] = 0.0f;
                     } else {
                         ngbid[p][msk][b] = -1;
                     }
                }
            }
        }

        // diffusion step
        float[][] prev = new float[np][nmask];
	    for (int t=0;t<maxiter;t++) {
	        System.out.print("Iteration "+t);
	        float diff=0.0f;
	    
	        for (int msk=0;msk<nmask;msk++) {
                int xyz = idmap[msk];
                for (int p=0;p<np;p++) {
                    prev[p][msk] = posterior[p][xyz];
                }
            }
	        for (int msk=0;msk<nmask;msk++) {
                int xyz = idmap[msk];
                
                for (int p=0;p<np;p++) {
	                float num = 0.0f;
	                float den = 0.0f;
                    for (int b=0;b<ngb;b++) if (ngbid[p][msk][b]>-1) {
	                    int idx = ngbid[p][msk][b];
	                    float maxp = 0.0f;
                        for (int q=0;q<np;q++) if (q!=p) maxp = Numerics.max(maxp, prev[q][msk]);
                        
                        num += proba[p][idx]*(prev[p][msk]-maxp);
	                    den += proba[p][idx];
	                }
	                // should be globally stable, but maybe not spreading far enough?
	                //posterior[p][xyz] = priors[p][msk] + ratio*num/den;
	                // tends to saturate but propagates nicely
	                posterior[p][xyz] += ratio*num/den;
	                
	                posterior[p][xyz] = Numerics.bounded(posterior[p][xyz],0.0f,1.0f);
                        
                    if (Numerics.abs(posterior[p][xyz]-prev[p][msk])>diff)
                        diff = Numerics.abs(posterior[p][xyz]-prev[p][msk]);
                }
            }
            System.out.println(": "+diff);
            
            if (diff<maxdiff) t=maxiter;
        }
		// compute the clustering
		clustering = new int[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    // combine with probability of existence?
		    for (int p=0;p<np;p++) posterior[p][xyz] = (float)FastMath.sqrt(posterior[p][xyz]*proba[p][xyz]);
		    int nbest = -1;
		    float best = 0.0f;
		    for (int p=0;p<np;p++) if (posterior[p][xyz]>best) {
		        best = posterior[p][xyz];
		        nbest = p;
		    }
		    clustering[xyz] = nbest+1;
		    //for (int p=0;p<np;p++) posterior[p][xyz] *= proba[p][xyz];
		}
		return;
	}
}
