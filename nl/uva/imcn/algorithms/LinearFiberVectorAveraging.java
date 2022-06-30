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

public class LinearFiberVectorAveraging {
	
	// jist containers
	private float[][] vecImage;
	
	private int orig = 7;
	private int kept = 14;
	private int ngb = 3;
	private int search = 1;
	private float threshold = 0.001f;
		
	// global variables
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;
	
	private final int X=0;
	private final int Y=1;
	private final int Z=2;
	
	private static final boolean debug=true;
	
	//set inputs
	public final void setVectorImageAt(int n, float[] val) { vecImage[n] = val; }

	public final void setOriginalVectorNumber(int val) { 
	    orig = val; 
	    vecImage = new float[6*orig][];
	}
	public final void setKeptVectorNumber(int val) { kept = val; }
	public final void setNumberOfNeighbors(int val) { ngb = val; }
	public final void setSearchRadius(int val) { search = val; }
	public final void setVectorThreshold(float val) { threshold = val; }
		
	// set generic inputs	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
			
	// create outputs
	public final float[] getAveragedVectorImageAt(int n) { return vecImage[n];}
	
	public void execute(){
		BasicInfo.displayMessage("linear fiber vector averaging:\n");
		
		// 1. rebuild the angular information to compute similarity scores
		float[][] angle = new float[2*orig][nxyz];
		boolean[] mask = new boolean[nxyz];
		
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
		    int idx = x+nx*y+nx*ny*z;
		    for (int n=0;n<orig;n++) {
                float norm = (float)FastMath.sqrt(vecImage[X+3*n][idx]*vecImage[X+3*n][idx]
                                                 +vecImage[Y+3*n][idx]*vecImage[Y+3*n][idx]
                                                 +vecImage[Z+3*n][idx]*vecImage[Z+3*n][idx]);
                
                float corr = (float)FastMath.sqrt(vecImage[X+3*n][idx]*vecImage[X+3*n][idx]
                                                 +vecImage[Y+3*n][idx]*vecImage[Y+3*n][idx]);
                
                angle[X+2*n][idx] = vecImage[X+3*n][idx]*norm/Numerics.max(0.0001f, corr);
                angle[Y+2*n][idx] = vecImage[Y+3*n][idx]*norm/Numerics.max(0.0001f, corr);
                
                if (n==0) { 
                    if (norm>threshold) mask[idx] = true;
                    else mask[idx] = false;
                }
            }
		}
		
		// 2. Find for each location the closest neighbor voxel in each section
		float[][] keptVec = new float[6*kept][nxyz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) if (mask[x+nx*y+nx*ny*z]) {
		    int idx = x+nx*y+nx*ny*z;
		    
		    // list of best neighbors (including self)
		    int[] idngb = new int[2*ngb+1];
		    for (int dz=-ngb;dz<=ngb;dz++) idngb[dz+ngb] = -1;
		    idngb[ngb] = idx;
		    
		    // one ngb per section
		    for (int dz=-ngb;dz<=ngb;dz++) if (dz!=0 && z+dz>=0 && z+dz<nz) {
		        // search window in plane for maximum angular similarity
		        float maxscore = 0.0f;
		        for (int dx=-search;dx<=search;dx++) for (int dy=-search;dy<=search;dy++) {
		            if (x+dx>=0 && x+dx<nx && y+dy>=0 && y+dy<ny) {
                        int idn = idx + dx + nx*dy + nx*ny*dz;
                        if (mask[idn]) {
                            float score = 0.0f;
                            for (int n=0;n<orig;n++) {
                                score += Numerics.abs(angle[X+2*n][idx]*angle[X+2*n][idn]
                                                     +angle[Y+2*n][idx]*angle[Y+2*n][idn]);
                            }
                            if (score>maxscore) {
                                maxscore = score;
                                idngb[dz+ngb] = idn;
                            }
                        }
                    }
                }
            }
            
            // compute probability score
            float[][] pscore = new float[2*ngb+1][orig];
            for (int dz=-ngb;dz<=ngb;dz++) if (idngb[dz+ngb]!=-1) {
                for (int n=0;n<orig;n++) {
                    int idn = idngb[dz+ngb];
                    pscore[dz+ngb][n] = (float)FastMath.sqrt(vecImage[X+3*n][idn]*vecImage[X+3*n][idn]
                                                            +vecImage[Y+3*n][idn]*vecImage[Y+3*n][idn]
                                                            +vecImage[Z+3*n][idn]*vecImage[Z+3*n][idn]);
                }
            }
                         
		    // select best kept values by probability score
		    for (int k=0;k<kept;k++) {
		        float maxscore = 0.0f;
		        int dzb = -ngb-1;
		        int onb = -1;
		        for (int dz=-ngb;dz<=ngb;dz++) if (idngb[dz+ngb]!=-1) {
		            for (int on=0;on<orig;on++) {
		                if (pscore[dz+ngb][on]>maxscore) {
		                    maxscore = pscore[dz+ngb][on];
		                    dzb = dz;
		                    onb = on;
		                }
		            }
		        }
		        if (onb>-1) {
                    int idb = idngb[dzb+ngb];
                    
                    // copy the result over (and z-flipped version)
                    keptVec[X+3*k][idx] = vecImage[X+3*onb][idb];
                    keptVec[Y+3*k][idx] = vecImage[Y+3*onb][idb];
                    keptVec[Z+3*k][idx] = vecImage[Z+3*onb][idb];
                    
                    keptVec[X+3*k+3*kept][idx] = vecImage[X+3*onb][idb];
                    keptVec[Y+3*k+3*kept][idx] = vecImage[Y+3*onb][idb];
                    keptVec[Z+3*k+3*kept][idx] = -vecImage[Z+3*onb][idb];
                    
                    // wipe value
                    pscore[dzb+ngb][onb] = 0.0f;
                } else {
                    k = kept;
                }
            }
        }
        vecImage = keptVec;
		    
		return;
    }

}
