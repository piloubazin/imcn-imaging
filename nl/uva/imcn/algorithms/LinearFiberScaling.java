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

public class LinearFiberScaling {
	
	// jist containers
	private float[] probaImage;
	private float[] thetaImage;
	private float[] lengthImage;
	
	private int scaling = 7;
	private int kept = 5;
	private float detectionThreshold = 1e-9f;
		
	// global variables
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;
	
	private static final boolean debug=true;
	
	//set inputs
	public final void setProbaImage(float[] val) { probaImage = val; }
	public final void setThetaImage(float[] val) { thetaImage = val; }
	public final void setLengthImage(float[] val) { lengthImage = val; }

	public final void setScaling(int val) { scaling = val; }
	public final void setNumberKept(int val) { kept = val; }
	public final void setDetectionThreshold(float val) { detectionThreshold = val; }
		
	// set generic inputs	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
			
	// create outputs
	public final float[] getScaledProbabilityImage() { return probaImage;}
	public final float[] getScaledLengthImage() { return lengthImage;}
	public final float[] getScaledAngleImage() { return thetaImage;}
	
	public void execute(){
		BasicInfo.displayMessage("linear fiber scaling:\n");
		
		// clean up length zero patterns
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
		    int idn = x+nx*y+nx*ny*z;
		    if (lengthImage[idn]==0) {
		        probaImage[idn] = 0.0f;
		        thetaImage[idn] = 0.0f;
		    }
		}
		
		// import the inputImage data into 1D arrays: already done
		int mx = Numerics.ceil((float)nx/scaling);
		int my = Numerics.ceil((float)ny/scaling);
		
		BasicInfo.displayMessage("...scale data from "+nx+"x"+ny+" to "+mx+"x"+my);
		
		float[] probasc = new float[mx*my*nz*kept];
		float[] thetasc = new float[mx*my*nz*kept];
		float[] lengthsc = new float[mx*my*nz*kept];
		for (int x=0;x<mx;x++) for (int y=0;y<my;y++) for (int z=0;z<nz;z++) {
		    // find the largest groups
		    int idm = x+mx*y+mx*my*z;
		    for (int k=0;k<kept;k++) {
		        // first locate the highest probability
		        float maxproba = 0.0f;
		        for (int dx=0;dx<scaling;dx++) for (int dy=0;dy<scaling;dy++) {
		            if (x*scaling+dx<nx && y*scaling+dy<ny) {
		                int idn = x*scaling+dx + nx*(y*scaling+dy) + nx*ny*z;
		                if (probaImage[idn]>maxproba) maxproba = probaImage[idn];
		            }
		        }
		        // then group the corresponding voxels
		        if (maxproba>0) {
		            //int nb=0;
                    for (int dx=0;dx<scaling;dx++) for (int dy=0;dy<scaling;dy++) {
                        if (x*scaling+dx<nx && y*scaling+dy<ny) {
                            int idn = x*scaling+dx + nx*(y*scaling+dy) + nx*ny*z;
                            if (probaImage[idn]>=maxproba-detectionThreshold) {
                                probasc[idm+mx*my*nz*k] = maxproba;
                                thetasc[idm+mx*my*nz*k] = thetaImage[idn];
                                lengthsc[idm+mx*my*nz*k] = lengthImage[idn];
                                //nb++;
                                // reset probability once it has been used
                                // not OK: used in different regions??
                                probaImage[idn] = 0.0f;
                            }
                        }
                    }
                    //if (nb>0) {
                    //    thetasc[idm+mx*my*nz*k] /= nb;
                    //    lengthsc[idm+mx*my*nz*k] /= nb;
                    //}
                }
            }
        }
        probaImage = probasc;
        thetaImage = thetasc;
        lengthImage = lengthsc;
		    
		return;
    }

}
