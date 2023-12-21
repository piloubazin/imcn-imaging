package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import nl.uva.imcn.structures.*;
import nl.uva.imcn.libraries.*;
import nl.uva.imcn.methods.*;

import org.apache.commons.math3.util.FastMath;


/*
 * @author Pierre-Louis Bazin
 */
public class LevelsetCurvature {

	// jist containers
	private float[] lvlImage=null;
	private float distance=1.0f;
	private int kernelParam=3;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;

	private float[][] curv;
	private boolean[] mask;
		
	// for debug and display
	private static final boolean		debug=true;
	private static final boolean		verbose=true;

	// create inputs
	public final void setLevelsetImage(float[] val) { lvlImage = val; }
	public final void setMaxDistance(float val) { distance = val; }
	public final void setKernelParameter(int val) { kernelParam = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
				
	// create outputs on demand
	public final float[] getMeanCurvatureImage() { 
	    float[] metricImage = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
            metricImage[xyz] = 0.5f*(curv[0][xyz]+curv[4][xyz]);
        }	
	    return metricImage; 
	}
	public final float[] getGaussCurvatureImage() { 
	    float[] metricImage = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
            metricImage[xyz] = curv[0][xyz]*curv[4][xyz];
        }	
	    return metricImage; 
	}
	public final float[] getShapeIndexImage() { 
	    float[] metricImage = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
            metricImage[xyz] = (float)(2.0/FastMath.PI*FastMath.atan2(curv[0][xyz]+curv[4][xyz],Numerics.abs(curv[0][xyz]-curv[4][xyz])));
        }	
	    return metricImage; 
	}
	public final float[] getCurvednessImage() { 
	    float[] metricImage = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
            metricImage[xyz] = (float)FastMath.sqrt(0.5*(curv[0][xyz]*curv[0][xyz]+curv[4][xyz]*curv[4][xyz]));
        }	
	    return metricImage; 
	}


	public void execute(){
		
	    // set up the computation mask
        mask = new boolean[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) if (Numerics.abs(lvlImage[xyz])<distance) mask[xyz] = true;
        
        curv = ImageGeometry.quadricCurvatureEstimates(lvlImage, mask, mask, kernelParam, nx,ny,nz);
	}

}
