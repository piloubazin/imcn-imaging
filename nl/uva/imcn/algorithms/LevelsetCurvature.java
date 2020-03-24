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

	private float[] mcurvImage;
	private float[] gcurvImage;
		
	// for debug and display
	private static final boolean		debug=true;
	private static final boolean		verbose=true;

	// create inputs
	public final void setLevelsetImage(float[] val) { lvlImage = val; }
	public final void setMaxDistance(float val) { distance = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
				
	// create outputs
	public final float[] getMeanCurvatureImage() { return mcurvImage; }
	public final float[] getGaussCurvatureImage() { return gcurvImage; }

	public void execute(){
		
	    // set up the computation mask
        boolean[] mask = new boolean[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) if (Numerics.abs(lvlImage[xyz])<distance) mask[xyz] = true;
        
        float[][] curv = ImageGeometry.quadricCurvatureEstimates(lvlImage, mask, mask, kernelParam, nx,ny,nz);

        mcurvImage = new float[nxyz];
        gcurvImage = new float[nxyz];

        for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
            mcurvImage[xyz] = 0.5f*(curv[0][xyz]+curv[4][xyz]);
            gcurvImage[xyz] = curv[0][xyz]*curv[4][xyz];
        }
        
	}

}
