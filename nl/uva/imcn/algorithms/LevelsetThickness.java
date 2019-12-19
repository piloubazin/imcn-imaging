package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import nl.uva.imcn.structures.*;
import nl.uva.imcn.libraries.*;
import nl.uva.imcn.methods.*;

import org.apache.commons.math3.util.FastMath;


/*
 * @author Pierre-Louis Bazin
 */
public class LevelsetThickness {

	// jist containers
	private float[] inputImage=null;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;

	private String featureParam = "probability_map";
	private static final String[] featureTypes = {"signed_distance","probability_map"}; 
	
	private float[] medialImage;
	private float[] distImage;
	private float[] thickImage;
	
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
	
	// computation variables
	private boolean[][][] obj = new boolean[3][3][3];
	private CriticalPointLUT lut;
	private BinaryHeap3D	heap;
	private String	lutdir = null;
	
	// for debug and display
	private static final boolean		debug=true;
	private static final boolean		verbose=true;

	// create inputs
	public final void setShapeImage(float[] val) { inputImage = val; }
	public final void setShapeImageType(String val) { featureParam = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
				
	// create outputs
	public final float[] geThicknessImage() { return thickImage; }
	public final float[] getMedialAxisImage() { return medialImage; }
	public final float[] getMedialDistanceImage() { return distImage; }

	public void execute(){
		
	    // if needed, convert to levelset
        boolean[] mask = new boolean[nxyz];
        for (int n=0;n<nxyz;n++) mask[n] = true;
	    float[] levelset;
		if (featureParam.equals("probability_map")) {
		    float[] proba = inputImage;
		    levelset = new float[nxyz];
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (proba[xyz]>=0.5f && (proba[xyz+1]<0.5f || proba[xyz-1]<0.5f
                                           || proba[xyz+nx]<0.5f || proba[xyz-nx]<0.5f
                                           || proba[xyz+nx*ny]<0.5f || proba[xyz-nx*ny]<0.5f)) 
                    levelset[xyz] = 0.5f-proba[xyz];
                else if (proba[xyz]<0.5f && (proba[xyz+1]>=0.5f || proba[xyz-1]>=0.5f
                                               || proba[xyz+nx]>=0.5f || proba[xyz-nx]>=0.5f
                                               || proba[xyz+nx*ny]>=0.5f || proba[xyz-nx*ny]>=0.5f))
                    levelset[xyz] = 0.5f-proba[xyz];
                else if (proba[xyz]>=0.5f) levelset[xyz] = -1.0f;
                else levelset[xyz] = +1.0f;
            }
            inputImage = null;
            InflateGdm gdm = new InflateGdm(levelset, nx, ny, nz, rx, ry, rz, mask, 0.4f, 0.4f, "no", null);
            gdm.evolveNarrowBand(0, 1.0f);
            levelset = gdm.getLevelSet();
		} else {
		    levelset = inputImage;
		}
		
		// compute the gradient norm
		medialImage = new float[nxyz];
		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
		    int xyz = x+nx*y+nx*ny*z;
		    if (levelset[xyz]<=0) {
                double grad = 0.0;
                grad += 0.25*(levelset[xyz+1]-levelset[xyz-1])*(levelset[xyz+1]-levelset[xyz-1]);
                grad += 0.25*(levelset[xyz+nx]-levelset[xyz-nx])*(levelset[xyz+nx]-levelset[xyz-nx]);
                grad += 0.25*(levelset[xyz+nx*ny]-levelset[xyz-nx*ny])*(levelset[xyz+nx*ny]-levelset[xyz-nx*ny]);
                
                medialImage[xyz] = (float)Numerics.max(0.0, 1.0-FastMath.sqrt(grad));
            } else {
                medialImage[xyz] = 0.0f;
            }
        }
         
		// use to build a distance function
        distImage = new float[nxyz];
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (medialImage[xyz]>=0.5f && (medialImage[xyz+1]<0.5f || medialImage[xyz-1]<0.5f
                                       || medialImage[xyz+nx]<0.5f || medialImage[xyz-nx]<0.5f
                                       || medialImage[xyz+nx*ny]<0.5f || medialImage[xyz-nx*ny]<0.5f)) 
                distImage[xyz] = 0.5f-medialImage[xyz];
            else if (medialImage[xyz]<0.5f && (medialImage[xyz+1]>=0.5f || medialImage[xyz-1]>=0.5f
                                           || medialImage[xyz+nx]>=0.5f || medialImage[xyz-nx]>=0.5f
                                           || medialImage[xyz+nx*ny]>=0.5f || medialImage[xyz-nx*ny]>=0.5f))
                distImage[xyz] = 0.5f-medialImage[xyz];
            else if (medialImage[xyz]>=0.5f) distImage[xyz] = -1.0f;
            else distImage[xyz] = +1.0f;
        }
        InflateGdm gdm = new InflateGdm(distImage, nx, ny, nz, rx, ry, rz, mask, 0.4f, 0.4f, "no", null);
        gdm.evolveNarrowBand(0, 1.0f);
        distImage = gdm.getLevelSet();

		// add distances to get thickness
		thickImage = new float[nxyz];
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (levelset[xyz]<=0) {
                thickImage[xyz] = distImage[xyz]-levelset[xyz];
            }
        }
	}

}
