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
	private float[] shapeImage=null;
	private int[] labelImage=null;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;

	private String featureParam = "probability_map";
	private static final String[] featureTypes = {"signed_distance","probability_map","parcellation"}; 
	
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
	public final void setShapeImage(float[] val) { shapeImage = val; }
	public final void setLabelImage(int[] val) { labelImage = val; }
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
	    if (featureParam.equals("parcellation")) multiLabelThickness();
        else structureThickness();
	}
	
	public void structureThickness(){
		
	    // if needed, convert to levelset
        boolean[] mask = new boolean[nxyz];
        for (int n=0;n<nxyz;n++) mask[n] = true;

		float rmax = Numerics.max(rx,ry,rz);
	    
		float[] levelset;
		if (featureParam.equals("probability_map")) {
		    float[] proba = shapeImage;
		    
		    levelset = new float[nxyz];
            for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (x>0 && x<nx-1 && y>0 && y<ny-1 && z>0 && z<nz-1) {
                    if (proba[xyz]>=0.5f && (proba[xyz+1]<0.5f || proba[xyz-1]<0.5f
                                               || proba[xyz+nx]<0.5f || proba[xyz-nx]<0.5f
                                               || proba[xyz+nx*ny]<0.5f || proba[xyz-nx*ny]<0.5f)) 
                        levelset[xyz] = 0.5f-proba[xyz];
                    else if (proba[xyz]<0.5f && (proba[xyz+1]>=0.5f || proba[xyz-1]>=0.5f
                                                   || proba[xyz+nx]>=0.5f || proba[xyz-nx]>=0.5f
                                                   || proba[xyz+nx*ny]>=0.5f || proba[xyz-nx*ny]>=0.5f))
                        
                        levelset[xyz] = -0.5f;
                    else if (proba[xyz]>=0.5f) levelset[xyz] = -1.0f;
                    else levelset[xyz] = +1.0f;
                } else levelset[xyz] = +1.0f;
            }
            // second pass to bring the outside at the correct distance
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (proba[xyz]<0.5f && (proba[xyz+1]>=0.5f || proba[xyz-1]>=0.5f
                                       || proba[xyz+nx]>=0.5f || proba[xyz-nx]>=0.5f
                                       || proba[xyz+nx*ny]>=0.5f || proba[xyz-nx*ny]>=0.5f))
                    
                    levelset[xyz] = ObjectTransforms.fastMarchingOutsideNeighborDistance(levelset, xyz, nx,ny,nz, rx,ry,rz);
			}
            shapeImage = null;
            levelset = ObjectTransforms.fastMarchingDistanceFunction(levelset, 10.0f*rmax, nx, ny, nz, rx, ry, rz);
		} else {
		    levelset = shapeImage;
		}
		
		// compute the gradient norm
		medialImage = new float[nxyz];
		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
		    int xyz = x+nx*y+nx*ny*z;
		    if (levelset[xyz]<=0) {
                double grad = 0.0;
                grad += 0.25*(levelset[xyz+1]-levelset[xyz-1])*(levelset[xyz+1]-levelset[xyz-1])/(rx/rmax);
                grad += 0.25*(levelset[xyz+nx]-levelset[xyz-nx])*(levelset[xyz+nx]-levelset[xyz-nx])/(ry/rmax);
                grad += 0.25*(levelset[xyz+nx*ny]-levelset[xyz-nx*ny])*(levelset[xyz+nx*ny]-levelset[xyz-nx*ny])/(rz/rmax);
                
                medialImage[xyz] = (float)Numerics.max(0.0, 1.0-FastMath.sqrt(grad));
            } else {
                medialImage[xyz] = 0.0f;
            }
        }
         
		// use to build a distance function
        distImage = new float[nxyz];
        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (x>0 && x<nx-1 && y>0 && y<ny-1 && z>0 && z<nz-1) {
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
            } else {
                distImage[xyz] = +1.0f;
            }
        }
        distImage = ObjectTransforms.fastMarchingDistanceFunction(distImage, 10.0f*rmax, nx, ny, nz, rx, ry, rz);

		// add distances to get thickness
		thickImage = new float[nxyz];
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (levelset[xyz]<=0) {
                thickImage[xyz] = 2.0f*(distImage[xyz]-levelset[xyz])*rmax;
            }
        }
	}

	public void multiLabelThickness(){
		
	    boolean[] mask = new boolean[nxyz];
        for (int n=0;n<nxyz;n++) mask[n] = true;
        
        float rmax = Numerics.max(rx,ry,rz);
		
        // iterate over all structures
        int nlb = ObjectLabeling.countLabels(labelImage, nx, ny, nz);
	    int[] lbl = ObjectLabeling.listLabels(labelImage, nx, ny, nz);
	    
        thickImage = new float[nxyz];
        medialImage = new float[nxyz];
        distImage = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) distImage[xyz] = 10.0f*rmax;
        
	    for (int n=0;n<nlb;n++) if (lbl[n]>0) {
	       
            float[] levelset= new float[nxyz];
            for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (x>0 && x<nx-1 && y>0 && y<ny-1 && z>0 && z<nz-1) {
                    if (labelImage[xyz]==lbl[n] && (labelImage[xyz+1]!=lbl[n] || labelImage[xyz-1]!=lbl[n]
                                                 || labelImage[xyz+nx]!=lbl[n] || labelImage[xyz-nx]!=lbl[n]
                                                 || labelImage[xyz+nx*ny]!=lbl[n] || labelImage[xyz-nx*ny]!=lbl[n]) )
                        levelset[xyz] = -0.5f;
                    else if (labelImage[xyz]!=lbl[n] && (labelImage[xyz+1]==lbl[n] || labelImage[xyz-1]==lbl[n]
                                                      || labelImage[xyz+nx]==lbl[n] || labelImage[xyz-nx]==lbl[n]
                                                      || labelImage[xyz+nx*ny]==lbl[n] || labelImage[xyz-nx*ny]==lbl[n]) )
                        levelset[xyz] = 0.5f;
                    else if (labelImage[xyz]==lbl[n]) levelset[xyz] = -1.0f;
                    else levelset[xyz] = +1.0f;
                } else {
                    levelset[xyz] = +1.0f;
                }
            }
            // second pass to bring the outside at the correct distance
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (labelImage[xyz]!=lbl[n] && (labelImage[xyz+1]==lbl[n] || labelImage[xyz-1]==lbl[n]
                                             || labelImage[xyz+nx]==lbl[n] || labelImage[xyz-nx]==lbl[n]
                                             || labelImage[xyz+nx*ny]==lbl[n] || labelImage[xyz-nx*ny]==lbl[n]) )
                    
                    levelset[xyz] = ObjectTransforms.fastMarchingOutsideNeighborDistance(levelset, xyz, nx,ny,nz, rx,ry,rz);
			}
            levelset = ObjectTransforms.fastMarchingDistanceFunction(levelset, 10.0f*rmax, nx, ny, nz, rx, ry, rz);
            
            // compute the gradient norm
            float[] medial = new float[nxyz];
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (levelset[xyz]<=0) {
                    double grad = 0.0;
                    grad += 0.25*(levelset[xyz+1]-levelset[xyz-1])*(levelset[xyz+1]-levelset[xyz-1])/(rx/rmax);
                    grad += 0.25*(levelset[xyz+nx]-levelset[xyz-nx])*(levelset[xyz+nx]-levelset[xyz-nx])/(ry/rmax);
                    grad += 0.25*(levelset[xyz+nx*ny]-levelset[xyz-nx*ny])*(levelset[xyz+nx*ny]-levelset[xyz-nx*ny])/(rz/rmax);
                    
                    medial[xyz] = (float)Numerics.max(0.0, 1.0-FastMath.sqrt(grad));
                } else {
                    medial[xyz] = 0.0f;
                }
            }
             
            // use to build a distance function
            float[] dist = new float[nxyz];
            for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (x>0 && x<nx-1 && y>0 && y<ny-1 && z>0 && z<nz-1) {
                    if (medial[xyz]>=0.5f && (medial[xyz+1]<0.5f || medial[xyz-1]<0.5f
                                           || medial[xyz+nx]<0.5f || medial[xyz-nx]<0.5f
                                           || medial[xyz+nx*ny]<0.5f || medial[xyz-nx*ny]<0.5f)) 
                        dist[xyz] = 0.5f-medial[xyz];
                    else if (medial[xyz]<0.5f && (medial[xyz+1]>=0.5f || medial[xyz-1]>=0.5f
                                               || medial[xyz+nx]>=0.5f || medial[xyz-nx]>=0.5f
                                               || medial[xyz+nx*ny]>=0.5f || medial[xyz-nx*ny]>=0.5f))
                        dist[xyz] = 0.5f-medial[xyz];
                    else if (medial[xyz]>=0.5f) dist[xyz] = -1.0f;
                    else dist[xyz] = +1.0f;
                } else {
                    dist[xyz] = +1.0f;
                }
            }
            dist = ObjectTransforms.fastMarchingDistanceFunction(dist, 10.0f*rmax, nx, ny, nz, rx, ry, rz);

            // combine maps to get thickness
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (labelImage[xyz]==lbl[n]) {
                    thickImage[xyz] = 2.0f*(dist[xyz]-levelset[xyz])*rmax;
                    distImage[xyz] = dist[xyz];
                    medialImage[xyz] = medial[xyz];
                }
            }
        }
	}

}
