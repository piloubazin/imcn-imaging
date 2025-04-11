package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import nl.uva.imcn.structures.*;
import nl.uva.imcn.libraries.*;
import nl.uva.imcn.methods.*;

import org.apache.commons.math3.util.FastMath;


/*
 * @author Pierre-Louis Bazin
 */
public class ImageShapeOperator {

	// jist containers
	private float[] image=null;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;

	private float[] shapeImage;

	private final static int X = 0;
    private final static int Y = 1;
    private final static int Z = 2;

	// for debug and display
	private static final boolean		debug=true;
	private static final boolean		verbose=true;

	// create inputs
	public final void setInputImage(float[] val) { image = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
				
	// create outputs
	public final float[] getShapeOperatorImage() { return shapeImage; }
	
	public void execute(){
		float[][] first = new float[3][3];
		float[][] second = new float[3][3];
		shapeImage = new float[nxyz*6];
		
		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
		    int xyz = x+nx*y+nx*ny*z;
		    if (image[xyz]!=0) {
		        float dx = 0.5f*(image[xyz+1]-image[xyz-1]);
		        float dy = 0.5f*(image[xyz+nx]-image[xyz-nx]);
		        float dz = 0.5f*(image[xyz+nx*ny]-image[xyz-nx*ny]);
		        
		        first[X][X] = dx*dx;
		        first[Y][Y] = dy*dy;
		        first[Z][Z] = dz*dz;
		        first[X][Y] = dx*dy;
		        first[Y][Z] = dy*dz;
		        first[Z][X] = dz*dx;
		        first[X][Z] = dx*dz;
		        first[Z][Y] = dz*dy;
                first[Y][X] = dy*dx;
                
                Matrix3D.invert(first);
                
                float dxx = image[xyz+1] - 2.0f*image[xyz] + image[xyz-1];
                float dyy = image[xyz+nx] - 2.0f*image[xyz] + image[xyz-nx];
                float dzz = image[xyz+nx*ny] - 2.0f*image[xyz] + image[xyz-nx*ny];
                
                float dxy = 0.25f*(image[xyz+1+nx] - image[xyz-1+nx] - image[xyz+1-nx] + image[xyz-1-nx]);
                float dyz = 0.25f*(image[xyz+nx+nx*ny] - image[xyz-nx+nx*ny] - image[xyz+nx-nx*ny] + image[xyz-nx-nx*ny]);
                float dzx = 0.25f*(image[xyz+nx*ny+1] - image[xyz-nx*ny+1] - image[xyz+nx*ny-1] + image[xyz-nx*ny-1]);
                
		        second[X][X] = dxx;
		        second[Y][Y] = dyy;
		        second[Z][Z] = dzz;
		        second[X][Y] = dxy;
		        second[Y][Z] = dyz;
		        second[Z][X] = dzx;
		        second[X][Z] = dzx;
		        second[Z][Y] = dyz;
                second[Y][X] = dxy;
                
                // combine forms to create operator
                shapeImage[xyz+0*nxyz] = first[X][X]*second[X][X] + first[X][Y]*second[Y][X] + first[X][Z]*second[Z][X];
                shapeImage[xyz+1*nxyz] = first[Y][X]*second[X][Y] + first[Y][Y]*second[Y][Y] + first[Y][Z]*second[Z][Y];
                shapeImage[xyz+2*nxyz] = first[Z][X]*second[X][Z] + first[Z][Y]*second[Y][Z] + first[Z][Z]*second[Z][Z];
                shapeImage[xyz+3*nxyz] = first[X][X]*second[X][Y] + first[X][Y]*second[Y][Y] + first[X][Z]*second[Z][Y];
                shapeImage[xyz+4*nxyz] = first[X][X]*second[X][Z] + first[X][Y]*second[Y][Z] + first[X][Z]*second[Z][Z];
                shapeImage[xyz+5*nxyz] = first[Y][X]*second[X][Z] + first[Y][Y]*second[Y][Z] + first[Y][Z]*second[Z][Z];
            }
                
        }
        
	}

}
