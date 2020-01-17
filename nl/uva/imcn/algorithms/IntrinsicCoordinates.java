package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import nl.uva.imcn.structures.*;
import nl.uva.imcn.libraries.*;
import nl.uva.imcn.methods.*;

import org.apache.commons.math3.util.FastMath;
import Jama.*;


/*
 * @author Pierre-Louis Bazin
 */
public class IntrinsicCoordinates {

	// jist containers
	private int[] labelImage=null;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;

	private String systemParam = "centroid_pca";
	private static final String[] systemTypes = {"centroid_pca","voxel_pca"}; 
	
	private float[] coordImage;
	private int[] transImage;
	
	// numerical quantities
	private static final	float	INVSQRT2 = (float)(1.0/FastMath.sqrt(2.0));
	private static final	float	INVSQRT3 = (float)(1.0/FastMath.sqrt(3.0));
	private static final	float	SQRT2 = (float)FastMath.sqrt(2.0);
	private static final	float	SQRT3 = (float)FastMath.sqrt(3.0);

	// direction labeling		
	public	static	final	byte	X = 0;
	public	static	final	byte	Y = 1;
	public	static	final	byte	Z = 2;
	public	static	final	byte	T = 3;
	
	// for debug and display
	private static final boolean		debug=true;
	private static final boolean		verbose=true;

	// create inputs
	public final void setLabelImage(int[] val) { labelImage = val; }
	public final void setSystemType(String val) { systemParam = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
				
	// create outputs
	public final float[] geCoordinateImage() { return coordImage; }
	public final int[] getTransformedImage() { return transImage; }

	public void execute(){
		
	    // 1. build centroid list
	    int nlb = ObjectLabeling.countLabels(labelImage, nx, ny, nz);
	    int[] lbl = ObjectLabeling.listLabels(labelImage, nx, ny, nz);
	    double[][] centroids = new double[nlb+1][4];
	    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (labelImage[xyz]>0) {
                for (int lb=0;lb<nlb;lb++) if (labelImage[xyz]==lbl[lb]) {
                    centroids[lb][X] += x;
                    centroids[lb][Y] += y;
                    centroids[lb][Z] += z;
                    centroids[lb][T] += 1;
                    lb = nlb;
                }
            }
	    }
        for (int lb=0;lb<nlb;lb++) {
            centroids[lb][X] /= centroids[lb][T];
            centroids[lb][Y] /= centroids[lb][T];
            centroids[lb][Z] /= centroids[lb][T];
        }
        for (int lb=0;lb<nlb;lb++) {
            centroids[nlb][X] += centroids[lb][X]/nlb;
            centroids[nlb][Y] += centroids[lb][Y]/nlb;
            centroids[nlb][Z] += centroids[lb][Z]/nlb;
        }
	    
	    // 2. shape PCA on the centroids
		double[][] shape = new double[3][nlb];
	    for (int lb=0;lb<nlb;lb++) {
	        shape[X][lb] = centroids[lb][X]-centroids[nlb][X];
	        shape[Y][lb] = centroids[lb][Y]-centroids[nlb][Y];
	        shape[Z][lb] = centroids[lb][Z]-centroids[nlb][Z];
	    }
		Matrix M = new Matrix(shape);
		SingularValueDecomposition svd = M.svd();
		Matrix U = svd.getU();
		
		System.out.println("eigenvalues:");
		double[] eigenval = new double[3];
		for (int n=0;n<3;n++) {
			eigenval[n] = svd.getSingularValues()[n]*svd.getSingularValues()[n];
			System.out.print(eigenval[n]+", ");
		}
		double[][] axes = new double[3][3];
		for (int n=0;n<3;n++) {
		    axes[X][n] = eigenval[n]*U.get(X,n);
		    axes[Y][n] = eigenval[n]*U.get(Y,n);
		    axes[Z][n] = eigenval[n]*U.get(Z,n);
		}
		// build the corresponding coordinate system
		coordImage = new float[3*nxyz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
		    int xyz = x+nx*y+nx*ny*z;
            coordImage[xyz+X*nxyz] = (float)( (x-centroids[nlb][X])*axes[X][X] 
                                            + (y-centroids[nlb][Y])*axes[Y][X] 
                                            + (z-centroids[nlb][Z])*axes[Z][X] ); 
                              
            coordImage[xyz+Y*nxyz] = (float)( (x-centroids[nlb][X])*axes[X][Y] 
                                            + (y-centroids[nlb][Y])*axes[Y][Y] 
                                            + (z-centroids[nlb][Z])*axes[Z][Y] ); 

            coordImage[xyz+Z*nxyz] = (float)( (x-centroids[nlb][X])*axes[X][Z] 
                                            + (y-centroids[nlb][Y])*axes[Y][Z] 
                                            + (z-centroids[nlb][Z])*axes[Z][Z] ); 

		}
		// map the data onto it
		transImage = new int[nxyz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
		    int xyz = x+nx*y+nx*ny*z;
		    
		    float xc = 3.0f*(x-nx/2.0f)/nx;
		    float yc = 3.0f*(y-ny/2.0f)/ny;
		    float zc = 3.0f*(z-nz/2.0f)/nz;
		    
		    float xi = (float)(xc*axes[X][X] + yc*axes[X][Y] + zc*axes[X][Z] + centroids[nlb][X]);
		    float yi = (float)(xc*axes[Y][X] + yc*axes[Y][Y] + zc*axes[Y][Z] + centroids[nlb][Y]);
		    float zi = (float)(xc*axes[Z][X] + yc*axes[Z][Y] + zc*axes[Z][Z] + centroids[nlb][Z]);
		    
		    transImage[xyz] = ImageInterpolation.nearestNeighborClosestInterpolation(labelImage, xi,yi,zi, nx,ny,nz);
		}
	}
}
