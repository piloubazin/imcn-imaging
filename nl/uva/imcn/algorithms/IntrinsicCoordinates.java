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
	public final float[] getCoordinateImage() { return coordImage; }
	public final int[] getTransformedImage() { return transImage; }

	public void execute(){
	    if (systemParam.equals("centroid_pca")) centroidPCA();
	    else if (systemParam.equals("weighted_pca")) weightedPCA();
	    else voxelPCA();
	}
	
	private final void centroidPCA() { 
	    // 1. build centroid list
	    int nlb = ObjectLabeling.countLabels(labelImage, nx, ny, nz);
	    int[] lbl = ObjectLabeling.listOrderedLabels(labelImage, nx, ny, nz);
	    System.out.println("labels: "+nlb);
		
	    double[][] centroids = new double[nlb][4];
	    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
            int xyz = x+nx*y+nx*ny*z;
            for (int lb=1;lb<nlb;lb++) if (labelImage[xyz]==lbl[lb]) {
                centroids[lb][X] += x;
                centroids[lb][Y] += y;
                centroids[lb][Z] += z;
                centroids[lb][T] += 1;
                lb = nlb;
            }
	    }
        for (int lb=1;lb<nlb;lb++) {
            centroids[lb][X] /= centroids[lb][T];
            centroids[lb][Y] /= centroids[lb][T];
            centroids[lb][Z] /= centroids[lb][T];
        }
        for (int lb=1;lb<nlb;lb++) {
            centroids[0][X] += centroids[lb][X]/nlb;
            centroids[0][Y] += centroids[lb][Y]/nlb;
            centroids[0][Z] += centroids[lb][Z]/nlb;
        }
	    System.out.println("centroids:");
		for (int lb=0;lb<nlb;lb++) {
            System.out.println(lbl[lb]+" : ("+centroids[lb][X]+", "+centroids[lb][Y]+", "+centroids[lb][Z]+")");
        }
        
	    // 2. shape PCA on the centroids
		double[][] shape = new double[3][nlb-1];
	    for (int lb=1;lb<nlb;lb++) {
	        shape[X][lb-1] = centroids[lb][X]-centroids[0][X];
	        shape[Y][lb-1] = centroids[lb][Y]-centroids[0][Y];
	        shape[Z][lb-1] = centroids[lb][Z]-centroids[0][Z];
	    }
		Matrix M = new Matrix(shape);
		SingularValueDecomposition svd = M.svd();
		Matrix U = svd.getU();
		
		System.out.println("singular values:");
		double[] singval = new double[3];
		for (int n=0;n<3;n++) {
			singval[n] = svd.getSingularValues()[n];
			System.out.print(singval[n]+", ");
		}
		double[][] axes = new double[3][3];
		for (int n=0;n<3;n++) {
		    axes[X][n] = singval[n]*U.get(X,n);
		    axes[Y][n] = singval[n]*U.get(Y,n);
		    axes[Z][n] = singval[n]*U.get(Z,n);
		}
		// build the corresponding coordinate system
		coordImage = new float[3*nxyz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
		    int xyz = x+nx*y+nx*ny*z;
            coordImage[xyz+X*nxyz] = (float)( (x-centroids[0][X])*axes[X][X] 
                                            + (y-centroids[0][Y])*axes[Y][X] 
                                            + (z-centroids[0][Z])*axes[Z][X] ); 
                              
            coordImage[xyz+Y*nxyz] = (float)( (x-centroids[0][X])*axes[X][Y] 
                                            + (y-centroids[0][Y])*axes[Y][Y] 
                                            + (z-centroids[0][Z])*axes[Z][Y] ); 

            coordImage[xyz+Z*nxyz] = (float)( (x-centroids[0][X])*axes[X][Z] 
                                            + (y-centroids[0][Y])*axes[Y][Z] 
                                            + (z-centroids[0][Z])*axes[Z][Z] ); 

		}
		// map the data onto it
		transImage = new int[nxyz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
		    int xyz = x+nx*y+nx*ny*z;
		    
		    float xc = 2.0f*(x-nx/2.0f)/nx;
		    float yc = 2.0f*(y-ny/2.0f)/ny;
		    float zc = 2.0f*(z-nz/2.0f)/nz;
		    
		    float xi = (float)(xc*axes[X][X] + yc*axes[X][Y] + zc*axes[X][Z] + centroids[0][X]);
		    float yi = (float)(xc*axes[Y][X] + yc*axes[Y][Y] + zc*axes[Y][Z] + centroids[0][Y]);
		    float zi = (float)(xc*axes[Z][X] + yc*axes[Z][Y] + zc*axes[Z][Z] + centroids[0][Z]);
		    
		    transImage[xyz] = ImageInterpolation.nearestNeighborClosestInterpolation(labelImage, xi,yi,zi, nx,ny,nz);
		}
	}

	private final void weightedPCA() { 
	    // 1. estimate volumes for weighting
	    int nlb = ObjectLabeling.countLabels(labelImage, nx, ny, nz);
	    int[] lbl = ObjectLabeling.listOrderedLabels(labelImage, nx, ny, nz);
	    System.out.println("labels: "+nlb);
		
	    double[] weight = new double[nlb];
	    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (labelImage[xyz]>lbl[0]) {
                for (int lb=1;lb<nlb;lb++) if (labelImage[xyz]==lbl[lb]) {
                    weight[lb]++;
                    lb = nlb;
                }
            }
	    }
        for (int lb=1;lb<nlb;lb++) {
            weight[lb] = 1.0/weight[lb]/(nlb-1.0);
        }
        
        // 2. build covariance matrix
        double[] avg = new double[3];
        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (labelImage[xyz]>lbl[0]) {
                for (int lb=1;lb<nlb;lb++) if (labelImage[xyz]==lbl[lb]) {
                    avg[X] += weight[lb]*x;
                    avg[Y] += weight[lb]*y;
                    avg[Z] += weight[lb]*z;
                    lb = nlb;
                }
            }
        }
        
	    double[][] covar = new double[3][3];
	    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (labelImage[xyz]>lbl[0]) {
                for (int lb=1;lb<nlb;lb++) if (labelImage[xyz]==lbl[lb]) {
                    covar[X][X] += weight[lb]*(x-avg[X])*(x-avg[X]);
                    covar[X][Y] += weight[lb]*(x-avg[X])*(y-avg[Y]);
                    covar[X][Z] += weight[lb]*(x-avg[X])*(z-avg[Z]);

                    covar[Y][X] += weight[lb]*(y-avg[Y])*(x-avg[X]);
                    covar[Y][Y] += weight[lb]*(y-avg[Y])*(y-avg[Y]);
                    covar[Y][Z] += weight[lb]*(y-avg[Y])*(z-avg[Z]);

                    covar[Z][X] += weight[lb]*(z-avg[Z])*(x-avg[X]);
                    covar[Z][Y] += weight[lb]*(z-avg[Z])*(y-avg[Y]);
                    covar[Z][Z] += weight[lb]*(z-avg[Z])*(z-avg[Z]);
                    lb = nlb;
                }
            }
        }
                    
	    // 3. shape PCA on the covarianca
		Matrix M = new Matrix(covar);
		SingularValueDecomposition svd = M.svd();
		Matrix U = svd.getU();
		
		System.out.println("singular values:");
		double[] singval = new double[3];
		for (int n=0;n<3;n++) {
			singval[n] = FastMath.sqrt(svd.getSingularValues()[n]);
			System.out.print(singval[n]+", ");
		}
		double[][] axes = new double[3][3];
		for (int n=0;n<3;n++) {
		    axes[X][n] = singval[n]*U.get(X,n);
		    axes[Y][n] = singval[n]*U.get(Y,n);
		    axes[Z][n] = singval[n]*U.get(Z,n);
		}
		// build the corresponding coordinate system
		coordImage = new float[3*nxyz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
		    int xyz = x+nx*y+nx*ny*z;
            coordImage[xyz+X*nxyz] = (float)( (x-avg[X])*axes[X][X] 
                                            + (y-avg[Y])*axes[Y][X] 
                                            + (z-avg[Z])*axes[Z][X] ); 
                              
            coordImage[xyz+Y*nxyz] = (float)( (x-avg[X])*axes[X][Y] 
                                            + (y-avg[Y])*axes[Y][Y] 
                                            + (z-avg[Z])*axes[Z][Y] ); 

            coordImage[xyz+Z*nxyz] = (float)( (x-avg[X])*axes[X][Z] 
                                            + (y-avg[Y])*axes[Y][Z] 
                                            + (z-avg[Z])*axes[Z][Z] ); 

		}
		// map the data onto it
		transImage = new int[nxyz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
		    int xyz = x+nx*y+nx*ny*z;
		    
		    float xc = 2.0f*(x-nx/2.0f)/nx;
		    float yc = 2.0f*(y-ny/2.0f)/ny;
		    float zc = 2.0f*(z-nz/2.0f)/nz;
		    
		    float xi = (float)(xc*axes[X][X] + yc*axes[X][Y] + zc*axes[X][Z] + avg[X]);
		    float yi = (float)(xc*axes[Y][X] + yc*axes[Y][Y] + zc*axes[Y][Z] + avg[Y]);
		    float zi = (float)(xc*axes[Z][X] + yc*axes[Z][Y] + zc*axes[Z][Z] + avg[Z]);
		    
		    transImage[xyz] = ImageInterpolation.nearestNeighborClosestInterpolation(labelImage, xi,yi,zi, nx,ny,nz);
		}
	}

	private final void voxelPCA() { 
	    // 1. estimate volumes for weighting
	    int nlb = ObjectLabeling.countLabels(labelImage, nx, ny, nz);
	    int[] lbl = ObjectLabeling.listOrderedLabels(labelImage, nx, ny, nz);
	    System.out.println("labels: "+nlb);
		
	    double weight = 0.0;
	    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (labelImage[xyz]>lbl[0]) {
                weight++;
            }
	    }
        weight = 1.0/weight;
        
        // 2. build covariance matrix
        double[] avg = new double[3];
        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (labelImage[xyz]>lbl[0]) {
                avg[X] += weight*x;
                avg[Y] += weight*y;
                avg[Z] += weight*z;
            }
        }
        
	    double[][] covar = new double[3][3];
	    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if (labelImage[xyz]>lbl[0]) {
                covar[X][X] += weight*(x-avg[X])*(x-avg[X]);
                covar[X][Y] += weight*(x-avg[X])*(y-avg[Y]);
                covar[X][Z] += weight*(x-avg[X])*(z-avg[Z]);

                covar[Y][X] += weight*(y-avg[Y])*(x-avg[X]);
                covar[Y][Y] += weight*(y-avg[Y])*(y-avg[Y]);
                covar[Y][Z] += weight*(y-avg[Y])*(z-avg[Z]);

                covar[Z][X] += weight*(z-avg[Z])*(x-avg[X]);
                covar[Z][Y] += weight*(z-avg[Z])*(y-avg[Y]);
                covar[Z][Z] += weight*(z-avg[Z])*(z-avg[Z]);
            }
        }
                    
	    // 3. shape PCA on the covarianca
		Matrix M = new Matrix(covar);
		SingularValueDecomposition svd = M.svd();
		Matrix U = svd.getU();
		
		System.out.println("singular values:");
		double[] singval = new double[3];
		for (int n=0;n<3;n++) {
			singval[n] = FastMath.sqrt(svd.getSingularValues()[n]);
			System.out.print(singval[n]+", ");
		}
		double[][] axes = new double[3][3];
		for (int n=0;n<3;n++) {
		    axes[X][n] = singval[n]*U.get(X,n);
		    axes[Y][n] = singval[n]*U.get(Y,n);
		    axes[Z][n] = singval[n]*U.get(Z,n);
		}
		// build the corresponding coordinate system
		coordImage = new float[3*nxyz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
		    int xyz = x+nx*y+nx*ny*z;
            coordImage[xyz+X*nxyz] = (float)( (x-avg[X])*axes[X][X] 
                                            + (y-avg[Y])*axes[Y][X] 
                                            + (z-avg[Z])*axes[Z][X] ); 
                              
            coordImage[xyz+Y*nxyz] = (float)( (x-avg[X])*axes[X][Y] 
                                            + (y-avg[Y])*axes[Y][Y] 
                                            + (z-avg[Z])*axes[Z][Y] ); 

            coordImage[xyz+Z*nxyz] = (float)( (x-avg[X])*axes[X][Z] 
                                            + (y-avg[Y])*axes[Y][Z] 
                                            + (z-avg[Z])*axes[Z][Z] ); 

		}
		// map the data onto it
		transImage = new int[nxyz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
		    int xyz = x+nx*y+nx*ny*z;
		    
		    float xc = 2.0f*(x-nx/2.0f)/nx;
		    float yc = 2.0f*(y-ny/2.0f)/ny;
		    float zc = 2.0f*(z-nz/2.0f)/nz;
		    
		    float xi = (float)(xc*axes[X][X] + yc*axes[X][Y] + zc*axes[X][Z] + avg[X]);
		    float yi = (float)(xc*axes[Y][X] + yc*axes[Y][Y] + zc*axes[Y][Z] + avg[Y]);
		    float zi = (float)(xc*axes[Z][X] + yc*axes[Z][Y] + zc*axes[Z][Z] + avg[Z]);
		    
		    transImage[xyz] = ImageInterpolation.nearestNeighborClosestInterpolation(labelImage, xi,yi,zi, nx,ny,nz);
		}
	}
}
