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
	private static final String[] systemTypes = {"centroid_pca","voxel_pca","weighted_pca","weighted_som"}; 
	
	private float[] coordImage;
	private int[] transImage;
	
	// som parameters
	private int somDim = 3;
	private int somSize = 15;
	private int learningTime = 1000;
	private int totalTime = 2000;
	
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
	
	public final void setSomDimension(int val) { somDim = val; }
	public final void setSomSize(int val) { somSize = val; }
	public final void setLearningTime(int val) { learningTime = val; }
	public final void setTotalTime(int val) { totalTime = val; }
	
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
	    else if (systemParam.equals("weighted_som")) weightedSOM();
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
	
	private final void weightedSOM() {
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
	    double maxweight = 0.0;
        for (int lb=1;lb<nlb;lb++) {
            weight[0] += weight[lb];
            //weight[lb] = 1.0/weight[lb]/(nlb-1.0);
            // probabilities closer to 1 are nicer for the algorithm
            weight[lb] = 1.0/weight[lb]/(nlb-1.0);
            if (weight[lb]>maxweight) maxweight = weight[lb];
        }
        for (int lb=1;lb<nlb;lb++) {
            weight[lb] /= maxweight;
        }
        // 2. reformat the data points and probas
        int npt = (int)weight[0];
	    float[][] data = new float[npt][3];
	    float[] proba = new float[npt];
	    int pt=0;
	    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
	        int xyz = x+nx*y+nx*ny*z;
	        if (labelImage[xyz]>lbl[0]) {
                for (int lb=1;lb<nlb;lb++) if (labelImage[xyz]==lbl[lb]) {
                    data[pt][0] = x;
                    data[pt][1] = y;
                    data[pt][2] = z;
                    proba[pt] = (float)weight[lb];
                    pt++;
                    lb = nlb;
                }
            }
        }
	    
	    BasicSom algorithm = new BasicSom(data, proba, null, npt, 3, somDim, somSize, learningTime, totalTime);
	    algorithm.run_som3D();

		// build the corresponding coordinate system
		float[][] mapping = algorithm.interpolateSomOnData3D();
		coordImage = new float[3*nxyz];
		pt=0;
	    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
	        int xyz = x+nx*y+nx*ny*z;
	        if (labelImage[xyz]>lbl[0]) {
                coordImage[xyz+X*nxyz] = mapping[pt][X]; 
                coordImage[xyz+Y*nxyz] = mapping[pt][Y];
                coordImage[xyz+Z*nxyz] = mapping[pt][Z]; 
                pt++;
            }
		}
		
		System.out.println("som output");
		// output: warp som grid onto surface space
		float[][] som = algorithm.getSomWeights();

		/*
		System.out.println("som points: "+nx*ny*nz);
		mappedSomPoints = new float[nx*ny*nz*3];
		for (int xy=0;xy<nx*ny;xy++) {
		     mappedSomPoints[0+3*xy] = som[0][xy];
		     mappedSomPoints[1+3*xy] = som[1][xy];
		     mappedSomPoints[2+3*xy] = som[2][xy];
		}
		
		int ntriangles = (nx-1)*(ny-1)*(nz-1)*6;
		System.out.println("som triangles: "+ntriangles);
		mappedSomTriangles = new int[3*ntriangles];
		int tr=0;
		for (int x=0;x<nx-1;x++) {
		     for (int y=0;y<ny-1;y++) {
		         for (int z=0;z<nz-1;z++) {
                     // top XY
                     mappedSomTriangles[0+tr] = (x+0)+nx*(y+0)+nx*ny*(z+0);
                     mappedSomTriangles[1+tr] = (x+1)+nx*(y+0)+nx*ny*(z+0);
                     mappedSomTriangles[2+tr] = (x+0)+nx*(y+1)+nx*ny*(z+0);
                     tr+=3;
                     // bottom XY
                     mappedSomTriangles[0+tr] = (x+1)+nx*(y+0)+nx*ny*(z+0);
                     mappedSomTriangles[1+tr] = (x+0)+nx*(y+1)+nx*ny*(z+0);
                     mappedSomTriangles[2+tr] = (x+1)+nx*(y+1)+nx*ny*(z+0);
                     tr+=3;
                     // top YZ
                     mappedSomTriangles[0+tr] = (x+0)+nx*(y+0)+nx*ny*(z+0);
                     mappedSomTriangles[1+tr] = (x+0)+nx*(y+1)+nx*ny*(z+0);
                     mappedSomTriangles[2+tr] = (x+0)+nx*(y+0)+nx*ny*(z+1);
                     tr+=3;
                     // bottom YZ
                     mappedSomTriangles[0+tr] = (x+0)+nx*(y+1)+nx*ny*(z+0);
                     mappedSomTriangles[1+tr] = (x+0)+nx*(y+0)+nx*ny*(z+1);
                     mappedSomTriangles[2+tr] = (x+0)+nx*(y+1)+nx*ny*(z+1);
                     tr+=3;
                     // top ZX
                     mappedSomTriangles[0+tr] = (x+0)+nx*(y+0)+nx*ny*(z+0);
                     mappedSomTriangles[1+tr] = (x+0)+nx*(y+0)+nx*ny*(z+1);
                     mappedSomTriangles[2+tr] = (x+1)+nx*(y+0)+nx*ny*(z+0);
                     tr+=3;
                     // bottom ZX
                     mappedSomTriangles[0+tr] = (x+0)+nx*(y+0)+nx*ny*(z+1);
                     mappedSomTriangles[1+tr] = (x+1)+nx*(y+0)+nx*ny*(z+0);
                     mappedSomTriangles[2+tr] = (x+1)+nx*(y+0)+nx*ny*(z+1);
                     tr+=3;
                }
		     }
		}
		System.out.println("som values: "+nx*ny*nz);
		mappedSomValues = new float[nx*ny*nz];
		for (int x=0;x<nx;x++) {
		    for (int y=0;y<ny;y++) {
		        for (int z=0;z<nz;z++) {
		            int xyz = x+nx*y+nx*ny*z;
                    mappedSomValues[xyz] = labelImage[xyz];
                }
            }
		}
		System.out.println("done");
		*/
		// map the data onto it
		transImage = new int[nxyz];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
		    int xyz = x+nx*y+nx*ny*z;
		    
		    float xc = somSize*(float)x/nx;
		    float yc = somSize*(float)y/ny;
		    float zc = somSize*(float)z/nz;
		    
		    float xi = ImageInterpolation.linearClosestInterpolation(som[X], xc,yc,zc, somSize,somSize,somSize);
		    float yi = ImageInterpolation.linearClosestInterpolation(som[Y], xc,yc,zc, somSize,somSize,somSize);
		    float zi = ImageInterpolation.linearClosestInterpolation(som[Z], xc,yc,zc, somSize,somSize,somSize);
		    
		    transImage[xyz] = ImageInterpolation.nearestNeighborInterpolation(labelImage, 0, xi,yi,zi, nx,ny,nz);
		    //transImage[xyz] = (int)Numerics.min(xi,yi,zi);
		}
	    
	}
}
