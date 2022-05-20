package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import nl.uva.imcn.structures.*;
import nl.uva.imcn.libraries.*;
import nl.uva.imcn.methods.*;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.linear.*;
//import Jama.*;
//import org.jblas.*;

/*
 * @author Pierre-Louis Bazin
 */
public class SpectralShapeEmbedding {

	// jist containers
	private int[] labelImage=null;
	private float[][] contrastImages = null;
	private float[] contrastDev = null;
	private int nc=0;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;

	private String systemParam = "single";
	private static final String[] systemTypes = {"single","joint"}; 
	private float spaceDev = 10.0f;
	private boolean sparse=true;
	private int msize = 800;
	private static final String[] refAxTypes = {"none","X","Y","Z"};
	private String refAxis = "none";
	
	private float[] coordImage;
	
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
	
	public final void setSpatialDev(float val) { spaceDev = val; }
	public final void setSystemType(String val) { systemParam = val; }
	
	public final void setContrastNumber(int val) { 
	    nc = val;
	    contrastImages = new float[nc][];
	    contrastDev = new float[nc];
	}
	public final void setContrastImageAt(int num, float[] val) { contrastImages[num] = val; }
	public final void setContrastDevAt(int num, float val) { contrastDev[num] = val; }
	public final void setMatrixSize(int val) { msize = val; }
	
	public final void setReferenceAxis(String val) { refAxis = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
				
	// create outputs
	public final float[] getCoordinateImage() { return coordImage; }

	public void execute(){
	    if (systemParam.equals("single")) singleShapeEmbedding();
	    else if (systemParam.equals("joint")) jointShapeEmbedding();
	}
	
	private final void singleShapeEmbedding() { 
	    
	    // 1. build label list
	    int nlb = ObjectLabeling.countLabels(labelImage, nx, ny, nz);
	    int[] lbl = ObjectLabeling.listOrderedLabels(labelImage, nx, ny, nz);
	    System.out.println("labels: "+nlb);
		
	    spaceDev *= spaceDev;
	    if (nc>0) {
	        for (int c=0;c<nc;c++) {
	            contrastDev[c] *= contrastDev[c];
	        }
	    }
	    
	    coordImage = new float[4*nxyz];
            
	    for (int n=0;n<nlb;n++) if (n>0) {
	        // compute embeddings independetly for each structure
	        int lb = lbl[n];
	        System.out.println("process label "+lb);
	        
	        int vol=0;
	        for (int xyz=0;xyz<nxyz;xyz++) if (labelImage[xyz]==lb) vol++;
	        System.out.println("base volume: "+vol);
            
	        // if volume is too big, subsample: no need to work with giant matrices?
	        int sub=1;
	        if (vol>500) sub=2;
	        if (vol>4000) sub=3;
	        if (vol>13500) sub=4;
	        if (vol>32000) sub=5;
	        if (vol>62500) sub=6;
	        if (vol>108000) sub=7;
	        if (vol>171500) sub=8;
	        if (vol>256000) sub=9;
	        if (vol>364500) sub=10;
	        if (vol>500000) sub=11;
	        if (vol>665500) sub=12;
	        
	        // explict target instead?
	        sub = Numerics.max(1, Numerics.round(FastMath.cbrt(vol/msize)));
	        
	        if (sub>1) {
	            vol=0;
	            for (int x=0;x<nx;x+=sub) for (int y=0;y<ny;y+=sub) for (int z=0;z<nz;z+=sub) {
                    int xyz = x+nx*y+nx*ny*z;
                    float npt=0.0f;
                    for (int dx=0;dx<sub;dx++) for (int dy=0;dy<sub;dy++) for (int dz=0;dz<sub;dz++) {
                        if (x+dx<nx && y+dy<ny && z+dz<nz) {
                            int ngb = xyz+dx+nx*dy+nx*ny*dz;
                            if (labelImage[ngb]==lb) {
                                npt++;
                            }
                        }
                    }
                    if (npt>0) vol++;
                }
            }
            System.out.println("final volume: "+vol+" ("+sub+")");
            
            // build coordinate and contrast arrays
            float[][] coord = new float[3][vol];
            float[][] contrasts = null;
            if (nc>0) contrasts = new float[nc][vol];
            float[] weight = new float[vol];
            int v=0;
            for (int x=0;x<nx;x+=sub) for (int y=0;y<ny;y+=sub) for (int z=0;z<nz;z+=sub) {
                int xyz = x+nx*y+nx*ny*z;
                float npt=0.0f;
                for (int dx=0;dx<sub;dx++) for (int dy=0;dy<sub;dy++) for (int dz=0;dz<sub;dz++) {
                    if (x+dx<nx && y+dy<ny && z+dz<nz) {
                        int ngb = xyz+dx+nx*dy+nx*ny*dz;
                        if (labelImage[ngb]==lb) {
                            coord[X][v] += x;
                            coord[Y][v] += y;
                            coord[Z][v] += z;
                            if (nc>0) {
                                for (int c=0;c<nc;c++) {
                                    contrasts[c][v] += contrastImages[c][xyz];
                                }
                            }
                            npt++;
                        }
                    }
                }
                if (npt>0) {
                    coord[X][v] /= npt;
                    coord[Y][v] /= npt;
                    coord[Z][v] /= npt;
                    if (nc>0) {
                        for (int c=0;c<nc;c++) {
                            contrasts[c][v] /= npt;
                        }
                    }
                    weight[v] = npt/(sub*sub*sub);
                    v++;
                }
            }       
            System.out.println("..contrasts");
                    
            // build correlation matrix
            double[][] matrix = new double[vol][vol];
            for (int v1=0;v1<vol;v1++) for (int v2=v1+1;v2<vol;v2++) {
                double dist = (coord[X][v1]-coord[X][v2])*(coord[X][v1]-coord[X][v2])
                             +(coord[Y][v1]-coord[Y][v2])*(coord[Y][v1]-coord[Y][v2])
                             +(coord[Z][v1]-coord[Z][v2])*(coord[Z][v1]-coord[Z][v2]);
                // when computing a sparse version, only keep strict 26-C neighbors             
                if (sparse && dist>3.0*sub*sub) {
                    matrix[v1][v2] = 0.0;
                } else {
                    if (sparse) matrix[v1][v2] = 1.0/FastMath.sqrt(dist);
                    else matrix[v1][v2] = FastMath.exp(-0.5*dist/spaceDev);
                    
                    if (nc>0) {
                        double diff = 0.0;
                        for (int c=0;c<nc;c++) {
                            diff += (contrasts[c][v1]-contrasts[c][v2])
                                   *(contrasts[c][v1]-contrasts[c][v2])/contrastDev[c];
                        }
                        matrix[v1][v2] *= FastMath.exp(-0.5*diff);
                    }
                }
                matrix[v2][v1] = matrix[v1][v2];
            }
            System.out.println("..correlations");
            
            // build Laplacian
            double[] degree = new double[vol];
            for (int v1=0;v1<vol;v1++) {
                degree[v1] = 0.0;
                for (int v2=0;v2<vol;v2++) {
                    degree[v1] += matrix[v1][v2];
                }
            }
            for (int v1=0;v1<vol;v1++) {
                //matrix[v1][v1] = degree[v1]/weight[v1];
                //matrix[v1][v1] = 1.0/weight[v1];
                matrix[v1][v1] = 1.0;
                //matrix[v1][v1] = weight[v1];
            }
            for (int v1=0;v1<vol;v1++) for (int v2=v1+1;v2<vol;v2++) {
                //matrix[v1][v2] = -matrix[v1][v2]/weight[v1];
                //matrix[v2][v1] = -matrix[v2][v1]/weight[v2];
                //matrix[v1][v2] = -matrix[v1][v2]/(degree[v1]*weight[v1]);
                //matrix[v2][v1] = -matrix[v2][v1]/(degree[v2]*weight[v2]);
                matrix[v1][v2] = -matrix[v1][v2]/degree[v1];
                matrix[v2][v1] = -matrix[v2][v1]/degree[v2];
                //matrix[v1][v2] = -matrix[v1][v2]*weight[v1]/degree[v1];
                //matrix[v2][v1] = -matrix[v2][v1]*weight[v2]/degree[v2];
            }
            System.out.println("..Laplacian");
            
            // SVD? no, eigendecomposition (squared matrix)
            /*
            // Jama version: very slow for vol>>1
            Matrix M = new Matrix(matrix);
            SingularValueDecomposition svd = M.svd();
            //Matrix U = svd.getU();
            */
            
            RealMatrix mtx = null;
            if (sparse) {
                // Sparse matrix version from apache math? not very fast either
                mtx = new OpenMapRealMatrix(vol,vol);
                for (int v1=0;v1<vol;v1++) for (int v2=0;v2<vol;v2++) if (matrix[v1][v2]!=0) {
                    mtx.setEntry(v1,v2, matrix[v1][v2]);
                }
            } else {
                mtx = new Array2DRowRealMatrix(matrix);
            }
            EigenDecomposition eig = new EigenDecomposition(mtx);
            
            /*
            // jblas version?
            DoubleMatrix mtx = new DoubleMatrix(matrix);
            DoubleMatrix[] eig = Eigen.symmetricEigenvectors(mtx);
            */
            System.out.println("first four eigen values:");
            double[] eigval = new double[4];
            for (int s=0;s<4;s++) {
                eigval[s] = eig.getRealEigenvalues()[s];
                //eigval[s] = eig[1].get(s,s);
                System.out.print(eigval[s]+", ");
            }
            // tiled results: we should interpolate instead...
            // from mean coord to neighbors
            v=0;
            for (int x=0;x<nx;x+=sub) for (int y=0;y<ny;y+=sub) for (int z=0;z<nz;z+=sub) {
                int xyz = x+nx*y+nx*ny*z;
                float npt=0.0f;
                for (int dx=0;dx<sub;dx++) for (int dy=0;dy<sub;dy++) for (int dz=0;dz<sub;dz++) {
                    if (x+dx<nx && y+dy<ny && z+dz<nz) {
                        int ngb = xyz+dx+nx*dy+nx*ny*dz;
                        if (labelImage[ngb]==lb) {
                            /*
                            coordImage[xyz+X*nxyz] = (float)svd.getU().get(v,X);
                            coordImage[xyz+Y*nxyz] = (float)svd.getU().get(v,Y);
                            coordImage[xyz+Z*nxyz] = (float)svd.getU().get(v,Z);
                            coordImage[xyz+T*nxyz] = (float)svd.getU().get(v,T);
                            */
                            coordImage[ngb+X*nxyz] = (float)eig.getV().getEntry(v,X);
                            coordImage[ngb+Y*nxyz] = (float)eig.getV().getEntry(v,Y);
                            coordImage[ngb+Z*nxyz] = (float)eig.getV().getEntry(v,Z);
                            coordImage[ngb+T*nxyz] = (float)eig.getV().getEntry(v,T);
                            /*
                            coordImage[xyz+X*nxyz] = (float)eig[0].get(v,X);
                            coordImage[xyz+Y*nxyz] = (float)eig[0].get(v,Y);
                            coordImage[xyz+Z*nxyz] = (float)eig[0].get(v,Z);
                            coordImage[xyz+T*nxyz] = (float)eig[0].get(v,T);
                            */
                            npt++;
                        }
                    }
                }
                if (npt>0) v++;
            }
            // flip eigenvectors to common orientation if desired
            if (refAxis!="none") {
                float[] sign = new float[4];
                for (int x=0;x<nx;x+=sub) for (int y=0;y<ny;y+=sub) for (int z=0;z<nz;z+=sub) {
                    int xyz = x+nx*y+nx*ny*z;
                    if (labelImage[xyz]==lb) {
                        if (refAxis=="X") {
                            sign[1] += x*coordImage[xyz+Y*nxyz];
                            sign[2] += x*coordImage[xyz+Z*nxyz];
                            sign[3] += x*coordImage[xyz+T*nxyz];
                        } else if (refAxis=="Y") {
                            sign[1] += y*coordImage[xyz+Y*nxyz];
                            sign[2] += y*coordImage[xyz+Z*nxyz];
                            sign[3] += y*coordImage[xyz+T*nxyz];
                        } else if (refAxis=="Z") {
                            sign[1] += z*coordImage[xyz+Y*nxyz];
                            sign[2] += z*coordImage[xyz+Z*nxyz];
                            sign[3] += z*coordImage[xyz+T*nxyz];
                        }
                    }
                }
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                    int xyz = x+nx*y+nx*ny*z;
                    if (labelImage[xyz]==lb) {
                        if (sign[1]<0) coordImage[xyz+Y*nxyz] = -coordImage[xyz+Y*nxyz];
                        if (sign[2]<0) coordImage[xyz+Z*nxyz] = -coordImage[xyz+Z*nxyz];
                        if (sign[3]<0) coordImage[xyz+T*nxyz] = -coordImage[xyz+T*nxyz];
                    }
                }
            }
            
		}
		return;
	}

    private final void jointShapeEmbedding() { 
	    
    }
}