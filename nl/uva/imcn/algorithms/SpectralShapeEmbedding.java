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
	private int nlb=0;
	private int[] lbl = null;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;

	private String systemParam = "single";
	private static final String[] systemTypes = {"single","joint"}; 
	private float spaceDev = 10.0f;
	private boolean sparse=true;
	private int msize = 800;
	private static final String[] refAxTypes = {"none","X","Y","Z"};
	private String refAxis = "none";
	private boolean eigenGame = true;
	private double alpha = 1e-2;
	private double error = 1e-2;
	
	private float[] coordImage;
	private float[] flatmapImage;
	
	// numerical quantities
	private static final	float	INVSQRT2 = (float)(1.0/FastMath.sqrt(2.0));
	private static final	float	INVSQRT3 = (float)(1.0/FastMath.sqrt(3.0));
	private static final	float	SQRT2 = (float)FastMath.sqrt(2.0);
	private static final	float	SQRT3 = (float)FastMath.sqrt(3.0);

	// direction labeling		
	public	static	byte	X = 0;
	public	static	byte	Y = 1;
	public	static	byte	Z = 2;
	public	static	byte	T = 3;
	
	// for debug and display
	private static final boolean		debug=true;
	private static final boolean		verbose=true;

	// create inputs
	public final void setLabelImage(int[] val) { labelImage = val; }
	
	// if precomputed:
	public final void setCoordinateImage(float[] val) { coordImage = val; }
	
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
	
	public final void setEigenGame(boolean val, double a, double e) { eigenGame=val; alpha=a; error=e; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
				
	// create outputs
	public final float[] getCoordinateImage() { return coordImage; }
	public final float[] getFlatMapImage() { return flatmapImage; }
	public final int getLabelNumber() { return nlb; }

	public void execute(){
	    singleShapeRecursiveEmbedding();
        //singleShapeEmbedding();
	}
	
	public final void singleShapeEmbedding() { 
	    
	    // 1. build label list
	    nlb = ObjectLabeling.countLabels(labelImage, nx, ny, nz);
	    lbl = ObjectLabeling.listOrderedLabels(labelImage, nx, ny, nz);
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
	        /*
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
	        */
	        // explict target instead?
	        int sub = Numerics.max(1, Numerics.round(FastMath.cbrt(vol/msize)));
	        
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
            //float[] weight = new float[vol];
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
                    //weight[v] = npt/(sub*sub*sub);
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
                        /*
                        double diff = 0.0;
                        for (int c=0;c<nc;c++) {
                            diff += (contrasts[c][v1]-contrasts[c][v2])
                                   *(contrasts[c][v1]-contrasts[c][v2])/contrastDev[c];
                        }
                        matrix[v1][v2] *= FastMath.exp(-0.5*diff);
                        */
                        /*
                        boolean boundary = false;
                        for (int c=0;c<nc;c++) {
                            double diff = (contrasts[c][v1]-contrasts[c][v2])
                                         *(contrasts[c][v1]-contrasts[c][v2])/contrastDev[c];
                            if (diff>1) boundary=true;
                        }                        
                        if (boundary) matrix[v1][v2] *= 0.5;
                        */
                        /*
                        double diff = 0.0;
                        for (int c=0;c<nc;c++) {
                            diff += (contrasts[c][v1]-contrasts[c][v2])
                                   *(contrasts[c][v1]-contrasts[c][v2])/contrastDev[c];
                        }
                        diff /= nc;
                        matrix[v1][v2] = 1.0/(0.1+diff);
                        */
                        double avg=0.0;
                        for (int c=0;c<nc;c++) {
                            avg += (contrasts[c][v1]+contrasts[c][v2])/contrastDev[c];
                        }
                        avg /= 2.0*nc;
                        matrix[v1][v2] = 1.0/Numerics.max(0.01,avg);
                         
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
            // refine the result with eigenGame?
            if (eigenGame && sub>1) {
                vol=0;
                for (int xyz=0;xyz<nxyz;xyz++) if (labelImage[xyz]==lb) vol++;
                System.out.println("Eigen game base volume: "+vol);
                
                // build coordinate and contrast arrays
                coord = new float[3][vol];
                contrasts = null;
                if (nc>0) contrasts = new float[nc][vol];
                v=0;
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                    int xyz = x+nx*y+nx*ny*z;
                    if (labelImage[xyz]==lb) {
                        coord[X][v] = x;
                        coord[Y][v] = y;
                        coord[Z][v] = z;
                        if (nc>0) {
                            for (int c=0;c<nc;c++) {
                                contrasts[c][v] = contrastImages[c][xyz];
                            }
                        }
                        v++;
                    }
                }       
                System.out.println("..contrasts");
                        
                // build correlation matrix
                int nmtx = 0;
                for (int v1=0;v1<vol;v1++) for (int v2=v1+1;v2<vol;v2++) {
                    double dist = (coord[X][v1]-coord[X][v2])*(coord[X][v1]-coord[X][v2])
                                 +(coord[Y][v1]-coord[Y][v2])*(coord[Y][v1]-coord[Y][v2])
                                 +(coord[Z][v1]-coord[Z][v2])*(coord[Z][v1]-coord[Z][v2]);
                    // when computing a sparse version, only keep strict 6-C neighbors             
                    if (sparse && dist<=1.0) {
                        nmtx++;
                    }
                }
                System.out.println("non-zero components: "+nmtx);
                
                double[] mtxval = new double[nmtx];
                int[] mtxid1 = new int[nmtx];
                int[] mtxid2 = new int[nmtx];
                int[][] mtxinv = new int[6][vol];
                
                int id=0;
                for (int v1=0;v1<vol;v1++) for (int v2=v1+1;v2<vol;v2++) {
                    double dist = (coord[X][v1]-coord[X][v2])*(coord[X][v1]-coord[X][v2])
                                 +(coord[Y][v1]-coord[Y][v2])*(coord[Y][v1]-coord[Y][v2])
                                 +(coord[Z][v1]-coord[Z][v2])*(coord[Z][v1]-coord[Z][v2]);
                    // when computing a sparse version, only keep strict 6-C neighbors             
                    if (sparse && dist<=1.0) {
                        double coeff = 1.0/FastMath.sqrt(dist);
                        if (nc>0) {
                            /*
                            double diff = 0.0;
                            for (int c=0;c<nc;c++) {
                                diff += (contrasts[c][v1]-contrasts[c][v2])
                                       *(contrasts[c][v1]-contrasts[c][v2])/contrastDev[c];
                            }
                            coeff *= FastMath.exp(-0.5*diff);
                            */
                            /*
                            boolean boundary = false;
                            for (int c=0;c<nc;c++) {
                                double diff = (contrasts[c][v1]-contrasts[c][v2])
                                       *(contrasts[c][v1]-contrasts[c][v2])/contrastDev[c];
                                if (diff>1) boundary=true;
                            }
                            if (boundary) coeff *= 0.5;
                            */
                            /*
                            double diff = 0.0;
                            for (int c=0;c<nc;c++) {
                                diff += (contrasts[c][v1]-contrasts[c][v2])
                                       *(contrasts[c][v1]-contrasts[c][v2])/contrastDev[c];
                            }
                            diff /= nc;
                            coeff = 1.0/(0.1+diff);
                            */
                            double avg=0.0;
                            for (int c=0;c<nc;c++) {
                                avg += (contrasts[c][v1]+contrasts[c][v2])/contrastDev[c];
                            }
                            avg /= 2.0*nc;
                            coeff = 1.0/Numerics.max(0.01,avg);
                        }
                        mtxval[id] = coeff;
                        mtxid1[id] = v1;
                        mtxid2[id] = v2; 
                        for (int c=0;c<6;c++) if (mtxinv[c][v1]==0) {
                            mtxinv[c][v1] = id+1;
                            c=6;
                        }
                        for (int c=0;c<6;c++) if (mtxinv[c][v2]==0) {
                            mtxinv[c][v2] = id+1;
                            c=6;
                        }
                        id++;
                    }
                }
                coord = null;
                contrasts = null;
                
                System.out.println("..correlations");
                
                // get initial vector guesses from subsampled data
                double[][] init = new double[4][vol];
                v=0;
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                    int xyz = x+nx*y+nx*ny*z;
                    if (labelImage[xyz]==lb) {
                        init[0][v] = coordImage[xyz+X*nxyz];
                        init[1][v] = coordImage[xyz+Y*nxyz];
                        init[2][v] = coordImage[xyz+Z*nxyz];
                        init[3][v] = coordImage[xyz+T*nxyz];
                        v++;
                    }
                }
                
                runSparseLaplacianEigenGame(mtxval, mtxid1, mtxid2, mtxinv, nmtx, vol, 4, init, alpha, error);
                              
                v=0;
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                    int xyz = x+nx*y+nx*ny*z;
                    if (labelImage[xyz]==lb) {
                        coordImage[xyz+Y*nxyz] = (float)init[1][v];
                        coordImage[xyz+Z*nxyz] = (float)init[2][v];
                        coordImage[xyz+T*nxyz] = (float)init[3][v];
                        v++;
                    }
                }
                
            }
            
            // flip eigenvectors to common orientation if desired
            if (!refAxis.equals("none")) {
                System.out.println("Orient to axis: "+refAxis);
                float x0 = 0.0f;
                float y0 = 0.0f;
                float z0 = 0.0f;
                int npt = 0;
                float[][] sign = new float[4][4];
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                    int xyz = x+nx*y+nx*ny*z;
                    if (labelImage[xyz]==lb) {
                        x0 += x;
                        y0 += y;
                        z0 += z;
                        npt++;
                    }
                }
                if (npt>0) {
                    x0 /= npt;
                    y0 /= npt;
                    z0 /= npt;
                }
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                    int xyz = x+nx*y+nx*ny*z;
                    if (labelImage[xyz]==lb) {
                        sign[1][X] += (x-x0)*coordImage[xyz+Y*nxyz];
                        sign[2][X] += (x-x0)*coordImage[xyz+Z*nxyz];
                        sign[3][X] += (x-x0)*coordImage[xyz+T*nxyz];

                        sign[1][Y] += (y-y0)*coordImage[xyz+Y*nxyz];
                        sign[2][Y] += (y-y0)*coordImage[xyz+Z*nxyz];
                        sign[3][Y] += (y-y0)*coordImage[xyz+T*nxyz];

                        sign[1][Z] += (z-z0)*coordImage[xyz+Y*nxyz];
                        sign[2][Z] += (z-z0)*coordImage[xyz+Z*nxyz];
                        sign[3][Z] += (z-z0)*coordImage[xyz+T*nxyz];
                    }
                }
                // use the strongest variation
                for (int s=1;s<4;s++) {
                    float center = x0-nx/2.0f;
                    float max = sign[s][X];
                    if (sign[s][Y]*sign[s][Y]>max*max) {
                        max = sign[s][Y];
                        center = y0-ny/2.0f;
                    }
                    if (sign[s][Z]*sign[s][Z]>max*max) {
                        max = sign[s][Z];
                        center = z0-nz/2.0f;
                    }
                    // dissimilar: negative sign
                    if (center<0 && max>0) sign[s][T] = -max;
                    if (center>0 && max<0) sign[s][T] = max;
                    // similar: positive sign
                    if (center<0 && max<0) sign[s][T] = -max;
                    if (center>0 && max>0) sign[s][T] = max;
                }
                System.out.println("Label "+n+" switching: "+sign[1][T]+", "+sign[2][T]+", "+sign[3][T]);
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                    int xyz = x+nx*y+nx*ny*z;
                    if (labelImage[xyz]==lb) {
                        if (sign[1][T]<0) coordImage[xyz+Y*nxyz] = -coordImage[xyz+Y*nxyz];
                        if (sign[2][T]<0) coordImage[xyz+Z*nxyz] = -coordImage[xyz+Z*nxyz];
                        if (sign[3][T]<0) coordImage[xyz+T*nxyz] = -coordImage[xyz+T*nxyz];
                    }
                }
            }
            
		}
		return;
	}

	public final void singleShapeRecursiveEmbedding() { 
	    
	    // 1. build label list
	    nlb = ObjectLabeling.countLabels(labelImage, nx, ny, nz);
	    lbl = ObjectLabeling.listOrderedLabels(labelImage, nx, ny, nz);
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
            
	        // if volume is too big, subsample to match explict target size
	        int sub = Numerics.max(1, Numerics.round(FastMath.cbrt(vol/msize)));
	        
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
            System.out.println("starting volume: "+vol+" ("+sub+")");
            
            // build coordinate and contrast arrays
            float[][] coord = new float[3][vol];
            float[][] contrasts = null;
            if (nc>0) contrasts = new float[nc][vol];
            //float[] weight = new float[vol];
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
                    //weight[v] = npt/(sub*sub*sub);
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
    
            // refine the result with eigenGame?
            //double vol0 = vol; 
            double vol0 = msize; 
            if (eigenGame) while (sub>1) {
                // downscale by 1
                sub--;
                
                // recompute volumes
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
                System.out.println("Eigen game volume: "+vol+" (ratio: "+vol/vol0+")");
                
                // build coordinate and contrast arrays
                coord = new float[3][vol];
                contrasts = null;
                if (nc>0) contrasts = new float[nc][vol];
                //weight = new float[vol];
                v=0;
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
                        //weight[v] = npt/(sub*sub*sub);
                        v++;
                    }
                }       
                System.out.println("..contrasts");
                        
                // build correlation matrix
                int nmtx = 0;
                for (int v1=0;v1<vol;v1++) for (int v2=v1+1;v2<vol;v2++) {
                    double dist = (coord[X][v1]-coord[X][v2])*(coord[X][v1]-coord[X][v2])
                                 +(coord[Y][v1]-coord[Y][v2])*(coord[Y][v1]-coord[Y][v2])
                                 +(coord[Z][v1]-coord[Z][v2])*(coord[Z][v1]-coord[Z][v2]);
                    // when computing a sparse version, only keep strict 6-C neighbors             
                    if (sparse && dist<=sub*sub) {
                        nmtx++;
                    }
                }
                System.out.println("non-zero components: "+nmtx);
                
                double[] mtxval = new double[nmtx];
                int[] mtxid1 = new int[nmtx];
                int[] mtxid2 = new int[nmtx];
                int[][] mtxinv = new int[6][vol];
                
                int id=0;
                for (int v1=0;v1<vol;v1++) for (int v2=v1+1;v2<vol;v2++) {
                    double dist = (coord[X][v1]-coord[X][v2])*(coord[X][v1]-coord[X][v2])
                                 +(coord[Y][v1]-coord[Y][v2])*(coord[Y][v1]-coord[Y][v2])
                                 +(coord[Z][v1]-coord[Z][v2])*(coord[Z][v1]-coord[Z][v2]);
                    // when computing a sparse version, only keep strict 6-C neighbors             
                    if (sparse && dist<=sub*sub) {
                        double coeff = 1.0/FastMath.sqrt(dist);
                        if (nc>0) {
                            double diff = 0.0;
                            for (int c=0;c<nc;c++) {
                                diff += (contrasts[c][v1]-contrasts[c][v2])
                                       *(contrasts[c][v1]-contrasts[c][v2])/contrastDev[c];
                            }
                            coeff *= FastMath.exp(-0.5*diff);
                        }
                        mtxval[id] = coeff;
                        mtxid1[id] = v1;
                        mtxid2[id] = v2; 
                        for (int c=0;c<6;c++) if (mtxinv[c][v1]==0) {
                            mtxinv[c][v1] = id+1;
                            c=6;
                        }
                        for (int c=0;c<6;c++) if (mtxinv[c][v2]==0) {
                            mtxinv[c][v2] = id+1;
                            c=6;
                        }
                        id++;
                    }
                }
                coord = null;
                contrasts = null;
                
                System.out.println("..correlations");
                
                // get initial vector guesses from subsampled data
                double[][] init = new double[4][vol];
                double[] norm = new double[4];
                v=0;
                for (int x=0;x<nx;x+=sub) for (int y=0;y<ny;y+=sub) for (int z=0;z<nz;z+=sub) {
                    int xyz = x+nx*y+nx*ny*z;
                    if (labelImage[xyz]==lb) {
                        init[0][v] = coordImage[xyz+X*nxyz];
                        init[1][v] = coordImage[xyz+Y*nxyz];
                        init[2][v] = coordImage[xyz+Z*nxyz];
                        init[3][v] = coordImage[xyz+T*nxyz];
                        
                        norm[0] += init[0][v]*init[0][v];
                        norm[1] += init[1][v]*init[1][v];
                        norm[2] += init[2][v]*init[2][v];
                        norm[3] += init[3][v]*init[3][v];
                        v++;
                    }
                }
                // rescale to ||V||=1
                for (int i=0;i<4;i++) {
                    norm[i] = FastMath.sqrt(norm[i]);
                    for (int vi=0;vi<vol;vi++) {
                        init[i][vi] /= norm[i];
                    }
                }
                
                //runSparseLaplacianEigenGame(mtxval, mtxid1, mtxid2, mtxinv, nmtx, vol, 4, init, alpha, error*FastMath.sqrt(vol/vol0));
                runSparseLaplacianEigenGame(mtxval, mtxid1, mtxid2, mtxinv, nmtx, vol, 4, init, alpha*FastMath.pow(vol/vol0,0.25), error*FastMath.pow(vol/vol0,0.25));
                //runSparseLaplacianEigenGame(mtxval, mtxid1, mtxid2, mtxinv, nmtx, vol, 4, init, alpha, error);

                // update the error parameter to reduce the number of steps? no keep to baseline
                // because the error is not updated
                //vol0 = vol;
                
                mtxval = null;
                mtxid1 = null;
                mtxid2 = null;
                mtxinv = null;
                              
                v=0;
                for (int x=0;x<nx;x+=sub) for (int y=0;y<ny;y+=sub) for (int z=0;z<nz;z+=sub) {
                    int xyz = x+nx*y+nx*ny*z;
                    float npt=0.0f;
                    for (int dx=0;dx<sub;dx++) for (int dy=0;dy<sub;dy++) for (int dz=0;dz<sub;dz++) {
                        if (x+dx<nx && y+dy<ny && z+dz<nz) {
                            int ngb = xyz+dx+nx*dy+nx*ny*dz;
                            if (labelImage[ngb]==lb) {
                                coordImage[ngb+X*nxyz] = (float)init[0][v];
                                coordImage[ngb+Y*nxyz] = (float)init[1][v];
                                coordImage[ngb+Z*nxyz] = (float)init[2][v];
                                coordImage[ngb+T*nxyz] = (float)init[3][v];
                                npt++;
                            }
                        }
                    }
                    if (npt>0) v++;
                }
                init = null;
                
            }
            
            // flip eigenvectors to common orientation if desired
            if (!refAxis.equals("none")) {
                System.out.println("Orient to axis: "+refAxis);
                float x0 = 0.0f;
                float y0 = 0.0f;
                float z0 = 0.0f;
                int npt = 0;
                float[][] sign = new float[4][4];
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                    int xyz = x+nx*y+nx*ny*z;
                    if (labelImage[xyz]==lb) {
                        x0 += x;
                        y0 += y;
                        z0 += z;
                        npt++;
                    }
                }
                if (npt>0) {
                    x0 /= npt;
                    y0 /= npt;
                    z0 /= npt;
                }
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                    int xyz = x+nx*y+nx*ny*z;
                    if (labelImage[xyz]==lb) {
                        sign[1][X] += (x-x0)*coordImage[xyz+Y*nxyz];
                        sign[2][X] += (x-x0)*coordImage[xyz+Z*nxyz];
                        sign[3][X] += (x-x0)*coordImage[xyz+T*nxyz];

                        sign[1][Y] += (y-y0)*coordImage[xyz+Y*nxyz];
                        sign[2][Y] += (y-y0)*coordImage[xyz+Z*nxyz];
                        sign[3][Y] += (y-y0)*coordImage[xyz+T*nxyz];

                        sign[1][Z] += (z-z0)*coordImage[xyz+Y*nxyz];
                        sign[2][Z] += (z-z0)*coordImage[xyz+Z*nxyz];
                        sign[3][Z] += (z-z0)*coordImage[xyz+T*nxyz];
                    }
                }
                // use the strongest variation
                for (int s=1;s<4;s++) {
                    float center = x0-nx/2.0f;
                    float max = sign[s][X];
                    if (sign[s][Y]*sign[s][Y]>max*max) {
                        max = sign[s][Y];
                        center = y0-ny/2.0f;
                    }
                    if (sign[s][Z]*sign[s][Z]>max*max) {
                        max = sign[s][Z];
                        center = z0-nz/2.0f;
                    }
                    // dissimilar: negative sign
                    if (center<0 && max>0) sign[s][T] = -max;
                    if (center>0 && max<0) sign[s][T] = max;
                    // similar: positive sign
                    if (center<0 && max<0) sign[s][T] = -max;
                    if (center>0 && max>0) sign[s][T] = max;
                }
                System.out.println("Label "+n+" switching: "+sign[1][T]+", "+sign[2][T]+", "+sign[3][T]);
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                    int xyz = x+nx*y+nx*ny*z;
                    if (labelImage[xyz]==lb) {
                        if (sign[1][T]<0) coordImage[xyz+Y*nxyz] = -coordImage[xyz+Y*nxyz];
                        if (sign[2][T]<0) coordImage[xyz+Z*nxyz] = -coordImage[xyz+Z*nxyz];
                        if (sign[3][T]<0) coordImage[xyz+T*nxyz] = -coordImage[xyz+T*nxyz];
                    }
                }
            }
            
		}
		return;
	}

    private final void runSparseLaplacianEigenGame(double[] mtval, int[] mtid1, int[] mtid2, int[][] mtinv, int nn0, int nm, int nv, double[][] init, double alpha, double error) {
        //double 		error = 1e-2;	// error tolerance
        //double      alpha = 1e-3;    // step size
        int iter;
        double[][] Mv = new double[nv][nm];
        double[] vMv = new double[nv];
        
        double[][] vect = init;
        
        // here assume the matrix is the upper diagonal of correlation matrix
        
        // build degree first
        double[] deg = new double[nm];
        // M_ii = 0
        for (int n=0;n<nm;n++) {
            deg[n] = 0.0;
        }
        // M_ij and M_ji
        for (int n=0;n<nn0;n++) {
            deg[mtid1[n]] += mtval[n];
            deg[mtid2[n]] += mtval[n];
        }
        
        for (int vi=0;vi<nv;vi++) {
            System.out.println("..eigenvector "+(vi+1));
        
            // compute new vectors based on 
            for (int n=0;n<nm;n++) {
                /* generic formula
                Mv[vi][n] = 0.0;
                for (int m=0;m<nm;m++)
                    Mv[vi][n] += matrix[n][m]*vect[vi][m];
                    */
                // diagonal term is 2-1, as lambda_0<=2 (graph Laplacian property)
                Mv[vi][n] = vect[vi][n];
                // off-diagonals
                for (int c=0;c<6;c++) if (mtinv[c][n]>0) {
                    if (mtid1[mtinv[c][n]-1]==n) {
                        // v1<v2
                        Mv[vi][n] += mtval[mtinv[c][n]-1]/deg[n]*vect[vi][mtid2[mtinv[c][n]-1]];
                    } else if (mtid2[mtinv[c][n]-1]==n) {
                        // v1<v2
                        Mv[vi][n] += mtval[mtinv[c][n]-1]/deg[n]*vect[vi][mtid1[mtinv[c][n]-1]];
                    }  
                }
                /*
                for (int m=0;m<nn0;m++) {
                    if (mtid1[m]==n) {
                        // v1<v2
                        Mv[vi][n] += mtval[m]/deg[n]*vect[vi][mtid2[m]];
                    } else if (mtid2[m]==n) {
                        // v1<v2
                        Mv[vi][n] += mtval[m]/deg[n]*vect[vi][mtid1[m]];
                    }
                }*/
            }
            
            // calculate required number of iterations
            double norm = 0.0;
            for (int n=0;n<nm;n++) norm += Mv[vi][n]*Mv[vi][n];
            System.out.println("norm: "+norm);
            
            double Ti = 5.0/4.0/Numerics.min(norm/4.0, error*error);
            System.out.println("-> "+Ti+" iterations");
            
            // pre-compute previous quantities?
            
            // main loop
            double[] grad = new double[nm];
            for (int t=0;t<Ti;t++) {
//            int t=0;
//            while (t<Ti && Numerics.abs(norm/4.0-1.0)>error*error) {
//                t++;
                //System.out.print(".");
                // pre-compute product
                double[] viMvj = new double[nv];
                for (int vj=0;vj<vi;vj++) {
                    viMvj[vj] = 0.0;
                    for (int m=0;m<nm;m++) viMvj[vj] += Mv[vj][m]*vect[vi][m];
                }
                // gradient computation
                for (int n=0;n<nm;n++) {
                    grad[n] = 2.0*Mv[vi][n];
                    for (int vj=0;vj<vi;vj++) {
                        //double prod = 0.0;
                        //for (int m=0;m<nm;m++) prod += Mv[vj][m]*vect[vi][m];
                        grad[n] -= 2.0*viMvj[vj]/vMv[vj]*Mv[vj][n];
                    }
                }
                // Riemannian projection
                double gradR = 0.0;
                for (int n=0;n<nm;n++)
                    gradR += grad[n]*vect[vi][n];
                
                // update
                norm = 0.0;
                for (int n=0;n<nm;n++) {
                    vect[vi][n] += alpha*(grad[n] - gradR*vect[vi][n]);
                    norm += vect[vi][n]*vect[vi][n];
                }
                norm = FastMath.sqrt(norm);
                
                // renormalize 
                for (int n=0;n<nm;n++) {
                    vect[vi][n] /= norm;
                }
                
                // recompute Mvi
                for (int n=0;n<nm;n++) {
                    /* replace by compressed matrix
                    Mv[vi][n] = 0.0;
                    for (int m=0;m<nm;m++)
                        Mv[vi][n] += matrix[n][m]*vect[vi][m];
                        */
                    // diagonal term is 2-1
                    Mv[vi][n] = vect[vi][n];
                    // off-diagonals
                    for (int c=0;c<6;c++) if (mtinv[c][n]>0) {
                        if (mtid1[mtinv[c][n]-1]==n) {
                            // v1<v2
                            Mv[vi][n] += mtval[mtinv[c][n]-1]/deg[n]*vect[vi][mtid2[mtinv[c][n]-1]];
                        } else if (mtid2[mtinv[c][n]-1]==n) {
                            // v1<v2
                            Mv[vi][n] += mtval[mtinv[c][n]-1]/deg[n]*vect[vi][mtid1[mtinv[c][n]-1]];
                        }  
                    }
                    /*
                    // off-diagonals
                    for (int m=0;m<nn0;m++) {
                        if (mtid1[m]==n) {
                            // v1<v2
                            Mv[vi][n] += mtval[m]/deg[mtid1[m]]*vect[vi][mtid2[m]];
                        } else if (mtid2[m]==n) {
                            // v1<v2
                            Mv[vi][n] += mtval[m]/deg[mtid2[m]]*vect[vi][mtid1[m]];
                        }
                    }*/
                }
    
                // recompute norm to stop earlier if possible?
                norm = 0.0;
                for (int n=0;n<nm;n++) norm += Mv[vi][n]*Mv[vi][n];
            }
            //System.out.println(" ("+t+" needed, norm: "+norm+")");
            System.out.println("norm: "+norm);
            
            // post-process: compute summary quantities for next eigenvector
            vMv[vi] = 0.0;
            for (int n=0;n<nm;n++) vMv[vi] += vect[vi][n]*Mv[vi][n];
        }
        
        // check the result
        System.out.println("final vector orthogonality");
        for (int v1=0;v1<nv-1;v1++) for (int v2=v1+1;v2<nv;v2++) {
            double prod=0.0;
            for (int m=0;m<nm;m++) prod += vect[v1][m]*vect[v2][m];
            System.out.println("v"+v1+" * v"+v2+" = "+prod);
        }
        System.out.println("final vector eigenscore");
        for (int v1=0;v1<nv;v1++) {
            double normvect=0.0;
            double normMv=0.0;
            double prod=0.0;
            for (int m=0;m<nm;m++) {
                normvect += vect[v1][m]*vect[v1][m];
                normMv += Mv[v1][m]*Mv[v1][m];
                prod += vect[v1][m]*Mv[v1][m];
            }
            System.out.println("v"+v1+" . Mv"+v1+" = "+prod/FastMath.sqrt(normvect*normMv)+" (lambda = "+normMv/normvect+")");
        }
    }

    public final void buildSpectralMaps(int size, boolean combined, int offset) { 
	    // 1. build label list
	    nlb = ObjectLabeling.countLabels(labelImage, nx, ny, nz);
	    lbl = ObjectLabeling.listOrderedLabels(labelImage, nx, ny, nz);
	    System.out.println("labels: "+nlb);
	    
	    if (combined) buildJointFlatMap(size);
	    else buildFlatMap(size, offset);
    }
    
    public final void buildSpectralProjectionMaps(int size, boolean combined) { 
	    // 1. build label list
	    nlb = ObjectLabeling.countLabels(labelImage, nx, ny, nz);
	    lbl = ObjectLabeling.listOrderedLabels(labelImage, nx, ny, nz);
	    System.out.println("labels: "+nlb);
	    
	    if (combined) buildJointProjectionFlatMap(size);
	    else buildPcaFlatMap(size);
    }
    
    private final void buildFlatMap(int dim, int offset) {
        // circular offset for different combinations
        if (offset==1) {
            X=0;
            Y=2;
            Z=3;
            T=1;
        } else if (offset==2) {
            X=0;
            Y=3;
            Z=1;
            T=2;
        } else {
            X=0;
            Y=1;
            Z=2;
            T=3;
        }
        
        // map dimensions
        /*
        float[] flatmapCount = new float[dim*dim*(nlb-1)];
        if (nc>0) flatmapImage = new float[dim*dim*(nlb-1)];
        else flatmapImage = flatmapCount;
        */
        flatmapImage = new float[dim*dim*(nlb-1)];
	    for (int n=0;n<nlb;n++) if (n>0) {
	        float minY = 0.0f;
	        float maxY = 0.0f;
	        float minZ = 0.0f;
	        float maxZ = 0.0f;
	        
	        // find min/max coordinates
	        for (int xyz=0;xyz<nxyz;xyz++) if (labelImage[xyz]==lbl[n]) {
	            if (coordImage[xyz+Y*nxyz]<minY) minY = coordImage[xyz+Y*nxyz];
	            if (coordImage[xyz+Y*nxyz]>maxY) maxY = coordImage[xyz+Y*nxyz];
	            if (coordImage[xyz+Z*nxyz]<minZ) minZ = coordImage[xyz+Z*nxyz];
	            if (coordImage[xyz+Z*nxyz]>maxZ) maxZ = coordImage[xyz+Z*nxyz];
	        }
	        float dY = (maxY-minY)/(dim+1.0f);
	        float dZ = (maxZ-minZ)/(dim+1.0f);
	        // for scale-preserving mappings?
	        //float d0 = Numerics.max(dY,dZ);
	        
	        // global representation as a 2D map spaced along a global scale?
	        // -> a single region map? no, too blobby use a two level approach instead
	        // (first, the local maps, then the global tiling / organization)
	        // centroid + distance full matrix -> global embedding
	        
	        //float[] dist = new float[dim*dim];
	        for (int xyz=0;xyz<nxyz;xyz++) if (labelImage[xyz]==lbl[n]) {
	            // find the correct bin
	            int binY = Numerics.floor((coordImage[xyz+Y*nxyz]-minY)/dY);
	            int binZ = Numerics.floor((coordImage[xyz+Z*nxyz]-minZ)/dZ);
	            //int binY = Numerics.floor((coordImage[xyz+Y*nxyz]-minY)/d0);
	            //int binZ = Numerics.floor((coordImage[xyz+Z*nxyz]-minZ)/d0);
	            
	            binY = Numerics.bounded(binY,0,dim-1);
	            binZ = Numerics.bounded(binZ,0,dim-1);
	            //float newdist = Numerics.square((coordImage[xyz+Y*nxyz]-minY)/dY-binY)
	            //               +Numerics.square((coordImage[xyz+Z*nxyz]-minZ)/dZ-binZ);
	            int idmap = binY+dim*binZ+dim*dim*(n-1);
	            /*
	            // simply count the number of voxels represented
	            flatmapCount[idmap]++;
	            // average contrast
	            if (nc>0) flatmapImage[idmap] += contrastImages[0][xyz];
	            */
	            if (nc>0) flatmapImage[idmap] = Numerics.max(flatmapImage[idmap], contrastImages[0][xyz]);
	            else flatmapImage[idmap]++;
	        }
	    }
	    /*
	    if (nc>0) {
	        for (int i=0;i<dim*dim*(nlb-1);i++) {
	            if (flatmapCount[i]>0) flatmapImage[i] /= flatmapCount[i];
	        }
	    }
	    */
	    return;
    }

    private final void buildJointFlatMap(int dim) {
        
        System.out.println("Joint flat maps");
        
        // first define the global Laplacian embedding
        float[][] centroids = new float[nlb-1][3];
        for (int n=0;n<nlb;n++) if (n>0) {
            int npt=0;
            for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (labelImage[xyz]==lbl[n]) {    
                    centroids[n-1][X] += x;
                    centroids[n-1][Y] += y;
                    centroids[n-1][Z] += z;
                    npt++;
                }
            }
            if (npt>0) {
                centroids[n-1][X] /= npt;
                centroids[n-1][Y] /= npt;
                centroids[n-1][Z] /= npt;
            }
        }
        System.out.println("..centroids");
        
        double maxdist = 50.0;
        double[][] matrix = new double[nlb-1][nlb-1];
        for (int l1=1;l1<nlb;l1++) for (int l2=l1+1;l2<nlb;l2++) {
            double dist = (centroids[l1-1][X]-centroids[l2-1][X])*(centroids[l1-1][X]-centroids[l2-1][X])
                         +(centroids[l1-1][Y]-centroids[l2-1][Y])*(centroids[l1-1][Y]-centroids[l2-1][Y])
                         +(centroids[l1-1][Z]-centroids[l2-1][Z])*(centroids[l1-1][Z]-centroids[l2-1][Z]);
            // when computing a sparse version, only keep strict 26-C neighbors             
            if (sparse && dist>3.0*maxdist*maxdist) {
                matrix[l1-1][l2-1] = 0.0;
            } else {
                if (sparse) matrix[l1-1][l2-1] = 1.0/FastMath.sqrt(dist);
                else matrix[l1-1][l2-1] = FastMath.exp(-0.5*dist/maxdist);                
            }
            matrix[l2-1][l1-1] = matrix[l1-1][l2-1];
        }
        // rescale to have values close to 1? or not needed?    
        System.out.println("..correlations");
         
        // build Laplacian
        double[] degree = new double[nlb-1];
        for (int l1=0;l1<nlb-1;l1++) {
            degree[l1] = 0.0;
            for (int l2=0;l2<nlb-1;l2++) {
                degree[l1] += matrix[l1][l2];
            }
        }
        for (int l1=0;l1<nlb-1;l1++) {
            matrix[l1][l1] = 1.0;
        }
        for (int l1=0;l1<nlb-1;l1++) for (int l2=l1+1;l2<nlb-1;l2++) {
            matrix[l1][l2] = -matrix[l1][l2]/degree[l1];
            matrix[l2][l1] = -matrix[l2][l1]/degree[l2];
        }
        System.out.println("..Laplacian");
        RealMatrix mtx = null;
        mtx = new Array2DRowRealMatrix(matrix);
        EigenDecomposition eig = new EigenDecomposition(mtx);
        
        System.out.println("first four eigen values:");
        double[] eigval = new double[4];
        for (int s=0;s<4;s++) {
            eigval[s] = eig.getRealEigenvalues()[s];
            System.out.print(eigval[s]+", ");
        }
        // each centroid location maps to a 2D location
        double[] frameY = new double[nlb-1];
        double[] frameZ = new double[nlb-1];
        double fminY = 0.0;
        double fmaxY = 0.0;
        double fminZ = 0.0;
        double fmaxZ = 0.0;
        
        for (int l=0;l<nlb-1;l++) {
            frameY[l] = eig.getV().getEntry(l,Y);
            frameZ[l] = eig.getV().getEntry(l,Z);
            if (frameY[l]<fminY) fminY = frameY[l];
            if (frameY[l]>fmaxY) fmaxY = frameY[l];
            if (frameZ[l]<fminZ) fminZ = frameZ[l];
            if (frameZ[l]>fmaxZ) fmaxZ = frameZ[l];
        }
        double dfY = (fmaxY-fminY)/(dim+1.0);
        double dfZ = (fmaxZ-fminZ)/(dim+1.0);
	    double df0 = Numerics.max(dfY,dfZ);
	        
        // find common scaling dimension
        float d0 = 0.0f;
	    for (int n=0;n<nlb;n++) if (n>0) {
	        // main location on map
	        int fY = Numerics.floor((frameY[n-1]-fminY)/df0);
	        int fZ = Numerics.floor((frameZ[n-1]-fminZ)/df0);
	        
	        float minY = 0.0f;
	        float maxY = 0.0f;
	        float minZ = 0.0f;
	        float maxZ = 0.0f;
	        
	        // find min/max coordinates
	        for (int xyz=0;xyz<nxyz;xyz++) if (labelImage[xyz]==lbl[n]) {
	            if (coordImage[xyz+Y*nxyz]<minY) minY = coordImage[xyz+Y*nxyz];
	            if (coordImage[xyz+Y*nxyz]>maxY) maxY = coordImage[xyz+Y*nxyz];
	            if (coordImage[xyz+Z*nxyz]<minZ) minZ = coordImage[xyz+Z*nxyz];
	            if (coordImage[xyz+Z*nxyz]>maxZ) maxZ = coordImage[xyz+Z*nxyz];
	        }
	        float dY = (float)(2.0*FastMath.sqrt(nlb-1)*(maxY-minY)/(dim+1.0f));
	        float dZ = (float)(2.0*FastMath.sqrt(nlb-1)*(maxZ-minZ)/(dim+1.0f));
	        // for scale-preserving mappings? how about a global scale? or scaling by volume?
	        d0 = Numerics.max(d0,dY,dZ);
	        
	        // global representation as a 2D map spaced along a global scale?
	        // -> a single region map? no, too blobby use a two level approach instead
	        // (first, the local maps, then the global tiling / organization)
	        // centroid + distance full matrix -> global embedding
	    }
	    
        // map dimensions
        flatmapImage = new float[dim*dim];
	    for (int n=0;n<nlb;n++) if (n>0) {
	        // main location on map
	        int fY = Numerics.floor((frameY[n-1]-fminY)/df0);
	        int fZ = Numerics.floor((frameZ[n-1]-fminZ)/df0);
	        
	        float minY = 0.0f;
	        float maxY = 0.0f;
	        float minZ = 0.0f;
	        float maxZ = 0.0f;
	        
	        // find min/max coordinates
	        for (int xyz=0;xyz<nxyz;xyz++) if (labelImage[xyz]==lbl[n]) {
	            if (coordImage[xyz+Y*nxyz]<minY) minY = coordImage[xyz+Y*nxyz];
	            if (coordImage[xyz+Y*nxyz]>maxY) maxY = coordImage[xyz+Y*nxyz];
	            if (coordImage[xyz+Z*nxyz]<minZ) minZ = coordImage[xyz+Z*nxyz];
	            if (coordImage[xyz+Z*nxyz]>maxZ) maxZ = coordImage[xyz+Z*nxyz];
	        }
	        float dY = (float)(2.0*FastMath.sqrt(nlb-1)*(maxY-minY)/(dim+1.0f));
	        float dZ = (float)(2.0*FastMath.sqrt(nlb-1)*(maxZ-minZ)/(dim+1.0f));
	        // for scale-preserving mappings? how about a global scale? or scaling by volume?
	        d0 = Numerics.min(dY,dZ);
	        
	        for (int xyz=0;xyz<nxyz;xyz++) if (labelImage[xyz]==lbl[n]) {
	            // find the correct bin
	            int binY = Numerics.floor((coordImage[xyz+Y*nxyz]-minY)/d0);
	            int binZ = Numerics.floor((coordImage[xyz+Z*nxyz]-minZ)/d0);
	            
	            binY = Numerics.bounded(fY+binY,0,dim-1);
	            binZ = Numerics.bounded(fZ+binZ,0,dim-1);
	            
	            int idmap = binY+dim*binZ;
	            
	            // simply count the number of voxels represented
	            flatmapImage[idmap]=lbl[n];
	        }
	    }
	    return;
    }

    private final void buildPcaFlatMap(int dim) {
        // map dimensions
        flatmapImage = new float[dim*dim*(nlb-1)];
        
	    for (int n=0;n<nlb;n++) if (n>0) {
	        // build orientation PCA weighted by first component
	        double[] center = new double[4];
	        
	        // first the center: give more weight to regions near zero
	        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (labelImage[xyz]==lbl[n]) {
                    double weight = 1.0/Numerics.max(1e-6,Numerics.abs(coordImage[xyz+Y*nxyz]));
                    center[X] += weight*x;
                    center[Y] += weight*y;
                    center[Z] += weight*z;
                    center[T] += weight;
                }
            }
            center[X] /= center[T];
            center[Y] /= center[T];
            center[Z] /= center[T];
            
            // second the orientation matrix giving more wieght to regions far from zero
            double[][] orient = new double[3][3];
            double sumweight = 0.0;
            
	        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (labelImage[xyz]==lbl[n]) {
                    double weight = Numerics.max(1e-6,Numerics.abs(coordImage[xyz+Y*nxyz]));
                    orient[X][X] += weight*(x-center[X])*(x-center[X]);
                    orient[Y][Y] += weight*(y-center[Y])*(y-center[Y]);
                    orient[Z][Z] += weight*(z-center[Z])*(z-center[Z]);
                    orient[X][Y] += weight*(x-center[X])*(y-center[Y]);
                    orient[Y][Z] += weight*(y-center[Y])*(z-center[Z]);
                    orient[Z][X] += weight*(z-center[Z])*(x-center[X]);
                    sumweight += weight;
                }
            }
            orient[Y][X] = orient[X][Y];
            orient[Z][Y] = orient[Y][Z];
            orient[X][Z] = orient[Z][X];
            for (int i=0;i<3;i++) for (int j=0;j<3;j++) orient[i][j] /= sumweight;
            
            // PCA: main 2 directions define the optimal cutting plane to include most of the first gradient variations
            RealMatrix mtx = new Array2DRowRealMatrix(orient);
            EigenDecomposition eig = new EigenDecomposition(mtx);
            System.out.println("eigen values:");
            double[] eigval = new double[3];
            for (int s=0;s<3;s++) {
                eigval[s] = eig.getRealEigenvalues()[s];
                System.out.print(eigval[s]+", ");
            }
            // project coordinates along the largest two PCA components
            double[] dirY = new double[3];
            double[] dirZ = new double[3];
            for (int s=0;s<3;s++) {
                dirY[s] = eig.getV().getEntry(s,Y);
                dirZ[s] = eig.getV().getEntry(s,Z);
            }
	        double minY = 0.0;
	        double maxY = 0.0;
	        double minZ = 0.0;
	        double maxZ = 0.0;
	        
	        // find min/max coordinates
	        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (labelImage[xyz]==lbl[n]) {
                    double coordY = (x-center[X])*dirY[X] + (y-center[Y])*dirY[Y] + (z-center[Z])*dirY[Z];
                    double coordZ = (x-center[X])*dirZ[X] + (y-center[Y])*dirZ[Y] + (z-center[Z])*dirZ[Z];
                                      
                    if (coordY<minY) minY = coordY;
                    if (coordY>maxY) maxY = coordY;
                    if (coordZ<minZ) minZ = coordZ;
                    if (coordZ>maxZ) maxZ = coordZ;
                }
	        }
	        double dY = (maxY-minY)/(dim+1.0f);
	        double dZ = (maxZ-minZ)/(dim+1.0f);
	        // for scale-preserving mappings?
	        double d0 = Numerics.max(dY,dZ);
	        
	        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (labelImage[xyz]==lbl[n]) {
                    double coordY = (x-center[X])*dirY[X] + (y-center[Y])*dirY[Y] + (z-center[Z])*dirY[Z];
                    double coordZ = (x-center[X])*dirZ[X] + (y-center[Y])*dirZ[Y] + (z-center[Z])*dirZ[Z];
                    // find the correct bin
                    int binY = Numerics.floor((coordY-minY)/d0);
                    int binZ = Numerics.floor((coordZ-minZ)/d0);
                    
                    binY = Numerics.bounded(binY,0,dim-1);
                    binZ = Numerics.bounded(binZ,0,dim-1);
                    int idmap = binY+dim*binZ+dim*dim*(n-1);
	            
                    // simply count the number of voxels represented
                    flatmapImage[idmap]++;
                }
	        }
	    }
	    return;
    }

    private final void buildJointProjectionFlatMap(int dim) {
        
        System.out.println("Joint flat maps");
        
        // first define the global Laplacian embedding
        float[][] centroids = new float[nlb-1][3];
        for (int n=0;n<nlb;n++) if (n>0) {
            int npt=0;
            for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (labelImage[xyz]==lbl[n]) {    
                    centroids[n-1][X] += x;
                    centroids[n-1][Y] += y;
                    centroids[n-1][Z] += z;
                    npt++;
                }
            }
            if (npt>0) {
                centroids[n-1][X] /= npt;
                centroids[n-1][Y] /= npt;
                centroids[n-1][Z] /= npt;
            }
        }
        System.out.println("..centroids");
        
        double maxdist = 50.0;
        double[][] matrix = new double[nlb-1][nlb-1];
        for (int l1=1;l1<nlb;l1++) for (int l2=l1+1;l2<nlb;l2++) {
            double dist = (centroids[l1-1][X]-centroids[l2-1][X])*(centroids[l1-1][X]-centroids[l2-1][X])
                         +(centroids[l1-1][Y]-centroids[l2-1][Y])*(centroids[l1-1][Y]-centroids[l2-1][Y])
                         +(centroids[l1-1][Z]-centroids[l2-1][Z])*(centroids[l1-1][Z]-centroids[l2-1][Z]);
            // when computing a sparse version, only keep strict 26-C neighbors             
            if (sparse && dist>3.0*maxdist*maxdist) {
                matrix[l1-1][l2-1] = 0.0;
            } else {
                if (sparse) matrix[l1-1][l2-1] = 1.0/FastMath.sqrt(dist);
                else matrix[l1-1][l2-1] = FastMath.exp(-0.5*dist/maxdist);                
            }
            matrix[l2-1][l1-1] = matrix[l1-1][l2-1];
        }
        // rescale to have values close to 1? or not needed?    
        System.out.println("..correlations");
         
        // build Laplacian
        double[] degree = new double[nlb-1];
        for (int l1=0;l1<nlb-1;l1++) {
            degree[l1] = 0.0;
            for (int l2=0;l2<nlb-1;l2++) {
                degree[l1] += matrix[l1][l2];
            }
        }
        for (int l1=0;l1<nlb-1;l1++) {
            matrix[l1][l1] = 1.0;
        }
        for (int l1=0;l1<nlb-1;l1++) for (int l2=l1+1;l2<nlb-1;l2++) {
            matrix[l1][l2] = -matrix[l1][l2]/degree[l1];
            matrix[l2][l1] = -matrix[l2][l1]/degree[l2];
        }
        System.out.println("..Laplacian");
        RealMatrix mtx = null;
        mtx = new Array2DRowRealMatrix(matrix);
        EigenDecomposition eig = new EigenDecomposition(mtx);
        
        System.out.println("first four eigen values:");
        double[] eigval = new double[4];
        for (int s=0;s<4;s++) {
            eigval[s] = eig.getRealEigenvalues()[s];
            System.out.print(eigval[s]+", ");
        }
        // each centroid location maps to a 2D location
        double[] frameY = new double[nlb-1];
        double[] frameZ = new double[nlb-1];
        double fminY = 0.0;
        double fmaxY = 0.0;
        double fminZ = 0.0;
        double fmaxZ = 0.0;
        
        for (int l=0;l<nlb-1;l++) {
            frameY[l] = eig.getV().getEntry(l,Y);
            frameZ[l] = eig.getV().getEntry(l,Z);
            if (frameY[l]<fminY) fminY = frameY[l];
            if (frameY[l]>fmaxY) fmaxY = frameY[l];
            if (frameZ[l]<fminZ) fminZ = frameZ[l];
            if (frameZ[l]>fmaxZ) fmaxZ = frameZ[l];
        }
        double dfY = (fmaxY-fminY)/(dim+1.0);
        double dfZ = (fmaxZ-fminZ)/(dim+1.0);
	    double df0 = Numerics.max(dfY,dfZ);
	        
        // map dimensions
        flatmapImage = new float[dim*dim];
        double d0=0.0;
	    for (int n=0;n<nlb;n++) if (n>0) {
	        // main location on map
	        int fY = Numerics.floor((frameY[n-1]-fminY)/df0);
	        int fZ = Numerics.floor((frameZ[n-1]-fminZ)/df0);
	        
	        // build orientation PCA weighted by first component
	        double[] center = new double[4];
	        
	        // first the center: give more weight to regions near zero
	        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (labelImage[xyz]==lbl[n]) {
                    double weight = 1.0/Numerics.max(1e-6,Numerics.abs(coordImage[xyz+Y*nxyz]));
                    center[X] += weight*x;
                    center[Y] += weight*y;
                    center[Z] += weight*z;
                    center[T] += weight;
                }
            }
            center[X] /= center[T];
            center[Y] /= center[T];
            center[Z] /= center[T];
            
            // second the orientation matrix giving more wieght to regions far from zero
            double[][] orient = new double[3][3];
            double sumweight = 0.0;
            
	        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (labelImage[xyz]==lbl[n]) {
                    double weight = Numerics.max(1e-6,Numerics.abs(coordImage[xyz+Y*nxyz]));
                    orient[X][X] += weight*(x-center[X])*(x-center[X]);
                    orient[Y][Y] += weight*(y-center[Y])*(y-center[Y]);
                    orient[Z][Z] += weight*(z-center[Z])*(z-center[Z]);
                    orient[X][Y] += weight*(x-center[X])*(y-center[Y]);
                    orient[Y][Z] += weight*(y-center[Y])*(z-center[Z]);
                    orient[Z][X] += weight*(z-center[Z])*(x-center[X]);
                    sumweight += weight;
                }
            }
            orient[Y][X] = orient[X][Y];
            orient[Z][Y] = orient[Y][Z];
            orient[X][Z] = orient[Z][X];
            for (int i=0;i<3;i++) for (int j=0;j<3;j++) orient[i][j] /= sumweight;
            
            // PCA: main 2 directions define the optimal cutting plane to include most of the first gradient variations
            mtx = new Array2DRowRealMatrix(orient);
            eig = new EigenDecomposition(mtx);
            System.out.println("eigen values:");
            eigval = new double[3];
            for (int s=0;s<3;s++) {
                eigval[s] = eig.getRealEigenvalues()[s];
                System.out.print(eigval[s]+", ");
            }
            // project coordinates along the largest two PCA components
            double[] dirY = new double[3];
            double[] dirZ = new double[3];
            for (int s=0;s<3;s++) {
                dirY[s] = eig.getV().getEntry(s,Y);
                dirZ[s] = eig.getV().getEntry(s,Z);
            }
	        double minY = 0.0;
	        double maxY = 0.0;
	        double minZ = 0.0;
	        double maxZ = 0.0;
	        
	        // find min/max coordinates
	        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (labelImage[xyz]==lbl[n]) {
                    double coordY = (x-center[X])*dirY[X] + (y-center[Y])*dirY[Y] + (z-center[Z])*dirY[Z];
                    double coordZ = (x-center[X])*dirZ[X] + (y-center[Y])*dirZ[Y] + (z-center[Z])*dirZ[Z];
                                      
                    if (coordY<minY) minY = coordY;
                    if (coordY>maxY) maxY = coordY;
                    if (coordZ<minZ) minZ = coordZ;
                    if (coordZ>maxZ) maxZ = coordZ;
                }
	        }
	        double dY = (float)(2.0*FastMath.sqrt(nlb-1)*(maxY-minY)/(dim+1.0f));
	        double dZ = (float)(2.0*FastMath.sqrt(nlb-1)*(maxZ-minZ)/(dim+1.0f));
	        // for scale-preserving mappings?
	        d0 = Numerics.max(dY,dZ);
	        
	        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                int xyz = x+nx*y+nx*ny*z;
                if (labelImage[xyz]==lbl[n]) {
                    double coordY = (x-center[X])*dirY[X] + (y-center[Y])*dirY[Y] + (z-center[Z])*dirY[Z];
                    double coordZ = (x-center[X])*dirZ[X] + (y-center[Y])*dirZ[Y] + (z-center[Z])*dirZ[Z];
                    // find the correct bin
                    int binY = Numerics.floor((coordY-minY)/d0);
                    int binZ = Numerics.floor((coordZ-minZ)/d0);
                    
                    binY = Numerics.bounded(fY+binY,0,dim-1);
                    binZ = Numerics.bounded(fZ+binZ,0,dim-1);
                    int idmap = binY+dim*binZ;
	            
                    // simply count the number of voxels represented
                    flatmapImage[idmap]=lbl[n];
                }
	        }
	    }
	    return;
    }


}