package nl.uva.imcn.methods;

import nl.uva.imcn.utilities.*;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.random.UnitSphereRandomVectorGenerator;

/**
 *
 *  This algorithm computes K highest or lowest eigenvectors of a (possibly big) matrix
 *  using the sequential eigengame approach (could be further improved with parallelism)
 *
 *	@version    June 2021
 *	@author     Pierre-Louis Bazin
 *		
 *
 */
 
public class EigenGame {
		
	// numerical quantities
	private static final	double   INF=1e30;
	private static final	double   ZERO=1e-30;
	
	// data buffers
	private 	double[][]			matrix;  			// original data
	private 	double[][]		    vect;				// eigenvectors
	private		static	int			nm,mm;   		        // matrix dimensions
	
	// parameters
	private 	int 		nv;    // number of eigenvectors
	private 	double 		error = 1e-6;	// error tolerance
	private     boolean     largest=true;    // whether to look for largest or lowest eigenvalues
	private     double      alpha = 1e-3;    // step size
			
	// computation variables
	private    double[][]    Mv;
	private    double[]     vMv;
	private    UnitSphereRandomVectorGenerator random;
	
	// for debug
	static final boolean		debug=true;
	static final boolean		verbose=true;
	
	/**
	 *  constructor
	 *	note: all images passed to the algorithm are just linked, not copied
	 */
	public EigenGame(double[][] matrix_, int nv_, boolean largest_) {
		
		matrix = matrix_;
		
		nm = matrix.length;
		mm = matrix[0].length;
		
		nv = nv_;		
		largest = largest_;

		// init all the new arrays
		try {
			vect = new double[nv][nm];
			Mv = new double[nv][nm];
			vMv = new double[nv];
			
			random = new UnitSphereRandomVectorGenerator(nm);
		} catch (OutOfMemoryError e){
			System.out.println(e.getMessage());
			return;
		}
		
		if (nm==mm && largest) {
		    for (int vi=0;vi<nv;vi++) {
                runSquareMatrixEigenGame(vi);
            }
        } else if (nm==mm && !largest) {
            runSquareMatrixEigenGame(0);
            flipSquareMatrixEigenvalues();
		    for (int vi=1;vi<nv;vi++) {
                runSquareMatrixEigenGame(vi);
            }
        }
	}

	/**
	 *  constructor
	 *	note: all images passed to the algorithm are just linked, not copied
	 */
	public EigenGame(double[][] matrix_, int nv_, double[][] init, boolean largest_) {
		
		matrix = matrix_;
		
		nm = matrix.length;
		mm = matrix[0].length;
		
		nv = nv_;		
		largest = largest_;

		// init all the new arrays
		try {
			vect = new double[nv][nm];
			Mv = new double[nv][nm];
			vMv = new double[nv];
			
			random = new UnitSphereRandomVectorGenerator(nm);
		} catch (OutOfMemoryError e){
			System.out.println(e.getMessage());
			return;
		}
		
		if (nm==mm && largest) {
		    for (int vi=0;vi<nv;vi++) {
                runSquareMatrixEigenGame(vi, init[vi]);
            }
        } else if (nm==mm && !largest) {
            flipSquareMatrixEigenvalues();
		    for (int vi=1;vi<nv;vi++) {
                runSquareMatrixEigenGame(vi, init[vi]);
            }
        }
	}

    /** accessor for computed data */ 
    public final double[][] getEigenvectors() { return vect; }
	
    /** accessor for computed data */ 
    public final double[] getEigenvectorAt(int n) { return vect[n]; }
	
    /** 
	 *  compute the eigenvectors for a square matrix with the basic sequential eigengame algorithm
	 */
    final public void runSquareMatrixEigenGame(int vi) {
        
        // random initialization
        double[] init = random.nextVector();
        
        runSquareMatrixEigenGame(vi, init);
    }
        
    final public void runSquareMatrixEigenGame(int vi, double[] init) {
        int iter;
        
        // random initialization
        vect[vi] = init;
        
        for (int n=0;n<nm;n++) {
            Mv[vi][n] = 0.0;
            for (int m=0;m<nm;m++)
                Mv[vi][n] += matrix[n][m]*vect[vi][m];
        }
        
        // calculate required number of iterations
        double norm = 0.0;
        for (int n=0;n<nm;n++) norm += Mv[vi][n]*Mv[vi][n];
        
        int Ti = Numerics.ceil(5.0/4.0/Numerics.min(norm/4.0, error*error));
        System.out.println("Eigenvector "+(vi+1)+": "+Ti+" iterations");
        
        // pre-compute previous quantities?
        
        // main loop
        double[] grad = new double[nm];
        for (int t=0;t<Ti;t++) {
            // gradient computation
            for (int n=0;n<nm;n++) {
                grad[n] = 2.0*Mv[vi][n];
                for (int vj=0;vj<vi;vj++) {
                    double prod = 0.0;
                    for (int m=0;m<nm;m++) prod += Mv[vj][m]*vect[vi][m];
                    grad[n] -= 2.0*prod/vMv[vj]*Mv[vj][n];
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
                Mv[vi][n] = 0.0;
                for (int m=0;m<nm;m++)
                    Mv[vi][n] += matrix[n][m]*vect[vi][m];
            }
        }
        
        // post-process: compute summary quantities for next step
        vMv[vi] = 0.0;
        for (int n=0;n<nm;n++) vMv[vi] += vect[vi][n]*Mv[vi][n];

        // done??
    }

    /** 
	 *  replace the matrix by lambda_1 I - M to get the lowest eigenvectors.
	 */
    final public void flipSquareMatrixEigenvalues() {
        // first eigenvalue: use the trace instead
        double lambda = 0.0;
        for (int n=0;n<nm;n++) {
            lambda += matrix[n][n];
        }
        
        // new matrix
        for (int n=0;n<nm;n++) {
            for (int m=0;m<nm;m++) {
                if (n==m) {
                    matrix[n][m] = lambda - matrix[n][m];
                } else {
                    matrix[n][m] = - matrix[n][m];
                }
            }
        }
        // done??
    }

}
