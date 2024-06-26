package nl.uva.imcn.methods;

import nl.uva.imcn.utilities.*;
import org.apache.commons.math3.util.FastMath;
import Jama.*;

/**
 *
 *  This algorithm performs classical self-organizing maps
 *  based on Kohonen, 2001
 *
 *	@version    Dec 2018
 *	@author     Pierre-Louis Bazin
 *		
 *
 */
 
public class BasicSom {
		
	// numerical quantities
	private static final	float   INF=1e30f;
	private static final	float   ZERO=1e-30f;
	
	// data buffers
	private 	float[][]	    data;  			// original data
	private 	float[]	        proba;  	    // original data probability
	private     int             dim;            // data dimensionality
	private		int			    ndata;   		// data size
	private 	float[][]		som;		    // som map
	private 	boolean[]		mask;   	    // data mask: true for data points
	private     int             dsom;           // som dimension
	private     int             nsom,nx,ny,nz;  // som shape
	private     int             iter;           // max number of iterations
	private     int             tlearn;           // max number of iterations
	
	// shpae parameters
	private    boolean[]    lattice;
	
	// computation variables
	private     float[]     prev;
	private     float       kernelAlpha;
	private     float       kernelSize;
	
	private     boolean     debug=true;
	
	/**
	 *  constructor
	 *	note: all images passed to the algorithm are just linked, not copied
	 */
	 
	public BasicSom(float[][] data_, float[] proba_, boolean [] mask_, 
					int ndata_, int dim_,
					int dsom_, int nsom_, int tlearn_, int iter_) {
		
		data = data_;
		proba = proba_;
		mask = mask_;

		ndata = ndata_;
		dim = dim_;
		
		dsom = dsom_;
		nx = nsom_;
		if (dsom>1) ny = nsom_; else ny = 1;
		if (dsom>2) nz = nsom_; else nz = 1;
		nsom = nx*ny*nz;
		
		iter = iter_;
		tlearn = tlearn_;
				
		if (mask==null) {
		    mask = new boolean[ndata];
		    for (int idx=0;idx<ndata;idx++) mask[idx] = true;
		}
		
		// init all the new arrays
		try {
		    som = new float[dim][nsom]; 
		    prev = new float[dim];
		    lattice = new boolean[nsom];
		} catch (OutOfMemoryError e){
			System.out.println(e.getMessage());
			return;
		}
		
		// build a circular lattice
		int nkept = 0;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
		    if ( (x-nx/2)*(x-nx/2)+(y-ny/2)*(y-ny/2)+(z-nz/2)*(z-nz/2) < (nx-1)*(nx-1)/4.0f) {
		        lattice[x+nx*y+nx*ny*z] = true;
		        nkept++;
		    } else {
		        lattice[x+nx*y+nx*ny*z] = false;
		    }
		}    
		// adjust iterations to reflect the SOM size
		tlearn = nkept;
		iter = 5*tlearn;
		
		if (debug) System.out.print("SOM:initialisation\n");
		if (debug) System.out.print(nkept+" points, adjusting iterations to: "+tlearn+", "+iter+"\n");
	}

	public final float[][] getSomWeights() { return som; }
	
	public final boolean[] getSomShape() { return lattice; }
	
	public final float[][] mapSomOnData2D() {
	    float[][] map = new float[ndata][2];
	    for (int n=0;n<ndata;n++) if (mask[n]) {
	        int node = findClosestNode(data[n]);
            int y0 = Numerics.floor(node/nx);
            int x0 = node - y0*nx;
	        map[n][0] = x0/(nx-1.0f);
	        map[n][1] = y0/(ny-1.0f);
	    }
	    return map;
	}
	
	public final float[][] mapSomOnData3D() {
	    float[][] map = new float[ndata][3];
	    for (int n=0;n<ndata;n++) if (mask[n]) {
	        int node = findClosestNode(data[n]);
            int z0 = Numerics.floor(node/(nx*ny));
            int y0 = Numerics.floor((node-nx*ny*z0)/nx);
            int x0 = node - z0*nx*ny - y0*nx;
	        map[n][0] = x0/(nx-1.0f);
	        map[n][1] = y0/(ny-1.0f);
	        map[n][2] = z0/(nz-1.0f);
	    }
	    return map;
	}
	
	public final float[][] interpolateSomOnData2D() {
	    float[][] map = new float[ndata][2];
	    double[] distances = new double[3];
	    int[] nodes = new int[3];
	    for (int n=0;n<ndata;n++) if (mask[n]) {
	        findClosestNodes(data[n], distances, nodes, 3);
	        float mapweight = 0.0f;
	        for (int b=0;b<3;b++) {
                double y0 = Numerics.floor(nodes[b]/nx);
                double x0 = nodes[b] - y0*nx;
                map[n][0] += (float)(x0/(nx-1.0)/distances[b]);
                map[n][1] += (float)(y0/(ny-1.0)/distances[b]);
                mapweight += (float)(1.0/distances[b]);
            }
            map[n][0] /= mapweight;
            map[n][1] /= mapweight;
	    }
	    return map;
	}
	
	public final float[][] interpolateSomOnData3D() {
	    float[][] map = new float[ndata][3];
	    double[] distances = new double[4];
	    int[] nodes = new int[4];
	    for (int n=0;n<ndata;n++) if (mask[n]) {
	        findClosestNodes(data[n], distances, nodes, 4);
	        float mapweight = 0.0f;
	        for (int b=0;b<4;b++) {
                double z0 = Numerics.floor(nodes[b]/(nx*ny));
                double y0 = Numerics.floor((nodes[b]-nx*ny*z0)/nx);
                double x0 = nodes[b] - z0*nx*ny - y0*nx;
                map[n][0] += (float)(x0/(nx-1.0)/distances[b]);
                map[n][1] += (float)(y0/(ny-1.0)/distances[b]);
                map[n][2] += (float)(z0/(nz-1.0)/distances[b]);
                mapweight += (float)(1.0/distances[b]);
            }
            map[n][0] /= mapweight;
            map[n][1] /= mapweight;
            map[n][2] /= mapweight;
	    }
	    return map;
	}
	
	/**
	 *   main routine
	 */
	final public void run_som2D() {
	    
	    // 1. random weight initialization
	    //initializeRandomSom();
	    
	    // or PCA-based
	    initializePcaSom2D();
	    
	    // 2. main loop: sample, update
	    for (int t=0;t<iter;t++) {
	        // randomly pick a value
	        int idx = Numerics.floor(ndata*FastMath.random());
	        if (idx==ndata) idx=ndata-1;
	        
	        boolean sample = false;
	        if (mask[idx]) {
	            if (proba!=null) {
	                // only sample if above proba
	                if (proba[idx]>FastMath.random()) {
	                    sample = true;
	                }
	            } else {
	                sample = true;
	            }
	        }
	        if (sample) {        
                int node = findClosestNode(data[idx]);
                float diff = updateNodeNeighborhood2D(t, node, data[idx]); 
                System.out.println("iteration "+t+" (alpha: "+kernelAlpha+", ngb: "+kernelSize+"): "+diff);
            } else {
                // skip and resample
                t--;
            }
	    }
	    
	    return;
	}
	
	/**
	 *   main routine
	 */
	final public void run_som3D() {
	    
	    // 1. random weight initialization
	    //initializeRandomSom();
	    
	    // or PCA-based
	    initializePcaSom3D();
	    
	    // 2. main loop: sample, update
	    for (int t=0;t<iter;t++) {
	        // randomly pick a value
	        int idx = Numerics.floor(ndata*FastMath.random());
	        if (idx==ndata) idx=ndata-1;
	        
	        boolean sample = false;
	        if (mask[idx]) {
	            if (proba!=null) {
	                // only sample if above proba
	                if (proba[idx]>FastMath.random()) {
	                    sample = true;
	                }
	            } else {
	                sample = true;
	            }
	        }
	        if (sample) {        
                int node = findClosestNode(data[idx]);
                float diff = updateNodeNeighborhood3D(t, node, data[idx]); 
                System.out.println("iteration "+t+" (alpha: "+kernelAlpha+", ngb: "+kernelSize+"): "+diff);
            } else {
                // skip and resample
                t--;
            }
	    }
	    
	    return;
	}
	
	
	/**
	 *  random initialization
	 *    uniform sampling in each dimension separately
	 */
	final public void initializeRandomSom() {
	    float[] datamin = new float[dim];
	    float[] datamax = new float[dim];
	    for (int d=0;d<dim;d++) {
	        datamin[d] = INF;
	        datamax[d] = -INF;
	    }
	    for (int n=0;n<ndata;n++) if (mask[n]) {
	        for (int d=0;d<dim;d++) {
                if (data[n][d] > datamax[d]) datamax[d] = data[n][d];
                if (data[n][d] < datamin[d]) datamin[d] = data[n][d];
            }
        }
        for (int s=0;s<nsom;s++) if (lattice[s]) for (int d=0;d<dim;d++) {
            som[d][s] = (float)(datamin[d] + (datamax[d]-datamin[d])*FastMath.random());
        }
	}
   
	/**
	 *  PCA initialization
	 *   from the data values themselves
	 */
	final public void initializePcaSom2D() {
	    float[] avg = new float[dim];
	    for (int d=0;d<dim;d++) {
	       avg[d] = 0.0f;
	    }
	    double nsample = 0;
	    if (proba!=null) {
            for (int n=0;n<ndata;n++) if (mask[n]) {
                for (int d=0;d<dim;d++) {
                    avg[d] += proba[n]*data[n][d];
                }
                nsample+=proba[n];
            }
	    } else {
            for (int n=0;n<ndata;n++) if (mask[n]) {
                for (int d=0;d<dim;d++) {
                    avg[d] += data[n][d];
                }
                nsample++;
            }
        }
        for (int d=0;d<dim;d++) {
            avg[d] /= (float)nsample;
        }
        
        double[][] covar = new double[dim][dim];
	    if (proba!=null) {
            for (int n=0;n<ndata;n++) if (mask[n]) {
                for (int i=0;i<dim;i++) {
                    for (int j=i;j<dim;j++) {
                        covar[i][j] += proba[n]*(data[n][i]-avg[i])*(data[n][j]-avg[j])/nsample;
                    }
                }
            }
	    } else {
            for (int n=0;n<ndata;n++) if (mask[n]) {
                for (int i=0;i<dim;i++) {
                    for (int j=i;j<dim;j++) {
                        covar[i][j] += (data[n][i]-avg[i])*(data[n][j]-avg[j])/nsample;
                    }
                }
            }
        }
	    for (int i=0;i<dim;i++) {
	        for (int j=0;j<i;j++) {
	            covar[i][j] = covar[j][i];
	        }
	    }
	    Matrix M = new Matrix(covar);
        SingularValueDecomposition svd = M.svd();
                    
        // keep the first two for 2D SOM
        float[] eig1 = new float[dim];
        float[] eig2 = new float[dim];
        double sig1 = FastMath.sqrt(svd.getSingularValues()[0]);
        double sig2 = FastMath.sqrt(svd.getSingularValues()[1]);
        
        for (int i=0;i<dim;i++) {
            eig1[i] = (float)(sig1*svd.getV().get(i,0));                        
            eig2[i] = (float)(sig2*svd.getV().get(i,1));
        }
	    // map the avg +/- eigenvalues to som dimensions
        for (int x=0;x<nx;x++) {
            float dx = 2.0f*(x-(nx-1.0f)/2.0f)/((nx-1.0f)/2.0f);
            for (int y=0;y<ny;y++) {
                float dy = 2.0f*(y-(ny-1.0f)/2.0f)/((ny-1.0f)/2.0f);
                if (lattice[x+nx*y]) {
                    for (int d=0;d<dim;d++) {
                        som[d][x+nx*y] = avg[d] + dx*eig1[d] + dy*eig2[d];
                    }
                }
            }
        }
	}
   
	/**
	 *  PCA initialization
	 *   from the data values themselves
	 */
	final public void initializePcaSom3D() {
	    float[] avg = new float[dim];
	    for (int d=0;d<dim;d++) {
	       avg[d] = 0.0f;
	    }
	    double nsample = 0;
	    if (proba!=null) {
            for (int n=0;n<ndata;n++) if (mask[n]) {
                for (int d=0;d<dim;d++) {
                    avg[d] += proba[n]*data[n][d];
                }
                nsample+=proba[n];
            }
	    } else {
            for (int n=0;n<ndata;n++) if (mask[n]) {
                for (int d=0;d<dim;d++) {
                    avg[d] += data[n][d];
                }
                nsample++;
            }
        }
        for (int d=0;d<dim;d++) {
            avg[d] /= (float)nsample;
        }
        
        double[][] covar = new double[dim][dim];
	    if (proba!=null) {
            for (int n=0;n<ndata;n++) if (mask[n]) {
                for (int i=0;i<dim;i++) {
                    for (int j=i;j<dim;j++) {
                        covar[i][j] += proba[n]*(data[n][i]-avg[i])*(data[n][j]-avg[j])/nsample;
                    }
                }
            }
	    } else {
            for (int n=0;n<ndata;n++) if (mask[n]) {
                for (int i=0;i<dim;i++) {
                    for (int j=i;j<dim;j++) {
                        covar[i][j] += (data[n][i]-avg[i])*(data[n][j]-avg[j])/nsample;
                    }
                }
            }
        }
	    for (int i=0;i<dim;i++) {
	        for (int j=0;j<i;j++) {
	            covar[i][j] = covar[j][i];
	        }
	    }
	    Matrix M = new Matrix(covar);
        SingularValueDecomposition svd = M.svd();
                    
        // keep all three for 3D SOM
        float[] eig1 = new float[dim];
        float[] eig2 = new float[dim];
        float[] eig3 = new float[dim];
        double sig1 = FastMath.sqrt(svd.getSingularValues()[0]);
        double sig2 = FastMath.sqrt(svd.getSingularValues()[1]);
        double sig3 = FastMath.sqrt(svd.getSingularValues()[2]);
        
        for (int i=0;i<dim;i++) {
            eig1[i] = (float)(sig1*svd.getV().get(i,0));                        
            eig2[i] = (float)(sig2*svd.getV().get(i,1));
            eig3[i] = (float)(sig3*svd.getV().get(i,2));
        }
	    // map the avg +/- eigenvalues to som dimensions
        for (int x=0;x<nx;x++) {
            float dx = 2.0f*(x-(nx-1.0f)/2.0f)/((nx-1.0f)/2.0f);
            for (int y=0;y<ny;y++) {
                float dy = 2.0f*(y-(ny-1.0f)/2.0f)/((ny-1.0f)/2.0f);
                for (int z=0;z<nz;z++) {
                    float dz = 2.0f*(z-(nz-1.0f)/2.0f)/((nz-1.0f)/2.0f);
                        if (lattice[x+nx*y+nx*ny*z]) {
                        for (int d=0;d<dim;d++) {
                            som[d][x+nx*y+nx*ny*z] = avg[d] + dx*eig1[d] + dy*eig2[d] + dz*eig3[d];
                        }
                    }
                }
            }
        }
	}
   
    /** 
	 *  find closest point to data
	 */
    final public int findClosestNode(float[] val) {
        double distance, dist;
        int best = -1;

		distance = INF;
		for (int n=0;n<nsom;n++) if (lattice[n]) {
		    dist = 0.0f;
		    for (int d=0;d<dim;d++) dist += (som[d][n]-val[d])*(som[d][n]-val[d]);
		    if (dist<distance) {
		        distance = dist;
		        best = n;
		    }
		}
		return best;
    }
   
    /** 
	 *  find N closest point to data (unsorted, for interpolation purposes)
	 */
    final public void findClosestNodes(float[] val, double[] distance, int[] best, int nb) {
        double dist;
        for (int b=0;b<nb;b++) {
            distance[b] = INF;
            best[b] = -1;
        }

		for (int n=0;n<nsom;n++) if (lattice[n]) {
		    dist = 0.0f;
		    for (int d=0;d<dim;d++) dist += (som[d][n]-val[d])*(som[d][n]-val[d]);
		    
		    int changed = -1;
		    for (int b=0;b<nb;b++) {
                if (dist<distance[b]) {
                    if (changed==-1) changed = b;
                    else if (distance[b]>distance[changed]) changed = b;
                }
            }
            if (changed>-1) {
                distance[changed] = dist;
                best[changed] = n;
            }
		}
		return;
    }
    
    /**
     *  update within a neighborhood (specific to SOM shape)
     */
    final public float updateNodeNeighborhood2D(int t, int node, float[] val) {
        // get the coordinates in the SOM
        int y0 = Numerics.floor(node/nx);
        int x0 = node - y0*nx;
        
        // update the kernel functions
        kernelUpdate(t);
        
        // find value boundaries
        int xmin = Numerics.max(0, Numerics.floor(x0-kernelSize));
        int xmax = Numerics.min(nx, Numerics.ceil(x0+kernelSize+1));
        int ymin = Numerics.max(0, Numerics.floor(y0-kernelSize));
        int ymax = Numerics.min(ny, Numerics.ceil(y0+kernelSize+1));
        
        // compute the update
        float maxdiff = 0.0f;
        for (int x=xmin;x<xmax;x++) for (int y=ymin;y<ymax;y++) if (lattice[x+nx*y]) {
            float dist = (x-x0)*(x-x0)+(y-y0)*(y-y0);   
            
            float weight = kernelWeight(dist);
            float diff = 0.0f;
            for (int d=0;d<dim;d++) {
                float delta = weight*(val[d]-som[d][x+nx*y]);
                som[d][x+nx*y] += delta;
                diff += delta*delta;
            }
            if (diff>maxdiff) maxdiff = diff;
        }

        return maxdiff;
    }
    
    /**
     *  update within a neighborhood (specific to SOM shape)
     */
    final public float updateNodeNeighborhood3D(int t, int node, float[] val) {
        // get the coordinates in the SOM
        int z0 = Numerics.floor(node/(nx*ny));
        int y0 = Numerics.floor((node-z0*nx*ny)/nx);
        int x0 = node - z0*nx*ny - y0*nx;
        
        // update the kernel functions
        kernelUpdate(t);
        
        // find value boundaries
        int xmin = Numerics.max(0, Numerics.floor(x0-kernelSize));
        int xmax = Numerics.min(nx, Numerics.ceil(x0+kernelSize+1));
        int ymin = Numerics.max(0, Numerics.floor(y0-kernelSize));
        int ymax = Numerics.min(ny, Numerics.ceil(y0+kernelSize+1));
        int zmin = Numerics.max(0, Numerics.floor(z0-kernelSize));
        int zmax = Numerics.min(nz, Numerics.ceil(z0+kernelSize+1));
        
        // compute the update
        float maxdiff = 0.0f;
        for (int x=xmin;x<xmax;x++) for (int y=ymin;y<ymax;y++) for (int z=zmin;z<zmax;z++) if (lattice[x+nx*y+nx*ny*z]) {
            float dist = (x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0);   
            
            float weight = kernelWeight(dist);
            float diff = 0.0f;
            for (int d=0;d<dim;d++) {
                float delta = weight*(val[d]-som[d][x+nx*y+nx*ny*z]);
                som[d][x+nx*y+nx*ny*z] += delta;
                diff += delta*delta;
            }
            if (diff>maxdiff) maxdiff = diff;
        }

        return maxdiff;
    }
    
    /**
     *  kernel functions
     */
    final private void kernelUpdate(int t) {
        if (t<tlearn) {
            kernelAlpha = (0.9f*(1.0f - t/(float)tlearn)+0.1f);
            kernelSize = Numerics.ceil(nx/2.0f-2.0f)*(1.0f - t/(float)tlearn) + 2;
        } else {
            kernelAlpha = 0.1f*(1.0f - (t-tlearn)/(iter-tlearn+1.0f))+0.02f;
            kernelSize = 3;
        }
    }
     
    final private float kernelWeight(float dist) {
        return kernelAlpha*(float)FastMath.exp( -0.5*dist/(kernelSize*kernelSize/9.0));
    }
    
 }
