package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.Ngb;
import nl.uva.imcn.utilities.Numerics;
import nl.uva.imcn.utilities.BasicInfo;
import nl.uva.imcn.libraries.ImageFilters;
import nl.uva.imcn.libraries.ImageStatistics;
import nl.uva.imcn.libraries.ObjectExtraction;
import nl.uva.imcn.libraries.Morphology;
import nl.uva.imcn.structures.*;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.commons.math3.special.Erf;
//import org.apache.commons.math3.analysis.*;


public class LinearFiberVectorEstimation {
	
	// input images
	private float[][] probaImage;
	private float[][] thetaImage;
	private float[][] lambdaImage;
	private int n0 = 7;
	
	// direction prior
	private float[][] priorImage = null;
	private int np = 3;
	
	// masking and reindexing
	private boolean[] mask = null;
	private int nmask;
	private int[] map;
	private int[] inv;
	
	// result vectors
	private float[][] vecImage;
	private int nv = 3;
	
	// neighborhood size
	private int ngb = 3;
	private int search = 1;
	
	// priors on stdev
	private float deltaSpace = 1.0f;
	private float deltaDepth = 3.0f;
	private float deltaTheta = 5.0f;
	private float deltaPrior = 10.0f;
	
	// search parameters
	float initThreshold = 0.25f;
	
	boolean expandMask=true;
	
	// global variables
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;
	
	// orientation variables for the prior only: set to match coronal orientation instead of axial
	private final int X=0;
	private final int Y=2;
	private final int Z=1;
	
	private static final boolean debug=true;
	
	//set inputs
	public final void setPriorNumber(int val) { 
	    np = val; 
	    priorImage = new float[3*np][];
	}
	public final void setPriorImageAt(int n, float[] val) { priorImage[n] = val; }

	public final void setInputNumber(int val) { 
	    n0 = val; 
	    probaImage = new float[n0][];
	    thetaImage = new float[n0][];
	    lambdaImage = new float[n0][];
	}
	public final void setProbaImageAt(int n, float[] val) { probaImage[n] = val; }
	public final void setThetaImageAt(int n, float[] val) { thetaImage[n] = val; }
	public final void setLambdaImageAt(int n, float[] val) { lambdaImage[n] = val; }

	public final void setVectorNumber(int val) { 
	    nv = val; 
	    vecImage = new float[3*nv][];
	}

	public final void setComputationMask(int[] val) { 
	    mask = new boolean[nxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        mask[xyz] = (val[xyz]>0);
        }
	}

	public final void setNumberOfNeighbors(int val) { ngb = val; }
	public final void setSearchRadius(int val) { search = val; }
	public final void setInitialThreshold(float val) { initThreshold = val; }
	
	public final void setSpatialScale(float val) { deltaSpace = val; }
	public final void setDepthScale(float val) { deltaDepth = val; }
	public final void setThetaScale(float val) { deltaTheta = val; }
	public final void setPriorScale(float val) { deltaPrior = val; }
		
	// set generic inputs	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
			
	// create outputs
	public final float[] getVectorImageAt(int n) { return vecImage[n];}
	
	public void execute(){
		BasicInfo.displayMessage("linear fiber vector estimation:\n");
		
		if (mask==null) {
            mask = new boolean[nxyz];
            for (int xyz=0;xyz<nxyz;xyz++) {
                mask[xyz] = (probaImage[0][xyz]>0);
            }
        }
        // expand to include missing data in neighborhood
        // but remove a voxel neighborhood at the border to avoid border effects
        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if ( (mask[xyz]) && (x==0 || x==nx-1 || y==0 || y==ny-1 || z==0 || z==nz-1) ) {
                mask[xyz] = false;
            }
        }
        mask = Morphology.dilateObject(mask,nx,ny,nz,ngb,ngb,search);
                
        // build indexing and smaller arrays
        nmask=0;
        for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) nmask++;
        
        map = new int[nmask];
        inv = new int[nxyz];
        int nidx=0;
        for (int xyz=0;xyz<nxyz;xyz++) {
            if (mask[xyz]) {
                map[nidx] = xyz;
                inv[xyz] = nidx;
                nidx++;
            }
        }
            
		// rescale all inputs
		float[][] probaImg = new float[n0][nmask];
		for (int n=0;n<n0;n++) for (int id=0;id<nmask;id++) {
		    probaImg[n][id] = probaImage[n][map[id]];
		}
		probaImage = probaImg;
		
		float[][] thetaImg = new float[n0][nmask];
		for (int n=0;n<n0;n++) for (int id=0;id<nmask;id++) {
		    thetaImg[n][id] = thetaImage[n][map[id]];
		}
		thetaImage = thetaImg;
		
		float[][] lambdaImg = new float[n0][nmask];
		for (int n=0;n<n0;n++) for (int id=0;id<nmask;id++) {
		    lambdaImg[n][id] = lambdaImage[n][map[id]];
		}
		lambdaImage = lambdaImg;
		
		float[][] priorImg = new float[3*np][nmask];
		for (int n=0;n<3*np;n++) for (int id=0;id<nmask;id++) {
		    priorImg[n][id] = priorImage[n][map[id]];
		}
		priorImage = priorImg;
		
		// convert angles to radians for convenience
		for (int id=0;id<nmask;id++) {
		//for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int n=0;n<n0;n++) {
		        thetaImage[n][id] *= (float)(FastMath.PI/180.0);
		        if (thetaImage[n][id]<0) thetaImage[n][id] += FastMath.PI;              
		    }
		}
		deltaTheta *= (float)(FastMath.PI/180.0);
		deltaPrior *= (float)(FastMath.PI/180.0);
		
		// rescale prior into [0,1]
		double maxprior = 0.0;
		for (int id=0;id<nmask;id++) {
		    for (int p=0;p<np;p++) {
		        double prior = FastMath.sqrt(priorImage[3*p+X][id]*priorImage[3*p+X][id]
		                                    +priorImage[3*p+Y][id]*priorImage[3*p+Y][id]
		                                    +priorImage[3*p+Z][id]*priorImage[3*p+Z][id]);
		        if (prior>maxprior) maxprior = prior;
		    }
		}
		if (maxprior>0) {
            for (int id=0;id<nmask;id++) {
                for (int p=0;p<np;p++) {
                    priorImage[3*p+X][id] /= (float)maxprior;
                    priorImage[3*p+Y][id] /= (float)maxprior;
                    priorImage[3*p+Z][id] /= (float)maxprior;
                }
            }
        }
        
		// map differences to variances
		deltaSpace *= deltaSpace;
		deltaDepth *= deltaDepth;
		deltaTheta *= deltaTheta;
		deltaPrior *= deltaPrior;
		
		/* debug: build basic joint prior maps
        // 1. Compute prior probability of matching data
		float[] priorMatch = new float[nmask];
		for (int id=0;id<nmask;id++) {
        //for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    float maxscore = 0.0f;
		    for (int p=0;p<np;p++) {
		        double prior = FastMath.sqrt(priorImage[3*p+X][id]*priorImage[3*p+X][id]
		                                    +priorImage[3*p+Y][id]*priorImage[3*p+Y][id]
		                                    +priorImage[3*p+Z][id]*priorImage[3*p+Z][id]);
		        
		        if (prior>maxscore) {
                    double angle = FastMath.atan2(priorImage[3*p+Y][id],priorImage[3*p+X][id]);
                    if (angle<0) angle += FastMath.PI;
                    
                    for (int n=0;n<n0;n++) {
                        //if (prior*probaImage[n][id]>maxscore) {
                        if (prior>maxscore) {
                            float score = probaImage[n][id]*(float)(prior*FastMath.exp(-0.5*Numerics.square(thetaImage[n][id]-angle)/deltaPrior));
                            //float score = (float)(prior*FastMath.exp(-0.5*Numerics.square(thetaImage[n][id]-angle)/deltaPrior));
                            if (score>maxscore) {
                                maxscore = score;
                            }
                        }
                    }
                }
		    }
		    priorMatch[id] = maxscore;
		}
		// 1b. Estimate pointwise vector models
		float[] jointDepth = new float[nmask];
		float[] jointTheta = new float[nmask];
		float[] jointProba = new float[nmask];
		for (int id=0;id<nmask;id++) {
        //for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    float maxscore = 0.0f;
		    float maxdepth = 0.0f;
		    float maxtheta = 0.0f;
		    float maxproba = 0.0f;
		    for (int p=0;p<np;p++) {
		    
		        double prior = FastMath.sqrt(priorImage[3*p+X][id]*priorImage[3*p+X][id]
		                                    +priorImage[3*p+Y][id]*priorImage[3*p+Y][id]
		                                    +priorImage[3*p+Z][id]*priorImage[3*p+Z][id]);
		        
		        if (prior>maxscore) {
                    double angle = FastMath.atan2(priorImage[3*p+Y][id],priorImage[3*p+X][id]);
                    if (angle<0) angle += FastMath.PI;
                    
                    for (int n=0;n<n0;n++) {
                        //if (prior*probaImage[n][id]>maxscore) {
                        if (prior>maxscore) {
                            //float score = probaImage[n][id]*(float)(prior*FastMath.exp(-0.5*Numerics.square(thetaImage[n][id]-angle)/deltaPrior));
                            float score = (float)(prior*FastMath.exp(-0.5*Numerics.square(thetaImage[n][id]-angle)/deltaPrior));
                            if (score>maxscore) {
                                maxscore = score;
                                // estimate corresponding depth
                                double lambda0 = FastMath.sqrt( (priorImage[3*p+X][id]*priorImage[3*p+X][id]
		                                                        +priorImage[3*p+Y][id]*priorImage[3*p+Y][id])
		                                                       /(priorImage[3*p+Z][id]*priorImage[3*p+Z][id]) );
		                        maxdepth = (float)( (probaImage[n][id]*lambdaImage[n][id] + score*lambda0)
		                                                    /(probaImage[n][id] + score) );
		                        maxtheta = (float)( (probaImage[n][id]*thetaImage[n][id] + score*angle)
		                                                    /(probaImage[n][id] + score) );
		                        maxproba = (float)( (probaImage[n][id]*probaImage[n][id] + score*prior)
		                                                    /(probaImage[n][id] + score) );
                            }
                        }
                    }
                }
		    }
		    priorMatch[id] = maxscore;
		    jointDepth[id] = maxdepth;
		    jointTheta[id] = maxtheta;
		    jointProba[id] = maxproba;
		}
		
		float[] bestNgb = new float[nmask];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
		    int xyz = x+nx*y+nx*ny*z;
		    int id0 = inv[xyz];
		    if (mask[xyz]) {
                float maxscore = 0.0f;
                // search neighborhood; different in x,y and z directions
                for (int dz=-ngb;dz<=ngb;dz++) if (z+dz>=0 && z+dz<nz) {
                    for (int dx=-search;dx<=search;dx++) for (int dy=-search;dy<=search;dy++) {
                        int idn = inv[xyz + dx + nx*dy + nx*ny*dz];
                        if (jointProba[idn]>maxscore) {
                            float diff = (dx*dx+dy*dy)/deltaSpace + dz*dz/deltaDepth;
                            float probaDist = (float)FastMath.exp(-0.5*diff);
                            
                            // only use first proba as highest sample? or weight also with angle difference?
                            // compute difference in angle
                            if (probaDist*jointProba[idn]>maxscore) {
                                float score = (float)FastMath.exp(-0.5*Numerics.square(jointTheta[id0]-jointTheta[idn])/deltaTheta);
                                score *= probaDist*jointProba[idn];
                                if (score>maxscore) maxscore = score;
                            }
                        }
                    }
                }
                bestNgb[id0] = maxscore;
            }
        }
        */

		// Main algorithm:
		//    Find maximum joint prior and similarity (as in step 1 above)
		//    Sort by best values, then depile and estimate posterior
		//    Perform these locally instead, then sort on neighborhood properties
		//    (seed as local maxima)
		
		//    Add neighbors to the ordering based on posterior + prior
		//    Keep depiling until done
		
		// 1. Stack candidates with high proba and high prior into queue
		BinaryHeap3DPair ordering = new BinaryHeap3DPair(nmask, Numerics.ceil(0.2*nmask), BinaryHeap3DPair.MAXTREE);
		for (int id=0;id<nmask;id++) {
		    // find best angle per voxel across directions 
		    for (byte n=0;n<n0;n++) {
		        // select all strong priors for the voxel, if multiple ones are valid
		        double maxscore = initThreshold;
		        byte pbest = -1;
                for (byte p=0;p<np;p++) {
                    // use only strong priors for now   
                    double prior = FastMath.sqrt(priorImage[3*p+X][id]*priorImage[3*p+X][id]
                                                +priorImage[3*p+Y][id]*priorImage[3*p+Y][id]
                                                +priorImage[3*p+Z][id]*priorImage[3*p+Z][id]);
                    // threshold or scaling by prior?
                    if (prior>maxscore) {
                        // 2D angle from prior
                        double angle = FastMath.atan2(priorImage[3*p+Y][id],priorImage[3*p+X][id]);
                        if (angle<0) angle += FastMath.PI;
                        // find corresponding angle in data (more likely and more similar detection results)   
                    
                        double score = prior*FastMath.exp(-0.5*Numerics.square(thetaImage[n][id]-angle)/deltaPrior);
                        if (score>maxscore) {
                            maxscore = score;
                            pbest = p;
                        }
                    }
                }
                if (pbest>-1 && maxscore>initThreshold) {
                    // estimate potential posterior
                    ordering.addValue((float)maxscore, id, n, id, pbest, id, (byte)-1);
                }
            }
        }
        System.out.println("Initial seed vectors: "+ordering.getCurrentSize());
		
        // Build the maps adding region growing
        float[][] jointDepth = new float[nv][nmask];
		float[][] jointTheta = new float[nv][nmask];
		float[][] jointProba = new float[nv][nmask];
		int nproc = 0;
		int nstep = nmask/100;
		while (ordering.isNotEmpty()) {
            float maxpropag = ordering.getFirst();
            int id0 = ordering.getFirstX();
            byte n0i = ordering.getFirstIdX();
            //int idn = ordering.getFirstY();
            byte n0p = ordering.getFirstIdY();
            int idn = ordering.getFirstZ();
            byte nJn = ordering.getFirstIdZ();
            
            ordering.removeFirst();
            
            // if already done or all kept values are set, skip
            if (probaImage[n0i][id0]>=0 && jointProba[nv-1][id0]==0) {
            
                // build joint maps:
                
                // 1. if needed, find best prior
                double pv0 = 0.0;
                double angle0 = 0.0;
                double prior0 = 0.0;
                double lambda0 = 0.0;
                
                // skip if there is a neighbor set already?? or not?
                //if (idn==id0) {
                if (n0p>-1) {
                    prior0 = FastMath.sqrt(priorImage[3*n0p+X][id0]*priorImage[3*n0p+X][id0]
                                          +priorImage[3*n0p+Y][id0]*priorImage[3*n0p+Y][id0]
                                          +priorImage[3*n0p+Z][id0]*priorImage[3*n0p+Z][id0]);
                    
                    angle0 = FastMath.atan2(priorImage[3*n0p+Y][id0],priorImage[3*n0p+X][id0]);
                    if (angle0<0) angle0 += FastMath.PI;
                        
                    pv0 = prior0*FastMath.exp(-0.5*Numerics.square(thetaImage[n0i][id0]-angle0)/deltaPrior);
                    
                    // estimate prior depth
                    lambda0 = FastMath.sqrt( (priorImage[3*n0p+X][id0]*priorImage[3*n0p+X][id0]
                                             +priorImage[3*n0p+Y][id0]*priorImage[3*n0p+Y][id0])
                                            /(priorImage[3*n0p+Z][id0]*priorImage[3*n0p+Z][id0]) );
                }
                
                // 2. if needed, add best neighbor
                double pngb = 0.0;
                double anglen = 0.0;
                double proban = 0.0;
                double depthn = 0.0;
                if (idn!=id0) {
                    pngb = jointProba[nJn][idn]*FastMath.exp(-0.5*Numerics.square(thetaImage[n0i][id0]-jointTheta[nJn][idn])/deltaTheta);
                    anglen = jointTheta[nJn][idn];
                    proban = jointProba[nJn][idn];
                    depthn = jointDepth[nJn][idn];
                }
                
                // 3. estimate joint values
                double p0 = probaImage[n0i][id0];
    
                
                // joint estimate
                float depth = (float)( (p0*lambdaImage[n0i][id0] + (1.0-p0)*pv0*lambda0 + (1.0-p0)*(1.0-prior0)*pngb*depthn)
                                        /(p0 + (1.0-p0)*pv0 + (1.0-p0)*(1.0-prior0)*pngb) );
                float theta = (float)( (p0*thetaImage[n0i][id0] + (1.0-p0)*pv0*angle0 + (1.0-p0)*(1.0-prior0)*pngb*anglen)
                                        /(p0 + (1.0-p0)*pv0 + (1.0-p0)*(1.0-prior0)*pngb) );
                float proba = (float)( (p0*probaImage[n0i][id0] + (1.0-p0)*pv0*prior0 + (1.0-p0)*(1.0-prior0)*pngb*proban)
                                        /(p0 + (1.0-p0)*pv0 + (1.0-p0)*(1.0-prior0)*pngb) );
                
                // 4. set as posterior in first empty spot
                byte nJi = -1;
                for (byte v=0;v<nv;v++) if (jointProba[v][id0]==0) {
                    jointDepth[v][id0] = depth;
                    jointTheta[v][id0] = theta;
                    jointProba[v][id0] = proba;
                    nJi = v;
                    v = (byte)nv;
                }
                // counter
                if (nproc%nstep==0) System.out.print("-");
                nproc++;
                
                // 5. make original map values negative to indicate it's been used
                // (need an offset as some values may be zero at the start)
                probaImage[n0i][id0] = -10.0f;
                
                // now add neighbors to the queue...
                int xyz = map[id0];
                // search neighborhood; different in x,y and z directions
                for (int dz=-ngb;dz<=ngb;dz++) {
                    for (int dx=-search;dx<=search;dx++) for (int dy=-search;dy<=search;dy++) {
                        int xyzn = xyz + dx + nx*dy + nx*ny*dz;
                        if (xyzn>0 && xyzn<nxyz && mask[xyzn]) {
                            int idm = inv[xyzn];
                            // check if already fully set
                            if (jointProba[nv-1][idm]==0) {
                                // distance factor (opt)
                                float diff = (dx*dx+dy*dy)/deltaSpace + dz*dz/deltaDepth;
                                double probaDist = jointProba[nJi][id0]*FastMath.exp(-0.5*diff);
                                
                                // use best matching of remaining
                                double maxscore = 0.0;
                                byte mbest = -1;
                                for (byte m=0;m<n0;m++) {
                                    if (probaDist*jointProba[nJi][id0]>maxscore) {
                                        double score = jointProba[nJi][id0]*probaDist
                                                        *FastMath.exp(-0.5*Numerics.square(jointTheta[nJi][id0]-thetaImage[m][idm])/deltaTheta);
                                    
                                        if (score>maxscore) {
                                            maxscore = score;
                                            mbest = m;
                                        }
                                    }
                                }
                                if (mbest>-1) {
                                    ordering.addValue((float)maxscore, idm, mbest, id0, (byte)-1, id0, nJi);
                                }
                            }
                        }
                    }
                }
            }            
        }
        /*
		// 2. Aggregate probabilities across priors
		float[] highestProba = new float[nmask];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
		    int xyz = x+nx*y+nx*ny*z;
		    int id0 = inv[xyz];
		    if (mask[xyz]) {
                float maxscore = 0.0f;
                // search neighborhood; different in x,y and z directions
                for (int dz=-ngb;dz<=ngb;dz++) if (z+dz>=0 && z+dz<nz) {
                    for (int dx=-search;dx<=search;dx++) for (int dy=-search;dy<=search;dy++) {
                        int idn = inv[xyz + dx + nx*dy + nx*ny*dz];
                        float diff = (dx*dx+dy*dy)/deltaSpace + dz*dz/deltaDepth;
                        float probaDist = (float)FastMath.exp(-0.5*diff);
                        
                        if (probaDist>maxscore) {
                            // only use first proba as highest sample? or weight also with angle difference?
                            // compute difference in angle
                            for (int n=0;n<n0;n++) {
                                if (probaDist*probaImage[n][idn]>maxscore) {
                                    for (int m=0;m<n0;m++) {
                                        float score = (float)FastMath.exp(-0.5*Numerics.square(thetaImage[m][id0]-thetaImage[n][idn])/deltaTheta);
                                        score *= probaDist*probaImage[n][idn];
                                        if (score>maxscore) maxscore = score;
                                    }
                                }
                            }
                        }
                    }
                }
                highestProba[id0] = maxscore;
            }
        }
        */

        /* TO DO
		// 3. From highest to lowest combined spatial and prior, estimate vectors
		// using already estimated vectors as better priors when available
		BinaryHeap3D ordering = new BinaryHeap3D(nmask, nmask, BinaryHeap3D.MAXTREE);

        // simply order them by size instead?
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
		    int xyz = x+nx*y+nx*ny*z;
		    if (mask[xyz]) {
                ordering.addValue(highestProba[xyz]+priorMatch[xyz], x, y, z);
            }
        }
        while (ordering.isNotEmpty()) {
            float maxpropag = ordering.getFirst();
            int x = ordering.getFirstX();
            int y = ordering.getFirstY();
            int z = ordering.getFirstZ();
            
            ordering.removeFirst();
            
            // estimate best vectors from inputs
            
            
            
        }
		*/
		probaImage = null;
		thetaImage = null;
		lambdaImage = null;
		priorImage = null;
		
		for (int v=0;v<nv;v++) {
            vecImage[v] = new float[nxyz];
            for (int id=0;id<nmask;id++) {
                vecImage[v][map[id]] = jointProba[v][id];
            }
            vecImage[v+nv] = new float[nxyz];
            for (int id=0;id<nmask;id++) {
                vecImage[v+nv][map[id]] = jointTheta[v][id];
            }
            vecImage[v+2*nv] = new float[nxyz];
            for (int id=0;id<nmask;id++) {
                vecImage[v+2*nv][map[id]] = jointDepth[v][id];
            }
        }
		
		return;
    }
    
    void generateWeightMaps() {
		// 1. Compute prior probability of matching data
		float[] priorMatch = new float[nmask];
		for (int id=0;id<nmask;id++) {
        //for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    float maxscore = 0.0f;
		    for (int p=0;p<np;p++) {
		        double prior = FastMath.sqrt(priorImage[3*p+X][id]*priorImage[3*p+X][id]
		                                    +priorImage[3*p+Y][id]*priorImage[3*p+Y][id]
		                                    +priorImage[3*p+Z][id]*priorImage[3*p+Z][id]);
		        
		        if (prior>maxscore) {
                    double angle = FastMath.atan2(priorImage[3*p+Y][id],priorImage[3*p+X][id]);
                    if (angle<0) angle += FastMath.PI;
                    
                    for (int n=0;n<n0;n++) {
                        //if (prior*probaImage[n][id]>maxscore) {
                        if (prior>maxscore) {
                            //float score = probaImage[n][id]*(float)(prior*FastMath.exp(-0.5*Numerics.square(thetaImage[n][id]-angle)/deltaPrior));
                            float score = (float)(prior*FastMath.exp(-0.5*Numerics.square(thetaImage[n][id]-angle)/deltaPrior));
                            if (score>maxscore) maxscore = score;
                        }
                    }
                }
		    }
		    priorMatch[id] = maxscore;
		}
		
		// 2. Aggregate probabilities across priors
		float[] highestProba = new float[nmask];
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
		    int xyz = x+nx*y+nx*ny*z;
		    int id0 = inv[xyz];
		    if (mask[xyz]) {
                float maxscore = 0.0f;
                // search neighborhood; different in x,y and z directions
                for (int dz=-ngb;dz<=ngb;dz++) if (z+dz>=0 && z+dz<nz) {
                    for (int dx=-search;dx<=search;dx++) for (int dy=-search;dy<=search;dy++) {
                        int idn = inv[xyz + dx + nx*dy + nx*ny*dz];
                        float diff = (dx*dx+dy*dy)/deltaSpace + dz*dz/deltaDepth;
                        float probaDist = (float)FastMath.exp(-0.5*diff);
                        
                        if (probaDist>maxscore) {
                            // only use first proba as highest sample? or weight also with angle difference?
                            // compute difference in angle
                            for (int n=0;n<n0;n++) {
                                if (probaDist*probaImage[n][idn]>maxscore) {
                                    for (int m=0;m<n0;m++) {
                                        float score = (float)FastMath.exp(-0.5*Numerics.square(thetaImage[m][id0]-thetaImage[n][idn])/deltaTheta);
                                        score *= probaDist*probaImage[n][idn];
                                        if (score>maxscore) maxscore = score;
                                    }
                                }
                            }
                        }
                    }
                }
                highestProba[id0] = maxscore;
            }
        }
        // debug: return the two maps
		probaImage = null;
		thetaImage = null;
		lambdaImage = null;
		priorImage = null;
		
		vecImage = new float[2][];
		vecImage[0] = new float[nxyz];
		for (int id=0;id<nmask;id++) {
		    vecImage[0][map[id]] = priorMatch[id];
		}
		vecImage[1] = new float[nxyz];
		for (int id=0;id<nmask;id++) {
		    vecImage[1][map[id]] = highestProba[id];
		}
		return;
	}
}
