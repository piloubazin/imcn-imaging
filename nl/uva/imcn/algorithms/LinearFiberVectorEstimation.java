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
	private int iter = -1;
	
	// priors on stdev
	private float deltaSpace = 1.0f;
	private float deltaDepth = 3.0f;
	private float deltaTheta = 5.0f;
	private float deltaPrior = 10.0f;
	
	// search parameters
	float initThreshold = 0.25f;
	float thickness = 20.0f;
	float offset = 5.0f;

	boolean expandMask=true;
	
	boolean weightProba=true;
	
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
	public final void setIterations(int val) { iter = val; }
	public final void setInitialThreshold(float val) { initThreshold = val; }
	public final void setImageThickness(float val) { thickness = val; }
	public final void setThicknessOffset(float val) { offset = val; }
	
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
        mask = Morphology.dilateObject(mask,nx,ny,nz,ngb,ngb,search);
         // but remove a voxel neighborhood at the border to avoid border effects
        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
            int xyz = x+nx*y+nx*ny*z;
            if ( (mask[xyz]) && (x==0 || x==nx-1 || y==0 || y==ny-1 || z==0 || z==nz-1) ) {
                mask[xyz] = false;
            }
        }
               
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
		
		// Main algorithm:
		//    Find maximum joint prior and similarity (as in step 1 above)
		//    Sort by best values, then depile and estimate posterior
		//    Perform these locally instead, then sort on neighborhood properties
		//    (seed as local maxima)
		
		//    Add neighbors to the ordering based on posterior + prior
		//    Keep depiling until done
		
		// 1. Estimate pointwise vector models from prior and local measurement
		if (deltaPrior>0) {
            for (int id=0;id<nmask;id++) {
                for (byte n=0;n<n0;n++) if (probaImage[n][id]>0) {
                    double maxscore = 0.0;
                    byte pbest=-1;
                    
                    double lambda0 = 0.0;
                    double theta0 = 0.0;
                    double proba0 = 0.0;
                    if (weightProba) {
                        for (byte p=0;p<np;p++) {		    
                            double prior = FastMath.sqrt(priorImage[3*p+X][id]*priorImage[3*p+X][id]
                                                        +priorImage[3*p+Y][id]*priorImage[3*p+Y][id]
                                                        +priorImage[3*p+Z][id]*priorImage[3*p+Z][id]);
                        
                            if (prior>maxscore) {
                                double angle = FastMath.atan2(priorImage[3*p+Y][id],priorImage[3*p+X][id]);
                                if (angle<0) angle += FastMath.PI;
                            
                                double score = prior;
                                if (probaImage[n][id]>0) {
                                    score *= FastMath.exp(-0.5*Numerics.square(thetaImage[n][id]-angle)/deltaPrior);
                                }
                                if (score>maxscore) {
                                    maxscore = score;
                                    pbest = p;
                                    theta0 = angle;
                                    proba0 = prior;
                                }
                            }
                        }
                    } else {
                        for (byte p=0;p<np;p++) {		    
                            double angle = FastMath.atan2(priorImage[3*p+Y][id],priorImage[3*p+X][id]);
                            if (angle<0) angle += FastMath.PI;
                        
                            double score = 0.0;
                            if (probaImage[n][id]>0) {
                                score = FastMath.exp(-0.5*Numerics.square(thetaImage[n][id]-angle)/deltaPrior);
                            }
                            if (score>maxscore) {
                                maxscore = score;
                                pbest = p;
                                theta0 = angle;
                                proba0 = FastMath.sqrt(priorImage[3*p+X][id]*priorImage[3*p+X][id]
                                                        +priorImage[3*p+Y][id]*priorImage[3*p+Y][id]
                                                        +priorImage[3*p+Z][id]*priorImage[3*p+Z][id]);
                            }
                        }
                    }                   
                    // joint estimates
                                    
                    // estimate corresponding depth
                    if (pbest>-1) {
                        lambda0 = offset + thickness*FastMath.sqrt( (priorImage[3*pbest+X][id]*priorImage[3*pbest+X][id]
                                                                    +priorImage[3*pbest+Y][id]*priorImage[3*pbest+Y][id])
                                                                   /(priorImage[3*pbest+Z][id]*priorImage[3*pbest+Z][id]) );
                    }
                    if (weightProba) {
                        // using proba as weight
                        lambdaImage[n][id] = (float)( (probaImage[n][id]*lambdaImage[n][id] + maxscore*lambda0)/(probaImage[n][id] + maxscore) );
                        thetaImage[n][id] = (float)( (probaImage[n][id]*thetaImage[n][id] + maxscore*theta0)/(probaImage[n][id] + maxscore) );
                        //probaImage[n][id] = (float)( (probaImage[n][id]*probaImage[n][id] + maxscore*proba0) /(probaImage[n][id] + maxscore) );
                        //probaImage[n][id] = (float)(0.5*(probaImage[n][id] + proba0) );
                        if (proba0>0) probaImage[n][id] = (float)( (probaImage[n][id] + maxscore)/(1.0 + maxscore/proba0) );
                    } else {
                        // using only the similarity as weight
                        lambdaImage[n][id] = (float)( (lambdaImage[n][id] + maxscore*lambda0)/(1.0 + maxscore) );
                        thetaImage[n][id] = (float)( (thetaImage[n][id] + maxscore*theta0)/(1.0 + maxscore) );
                        probaImage[n][id] = (float)( (probaImage[n][id] + maxscore*proba0)/(1.0 + maxscore) );
                    }
                }
            }
        }
		priorImage = null;
		/*
		// 1. Stack candidates with high proba and high prior into queue
		BinaryHeap4D ordering = new BinaryHeap4D(nmask, Numerics.ceil(0.2*nmask), BinaryHeap4D.MAXTREE);
		for (int id=0;id<nmask;id++) {
		    for (byte n=0;n<n0;n++) {
		        // find local maxima
		        double maxscore = initThreshold;
		        
		        if (probaImage[n][id]>maxscore) {
		            boolean better=false;
		            maxscore = probaImage[n][id];
		            
                    int xyz = map[id];
                    // search neighborhood; different in x,y and z directions
                    for (int dz=-ngb;dz<=ngb && !better;dz++) {
                        for (int dx=-search;dx<=search && !better;dx++) for (int dy=-search;dy<=search && !better;dy++) {
                            int xyzn = xyz + dx + nx*dy + nx*ny*dz;
                            if (xyzn>0 && xyzn<nxyz && mask[xyzn] && xyzn!=xyz) {
                                int idm = inv[xyzn];
                            
                                // distance factor (opt)
                                float diff = (dx*dx+dy*dy)/deltaSpace + dz*dz/deltaDepth;
                                double probaDist = FastMath.exp(-0.5*diff);
                                
                                if (probaDist>maxscore) {
                                    for (byte m=0;m<n0;m++) {
                                        // this might be just too strong
                                        //double score = probaDist*probaImage[m][idm]
                                        //                *FastMath.exp(-0.5*Numerics.square(thetaImage[n][id]-thetaImage[m][idm])/deltaTheta);
                                        // instead, keep whatever is a local maximum probability regardless of everything else
                                        double score = probaImage[m][idm];
                                    
                                        if (score>maxscore) {
                                            better = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // only add if no local value is higher
                    if (!better) {
                        ordering.addValue((float)maxscore, id, n, id, n);
                    }
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
            int n0i = ordering.getFirstY();
            int idn = ordering.getFirstZ();
            int nJn = ordering.getFirstK();
            
            ordering.removeFirst();
            
            // if already done or all kept values are set, skip
            if (probaImage[n0i][id0]>=0 && jointProba[nv-1][id0]==0) {
            
                // build joint maps:
                
                // if needed, find best neighbor
                double pngb = 0.0;
                double theta = 0.0;
                double proba = 0.0;
                double depth = 0.0;
                if (idn!=id0) {
                    pngb = FastMath.exp(-0.5*Numerics.square(thetaImage[n0i][id0]-jointTheta[nJn][idn])/deltaTheta);
                    theta = jointTheta[nJn][idn];
                    proba = jointProba[nJn][idn];
                    depth = jointDepth[nJn][idn];
                }
                
                // 3. estimate joint values
                double p0 = probaImage[n0i][id0];
                
                // joint estimate
                depth = (p0*lambdaImage[n0i][id0] + pngb*proba*depth)/(p0 +  pngb*proba);
                theta = (p0*thetaImage[n0i][id0] + pngb*proba*theta)/(p0 + pngb*proba);
                proba = (probaImage[n0i][id0] + pngb*proba)/(1.0 + pngb);
                
                // 4. set as posterior in first empty spot
                // (note that probability ranking is not necessarily preserved)
                byte nJi = -1;
                for (byte v=0;v<nv;v++) if (jointProba[v][id0]==0) {
                    jointDepth[v][id0] = (float)depth;
                    jointTheta[v][id0] = (float)theta;
                    jointProba[v][id0] = (float)proba;
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
                        if (xyzn>0 && xyzn<nxyz && mask[xyzn] && xyzn!=xyz) {
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
                                    // use local proba, or just the theta? initial score seems buggy..
                                    //if (probaDist*jointProba[nJi][id0]>maxscore) {
                                    //    double score = jointProba[nJi][id0]*probaDist
                                    //                    *FastMath.exp(-0.5*Numerics.square(jointTheta[nJi][id0]-thetaImage[m][idm])/deltaTheta);
                                    // both probas, not super-consistent (-> propagate preferrentially to measures with high detection over high angle match)
                                    //if (probaDist*probaImage[m][idm]>maxscore) {
                                    //    double score = probaImage[m][idm]*probaDist
                                    //                    *FastMath.exp(-0.5*Numerics.square(jointTheta[nJi][id0]-thetaImage[m][idm])/deltaTheta);
                                    // only the theta of neighbor to propagate into (allows to propagate into very low target probas..)
                                    if (probaDist>maxscore) {
                                        double score = probaDist
                                                        *FastMath.exp(-0.5*Numerics.square(jointTheta[nJi][id0]-thetaImage[m][idm])/deltaTheta);
                                    
                                        if (score>maxscore) {
                                            maxscore = score;
                                            mbest = m;
                                        }
                                    }
                                }
                                if (mbest>-1) {
                                    ordering.addValue((float)maxscore, idm, mbest, id0, nJi);
                                }
                            }
                        }
                    }
                }
            }            
        }*/
		// alternative: use a local diffusion technique (allows for smoothing across neighbors)
		// iterate between taking the local diffusion average and updating?
        float[][] jointDepth = new float[nv][nmask];
		float[][] jointTheta = new float[nv][nmask];
		float[][] jointProba = new float[nv][nmask];
		
		int nproc = 0;
		int nstep = nmask/100;
		for (int id=0;id<nmask;id++) {
		    for (byte n=0;n<n0;n++) {
		        
		        double sum = 0.0;
                double theta = 0.0;
                double proba = 0.0;
                double depth = 0.0;
		        
		        int xyz = map[id];
                // search neighborhood; different in x,y and z directions
                for (int dz=-ngb;dz<=ngb;dz++) {
                    for (int dx=-search;dx<=search;dx++) for (int dy=-search;dy<=search;dy++) {
                        int xyzn = xyz + dx + nx*dy + nx*ny*dz;
                        if (xyzn>0 && xyzn<nxyz && mask[xyzn]) {
                            int idm = inv[xyzn];
                        
                            // distance factor (opt)
                            float diff = (dx*dx+dy*dy)/deltaSpace + dz*dz/deltaDepth;
                            double probaDist = FastMath.exp(-0.5*diff);
                            
                            if (probaDist>initThreshold) {
                                double maxscore = initThreshold;
                                int mb = -1;
                                for (byte m=0;m<n0;m++) {
                                    double score = probaDist;
                                    if (weightProba) score *= probaImage[m][idm];
                                    if (probaImage[n][id]>0) score *= FastMath.exp(-0.5*Numerics.square(thetaImage[n][id]-thetaImage[m][idm])/deltaTheta);
                                    
                                    if (score>maxscore) {
                                        maxscore = score;
                                        mb = m;
                                    }
                                }
                                if (mb>-1) {
                                    double pngb = probaDist;
                                    if (probaImage[n][id]>0) pngb *= FastMath.exp(-0.5*Numerics.square(thetaImage[n][id]-thetaImage[mb][idm])/deltaTheta);
                                    if (weightProba) {
                                        theta += pngb*probaImage[mb][idm]*thetaImage[mb][idm];
                                        depth += pngb*probaImage[mb][idm]*lambdaImage[mb][idm]; 
                                        proba += pngb*probaImage[mb][idm];
                                        sum += pngb;
                                    } else {
                                        theta += pngb*thetaImage[mb][idm];
                                        depth += pngb*lambdaImage[mb][idm]; 
                                        proba += pngb*probaImage[mb][idm];
                                        sum += pngb;
                                    }
                                }
                            }
                        }
                    }
                }
                // Estimate joint values
                double p0 = probaImage[n][id];
                
                // joint estimate
                if (weightProba) {
                    depth = (p0*lambdaImage[n][id] + depth)/(p0 + proba);
                    theta = (p0*thetaImage[n][id] + theta)/(p0 + proba);
                    proba = (probaImage[n][id] + proba)/(1.0 + sum);
                } else {
                    depth = (lambdaImage[n][id] + depth)/(1.0 + sum);
                    theta = (thetaImage[n][id] + theta)/(1.0 + sum);
                    proba = (probaImage[n][id] + proba)/(1.0 + sum);
                }
                // set as posterior: need to keep only the top values
                for (int v=0;v<nv;v++) if (jointProba[v][id]<=proba) {
                    // check for neighbors
                    for (int vp=nv-1;vp>v;vp--) {
                        if (jointProba[vp][id]<jointProba[vp-1][id]) {
                            jointDepth[vp][id] = jointDepth[vp-1][id];
                            jointTheta[vp][id] = jointTheta[vp-1][id];
                            jointProba[vp][id] = jointProba[vp-1][id];
                        }
                    }
                    // new value
                    jointDepth[v][id] = (float)depth;
                    jointTheta[v][id] = (float)theta;
                    jointProba[v][id] = (float)proba;
                    v = (byte)nv;
                    // counter
                    if (nproc%nstep==0) System.out.print("-");
                    nproc++;
                }
            }            
        }
        
        // clean up before building the full maps
		probaImage = null;
		thetaImage = null;
		lambdaImage = null;
		priorImage = null;

		// iterate ?
        for (int t=0;t<iter;t++) {
		    // copy previous result? just rename
		    /*
            for (int id=0;id<nmask;id++) {
                for (byte v=0;v<nv;v++) {
                    prevDepth[v][id] = jointDepth[v][id];
                    prevTheta[v][id] = jointTheta[v][id];
                    prevProba[v][id] = jointProba[v][id];
                    jointDepth[v][id] = 0.0f;
                    jointTheta[v][id] = 0.0f;
                    jointProba[v][id] = 0.0f;
                }
            }*/
            float[][] prevProba = jointProba;
            float[][] prevTheta = jointTheta;
            float[][] prevDepth = jointDepth;
            jointDepth = new float[nv][nmask];
            jointTheta = new float[nv][nmask];
            jointProba = new float[nv][nmask];            
            
            for (int id=0;id<nmask;id++) {
                for (byte v=0;v<nv;v++) {
                    
                    double sum = 0.0;
                    double theta = 0.0;
                    double proba = 0.0;
                    double depth = 0.0;
                    
                    int xyz = map[id];
                    // search neighborhood; different in x,y and z directions
                    for (int dz=-ngb;dz<=ngb;dz++) {
                        for (int dx=-search;dx<=search;dx++) for (int dy=-search;dy<=search;dy++) {
                            int xyzn = xyz + dx + nx*dy + nx*ny*dz;
                            if (xyzn>0 && xyzn<nxyz && mask[xyzn]) {
                                int idm = inv[xyzn];
                            
                                // distance factor (opt)
                                float diff = (dx*dx+dy*dy)/deltaSpace + dz*dz/deltaDepth;
                                double probaDist = FastMath.exp(-0.5*diff);
                                
                                if (probaDist>initThreshold) {
                                    double maxscore = initThreshold;
                                    int wb = -1;
                                    for (byte w=0;w<nv;w++) {
                                        double score = probaDist;
                                        if (weightProba) score *= prevProba[w][idm];
                                        if (prevProba[v][id]>0) score *= FastMath.exp(-0.5*Numerics.square(prevTheta[v][id]-prevTheta[w][idm])/deltaTheta);
                                        
                                        if (score>maxscore) {
                                            maxscore = score;
                                            wb = w;
                                        }
                                    }
                                    if (wb>-1) {
                                        double pngb = probaDist;
                                        if (prevProba[v][id]>0) pngb *= FastMath.exp(-0.5*Numerics.square(prevTheta[v][id]-prevTheta[wb][idm])/deltaTheta);
                                        if (weightProba) {
                                            theta += pngb*prevProba[wb][idm]*prevTheta[wb][idm];
                                            depth += pngb*prevProba[wb][idm]*prevDepth[wb][idm]; 
                                            proba += pngb*prevProba[wb][idm];
                                            sum += pngb;
                                        } else {
                                            theta += pngb*prevTheta[wb][idm];
                                            depth += pngb*prevDepth[wb][idm]; 
                                            proba += pngb*prevProba[wb][idm];
                                            sum += pngb;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Estimate joint values
                    double p0 = prevProba[v][id];
                    
                    // joint estimate
                    if (weightProba) {
                        depth = (p0*prevDepth[v][id] + depth)/(p0 + proba);
                        theta = (p0*prevTheta[v][id] + theta)/(p0 + proba);
                        proba = (prevProba[v][id] + proba)/(1.0 + sum);
                    } else {
                        depth = (prevDepth[v][id] + depth)/(1.0 + sum);
                        theta = (prevTheta[v][id] + theta)/(1.0 + sum);
                        proba = (prevProba[v][id] + proba)/(1.0 + sum);
                    }
                    // set as posterior: replace current value
                    jointDepth[v][id] = (float)depth;
                    jointTheta[v][id] = (float)theta;
                    jointProba[v][id] = (float)proba;
                    // counter
                    if (nproc%nstep==0) System.out.print("-");
                    nproc++;
                }            
            }
            prevProba = null;
            prevDepth = null;
            prevTheta = null;
		}
		/* raw measurements
		for (int v=0;v<nv;v++) {
            vecImage[v] = new float[nxyz];
            for (int id=0;id<nmask;id++) {
                //vecImage[v][map[id]] = probaImage[v][id];
                vecImage[v][map[id]] = jointProba[v][id];
            }
            vecImage[v+nv] = new float[nxyz];
            for (int id=0;id<nmask;id++) {
                //vecImage[v+nv][map[id]] = thetaImage[v][id];
                vecImage[v+nv][map[id]] = jointTheta[v][id];
            }
            vecImage[v+2*nv] = new float[nxyz];
            for (int id=0;id<nmask;id++) {
                //vecImage[v+2*nv][map[id]] = lambdaImage[v][id];
                vecImage[v+2*nv][map[id]] = jointDepth[v][id];
            }
        }*/
		
        // transform into vectors
		for (int v=0;v<nv;v++) {
            vecImage[3*v+X] = new float[nxyz];
            vecImage[3*v+Y] = new float[nxyz];
            vecImage[3*v+Z] = new float[nxyz];
            for (int id=0;id<nmask;id++) {
                double phi = FastMath.atan2(thickness, jointDepth[v][id]-offset);
                vecImage[3*v+X][map[id]] = (float)(-jointProba[v][id]*FastMath.cos(jointTheta[v][id])*FastMath.cos(phi) );
                vecImage[3*v+Y][map[id]] = (float)( jointProba[v][id]*FastMath.sin(jointTheta[v][id])*FastMath.cos(phi) );
                vecImage[3*v+Z][map[id]] = (float)( jointProba[v][id]*FastMath.sin(phi) );
            }
        }
		
		return;
    }
}
