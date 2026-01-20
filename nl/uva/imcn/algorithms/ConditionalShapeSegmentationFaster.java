package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import nl.uva.imcn.structures.*;
import nl.uva.imcn.libraries.*;
import nl.uva.imcn.methods.*;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.*;

//import Jama.*;

/*
 * @author Pierre-Louis Bazin
 */
public class ConditionalShapeSegmentationFaster {

	// data buffers
	private float[][][] lvlImages;
	private float[][][] sklImages;
	private float[][][] intensImages;
	private float[][] mapImages = null;
	
	private float[][] targetImages;
	
	private double[][][] condmean = null;
	private double[][][] condstdv = null;
	
	private boolean[] mask = null;;
	
	private int nsub;
	private int nobj;
	private int nc;
	private int nbest = 16;
	//private int nbest = 24;
	private int nskel = 4;
	private byte nbg = 1;
	private int ntc;
	
	private float deltaIn = 1.0f; // this parameter might have a big effect
	private float deltaOut = 0.0f;
	private float boundary = 10.0f; // tricky: it should be large enough not to crop enlarged structures
	// these have been tested: could be hard-coded
	private boolean modelBackground = true;
	private boolean cancelBackground = false;
	private boolean cancelAll = false;
	private boolean sumPosterior = false;
	private boolean maxPosterior = true;
	private int maxiter = 100;
	private float maxdiff = 0.1f;
	//private boolean topoParam = true;
	//private     String	            lutdir = null;
	// these have been tested: could be hard-coded
	private double top = 95.0;
	private boolean rescaleProbas = true;
	private boolean rescaleIntensities = true;
	private boolean modelHistogram = true;
	private boolean rescaleHistograms = true;
	private boolean prelogHistogram = false;
	private boolean postlogHistogram = false;
	
	// more things to tune? small & variable structures seem to vanish a bit fast
	private boolean scalePriors = true;
	private boolean shiftPriors = false;
	
	// trust more the intensities when more are available? yes
	//private boolean averageIntensityPriors = true;
	
	// weight up or down intensities compared to spatial priors?
	private float intensityImportance = 1.0f;
	private float intensityBaseline = 0.1f;
	
	// same thing for the volume priors
	private float volumeImportance = 1.0f;

	// use only a subset of intensities per structure?
	private boolean[][] contrastList = null;
	
	private final float INF = 1e9f;
	private final float ISQRT2 = (float)(1.0/FastMath.sqrt(2.0));
	private final float ISQRT3 = (float)(1.0/FastMath.sqrt(2.0));
	
	private final int connectivity = 26;
	
	// possibly to extend to entire distribution?
	// model size: nbins x nc x nobj x nobj
	// benefits: no probability computation, no a priori model
	private double[][][][] condhistogram = null;
	private double[][][] condmin = null;
	private double[][][] condmax = null;
	private int nbins=200;
	private float histogramSpread = 1.0f;
	
	private float[][] avgAtlasImages;
	private float[][] avgTargetImages;
	
	private double[][][][] trgcondhistogram = null;
	private double[] trgcondmin = null;
	private double[] trgcondmax = null;
	
	private boolean[][][] condpair = null;
	
	private float[] medstdv = null;
	
	private int[][]        spatialLabels;
	private float[][]      spatialProbas;
	
	private int[][]        skeletonLabels;
	private float[][]      skeletonProbas;
	
	private int[][]        intensityLabels;
	private float[][]      intensityProbas;
	
	private int[][]        combinedLabels;
	private float[][]      combinedProbas;
	
	private int[][]        jointLabels;
	private float[][]      jointProbas;
	
	private int[][]        diffusedLabels;
	private float[][]      diffusedProbas;
	
	private int[]          finalLabel;
	private float[]        finalProba;
	
	private float[][]      ngbw;
	private int[]          idmap;
	
	private float[]        logVolMean;
	private float[]        logVolStdv;
	
	private float[][]        logVolMean2;
	private float[][]        logVolStdv2;

	private int[][]        origLabels;
	private float[][]      origProbas;
	private int[][]        targetLabels;
	private float[][]      targetProbas;
	/* not used
	private double[]        boundaryDev;
	private double[]        objAvg;
	private double[]        objDev;
	*/
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;
	private int ndata;

	// in case the atlas and target spaces are different
	private float[]    map2atlas = null;
	private float[]    map2target = null;
	
	private int nax, nay, naz, naxyz;
	private float rax, ray, raz;

	private int ntx, nty, ntz, ntxyz;
	private float rtx, rty, rtz;

	public final void setNumberOfSubjectsObjectsBgAndContrasts(int sub,int obj,int bg, int cnt) {
	    nsub = sub;
	    nobj = obj;
	    nbg = (byte)bg;
	    nc = cnt;
	    lvlImages = new float[nsub][nobj][];
	    sklImages = new float[nsub][nobj][];
	    intensImages = new float[nsub][nc][];
	    targetImages = new float[nc][];
	}
	public final void setLevelsetImageAt(int sub, int obj, float[] val) { lvlImages[sub][obj] = val; 
	    //System.out.println("levelset ("+sub+", "+obj+") = "+lvlImages[sub][obj][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setSkeletonImageAt(int sub, int obj, float[] val) { sklImages[sub][obj] = val; 
	    //System.out.println("levelset ("+sub+", "+obj+") = "+lvlImages[sub][obj][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setContrastImageAt(int sub, int cnt, float[] val) { intensImages[sub][cnt] = val; 
	    //System.out.println("contrast ("+sub+", "+cnt+") = "+intensImages[sub][cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setMappingImageAt(int sub, float[] val) { 
	    if (mapImages==null) mapImages = new float[nsub][];
	    mapImages[sub] = val; 
	    //System.out.println("contrast ("+sub+", "+cnt+") = "+intensImages[sub][cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setTargetImageAt(int cnt, float[] val) { targetImages[cnt] = val; 
	    //System.out.println("target ("+cnt+") = "+targetImages[cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setMappingToTarget(float[] val) { map2target = val; 
	    //System.out.println("target ("+cnt+") = "+targetImages[cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setMappingToAtlas(float[] val) { map2atlas = val; 
	    //System.out.println("target ("+cnt+") = "+targetImages[cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setNumberOfTargetContrasts(int cnt) {
	    ntc = cnt;
	    avgAtlasImages = new float[nc][];
	    avgTargetImages = new float[ntc][];
	}
	public final void setAvgAtlasImageAt(int cnt, float[] val) { avgAtlasImages[cnt] = val; 
	    //System.out.println("target ("+cnt+") = "+targetImages[cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setAvgTargetImageAt(int cnt, float[] val) { avgTargetImages[cnt] = val; 
	    //System.out.println("target ("+cnt+") = "+targetImages[cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void initOrigTargetLabelings(int val) {
	    nbest = val;
	    origLabels = new int[nbest][];
	    origProbas = new float[nbest][];
	    targetLabels = new int[nbest][];
	    targetProbas = new float[nbest][];
	}
	public final void setOrigLabelsAt(int cnt, int[] val) { origLabels[cnt] = val; 
	    //System.out.println("target ("+cnt+") = "+targetImages[cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setOrigProbasAt(int cnt, float[] val) { origProbas[cnt] = val; 
	    //System.out.println("target ("+cnt+") = "+targetImages[cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setTargetLabelsAt(int cnt, int[] val) { targetLabels[cnt] = val; 
	    //System.out.println("target ("+cnt+") = "+targetImages[cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setTargetProbasAt(int cnt, float[] val) { targetProbas[cnt] = val; 
	    //System.out.println("target ("+cnt+") = "+targetImages[cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setShapeAtlasProbasAndLabels(float[] pval, int[] lval) {
	    // first estimate ndata
	    ndata = 0;
	    System.out.println("atlas size: "+nax+" x "+nay+" x "+naz+" ("+naxyz+")");
	    
	    if (map2target!=null) {
	        System.out.println("image size: "+ntx+" x "+nty+" x "+ntz+" ("+ntxyz+")");
            mask = new boolean[ntxyz]; 
            for (int xt=0;xt<ntx;xt++) for (int yt=0;yt<nty;yt++) for (int zt=0;zt<ntz;zt++) {
                int idx = xt + ntx*yt + ntx*nty*zt;
                int xa = Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nax-1);
                int ya = Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,nay-1);
                int za = Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,naz-1);
                //int xyz =       Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nx-1)
                //        +    nx*Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,ny-1)
                //        + nx*ny*Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,nz-1);
                int xyz = xa + nax*ya + nax*nay*za;
                if (lval[xyz]>0) {
                    ndata++;
                    mask[idx] = true;
                } else {
                    mask[idx] = false;
                }
            }
            System.out.println("work region size: "+ndata);
            // only keep 6-connected component to avoid border effects from coordinate mappings
            mask = ObjectLabeling.largestObject(mask, ntx, nty, ntz, 6); 
            ndata = 0;
            for (int idx=0;idx<ntxyz;idx++) if (mask[idx]) ndata++;
            
            System.out.println("adjusted work region size: "+ndata);
            // build ID map
            idmap = new int[ntxyz];
            int id = 0;
            for (int idx=0;idx<ntxyz;idx++) if (mask[idx]) {
                idmap[idx] = id;
                id++;
            }
            System.out.println("id map size: "+id);
            // pass the probabilities USING the idmap
            spatialProbas = new float[nbest][ndata];
            spatialLabels = new int[nbest][ndata];
            for (int xt=0;xt<ntx;xt++) for (int yt=0;yt<nty;yt++) for (int zt=0;zt<ntz;zt++) {
                int idx = xt + ntx*yt + ntx*nty*zt;
                if (mask[idx]) {
                    int xa = Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nax-1);
                    int ya = Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,nay-1);
                    int za = Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,naz-1);
                    //int xyz =       Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nx-1)
                    //        +    nx*Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,ny-1)
                    //        + nx*ny*Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,nz-1);
                    int xyz = xa + nax*ya + nax*nay*za;

                    for (int best=0;best<nbest;best++) {
                        spatialProbas[best][idmap[idx]] = pval[xyz+best*naxyz];
                        spatialLabels[best][idmap[idx]] = lval[xyz+best*naxyz];
                    }
                }
            }
            // compute the structure-specific volume scaling factors
            // problem: if volume of average is null (low overlap + small structures), result is likely incorrect
            /*float[] atlasVol = new float[nobj];
            for (int xyz=0;xyz<naxyz;xyz++) if (lval[xyz]>0) {
                int obj = Numerics.floor(lval[xyz]/100.0f);
                atlasVol[obj-nbg] += rax*ray*raz;
            }
            float[] mappedVol = new float[nobj];
            for (int idx=0;idx<ntxyz;idx++) if (mask[idx]) {
                int obj = Numerics.floor(spatialLabels[0][idmap[idx]]/100.0f);
                mappedVol[obj-nbg] += rtx*rty*rtz;
            }*/
            // use a region growing approach instead...
            int[] atlasLabels = atlasVolumeLabels(pval, lval);
            float[] atlasVol = new float[nobj];
            for (int xyz=0;xyz<naxyz;xyz++) if (atlasLabels[xyz]>0) {
                atlasVol[atlasLabels[xyz]] += rax*ray*raz;
            }
            System.out.print("atlas volumes: ");
            for (int obj=0;obj<nobj;obj++) System.out.print(obj+": "+(atlasVol[obj])+", ");
            System.out.println("(a)");
            
            float[] mappedVol = new float[nobj];
            for (int xt=0;xt<ntx;xt++) for (int yt=0;yt<nty;yt++) for (int zt=0;zt<ntz;zt++) {
                int idx = xt + ntx*yt + ntx*nty*zt;
                if (mask[idx]) {
                    int xa = Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nax-1);
                    int ya = Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,nay-1);
                    int za = Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,naz-1);
                    //int xyz =       Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nx-1)
                    //        +    nx*Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,ny-1)
                    //        + nx*ny*Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,nz-1);
                    int xyz = xa + nax*ya + nax*nay*za;

                    if (atlasLabels[xyz]>0) {
                        mappedVol[atlasLabels[xyz]] += rtx*rty*rtz;
                    }
                }
            }
            System.out.print("mapped volumes: ");
            for (int obj=0;obj<nobj;obj++) System.out.print(obj+": "+(mappedVol[obj])+", ");
            System.out.println("(m)");
            
            System.out.print("volume ratios: ");
            for (int obj=0;obj<nobj;obj++) System.out.print(obj+": "+(mappedVol[obj]/atlasVol[obj])+", ");
            System.out.println("(m/a)");
            for (int obj=0;obj<nobj;obj++) {
                //objVolumeMean[obj] *= mappedVol[obj]/atlasVol[obj];
                //objVolumeStdv[obj] *= mappedVol[obj]/atlasVol[obj];
                logVolMean[obj] += FastMath.log(mappedVol[obj]) - FastMath.log(atlasVol[obj]);
                // only applies to the mean, the variance is not affected (additive factor in log space)
                //logVolStdv[obj] += FastMath.log(mappedVol[obj]) - FastMath.log(atlasVol[obj]);
            }
            
            // adjust the volume boundary priors too...
            int[] atlasBoundaryLabels = atlasBoundaryLabels(pval, lval);
            float[][] atlasVol2 = new float[nobj][nobj];
            for (int xyz=0;xyz<naxyz;xyz++) if (atlasBoundaryLabels[xyz]>0) {
                int obj1 = Numerics.floor(atlasBoundaryLabels[xyz]/100)-1;
                int obj2 = atlasBoundaryLabels[xyz] - 100*(obj1+1) - 1;
                
                atlasVol2[obj1][obj2] += rax*ray*raz;
            }
            /*
            System.out.print("atlas boudnary volumes: ");
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) 
                System.out.print(obj1+" | "+obj2+" : "+(atlasVol2[obj1][obj2])+", ");
            System.out.println("(a)");
            */
            float[][] mappedVol2 = new float[nobj][nobj];
            for (int xt=0;xt<ntx;xt++) for (int yt=0;yt<nty;yt++) for (int zt=0;zt<ntz;zt++) {
                int idx = xt + ntx*yt + ntx*nty*zt;
                if (mask[idx]) {
                    int xa = Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nax-1);
                    int ya = Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,nay-1);
                    int za = Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,naz-1);
                    //int xyz =       Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nx-1)
                    //        +    nx*Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,ny-1)
                    //        + nx*ny*Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,nz-1);
                    int xyz = xa + nax*ya + nax*nay*za;

                    if (atlasBoundaryLabels[xyz]>0) {
                        int obj1 = Numerics.floor(atlasBoundaryLabels[xyz]/100)-1;
                        int obj2 = atlasBoundaryLabels[xyz] - 100*(obj1+1) - 1;
                        
                        mappedVol2[obj1][obj2] += rtx*rty*rtz;
                    }
                }
            }
            /*
            System.out.print("mapped volumes: ");
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) 
                System.out.print(obj1+" | "+obj2+" : "+(mappedVol2[obj1][obj2])+", ");
            System.out.println("(m)");
            
            System.out.print("volume ratios: ");
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                if (mappedVol2[obj1][obj2]>0 && atlasVol2[obj1][obj2]>0) {
                    System.out.print(obj1+" | "+obj2+" : "+(mappedVol2[obj1][obj2]/atlasVol2[obj1][obj2])+", ");
                } else {
                    System.out.print(obj1+" | "+obj2+" : n/a, ");
                }
            }
            System.out.println("(m/a)");
            */
            // adjust the volume priors
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                //objVolumeMean[obj] *= mappedVol[obj]/atlasVol[obj];
                //objVolumeStdv[obj] *= mappedVol[obj]/atlasVol[obj];
                logVolMean2[obj1][obj2] += FastMath.log(Numerics.max(1.0,mappedVol2[obj1][obj2])) - FastMath.log(Numerics.max(1.0,atlasVol2[obj1][obj2]));
                // only applies to the mean, the variance is not affected (additive factor in log space)
                //logVolStdv2[obj1][obj2] += FastMath.log(Numerics.max(1.0,mappedVol2[obj1][obj2])) - FastMath.log(Numerics.max(1.0,atlasVol2[obj1][obj2]));
            }
            // reset all indices to subject space
            nx = ntx; ny = nty; nz = ntz; nxyz = ntxyz;
            rx = rtx; ry = rty; rz = rtz;
            map2atlas = null;
	    } else {
            mask = new boolean[naxyz]; 
            for (int xyz=0;xyz<naxyz;xyz++) {
                if (lval[xyz]>0) {
                    mask[xyz] = true;
                    ndata++;
                } else {
                    mask[xyz] = false;
                }
            }
            System.out.println("work region size: "+ndata);
            // build ID map
            idmap = new int[naxyz];
            int id = 0;
            for (int xyz=0;xyz<naxyz;xyz++) if (mask[xyz]) {
                idmap[xyz] = id;
                id++;
            }
            // pass the probabilities USING the idmap
            spatialProbas = new float[nbest][ndata];
            spatialLabels = new int[nbest][ndata];
            for (int xyz=0;xyz<naxyz;xyz++) if (mask[xyz]) {
                for (int best=0;best<nbest;best++) {
                    spatialProbas[best][idmap[xyz]] = pval[xyz+best*naxyz];
                    spatialLabels[best][idmap[xyz]] = lval[xyz+best*naxyz];
                }
            }
            nx = nax; ny = nay; nz = naz; nxyz = naxyz;
            rx = rax; ry = ray; rz = raz;
        }
	}
	
	public final void setSkeletonAtlasProbasAndLabels(float[] pval, int[] lval) {
	    // first estimate ndata
	    System.out.println("atlas size: "+nax+" x "+nay+" x "+naz+" ("+naxyz+")");
	    
	    if (map2target!=null) {
	        System.out.println("image size: "+ntx+" x "+nty+" x "+ntz+" ("+ntxyz+")");
            System.out.println("work region size: "+ndata);
            // pass the probabilities USING the idmap
            skeletonProbas = new float[nskel][ndata];
            skeletonLabels = new int[nskel][ndata];
            for (int xt=0;xt<ntx;xt++) for (int yt=0;yt<nty;yt++) for (int zt=0;zt<ntz;zt++) {
                int idx = xt + ntx*yt + ntx*nty*zt;
                if (mask[idx]) {
                    int xa = Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nax-1);
                    int ya = Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,nay-1);
                    int za = Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,naz-1);
                    //int xyz =       Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nx-1)
                    //        +    nx*Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,ny-1)
                    //        + nx*ny*Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,nz-1);
                    int xyz = xa + nax*ya + nax*nay*za;

                    for (int best=0;best<nskel;best++) {
                        skeletonProbas[best][idmap[idx]] = pval[xyz+best*naxyz];
                        skeletonLabels[best][idmap[idx]] = lval[xyz+best*naxyz];
                    }
                }
            }
	    } else {
            System.out.println("work region size: "+ndata);
            // pass the probabilities USING the idmap
            skeletonProbas = new float[nskel][ndata];
            skeletonLabels = new int[nskel][ndata];
            for (int xyz=0;xyz<naxyz;xyz++) if (mask[xyz]) {
                for (int best=0;best<nskel;best++) {
                    skeletonProbas[best][idmap[xyz]] = pval[xyz+best*naxyz];
                    skeletonLabels[best][idmap[xyz]] = lval[xyz+best*naxyz];
                }
            }
        }
	}
	public final void setConditionalMeanAndStdv(float[] mean, float[] stdv) {
	    condmean = new double[nc][nobj][nobj];
	    condstdv = new double[nc][nobj][nobj];
  		condpair = new boolean[nc][nobj][nobj];
        for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) for (int c=0;c<nc;c++) {
	        condmean[c][obj1][obj2] = mean[obj1+obj2*nobj+c*nobj*nobj];
	        condstdv[c][obj1][obj2] = mean[obj1+obj2*nobj+c*nobj*nobj];
	        if (condstdv[c][obj1][obj2]>0) condpair[c][obj1][obj2] = true;
	        else condpair[c][obj1][obj2] = false;
	    }
	}

	public final void setConditionalHistogram(float[] val) {
	    //nbins = n;
	    condhistogram = new double[nc][nobj][nobj][nbins];
	    condmin = new double[nc][nobj][nobj];
	    condmax = new double[nc][nobj][nobj];
		condpair = new boolean[nc][nobj][nobj];
		logVolMean = new float[nobj];
		logVolStdv = new float[nobj];
	    logVolMean2 = new float[nobj][nobj];
		logVolStdv2 = new float[nobj][nobj];
	    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) for (int c=0;c<nc;c++) {
	        condpair[c][obj1][obj2] = false;
	        condmin[c][obj1][obj2] = val[obj2+obj1*nobj+nobj*nobj*0+nobj*nobj*(nbins+6)*c];
	        for (int bin=0;bin<nbins;bin++) {
	            condhistogram[c][obj1][obj2][bin] = val[obj2+obj1*nobj+nobj*nobj*(bin+1)+nobj*nobj*(nbins+6)*c];
	            if (condhistogram[c][obj1][obj2][bin]>0) condpair[c][obj1][obj2] = true;
	        }
	        condmax[c][obj1][obj2] = val[obj2+obj1*nobj+nobj*nobj*(nbins+1)+nobj*nobj*(nbins+6)*c];
	        if (obj1==obj2) {
	            logVolMean[obj1] = val[obj2+obj1*nobj+nobj*nobj*(nbins+2)+nobj*nobj*(nbins+6)*c];
	            logVolStdv[obj1] = val[obj2+obj1*nobj+nobj*nobj*(nbins+3)+nobj*nobj*(nbins+6)*c];
	        }
            logVolMean2[obj1][obj2] = val[obj2+obj1*nobj+nobj*nobj*(nbins+4)+nobj*nobj*(nbins+6)*c];
            logVolStdv2[obj1][obj2] = val[obj2+obj1*nobj+nobj*nobj*(nbins+5)+nobj*nobj*(nbins+6)*c];
	    }
	}
	
	public final void setContrastList(int[] cnt) {
	    contrastList = new boolean[nobj-nbg][nc];
	    for (int i=0;i<nobj-nbg;i++) for (int c=0;c<nc;c++) {
	        contrastList[i][c] = (cnt[i*nc+c]>0);
	    }
	}
	public final void setOptions(boolean mB, boolean cB, boolean cA, boolean sP, boolean mP) {
	    modelBackground = mB;
	    cancelBackground = cB;
	    cancelAll = cA;
	    sumPosterior = sP;
	    maxPosterior = mP;
	    if (modelBackground) nobj = nobj+nbg;
	}
	
	public final void setDiffusionParameters(int iter, float diff) {
	    maxiter = iter;
	    maxdiff = diff;
	}
	
	public final void setHistogramModeling(boolean val) {
	    modelHistogram = val;
	}
	
	public final void setHistogramSmoothing(float val) {
	    histogramSpread = val;
	}
	
	//public static final void setFollowSkeleton(boolean val) { skelParam=val; }
	//public final void setCorrectSkeletonTopology(boolean val) { topoParam=val; }
	//public final void setTopologyLUTdirectory(String val) { lutdir = val; }
    //public final void setAverageIntensityPriors(boolean val) { averageIntensityPriors=val; }

    public final void setIntensityImportancePrior(float val) { intensityImportance=val; }
    public final void setIntensityBaselinePrior(float val) { intensityBaseline=val; }

    public final void setVolumeImportancePrior(float val) { volumeImportance=val; }

    //
	public final void setAtlasDimensions(int x, int y, int z) { nax=x; nay=y; naz=z; naxyz=nax*nay*naz; }
	public final void setAtlasDimensions(int[] dim) { nax=dim[0]; nay=dim[1]; naz=dim[2]; naxyz=nax*nay*naz; }
	
	public final void setAtlasResolutions(float x, float y, float z) { rax=x; ray=y; raz=z; }
	public final void setAtlasResolutions(float[] res) { rax=res[0]; ray=res[1]; raz=res[2]; }
	
	public final void setTargetDimensions(int x, int y, int z) { ntx=x; nty=y; ntz=z; ntxyz=ntx*nty*ntz; }
	public final void setTargetDimensions(int[] dim) { ntx=dim[0]; nty=dim[1]; ntz=dim[2]; ntxyz=ntx*nty*ntz; }
	
	public final void setTargetResolutions(float x, float y, float z) { rtx=x; rty=y; rtz=z; }
	public final void setTargetResolutions(float[] res) { rtx=res[0]; rty=res[1]; rtz=res[2]; }
	
	// to be used for JIST definitions, generic info / help
	public final String getPackage() { return "IMCN Toolkit"; }
	public final String getCategory() { return "Segmentation"; }
	public final String getLabel() { return "Conditional Shape Segmentation"; }
	public final String getName() { return "ConditionalShapeSegmentation"; }

	public final String[] getAlgorithmAuthors() { return new String[]{"Pierre-Louis Bazin"}; }
	public final String getAffiliation() { return "Integrative Model-basec Cognitve Neuroscience unit, Universiteit van Amsterdam | Max Planck Institute for Human Cognitive and Brain Sciences"; }
	public final String getDescription() { return "Combines a collection of levelset surfaces and intensity maps into a condfitional segmentation"; }
	public final String getLongDescription() { return getDescription(); }
		
	public final String getVersion() { return "1.0"; };

	// create outputs
	public final int getBestDimension() { return nbest; }
	
	public final float[] getBestSpatialProbabilityMaps(int nval) {
	    nval = Numerics.min(nval,nbest);
        float[] images = new float[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = spatialProbas[best][idmap[xyz]];
            }
        }
        return images;
	}
	public final int[] getBestSpatialProbabilityLabels(int nval) {
        nval = Numerics.min(nval,nbest);
        int[] images = new int[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = spatialLabels[best][idmap[xyz]];
            }
        }
        return images;
	}

	public final float[] getBestSkeletonProbabilityMaps(int nval) {
	    nval = Numerics.min(nval,nbest);
        float[] images = new float[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = skeletonProbas[best][idmap[xyz]];
            }
        }
        return images;
	}
	public final int[] getBestSkeletonProbabilityLabels(int nval) {
        nval = Numerics.min(nval,nbest);
        int[] images = new int[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = skeletonLabels[best][idmap[xyz]];
            }
        }
        return images;
	}

	public final float[] getBestIntensityProbabilityMaps(int nval) {
        nval = Numerics.min(nval,nbest);
        float[] images = new float[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = intensityProbas[best][idmap[xyz]];
            }
        }
        return images;
	}
	public final int[] getBestIntensityProbabilityLabels(int nval) {
        nval = Numerics.min(nval,nbest);
        int[] images = new int[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = intensityLabels[best][idmap[xyz]];
            }
        }
        return images;
	}

    public final float[] getBestProbabilityMaps(int nval) { 
        nval = Numerics.min(nval,nbest);
        float[] images = new float[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = combinedProbas[best][idmap[xyz]];
            }
        }
        return images;
	}
	public final int[] getBestProbabilityLabels(int nval) { 
        nval = Numerics.min(nval,nbest);
        int[] images = new int[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = combinedLabels[best][idmap[xyz]];
            }
        }
        return images;
	}
	
    public final float[] getJointProbabilityMaps(int nval) { 
        nval = Numerics.min(nval,nbest);
        float[] images = new float[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = jointProbas[best][idmap[xyz]];
            }
        }
        return images;
	}
	public final int[] getJointProbabilityLabels(int nval) { 
        nval = Numerics.min(nval,nbest);
        int[] images = new int[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = jointLabels[best][idmap[xyz]];
            }
        }
        return images;
	}
	
	public final float[] getCertaintyProbability() { 
        float[] images = new float[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    images[xyz] = combinedProbas[0][idmap[xyz]]-combinedProbas[1][idmap[xyz]];
        }
        return images;
	}
    public final float[] getNeighborhoodMaps(int nval) { 
        float[] images = new float[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = ngbw[best][idmap[xyz]];
            }
        }
        return images;
	}
	public final int[] getFinalLabel() { return finalLabel; }
	
	public final float[] getFinalProba() { return finalProba; }
	
	public final float[] getConditionalMean() {
	    float[] val = new float[nc*nobj*nobj];
	    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) for (int c=0;c<nc;c++) {
	        val[obj1+obj2*nobj+c*nobj*nobj] = (float)condmean[c][obj1][obj2];
	    }
	    return val;
	}
	
	public final float[] getConditionalStdv() {
	    float[] val = new float[nc*nobj*nobj];
	    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) for (int c=0;c<nc;c++) {
	        val[obj1+obj2*nobj+c*nobj*nobj] = (float)condstdv[c][obj1][obj2];
	    }
	    return val;
	}
	
	public final float[] getConditionalHistogram() {
	    float[] val = new float[nc*nobj*nobj*(nbins+6)];
	    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) for (int c=0;c<nc;c++) {
	        val[obj2+obj1*nobj+0*nobj*nobj+c*nobj*nobj*(nbins+6)] = (float)condmin[c][obj1][obj2];
	        for (int bin=0;bin<nbins;bin++) {
	            val[obj2+obj1*nobj+(bin+1)*nobj*nobj+c*nobj*nobj*(nbins+6)] = (float)condhistogram[c][obj1][obj2][bin];
	        }
	        val[obj2+obj1*nobj+(nbins+1)*nobj*nobj+c*nobj*nobj*(nbins+6)] = (float)condmax[c][obj1][obj2];
	        if (obj1==obj2) {
	            val[obj2+obj1*nobj+(nbins+2)*nobj*nobj+c*nobj*nobj*(nbins+6)] = (float)logVolMean[obj1];
	            val[obj2+obj1*nobj+(nbins+3)*nobj*nobj+c*nobj*nobj*(nbins+6)] = (float)logVolStdv[obj1];
	        }
            val[obj2+obj1*nobj+(nbins+4)*nobj*nobj+c*nobj*nobj*(nbins+6)] = (float)logVolMean2[obj1][obj2];
            val[obj2+obj1*nobj+(nbins+5)*nobj*nobj+c*nobj*nobj*(nbins+6)] = (float)logVolStdv2[obj1][obj2];
	    }
	    return val;
	}
	
	public final float[] getTargetConditionalHistogram() {
	    float[] val = new float[ntc*nobj*nobj*(nbins+6)];
	    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) for (int tc=0;tc<ntc;tc++) {
	        val[obj2+obj1*nobj+0*nobj*nobj+tc*nobj*nobj*(nbins+6)] = (float)trgcondmin[tc];
	        for (int bin=0;bin<nbins;bin++) {
	            val[obj2+obj1*nobj+(bin+1)*nobj*nobj+tc*nobj*nobj*(nbins+6)] = (float)trgcondhistogram[tc][obj1][obj2][bin];
	        }
	        val[obj2+obj1*nobj+(nbins+1)*nobj*nobj+tc*nobj*nobj*(nbins+6)] = (float)trgcondmax[tc];
	        if (obj1==obj2) {
	            val[obj2+obj1*nobj+(nbins+2)*nobj*nobj+tc*nobj*nobj*(nbins+6)] = (float)logVolMean[obj1];
	            val[obj2+obj1*nobj+(nbins+3)*nobj*nobj+tc*nobj*nobj*(nbins+6)] = (float)logVolStdv[obj1];
	        }
            val[obj2+obj1*nobj+(nbins+4)*nobj*nobj+tc*nobj*nobj*(nbins+6)] = (float)logVolMean2[obj1][obj2];
            val[obj2+obj1*nobj+(nbins+5)*nobj*nobj+tc*nobj*nobj*(nbins+6)] = (float)logVolStdv2[obj1][obj2];
	    }
	    return val;
	}
	
	public final int getNumberOfBins() {
	    return nbins;
	}
	
	public final float computeAtlasPriors() {
	    nx = nax; ny = nay; nz = naz; nxyz = naxyz;
	    rx = rax; ry = ray; rz = raz;
	    
	    float[][][] levelsets = null; 
	    
	    // not correct: explicitly build the levelset of the background first, then crop it
	    if (modelBackground) {
            // adding the background: building a ring around the structures of interest
            // with also a sharp decay to the boundary
            levelsets = new float[nsub][nobj][];
            float[] background = new float[nxyz];
            boolean[] bgmask = new boolean[nxyz];
            for (int xyz=0;xyz<nxyz;xyz++) bgmask[xyz] = true;
            boundary = boundary/Numerics.max(rx,ry,rz);
            
            for (int sub=0;sub<nsub;sub++) {
                for (int xyz=0;xyz<nxyz;xyz++) {
                    float mindist = boundary;
                    for (int obj=0;obj<nobj-nbg;obj++) {
                        if (lvlImages[sub][obj][xyz]<mindist) mindist = lvlImages[sub][obj][xyz];
                    }
                    /*
                    if (mindist<boundary) {
                        background[xyz] = -mindist;
                    } else {
                        background[xyz] = mindist;
                    }*/
                    background[xyz] = -mindist;
                }
                InflateGdm gdm = new InflateGdm(background, nx, ny, nz, rx, ry, rz, bgmask, 0.4f, 0.4f, "no", null);
                gdm.evolveNarrowBand(0, 1.0f);
                 // separate background into main tissues?
                if (nbg>1) {
                    background = gdm.getLevelSet();
                    FuzzyCmeans fcm = new FuzzyCmeans();
                    fcm.setDimensions(nx, ny, nz);
                    fcm.setImage(intensImages[sub][0]);
                    int[] roi = new int[nxyz];
                    int roisize=0;
                    boolean[] outside = new boolean[nxyz];
                    for (int xyz=0;xyz<nxyz;xyz++) {
                        if (background[xyz]<=0 && background[xyz]>-boundary) {
                            roi[xyz] = 1;
                            roisize++;
                        } else 
                        if (background[xyz]<=-boundary) {
                            outside[xyz] = true;
                        }
                    }
                    System.out.print("bg size: "+roisize+"\n");
                    fcm.setMaskImage(roi);
                    fcm.maskImageNaNs();
                    fcm.setClusterNumber(nbg);
                    fcm.setSmoothing(0.001f);
                    fcm.setMaxDist(0.001f);
                    fcm.setMaxIter(150);
                    // hard-coded prior for R1 maps... not great
                    /* remove?
                    if (nbg==2) {
                        float[] t1prior = {0.25f, 0.75f};
                        fcm.setInitCentroids(t1prior);
                        fcm.setMaxIter(0);
                    } else
                    if (nbg==3) {
                        float[] t1prior = {0.25f, 0.75f, 1.25f};
                        fcm.setInitCentroids(t1prior);
                        fcm.setMaxIter(0);
                    }
                    */
                    fcm.execute();
                    int[] classif = fcm.getClassification();
                    for (int n=0;n<nbg;n++) {
                        background = fcm.getMembership(n);
                        for (int xyz=0;xyz<nxyz;xyz++) {
                            if (outside[xyz]) {
                                background[xyz] = -1.0f;
                            } else if (classif[xyz]==n+1) {
                                background[xyz] = - background[xyz];
                            } else {
                                background[xyz] = 1.0f - background[xyz];
                            }
                        }
                        gdm = new InflateGdm(background, nx, ny, nz, rx, ry, rz, bgmask, 0.4f, 0.4f, "no", null);
                        gdm.evolveNarrowBand(0, 1.0f);
                        levelsets[sub][n] = gdm.getLevelSet();
                    }
                } else {
                    levelsets[sub][0] = gdm.getLevelSet();
                }
                for (int obj=nbg;obj<nobj;obj++) {
                    levelsets[sub][obj] = lvlImages[sub][obj-nbg];
                }
            }
            //nobj = nobj+1;
            lvlImages = null;
        } else {
            levelsets = lvlImages;
		}
		// mask anything too far outside the structures of interest
		mask = new boolean[nxyz];
		ndata = 0;
		for (int xyz=0;xyz<nxyz;xyz++) {
		    float mindist = boundary;
		    // skip the background label (bg must be the first label always)
            for (int sub=0;sub<nsub;sub++) for (int obj=nbg;obj<nobj;obj++) {
                if (levelsets[sub][obj][xyz]<mindist) mindist = levelsets[sub][obj][xyz];
            }
            if (mindist<boundary) {
                mask[xyz] = true;
                ndata++;
            } else {
                mask[xyz] = false;
            }
        }
        // build ID map
        idmap = new int[ntxyz];
        int id = 0;
        for (int xyz=0;xyz<ntxyz;xyz++) if (mask[xyz]) {
            idmap[xyz] = id;
            id++;
        }
        System.out.println("masking: work region "+ndata+", compression: "+(ndata/(float)nxyz));
		
        // adapt number of kept values?
        
		System.out.println("compute joint conditional shape priors");
		spatialProbas = new float[nbest][ndata]; 
		spatialLabels = new int[nbest][ndata];
		
		int ctr = Numerics.floor(nsub/2);
        int dev = Numerics.floor(nsub/4);
                    
		double[] val = new double[nsub];
		//double iqrsum=0, iqrden=0;
		double stdsum=0, stdden=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    double[][] priors = new double[nobj][nobj];
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                //priors[obj1][obj2] = FastMath.exp( -0.5*med*med/(1.349*iqr*1.349*iqr) );
                // alternative idea: use a combination of mean and stdev as distance basis
                // -> take into account uncertainty better
                double mean = 0.0;
                for (int sub=0;sub<nsub;sub++) {
                    mean += Numerics.max(0.0, levelsets[sub][obj1][xyz]-deltaOut, levelsets[sub][obj2][xyz]-deltaIn);
                }
                mean /= nsub;
                double var = 0.0;
                for (int sub=0;sub<nsub;sub++) {
                    var += Numerics.square(mean-Numerics.max(0.0, levelsets[sub][obj1][xyz]-deltaOut, levelsets[sub][obj2][xyz]-deltaIn));
                }
                var = FastMath.sqrt(var/nsub);
                
                stdsum += var;
                stdden ++;
                
                double sigma2 = var+Numerics.max(deltaOut, deltaIn, 1.0);
                sigma2 *= sigma2;
                // when scaling by the variance, it penalizes more strongly variable regions -> they get a weaker prior
                // maybe a good thing? not entirely sure...
                if (shiftPriors)
                    priors[obj1][obj2] = FastMath.exp( -0.5*Numerics.square(Numerics.max(0.0,mean-var))/sigma2 );
                else
                    priors[obj1][obj2] = FastMath.exp( -0.5*mean*mean/sigma2 );
                
                if (scalePriors)
                    priors[obj1][obj2] = 1.0/FastMath.sqrt(2.0*FastMath.PI*sigma2)*priors[obj1][obj2];
 			}
            for (int best=0;best<nbest;best++) {
                int best1=0;
				int best2=0;
					
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                    if (priors[obj1][obj2]>priors[best1][best2]) {
						best1 = obj1;
						best2 = obj2;
					}
				}
				// check if best is zero: give null label in that case
				if (priors[best1][best2]>0) {
                    // sub optimal labeling, but easy to read
                    spatialLabels[best][idmap[xyz]] = 100*(best1+1)+(best2+1);
                    spatialProbas[best][idmap[xyz]] = (float)priors[best1][best2];
                } else {
                    for (int b=best;b<nbest;b++) {
                        spatialLabels[b][idmap[xyz]] = 0;
                        spatialProbas[b][idmap[xyz]] = 0.0f;
                    }
                    best = nbest;
                }                    
                // remove best value
                priors[best1][best2] = 0.0;
 		    }
		}
		//System.out.println("mean spatial iqr: "+(iqrsum/iqrden));
		System.out.println("mean spatial stdev: "+(stdsum/stdden));
		// levelsets are now discarded...
		// not yet! 
		//levelsets = null;
		
		// rescale top % in each shape and intensity priors
		float shapeMax = 1.0f;
		if (rescaleProbas){
            Percentile measure = new Percentile();
            val = new double[ndata];
            for (id=0;id<ndata;id++) val[id] = spatialProbas[0][id];
            shapeMax = (float)measure.evaluate(val, top);
            System.out.println("top "+top+"% shape probability: "+shapeMax);
            for (id=0;id<ndata;id++) for (int best=0;best<nbest;best++) {
                spatialProbas[best][id] = (float)Numerics.min(top/100.0*spatialProbas[best][id]/shapeMax, 1.0f);
            }		
		}
		
		System.out.println("compute joint conditional intensity priors");
		
		float[][][] contrasts = intensImages;
		float[][] medc = new float[nc][ndata];
		float[][] iqrc = new float[nc][ndata];
		
		System.out.println("1. estimate subjects distribution");
		double[] cntsum = new double[nc];
		double[] cntden = new double[nc];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int c=0;c<nc;c++) {
		        int nvalid=0;
                for  (int sub=0;sub<nsub;sub++) {
                    // count the number of non-zero values
                    if (contrasts[sub][c][xyz]!=0) nvalid++;
                }
                if (nvalid>0) {
                    val = new double[nvalid];
                    int ns=0;
                    for  (int sub=0;sub<nsub;sub++) if (contrasts[sub][c][xyz]!=0) {
                        val[ns] = contrasts[sub][c][xyz];
                        ns++;
                    }
                    /*
                    //System.out.println("values");
                    Percentile measure = new Percentile();
                    measure.setData(val);
                
                    medc[c][idmap[xyz]] = (float)measure.evaluate(50.0); 
                    //System.out.println("median "+medc[c][idmap[xyz]]);
                    iqrc[c][idmap[xyz]] = (float)(measure.evaluate(75.0) - measure.evaluate(25.0));
                    //System.out.println("iqr "+iqrc[c][idmap[xyz]]);
                    */
                    Numerics.sort(val);
                    int ctrv = Numerics.floor(nvalid/2);
                    int devv = Numerics.floor(nvalid/4);
                    double med, iqr;
                    if (nvalid%2==0) {
                        med = 0.5*(val[ctrv-1]+val[ctrv]);
                        iqr = val[ctrv+devv] - val[ctrv-1-devv];
                    } else {
                        med = val[ctrv];
                        iqr = val[ctrv+devv] - val[ctrv-devv];
                    }                   
                    medc[c][idmap[xyz]] = (float)med;
                    iqrc[c][idmap[xyz]] = (float)iqr;
                    
                    cntsum[c] += iqr;
                    cntden[c]++;
                } else {
                    medc[c][idmap[xyz]] = 0.0f;
                    iqrc[c][idmap[xyz]] = 0.0f;
                }                    
            }
        }
        for (int c=0;c<nc;c++) {
            System.out.println("mean iqr (contrast "+c+"): "+(cntsum[c]/cntden[c]));
		}
		
		System.out.println("2. compute conditional maps");
		
		condpair = new boolean[nc][nobj][nobj];
		if (modelHistogram) {
		    System.out.println("(use histograms for intensities)");
		    condmin = new double[nc][nobj][nobj];
		    condmax = new double[nc][nobj][nobj];
		    condhistogram = new double[nc][nobj][nobj][nbins];
		    
		    // min, max: percentile on median (to avoid spreading the values to outliers)
		    // same min, max for all object pairs => needed for fair comparison 
		    // (or normalize by volume, i.e. taking width into account
		    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
		        System.out.print("\n("+(obj1+1)+" | "+(obj2+1)+"): ");
                for (int c=0;c<nc;c++) {
                    condmin[c][obj1][obj2] = 1e9f;
                    condmax[c][obj1][obj2] = -1e9f;
                }
                for (int c=0;c<nc;c++) {
                    boolean existsPair = false;
                    // use median intensities to estimate [min,max]
                    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                        double med = medc[c][idmap[xyz]];
                        double iqr = iqrc[c][idmap[xyz]];
                        // assuming here that iqr==0 means masked regions
                        if (iqr>0) { 
                            // look only among non-zero priors for each region
                            for (int best=0;best<nbest;best++) {
                                if (spatialLabels[best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
                                    // found value: proceeed
                                    if (med<condmin[c][obj1][obj2]) condmin[c][obj1][obj2] = med;
                                    if (med>condmax[c][obj1][obj2]) condmax[c][obj1][obj2] = med;
                                    if (condmin[c][obj1][obj2]!=condmax[c][obj1][obj2]) existsPair = true;
                                }
                            }
                        }
                    }
                    if (existsPair) {
                        condpair[c][obj1][obj2] = true;
                        System.out.print("["+condmin[c][obj1][obj2]+" , "+condmax[c][obj1][obj2]+"]    ");
                    } else {
                        condmin[c][obj1][obj2] = 0;
                        condmax[c][obj1][obj2] = 0;
                        condpair[c][obj1][obj2] = false;
                        System.out.print("empty pair    ");
                    }
                }
            }
            // take the global min,max to make histograms comparable
		    for (int c=0;c<nc;c++) {
		        double cmin = condmin[c][0][0];
		        double cmax = condmax[c][0][0];
		        for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                    if (condpair[c][obj1][obj2]) {
                        if (condmin[c][obj1][obj2]<cmin) cmin = condmin[c][obj1][obj2];
                        if (condmax[c][obj1][obj2]>cmax) cmax = condmax[c][obj1][obj2];
                    }
                }
		        for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
		            condmin[c][obj1][obj2] = cmin;
		            condmax[c][obj1][obj2] = cmax;
		        }
		    }
		    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
		        for (int c=0;c<nc;c++) {
		            if (condpair[c][obj1][obj2]) {
                        for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                            double med = medc[c][idmap[xyz]];
                            double iqr = iqrc[c][idmap[xyz]];
                            // assuming here that iqr==0 means masked regions
                            if (iqr>0) { 
                                // look for non-zero priors
                                for (int best=0;best<nbest;best++) {
                                    if (spatialLabels[best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
                                        // found value: proceeed, if non-zero
                                        for (int sub=0;sub<nsub;sub++) if (contrasts[sub][c][xyz]!=0) {
                                            // adds uncertainties from mismatch between subject intensities and mean shape
                                            /*
                                            double psub = spatialProbas[best][idmap[xyz]]*1.0/FastMath.sqrt(2.0*FastMath.PI*1.349*iqr*1.349*iqr)
                                                               *FastMath.exp( -0.5*(contrasts[sub][c][xyz]-med)*(contrasts[sub][c][xyz]-med)/(1.349*iqr*1.349*iqr) );
                                            */
                                            double ldist = Numerics.max(levelsets[sub][obj1][xyz]-deltaOut, levelsets[sub][obj2][xyz]-deltaIn, 0.0);
                                            double ldelta = Numerics.max(deltaOut, deltaIn, 1.0);
                                            double pshape = FastMath.exp(-0.5*(ldist*ldist)/(ldelta*ldelta));
                                            double psub = pshape*1.0/FastMath.sqrt(2.0*FastMath.PI*1.349*iqr*1.349*iqr)
                                                               *FastMath.exp( -0.5*(contrasts[sub][c][xyz]-med)*(contrasts[sub][c][xyz]-med)/(1.349*iqr*1.349*iqr) );
                                            // add to the mean
                                            int bin = Numerics.bounded(Numerics.ceil( (contrasts[sub][c][xyz]-condmin[c][obj1][obj2])/(condmax[c][obj1][obj2]-condmin[c][obj1][obj2])*nbins)-1, 0, nbins-1);
                                            condhistogram[c][obj1][obj2][bin] += psub;
                                        }
                                        best=nbest;
                                    }
                                }
                            }
                        }
                        // log before smoothing?
                        if (prelogHistogram) {
                            for (int bin=0;bin<nbins;bin++) {
                                condhistogram[c][obj1][obj2][bin] = FastMath.log(1.0+condhistogram[c][obj1][obj2][bin]);
                            }
                        }
                        // smooth histograms to avoid sharp edge effects
                        if (histogramSpread>0) {
                            double var = histogramSpread*histogramSpread;
                            double[] tmphist = new double[nbins];
                            for (int bin1=0;bin1<nbins;bin1++) {
                                for (int bin2=0;bin2<nbins;bin2++) {
                                    tmphist[bin1] += condhistogram[c][obj1][obj2][bin2]*FastMath.exp(-0.5*(bin1-bin2)*(bin1-bin2)/var);
                                }
                            }
                            if (postlogHistogram) {
                                for (int bin=0;bin<nbins;bin++) condhistogram[c][obj1][obj2][bin] = FastMath.log(1.0+tmphist[bin]);
                            } else {
                                for (int bin=0;bin<nbins;bin++) condhistogram[c][obj1][obj2][bin] = tmphist[bin];
                            }
                        }
                        // normalize: sum over count x spread = 1
                        double sum = 0.0;
                        for (int bin=0;bin<nbins;bin++) sum += condhistogram[c][obj1][obj2][bin];   
                        //for (int bin=0;bin<nbins;bin++) condhistogram[c][obj1][obj2][bin] /= sum*(condmax[c][obj1][obj2]-condmin[c][obj1][obj2]);   
                        for (int bin=0;bin<nbins;bin++) condhistogram[c][obj1][obj2][bin] /= sum;   
                    } else {
                        for (int bin=0;bin<nbins;bin++) condhistogram[c][obj1][obj2][bin] = 0;   
                    }
                }
            }
        } else {
            // use spatial priors and subject variability priors to define conditional intensity
            // mean and stdev
            System.out.println("(use mean,stdev for intensities)");
            condmean = new double[nc][nobj][nobj];
            condstdv = new double[nc][nobj][nobj];
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                System.out.print("\n("+(obj1+1)+" | "+(obj2+1)+"): ");
                for (int c=0;c<nc;c++) {
                   // System.out.println("..mean");
                   double sum = 0.0;
                   double den = 0.0;
                   for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                       double med = medc[c][idmap[xyz]];
                       double iqr = iqrc[c][idmap[xyz]];
                       // assuming here that iqr==0 means masked regions
                       if (iqr>0) { 
                           // look for non-zero priors
                           for (int best=0;best<nbest;best++) {
                               if (spatialLabels[best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
                                   // found value: proceeed
                                   for (int sub=0;sub<nsub;sub++) if (contrasts[sub][c][xyz]!=0) {
                                       // adds uncertainties from mismatch between subject intensities and mean shape
                                       double ldist = Numerics.max(levelsets[sub][obj1][xyz]-deltaOut, levelsets[sub][obj2][xyz]-deltaIn, 0.0);
                                       double ldelta = Numerics.max(deltaOut, deltaIn, 1.0);
                                       double pshape = FastMath.exp(-0.5*(ldist*ldist)/(ldelta*ldelta));
                                       double psub = pshape*1.0/FastMath.sqrt(2.0*FastMath.PI*1.349*iqr*1.349*iqr)
                                                          *FastMath.exp( -0.5*(contrasts[sub][c][xyz]-med)*(contrasts[sub][c][xyz]-med)/(1.349*iqr*1.349*iqr) );
                                       // add to the mean
                                       sum += psub*contrasts[sub][c][xyz];
                                       den += psub;
                                   }
                                   best=nbest;
                               }
                           }
                       }
                   }
                   // build average
                   if (den>0) {
                       condmean[c][obj1][obj2] = sum/den;
                       condpair[c][obj1][obj2] = true;
                   } else {
                       System.out.print("empty pair        ");
                       condmean[c][obj1][obj2] = 0.0;
                       condpair[c][obj1][obj2] = false;
                   }
                   //System.out.println("..stdev");
                   double var = 0.0;
                   for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                       double med = medc[c][idmap[xyz]];
                       double iqr = iqrc[c][idmap[xyz]];
                       // assuming here that iqr==0 means masked regions
                       if (iqr>0) { 
                           // look for non-zero priors
                           for (int best=0;best<nbest;best++) {
                               if (spatialLabels[best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
                                   // found value: proceeed
                                   for (int sub=0;sub<nsub;sub++) if (contrasts[sub][c][xyz]!=0) {
                                       // adds uncertainties from mismatch between subject intensities and mean shape
                                       double ldist = Numerics.max(levelsets[sub][obj1][xyz]-deltaOut, levelsets[sub][obj2][xyz]-deltaIn, 0.0);
                                       double ldelta = Numerics.max(deltaOut, deltaIn, 1.0);
                                       double pshape = FastMath.exp(-0.5*(ldist*ldist)/(ldelta*ldelta));
                                       double psub = pshape*1.0/FastMath.sqrt(2.0*FastMath.PI*1.349*iqr*1.349*iqr)
                                                           *FastMath.exp( -0.5*(contrasts[sub][c][xyz]-med)*(contrasts[sub][c][xyz]-med)/(1.349*iqr*1.349*iqr) );
                                       // add to the mean
                                       var += psub*(contrasts[sub][c][xyz]-condmean[c][obj1][obj2])*(contrasts[sub][c][xyz]-condmean[c][obj1][obj2]);
                                   }
                                   best=nbest;
                               }
                           }
                       }
                   }
                   // build stdev
                   if (var==0) {
                       System.out.print("empty region        ");
                       condstdv[c][obj1][obj2] = 0;
                       condpair[c][obj1][obj2] = false;
                   } else if (den>0) {
                       condstdv[c][obj1][obj2] = FastMath.sqrt(var/den);
                       System.out.print(condmean[c][obj1][obj2]+" +/- "+condstdv[c][obj1][obj2]+"    ");
                       condpair[c][obj1][obj2] = true;
                   } else {
                       System.out.print("empty pair        ");
                       condstdv[c][obj1][obj2] = 0;
                       condpair[c][obj1][obj2] = false;
                   } 
                }
            }
        }
        // compute volume mean, stdv of each structure
        System.out.println("Volume statistics");
        logVolMean = new float[nobj];
        logVolStdv = new float[nobj];
        for (int obj=0;obj<nobj;obj++) {
            float[] vols = new float[nsub];
            float[] bnds = new float[nsub];
            for (int sub=0;sub<nsub;sub++) {
                vols[sub] = 0.0f;
                bnds[sub] = 0.0f;
                for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                    if (levelsets[sub][obj][xyz]<0) {
                        vols[sub]+=rx*ry*rz;
                        // check if boundary
                        if (  levelsets[sub][obj][xyz+1]>=0 || levelsets[sub][obj][xyz-1]>=0
                           || levelsets[sub][obj][xyz+nx]>=0 || levelsets[sub][obj][xyz-nx]>=0
                           || levelsets[sub][obj][xyz+nx*ny]>=0 || levelsets[sub][obj][xyz-nx*ny]>=0 ) {
                            bnds[sub]+=rx*ry*rz;
                        }
                        //for (int i=-1;i<=1;i++) for (int j=-1;j<=1;j++) for (int l=-1;l<=1;l++) {
                        //    if (levelsets[sub][obj][xyz+i+j*nx+l*nx*ny]>=0) {
                        //        bnds[sub]+=rx*ry*rz;
                        //        i=2; j=2; l=2;
                        //    }
                        //}
                    }
                }
                logVolMean[obj] += FastMath.log(Numerics.max(1.0,vols[sub]))/nsub;
            }
            for (int sub=0;sub<nsub;sub++) {
                logVolStdv[obj] += Numerics.square(FastMath.log(Numerics.max(1.0,vols[sub]))-logVolMean[obj])/(nsub-1.0f);
            }
            // use the max of boundary-based and groupwise variances, to avoid variance shrinkage
            double varsub = 0.0;
            for (int sub=0;sub<nsub;sub++) {
                varsub += 0.5*Numerics.square(FastMath.log(Numerics.max(1.0,vols[sub]+0.5*bnds[sub]))-logVolMean[obj])/(nsub-1.0)
                         +0.5*Numerics.square(FastMath.log(Numerics.max(1.0,vols[sub]-0.5*bnds[sub]))-logVolMean[obj])/(nsub-1.0);
            }
            // replace the classical variance  by this one, which is always slightly bigger (accounting for partial volume uncertainty)
            //logVolStdv[obj] = (float)Numerics.max(logVolStdv[obj],varsub);
            logVolStdv[obj] = (float)varsub;
            
            logVolStdv[obj] = (float)FastMath.sqrt(logVolStdv[obj]);
            System.out.println(obj+" : "+FastMath.exp(logVolMean[obj])
                                   +" ["+FastMath.exp(logVolMean[obj]-logVolStdv[obj])
                                   +", "+FastMath.exp(logVolMean[obj]+logVolStdv[obj]));
        }        
        // same for all interfaces
        System.out.println("Conditional volume statistics");
        logVolMean2 = new float[nobj][nobj];
        logVolStdv2 = new float[nobj][nobj];
        float[][][] vols = new float[nsub][nobj][nobj];
        for (int sub=0;sub<nsub;sub++) {
            for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                double best = INF;
                int best1 = -1;
                int best2 = -1;
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                    if (Numerics.max(0.0, levelsets[sub][obj1][xyz]-deltaOut, levelsets[sub][obj2][xyz]-deltaIn)<best) {
                        best = Numerics.max(0.0, levelsets[sub][obj1][xyz]-deltaOut, levelsets[sub][obj2][xyz]-deltaIn);
                        best1 = obj1;
                        best2 = obj2;
                    }
                }
                vols[sub][best1][best2]+=rx*ry*rz;
            }
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                logVolMean2[obj1][obj2] += FastMath.log(Numerics.max(1.0,vols[sub][obj1][obj2]))/nsub;
            }
        }
        for (int sub=0;sub<nsub;sub++) {
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                logVolStdv2[obj1][obj2] += Numerics.square(FastMath.log(Numerics.max(1.0,vols[sub][obj1][obj2]))-logVolMean2[obj1][obj2])/(nsub-1.0f);
            }
        }
        for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            logVolStdv2[obj1][obj2] = (float)FastMath.sqrt(logVolStdv2[obj1][obj2]);
            if (logVolMean2[obj1][obj2]>0) {
                System.out.println(obj1+"|"+obj2+" : "+FastMath.exp(logVolMean2[obj1][obj2])
                                                 +" ["+FastMath.exp(logVolMean2[obj1][obj2]-logVolStdv2[obj1][obj2])
                                                 +", "+FastMath.exp(logVolMean2[obj1][obj2]+logVolStdv2[obj1][obj2])+"]");
            }
        }
		// at this point the atlas data is not used anymore
		levelsets = null;
		contrasts = null;
		lvlImages = null;
		intensImages = null;
		System.out.println("\ndone");
		
		return shapeMax;
	}
	
	public final void computeSkeletonPriors(float scale) {
	    
		System.out.println("compute skeleton priors");

		skeletonProbas = new float[nskel][ndata]; 
		skeletonLabels = new int[nskel][ndata];
		double stdsum=0,stdden=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    double[] priors = new double[nobj];
            for (int obj=nbg;obj<nobj;obj++) {
                //priors[obj1][obj2] = FastMath.exp( -0.5*med*med/(1.349*iqr*1.349*iqr) );
                // alternative idea: use a combination of mean and stdev as distance basis
                // -> take into account uncertainty better
                double mean = 0.0;
                for (int sub=0;sub<nsub;sub++) {
                    mean += Numerics.max(0.0, sklImages[sub][obj-nbg][xyz]);
                }
                mean /= nsub;
                double var = 0.0;
                for (int sub=0;sub<nsub;sub++) {
                    var += Numerics.square(mean-Numerics.max(0.0, sklImages[sub][obj-nbg][xyz]));
                }
                var = FastMath.sqrt(var/nsub);
                
                stdsum += var;
                stdden ++;
                
                double sigma2 = var+Numerics.max(deltaOut, deltaIn, 1.0);
                sigma2 *= sigma2;
                // when scaling by the variance, it penalizes more strongly variable regions -> they get a weaker prior
                // maybe a good thing? not entirely sure...
                if (shiftPriors)
                    priors[obj] = FastMath.exp( -0.5*Numerics.square(Numerics.max(0.0,mean-var))/sigma2 );
                else
                    priors[obj] = FastMath.exp( -0.5*mean*mean/sigma2 );
                
                if (scalePriors)
                    priors[obj] = 1.0/FastMath.sqrt(2.0*FastMath.PI*sigma2)*priors[obj];
 			}
            for (int best=0;best<nskel;best++) {
                int best1=0;
					
                for (int obj=nbg;obj<nobj;obj++) {
                    if (priors[obj]>priors[best1]) {
						best1 = obj;
					}
				}
				// check if best is zero: give null label in that case
				if (priors[best1]>0) {
                    // sub optimal labeling, but easy to read
                    skeletonLabels[best][idmap[xyz]] = 101*(best1+1);
                    skeletonProbas[best][idmap[xyz]] = (float)priors[best1];
                } else {
                    for (int b=best;b<nskel;b++) {
                        skeletonLabels[b][idmap[xyz]] = 0;
                        skeletonProbas[b][idmap[xyz]] = 0.0f;
                    }
                    best = nbest;
                }                    
                // remove best value
                priors[best1] = 0.0;
 		    }
		}
		
		// rescale top % in each shape and intensity priors
		if (rescaleProbas){
            //Percentile measure = new Percentile();
            //double[] val = new double[ndata];
            //for (int id=0;id<ndata;id++) val[id] = skeletonProbas[0][id];
            //float skelMax = (float)measure.evaluate(val, top);
            //System.out.println("top "+top+"% skeleton probability: "+skelMax);
            System.out.println("rescale with shape probability scale: "+scale);
            for (int id=0;id<ndata;id++) for (int best=0;best<nskel;best++) {
                skeletonProbas[best][id] = (float)Numerics.min(top/100.0*skeletonProbas[best][id]/scale, 1.0f);
            }		
		}
		sklImages = null;
	}

	public final void computeSkeletonTensors(float scale) {
	    
		System.out.println("compute skeleton tensors");

		skeletonProbas = new float[nskel][ndata]; 
		skeletonLabels = new int[nskel][ndata];
		double stdsum=0,stdden=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    double[] priors = new double[nobj];
            for (int obj=nbg;obj<nobj;obj++) {
                //priors[obj1][obj2] = FastMath.exp( -0.5*med*med/(1.349*iqr*1.349*iqr) );
                // alternative idea: use a combination of mean and stdev as distance basis
                // -> take into account uncertainty better
                double mean = 0.0;
                for (int sub=0;sub<nsub;sub++) {
                    mean += Numerics.max(0.0, sklImages[sub][obj-nbg][xyz]);
                }
                mean /= nsub;
                double var = 0.0;
                for (int sub=0;sub<nsub;sub++) {
                    var += Numerics.square(mean-Numerics.max(0.0, sklImages[sub][obj-nbg][xyz]));
                }
                var = FastMath.sqrt(var/nsub);
                
                stdsum += var;
                stdden ++;
                
                double sigma2 = var+Numerics.max(deltaOut, deltaIn, 1.0);
                sigma2 *= sigma2;
                // when scaling by the variance, it penalizes more strongly variable regions -> they get a weaker prior
                // maybe a good thing? not entirely sure...
                if (shiftPriors)
                    priors[obj] = FastMath.exp( -0.5*Numerics.square(Numerics.max(0.0,mean-var))/sigma2 );
                else
                    priors[obj] = FastMath.exp( -0.5*mean*mean/sigma2 );
                
                if (scalePriors)
                    priors[obj] = 1.0/FastMath.sqrt(2.0*FastMath.PI*sigma2)*priors[obj];
 			}
            for (int best=0;best<nskel;best++) {
                int best1=0;
					
                for (int obj=nbg;obj<nobj;obj++) {
                    if (priors[obj]>priors[best1]) {
						best1 = obj;
					}
				}
				// check if best is zero: give null label in that case
				if (priors[best1]>0) {
                    // sub optimal labeling, but easy to read
                    skeletonLabels[best][idmap[xyz]] = 101*(best1+1);
                    skeletonProbas[best][idmap[xyz]] = (float)priors[best1];
                } else {
                    for (int b=best;b<nskel;b++) {
                        skeletonLabels[b][idmap[xyz]] = 0;
                        skeletonProbas[b][idmap[xyz]] = 0.0f;
                    }
                    best = nbest;
                }                    
                // remove best value
                priors[best1] = 0.0;
 		    }
		}
		
		// rescale top % in each shape and intensity priors
		if (rescaleProbas){
            //Percentile measure = new Percentile();
            //double[] val = new double[ndata];
            //for (int id=0;id<ndata;id++) val[id] = skeletonProbas[0][id];
            //float skelMax = (float)measure.evaluate(val, top);
            //System.out.println("top "+top+"% skeleton probability: "+skelMax);
            System.out.println("rescale with shape probability scale: "+scale);
            for (int id=0;id<ndata;id++) for (int best=0;best<nskel;best++) {
                skeletonProbas[best][id] = (float)Numerics.min(top/100.0*skeletonProbas[best][id]/scale, 1.0f);
            }		
		}
		sklImages = null;
	}
	
	public final void estimateTarget() {
		
		// compute the median of stdevs from atlas -> scale for image distances
		// use only the j|j labels -> intra class variations
		double[] stdevs = new double[nobj];
		medstdv= new float[nc];
		for (int c=0;c<nc;c++) {
		    int ndev=0;
		    for (int obj=0;obj<nobj;obj++) {
		        if (condpair[c][obj][obj]) {
		            if (modelHistogram) {
                        double sum = 0.0;
                        double den = 0.0;
                        for (int bin=0;bin<nbins;bin++) {
                            sum += condhistogram[c][obj][obj][bin]*(condmin[c][obj][obj]+bin*(condmax[c][obj][obj]-condmin[c][obj][obj])/nbins);
                            den += condhistogram[c][obj][obj][bin];
                        }
                        sum /= den;
                        double var = 0.0;
                        for (int bin=0;bin<nbins;bin++) {
                            double val = condmin[c][obj][obj]+bin*(condmax[c][obj][obj]-condmin[c][obj][obj])/nbins;
                            var += condhistogram[c][obj][obj][bin]*(val-sum)*(val-sum);
                        }
                        stdevs[ndev] = FastMath.sqrt(var/den);
                    } else {
                        stdevs[ndev] = condstdv[c][obj][obj];
                    }
                    ndev++;
                }
		    }
		    Percentile measure = new Percentile();
            medstdv[c] = (float)measure.evaluate(stdevs, 0, ndev, 50.0);
        }
        stdevs = null;

        for (int c=0;c<nc;c++) {
            System.out.println("median intra-class stdev (contrast "+c+"): "+medstdv[c]);
		}
        
        System.out.println("apply priors to target");
        
        float[][][] separateIntensProbas = new float[nc][nbest][ndata]; 
		int[][][] separateIntensLabels = new int[nc][nbest][ndata];
			
		// if target in a different space, resample
		float[][] target = null;
		if (map2atlas!=null) {
		    target = new float[nc][nxyz];
		    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		        int idx = Numerics.bounded(Numerics.round(map2atlas[xyz]),0,ntx-1)
		                + ntx*Numerics.bounded(Numerics.round(map2atlas[xyz+nxyz]),0,nty-1)
		                + ntx*nty*Numerics.bounded(Numerics.round(map2atlas[xyz+2*nxyz]),0,ntz-1);
		        for (int c=0;c<nc;c++) target[c][xyz] = targetImages[c][idx];
		    }
		    // replace original data for further processing
		    targetImages = target;
		} else {
		    target = targetImages;
        }
		
		// combine priors and contrasts posteriors (update the priors maps)
		/* here it would be best to compute directly the joint distribution, so missing contrasts are not used */
		/* also it would be faster to only compute the spatial priors once */
		/* !!! when histograms go to zero, the probabilities collapse -> add a uniform probability everywhere? */
		/* speed-up: compute spatial likelihood first, then each contrast if non-zero 
       for (int c=0;c<nc;c++) {
            for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
               double[][] likelihood = new double[nobj][nobj];
               for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                   // look for non-zero priors
                   // note: could be done only ONCE!
                   likelihood[obj1][obj2] = 0.0;
                   
                   // impose the leftout classes here
                   if (cancelBackground && obj1==0 && obj2==0) {
                        likelihood[obj1][obj2] = 0.0;
                   } else if (cancelAll && obj1==obj2) {
                        likelihood[obj1][obj2] = 0.0;
                   } else {                   
                       for (int best=0;best<nbest;best++) {
                           if (spatialLabels[best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
                               // multiply nc times to balance prior and posterior
                               //likelihood[obj1][obj2] = 1.0;
                               likelihood[obj1][obj2] = spatialProbas[best][idmap[xyz]];
                               best = nbest;
                           }
                       }
                       // use the skeleton as prior? not here, it's not used to restrict search space
                   }
                   if (likelihood[obj1][obj2]>0) {
                       if (condpair[c][obj1][obj2]) {
                           double pobjc = intensityBaseline/nbins;
                           double psum = intensityBaseline;
                           if (target[c][xyz]!=0) {
                               if (modelHistogram) {
                                   int bin = Numerics.bounded(Numerics.ceil( (target[c][xyz]-condmin[c][obj1][obj2])/(condmax[c][obj1][obj2]-condmin[c][obj1][obj2])*nbins)-1, 0, nbins-1);
                                   pobjc += medstdv[c]*condhistogram[c][obj1][obj2][bin];
                               } else {
                                   pobjc += medstdv[c]/FastMath.sqrt(2.0*FastMath.PI*condstdv[c][obj1][obj2]*condstdv[c][obj1][obj2])
                                                *FastMath.exp( -0.5*(target[c][xyz]-condmean[c][obj1][obj2])*(target[c][xyz]-condmean[c][obj1][obj2])
                                                                   /(condstdv[c][obj1][obj2]*condstdv[c][obj1][obj2]) );
                              }
                              psum += 1.0;
                           }
                           if (psum>0) {
                               likelihood[obj1][obj2] *= pobjc/psum;
                           }
                       } else {
                           // what to do here? does it ever happen?
                           //System.out.print("!");
                           double pobjc = 1.0/nbins;
                           likelihood[obj1][obj2] *= pobjc;
                       }
                   }
                }
                for (int best=0;best<nbest;best++) {
                    int best1=0;
                    int best2=0;
                        
                    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                        if (likelihood[obj1][obj2]>likelihood[best1][best2]) {
                            best1 = obj1;
                            best2 = obj2;
                        }
                    }
                    // now find the corresponding shape prior
                    double shapeprior = 1.0;
                    for (int sbest=0;sbest<nbest;sbest++) {
                       if (spatialLabels[sbest][idmap[xyz]]==100*(best1+1)+(best2+1)) {
                           shapeprior = spatialProbas[sbest][idmap[xyz]];
                           sbest = nbest;
                       }
                    }
                    // sub optimal labeling, but easy to read
                    separateIntensLabels[c][best][idmap[xyz]] = 100*(best1+1)+(best2+1);
                    separateIntensProbas[c][best][idmap[xyz]] = (float)(likelihood[best1][best2]/shapeprior);
                    // remove best value
                    likelihood[best1][best2] = 0.0;
                }
            }
            if (rescaleIntensities) {
                // rescale top % in each shape and intensity priors
                Percentile measure = new Percentile();
                 double[] val = new double[ndata];
                for (int id=0;id<ndata;id++) val[id] = separateIntensProbas[c][0][id];
                
                float intensMax = (float)measure.evaluate(val, top);
                System.out.println("top "+top+"% intensity probability (contrast "+c+"): "+intensMax);
                
                for (int id=0;id<ndata;id++) for (int best=0;best<nbest;best++) {
                    separateIntensProbas[c][best][id] = (float)Numerics.min(top/100.0*separateIntensProbas[c][best][id]/intensMax, 1.0f);
                }
            }
		}*/
        for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
            double[][] spatialprior = new double[nobj][nobj];
            /* speed-up: we can run through the best lists instead of the obj1 x obj2 matrix
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                // look for non-zero priors
                // note: could be done only ONCE!
                spatialprior[obj1][obj2] = 0.0;
                   
                // impose the leftout classes here
                if (cancelBackground && obj1==0 && obj2==0) {
                    spatialprior[obj1][obj2] = 0.0;
                } else if (cancelAll && obj1==obj2) {
                    spatialprior[obj1][obj2] = 0.0;
                } else {                   
                   for (int best=0;best<nbest;best++) {
                       if (spatialLabels[best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
                           // multiply nc times to balance prior and posterior
                           //likelihood[obj1][obj2] = 1.0;
                           spatialprior[obj1][obj2] = spatialProbas[best][idmap[xyz]];
                           best = nbest;
                       }
                   }
                   // use the skeleton as prior? not here, it's not used to restrict search space
                }
            }*/
            for (int best=0;best<nbest;best++) {
                int obj1 = Numerics.floor(spatialLabels[best][idmap[xyz]]/100.0)-1;
                int obj2 = spatialLabels[best][idmap[xyz]]-100*(obj1+1)-1;
                    
                // impose the leftout classes here
                if (cancelBackground && obj1==0 && obj2==0) {
                    spatialprior[obj1][obj2] = 0.0;
                } else if (cancelAll && obj1==obj2) {
                    spatialprior[obj1][obj2] = 0.0;
                } else {                   
                   spatialprior[obj1][obj2] = spatialProbas[best][idmap[xyz]];
                }
            }
            // build a likelihood per contrast for now
            double[][] likelihood = new double[nobj][nobj];
            for (int c=0;c<nc;c++) {
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                    if (spatialprior[obj1][obj2]>0) {
                        // initialize with spatial prior
                        likelihood[obj1][obj2] = spatialprior[obj1][obj2];
                        if (condpair[c][obj1][obj2]) {
                            double pobjc = intensityBaseline/nbins;
                            double psum = intensityBaseline;
                            if (target[c][xyz]!=0) {
                                if (modelHistogram) {
                                    int bin = Numerics.bounded(Numerics.ceil( (target[c][xyz]-condmin[c][obj1][obj2])/(condmax[c][obj1][obj2]-condmin[c][obj1][obj2])*nbins)-1, 0, nbins-1);
                                    pobjc += medstdv[c]*condhistogram[c][obj1][obj2][bin];
                                } else {
                                    pobjc += medstdv[c]/FastMath.sqrt(2.0*FastMath.PI*condstdv[c][obj1][obj2]*condstdv[c][obj1][obj2])
                                                *FastMath.exp( -0.5*(target[c][xyz]-condmean[c][obj1][obj2])*(target[c][xyz]-condmean[c][obj1][obj2])
                                                                   /(condstdv[c][obj1][obj2]*condstdv[c][obj1][obj2]) );
                                }
                                psum += 1.0;
                            }
                            if (psum>0) {
                                likelihood[obj1][obj2] *= pobjc/psum;
                            }
                        } else {
                           // what to do here? does it ever happen?
                           //System.out.print("!");
                           double pobjc = 1.0/nbins;
                           likelihood[obj1][obj2] *= pobjc;
                        }
                    }
                }
                for (int best=0;best<nbest;best++) {
                    int best1=0;
                    int best2=0;
                        
                    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                        if (likelihood[obj1][obj2]>likelihood[best1][best2]) {
                            best1 = obj1;
                            best2 = obj2;
                        }
                    }
                    // sub optimal labeling, but easy to read
                    separateIntensLabels[c][best][idmap[xyz]] = 100*(best1+1)+(best2+1);
                    separateIntensProbas[c][best][idmap[xyz]] = (float)(likelihood[best1][best2]/spatialprior[best1][best2]);
                    // remove best value
                    likelihood[best1][best2] = 0.0;
                }
            }
        }
        if (rescaleIntensities) {
            for (int c=0;c<nc;c++) {
                // rescale top % in each shape and intensity priors
                Percentile measure = new Percentile();
                double[] val = new double[ndata];
                for (int id=0;id<ndata;id++) val[id] = separateIntensProbas[c][0][id];
                
                float intensMax = (float)measure.evaluate(val, top);
                System.out.println("top "+top+"% intensity probability (contrast "+c+"): "+intensMax);
                
                for (int id=0;id<ndata;id++) for (int best=0;best<nbest;best++) {
                    separateIntensProbas[c][best][id] = (float)Numerics.min(top/100.0*separateIntensProbas[c][best][id]/intensMax, 1.0f);
                }
            }
		}
        
		// combine the multiple contrasts            
		System.out.println("combine intensity probabilities");
		intensityProbas = new float[nbest][ndata]; 
		intensityLabels = new int[nbest][ndata];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
            double[][] likelihood = new double[nobj][nobj];
            //System.out.print("contrasts: ");
            // VERY SLOW : faster option using only the best ones instead? problem of correctly setting zero values
            /* too slow?
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                likelihood[obj1][obj2] = 1.0;
                int ncontrast = 0;
                for (int c=0;c<nc;c++) {
                    // if we use all contrasts or if the right non-bg contrasts are used
                    if (contrastList==null
                         || (obj1>=nbg && contrastList[obj1-nbg][c]) 
                         || (obj2>=nbg && contrastList[obj2-nbg][c])
                         || (obj1<nbg && obj2<nbg) ) {
                        //System.out.print(obj1+"-"+obj2+":"+c+", ");
                        double val = 0.0;
                        for (int best=0;best<nbest;best++) {
                            if (separateIntensLabels[c][best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
                                val = separateIntensProbas[c][best][idmap[xyz]];
                                best = nbest;
                            }
                        }
                        likelihood[obj1][obj2] *= val;
                        ncontrast++;
                    }
                }
                likelihood[obj1][obj2] = FastMath.pow(likelihood[obj1][obj2], intensityImportance/ncontrast);
            }/* faster? */
            int[][] ncontrast = new int[nobj][nobj];
            for (int c=0;c<nc;c++) {
                for (int best=0;best<nbest;best++) {
                    int obj1 = Numerics.floor(separateIntensLabels[c][best][idmap[xyz]]/100.0)-1;
                    int obj2 = separateIntensLabels[c][best][idmap[xyz]]-100*(obj1+1)-1;
                    
                    // if we use all contrasts or if the right non-bg contrasts are used
                    if (contrastList==null
                         || (obj1>=nbg && contrastList[obj1-nbg][c]) 
                         || (obj2>=nbg && contrastList[obj2-nbg][c])
                         || (obj1<nbg && obj2<nbg) ) {
                    
                        if (ncontrast[obj1][obj2]==0) {
                            likelihood[obj1][obj2] = separateIntensProbas[c][best][idmap[xyz]];
                        } else {
                            likelihood[obj1][obj2] *= separateIntensProbas[c][best][idmap[xyz]];
                        }
                        ncontrast[obj1][obj2]++;
                    }
                }
            }
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                if (likelihood[obj1][obj2]>0) {
                    likelihood[obj1][obj2] = FastMath.pow(likelihood[obj1][obj2], intensityImportance/ncontrast[obj1][obj2]);
                }
            }/**/
            //System.out.print("\n");
            
            for (int best=0;best<nbest;best++) {
                int best1=0;
                int best2=0;
                    
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                    if (likelihood[obj1][obj2]>likelihood[best1][best2]) {
                        best1 = obj1;
                        best2 = obj2;
                    }
                }
                // sub optimal labeling, but easy to read
                intensityLabels[best][idmap[xyz]] = 100*(best1+1)+(best2+1);
                // scaling for multiplicative intensities (done when combining into posteriors)
                intensityProbas[best][idmap[xyz]] = (float)likelihood[best1][best2];
                
                // remove best value
                likelihood[best1][best2] = 0.0;
            }
        }
		if (!rescaleIntensities && rescaleProbas) {
            Percentile measure = new Percentile();
            double[] val = new double[ndata];
            for (int id=0;id<ndata;id++) val[id] = intensityProbas[0][id];
            float intensMax = (float)measure.evaluate(val, top);
            System.out.println("top "+top+"% global intensity probability: "+intensMax);
            for (int id=0;id<ndata;id++) for (int best=0;best<nbest;best++) {
                intensityProbas[best][id] = (float)Numerics.min(top/100.0*intensityProbas[best][id]/intensMax, 1.0f);
            }		
		}
		
		// posterior : merge both measures
		System.out.println("generate posteriors");
        combinedProbas = new float[nbest][ndata]; 
		combinedLabels = new int[nbest][ndata];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    double[][] posteriors = new double[nobj][nobj];
		    /* too slow? better to populate only the non-zero items 
		    // seems OK, actually. The previous step is really the killer here
		    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                // look for non-zero priors
                posteriors[obj1][obj2] = 0.0;
                
                for (int best=0;best<nbest;best++) {
                    if (spatialLabels[best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
                        // multiply nc times to balance prior and posterior
                        posteriors[obj1][obj2] = spatialProbas[best][idmap[xyz]];
                        best = nbest;
                    }
                }
                // use the skeleton as prior?
                if (obj1==obj2) {
                   for (int best=0;best<nskel;best++) {
                       if (skeletonLabels[best][idmap[xyz]]==101*(obj1+1)) {
                           // multiply nc times to balance prior and posterior
                           //likelihood[obj1][obj2] = 1.0;
                           if (skeletonProbas[best][idmap[xyz]]>posteriors[obj1][obj1]*posteriors[obj1][obj1])
                               posteriors[obj1][obj1] = FastMath.sqrt(skeletonProbas[best][idmap[xyz]]);
                               //posteriors[obj1][obj1] = skeletonProbas[best][idmap[xyz]]/Numerics.max(0.001f,posteriors[obj1][obj1]);
                           best = nskel;
                       }
                   }
                }
                if (posteriors[obj1][obj2]>0) {
                    double intensPrior = 0.0;
                    for (int best=0;best<nbest;best++) {
                        if (intensityLabels[best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
                            intensPrior = intensityProbas[best][idmap[xyz]];
                            best=nbest;
                        }
                    }
                    // now the intensity weights are computed beforehand, as the number of contrasts may vary per structure
                    //if (averageIntensityPriors) {
                        //posteriors[obj1][obj2] = posteriors[obj1][obj2]*FastMath.pow(intensPrior,intensityImportance/nc);
                        posteriors[obj1][obj2] = posteriors[obj1][obj2]*intensPrior;
                    //} else {
                    //    //posteriors[obj1][obj2] = FastMath.pow(posteriors[obj1][obj2]*FastMath.pow(intensPrior,intensityImportance),2.0/(intensityImportance*nc+1.0));
                    //    posteriors[obj1][obj2] = FastMath.pow(posteriors[obj1][obj2]*FastMath.pow(intensPrior,intensityImportance*nc),2.0/(intensityImportance*nc+1.0));
                    //}
                }
            }/* faster? */
            // spatial prior
            for (int best=0;best<nbest;best++) {
                int obj1 = Numerics.floor(spatialLabels[best][idmap[xyz]]/100.0)-1;
                int obj2 = spatialLabels[best][idmap[xyz]]-100*(obj1+1)-1;
                
                posteriors[obj1][obj2] = spatialProbas[best][idmap[xyz]];
            }
            // use the skeleton as prior?
            for (int best=0;best<nskel;best++) {
                int obj = Numerics.floor(skeletonLabels[best][idmap[xyz]]/101.0)-1;
                if (skeletonProbas[best][idmap[xyz]]>posteriors[obj][obj]*posteriors[obj][obj])
                    posteriors[obj][obj] = FastMath.sqrt(skeletonProbas[best][idmap[xyz]]);
            }
            // intensity posterior
            for (int best=0;best<nbest;best++) {
                int obj1 = Numerics.floor(intensityLabels[best][idmap[xyz]]/100.0)-1;
                int obj2 = intensityLabels[best][idmap[xyz]]-100*(obj1+1)-1;
            
                if (posteriors[obj1][obj2]>0) {
                    posteriors[obj1][obj2] = posteriors[obj1][obj2]*intensityProbas[best][idmap[xyz]];
                }
            }/**/
                    
            for (int best=0;best<nbest;best++) {
                int best1=0;
                int best2=0;
                    
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                    if (posteriors[obj1][obj2]>posteriors[best1][best2]) {
                        best1 = obj1;
                        best2 = obj2;
                    }
                }
                // sub optimal labeling, but easy to read
                combinedLabels[best][idmap[xyz]] = 100*(best1+1)+(best2+1);
                combinedProbas[best][idmap[xyz]] = (float)FastMath.sqrt(posteriors[best1][best2]);
                // remove best value
                posteriors[best1][best2] = 0.0;
 		    }
		}
	}
	
	public final void collapseToJointMaps() {
	    jointLabels = new int[nbest][ndata];
	    jointProbas = new float[nbest][ndata];
	    
        for (int id=0;id<ndata;id++) {
            double[][] posteriors = new double[nobj][nobj];
            for (int best=0;best<nbest;best++) {
                int obj1 = Numerics.floor(combinedLabels[best][id]/100)-1;
                int obj2 = combinedLabels[best][id]-(obj1+1)*100-1;
                if (obj1>-1 && obj2>-1)
                    posteriors[obj1][obj2] = Numerics.max(posteriors[obj1][obj2],combinedProbas[best][id]);
            }
            if (sumPosterior) {
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) if (obj2!=obj1) {
                   posteriors[obj1][obj1] += posteriors[obj1][obj2];
                   posteriors[obj1][obj2] = 0.0;
                }
            } else if (maxPosterior) {
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) if (obj2!=obj1) {
                    posteriors[obj1][obj1] = Numerics.max(posteriors[obj1][obj1],posteriors[obj1][obj2]);
                    posteriors[obj1][obj2] = 0.0;
                }
            }
            for (int best=0;best<nbest;best++) {
                int best1=0;
                    
                for (int obj1=0;obj1<nobj;obj1++) {
                    if (posteriors[obj1][obj1]>posteriors[best1][best1]) {
                        best1 = obj1;
                    }
                }
                // sub optimal labeling, but easy to read
                jointLabels[best][id] = best1;
                jointProbas[best][id] = (float)posteriors[best1][best1];
                // remove best value
                posteriors[best1][best1] = 0.0;
 		    }
        }
    }  
	// NOTE: this needs updating for slab data: TODO
	public final void mapAtlasTargetIntensityPriors() {
	    nx = nax; ny = nay; nz = naz; nxyz = naxyz;
	    rx = rax; ry = ray; rz = raz;
	    
        System.out.println("(use histograms for intensities)");
        trgcondmin = new double[ntc];
        trgcondmax = new double[ntc];
        trgcondhistogram = new double[ntc][nobj][nobj][nbins];
        
        for (int tc=0;tc<ntc;tc++) {
            trgcondmin[tc] = 1e9;
            trgcondmax[tc] = -1e9;
            for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		        if (avgTargetImages[tc][xyz]<trgcondmin[tc]) trgcondmin[tc] = avgTargetImages[tc][xyz];
		        if (avgTargetImages[tc][xyz]>trgcondmax[tc]) trgcondmax[tc] = avgTargetImages[tc][xyz];
		    }
		}
        for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            System.out.print("\n("+(obj1+1)+" | "+(obj2+1)+"): ");
            for (int tc=0;tc<ntc;tc++) {
                for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                    // look for non-zero priors
                    for (int best=0;best<nbest;best++) {
                        if (spatialLabels[best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
                            float prior = spatialProbas[best][idmap[xyz]];
                            for (int c=0;c<nc;c++) {
                                int bin = Numerics.bounded(Numerics.ceil( (avgAtlasImages[c][xyz]-condmin[c][obj1][obj2])/(condmax[c][obj1][obj2]-condmin[c][obj1][obj2])*nbins)-1, 0, nbins-1);
                                prior *= condhistogram[c][obj1][obj2][bin];
                            }
                            
                            int tbin = Numerics.bounded(Numerics.ceil( (avgTargetImages[tc][xyz]-trgcondmin[tc])/(trgcondmax[tc]-trgcondmin[tc])*nbins)-1, 0, nbins-1);
                            trgcondhistogram[tc][obj1][obj2][tbin] += prior;
                        }
                    }
                }
            }
        }
        for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            for (int tc=0;tc<ntc;tc++) {
                // smooth histograms to avoid sharp edge effects
                double var = histogramSpread*histogramSpread;
                double[] tmphist = new double[nbins];
                for (int bin1=0;bin1<nbins;bin1++) {
                    for (int bin2=0;bin2<nbins;bin2++) {
                        tmphist[bin1] += trgcondhistogram[tc][obj1][obj2][bin2]*FastMath.exp(-0.5*(bin1-bin2)*(bin1-bin2)/var);
                    }
                }
                for (int bin=0;bin<nbins;bin++) trgcondhistogram[tc][obj1][obj2][bin] = tmphist[bin];
                
                // normalize: sum over count x spread = 1
                double sum = 0.0;
                for (int bin=0;bin<nbins;bin++) sum += trgcondhistogram[tc][obj1][obj2][bin];   
                //for (int bin=0;bin<nbins;bin++) condhistogram[c][obj1][obj2][bin] /= sum*(condmax[c][obj1][obj2]-condmin[c][obj1][obj2]);   
                for (int bin=0;bin<nbins;bin++) if (sum>0) trgcondhistogram[tc][obj1][obj2][bin] /= sum;   
            }
        }
    	
		return;
	}

	public final void mapAtlasVolumePriors() {
	    
        System.out.println("(use transformed priors)");
        
        // single object volume priors
        for (int obj=0;obj<nobj;obj++) {
            System.out.print("\n("+(obj+1)+"): ");
            double avol = 0.0;
            for (int xyza=0;xyza<naxyz;xyza++) {
                // look for non-zero priors
                for (int best=0;best<nbest;best++) {
                    if (origLabels[best][xyza]>100*(obj+1) && origLabels[best][xyza]<100*(obj+2)) {
                        float prior = origProbas[best][xyza];
                        avol += prior*rax*ray*raz;
                    }
                }
            }
            System.out.print(avol+" -> ");
            double tvol = 0.0;
            for (int xyzt=0;xyzt<ntxyz;xyzt++) {
                // look for non-zero priors
                for (int best=0;best<nbest;best++) {
                    if (targetLabels[best][xyzt]>100*(obj+1) && targetLabels[best][xyzt]<100*(obj+2)) {
                        float prior = targetProbas[best][xyzt];
                        tvol += prior*rtx*rty*rtz;
                    }
                }
            }
            System.out.print(tvol+", ratio: "+tvol/avol);
            logVolMean[obj] += FastMath.log(Numerics.max(1.0,tvol)) - FastMath.log(Numerics.max(1.0,avol));
            logVolStdv[obj] += FastMath.log(Numerics.max(1.0,tvol)) - FastMath.log(Numerics.max(1.0,avol));
        }
        /* skip for now, not relevant anymore, and verrrry slow
        for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            System.out.print("\n("+(obj1+1)+" | "+(obj2+1)+"): ");
            double avol = 0.0;
            for (int xyza=0;xyza<naxyz;xyza++) {
                // look for non-zero priors
                for (int best=0;best<nbest;best++) {
                    if (origLabels[best][xyza]==100*(obj1+1)+(obj2+1)) {
                        float prior = origProbas[best][xyza];
                        avol += prior*rax*ray*raz;
                    }
                }
            }
            System.out.print(avol+" -> ");
            double tvol = 0.0;
            for (int xyzt=0;xyzt<ntxyz;xyzt++) {
                // look for non-zero priors
                for (int best=0;best<nbest;best++) {
                    if (targetLabels[best][xyzt]==100*(obj1+1)+(obj2+1)) {
                        float prior = targetProbas[best][xyzt];
                        tvol += prior*rtx*rty*rtz;
                    }
                }
            }
            System.out.print(tvol+", ratio: "+tvol/avol);
            logVolMean2[obj1][obj2] += FastMath.log(Numerics.max(1.0,tvol)) - FastMath.log(Numerics.max(1.0,avol));
            logVolStdv2[obj1][obj2] += FastMath.log(Numerics.max(1.0,tvol)) - FastMath.log(Numerics.max(1.0,avol));
        }*/
    	
		return;
	}
	
	public final void collapseConditionalMaps() {	    
        for (int id=0;id<ndata;id++) {
            double[][] posteriors = new double[nobj][nobj];
            for (int best=0;best<nbest;best++) {
                int obj1 = Numerics.floor(combinedLabels[best][id]/100)-1;
                int obj2 = combinedLabels[best][id]-(obj1+1)*100-1;
                if (obj1>-1 && obj2>-1)
                    posteriors[obj1][obj2] = Numerics.max(posteriors[obj1][obj2],combinedProbas[best][id]);
            }
            if (sumPosterior) {
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) if (obj2!=obj1) {
                   posteriors[obj1][obj1] += posteriors[obj1][obj2];
                   posteriors[obj1][obj2] = 0.0;
                }
            } else if (maxPosterior) {
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) if (obj2!=obj1) {
                    posteriors[obj1][obj1] = Numerics.max(posteriors[obj1][obj1],posteriors[obj1][obj2]);
                    posteriors[obj1][obj2] = 0.0;
                }
            }
            for (int best=0;best<nbest;best++) {
                int best1=0;
                for (int obj1=0;obj1<nobj;obj1++) {
                    if (posteriors[obj1][obj1]>posteriors[best1][best1]) {
                        best1 = obj1;
                    }
                }
                // sub optimal labeling, but easy to read
                combinedLabels[best][id] = best1;
                combinedProbas[best][id] = (float)posteriors[best1][best1];
                // remove best value
                posteriors[best1][best1] = 0.0;
 		    }
        }
    }

	public final void collapseSpatialPriorMaps() {	    
        for (int id=0;id<ndata;id++) {
            double[][] priors = new double[nobj][nobj];
            for (int best=0;best<nbest;best++) {
                int obj1 = Numerics.floor(spatialLabels[best][id]/100)-1;
                int obj2 = spatialLabels[best][id]-(obj1+1)*100-1;
                if (obj1>-1 && obj2>-1)
                    priors[obj1][obj2] =  Numerics.max(priors[obj1][obj2],spatialProbas[best][id]);
            }
            if (sumPosterior) {
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) if (obj2!=obj1) {
                   priors[obj1][obj1] += priors[obj1][obj2];
                   priors[obj1][obj2] = 0.0;
                }
            } else if (maxPosterior) {
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) if (obj2!=obj1) {
                    priors[obj1][obj1] = Numerics.max(priors[obj1][obj1],priors[obj1][obj2]);
                    priors[obj1][obj2] = 0.0;
                }
            }
            for (int best=0;best<nbest;best++) {
                int best1=0;
                for (int obj1=0;obj1<nobj;obj1++) {
                    if (priors[obj1][obj1]>priors[best1][best1]) {
                        best1 = obj1;
                    }
                }
                // sub optimal labeling, but easy to read
                spatialLabels[best][id] = best1;
                spatialProbas[best][id] = (float)priors[best1][best1];
                // remove best value
                priors[best1][best1] = 0.0;
 		    }
        }
    }
	
	public final void fastSimilarityDiffusion(int nngb) {
		
		float[][] target = targetImages;
		// add a local diffusion step?
		System.out.print("Diffusion step: \n");
		
		// graph = N-most likely neihgbors (based on target intensity)
		System.out.print("Build similarity neighborhood\n");
 		ngbw = new float[nngb+1][ndata];
		int[][] ngbi = new int[nngb][ndata];
		float[] ngbsim = new float[26];
		
		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
		    int xyz = x+nx*y+nx*ny*z;
		    if (mask[xyz]) {
		        for (byte d=0;d<26;d++) {
		            int ngb = Ngb.neighborIndex(d, xyz, nx,ny,nz);
		            if (mask[ngb]) {
		                ngbsim[d] = 1.0f;
		                for (int c=0;c<nc;c++) {
		                    //ngbsim[d] *= 1.0f/Numerics.max(1e-6,Numerics.abs(target[c][xyz]-target[c][ngb])/medstdv[c]);
                            ngbsim[d] *= (float)FastMath.exp( -0.5/nc*(target[c][xyz]-target[c][ngb])*(target[c][xyz]-target[c][ngb])
                                         /(medstdv[c]*medstdv[c]) );
                        }
                        //if (ngbsim[d]==0) System.out.print("!");
                    } else {
                        ngbsim[d] = 0.0f;
                    }
                }
                // choose the N best ones
                ngbw[nngb][idmap[xyz]] = 0.0f;
                for (int n=0;n<nngb;n++) {
                    byte best=0;
                        
                    for (byte d=0;d<26;d++)
                        if (ngbsim[d]>ngbsim[best]) 
                            best = d;
                    
                    ngbw[n][idmap[xyz]] = ngbsim[best];
                    ngbi[n][idmap[xyz]] = idmap[Ngb.neighborIndex(best, xyz, nx,ny,nz)];
                    ngbw[nngb][idmap[xyz]] += ngbsim[best];
                    
                    ngbsim[best] = 0.0f;
                }
                //if (ngbw[nngb][idmap[xyz]]==0) System.out.print("0");
            }
        }  
		System.out.print("\n");

		// diffusion only between i|j <-> i|j, not i|j <-> i|k, i|j <-> j|i
		
		float[][] diffusedProbas = new float[nbest][ndata]; 
		int[][] diffusedLabels = new int[nbest][ndata];
		
		// first copy the originals, then iterate on the copy?
		for (int id=0;id<ndata;id++) {
		    for (int best=0;best<nbest;best++) {
                diffusedProbas[best][id] = combinedProbas[best][id];
                diffusedLabels[best][id] = combinedLabels[best][id];
            }
        }
		    		
		double[][] diffused = new double[nobj][nobj];
        for (int t=0;t<maxiter;t++) {
		    for (int id=0;id<ndata;id++) if (ngbw[nngb][id]>0) {
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                    diffused[obj1][obj2] = 0.0;
                }
                
                for (int best=0;best<nbest;best++) {
                    int obj1 = Numerics.floor(combinedLabels[best][id]/100.0f)-1;
                    int obj2 = Numerics.round(combinedLabels[best][id]-100*(obj1+1)-1);
                     
                    if (diffused[obj1][obj2]==0) {
                        diffused[obj1][obj2] = combinedProbas[best][id];
                        float den = ngbw[nngb][id];
                        diffused[obj1][obj2] *= den;
                        
                        for (int n=0;n<nngb;n++) {
                            int ngb = ngbi[n][id];
                            float ngbmax = 0.0f;
                            // max over neighbors ( -> stop at first found)
                            for (int bestngb=0;bestngb<nbest;bestngb++) {
                                if (combinedLabels[bestngb][ngb]==100*(obj1+1)+(obj2+1)) {
                                    ngbmax = combinedProbas[bestngb][ngb];
                                    bestngb=nbest;
                                }
                            }
                            //if (ngbmax==0) System.out.print("0");
                            diffused[obj1][obj2] += ngbw[n][id]*ngbmax;
                            den += ngbw[n][id];
                        }
                        diffused[obj1][obj2] /= den;
                    }
                }
                for (int best=0;best<nbest;best++) {
                    int best1 = Numerics.floor(combinedLabels[best][id]/100.0f)-1;
                    int best2 = Numerics.round(combinedLabels[best][id]-100*(best1+1)-1);
                        
                    for (int next=0;next<nbest;next++) {
                        int obj1 = Numerics.floor(combinedLabels[next][id]/100.0f)-1;
                        int obj2 = Numerics.round(combinedLabels[next][id]-100*(obj1+1)-1);
                        if (diffused[obj1][obj2]>diffused[best1][best2]) {
                            best1 = obj1;
                            best2 = obj2;
                        }
                    }
                    // sub optimal labeling, but easy to read
                    diffusedLabels[best][id] = 100*(best1+1)+(best2+1);
                    diffusedProbas[best][id] = (float)diffused[best1][best2];
                    // remove best value
                    diffused[best1][best2] = 0.0;
                }
            }
            double diff = 0.0;
            for (int id=0;id<ndata;id++) for (int best=0;best<nbest;best++) {
                if (combinedLabels[best][id] == diffusedLabels[best][id]) {
                    //diff += Numerics.abs(diffusedProbas[best][idmap[xyz]]-combinedProbas[best][idmap[xyz]]);
                    diff += 0.0;
                } else {
                    //diff += Numerics.abs(diffusedProbas[best][idmap[xyz]]-combinedProbas[best][idmap[xyz]]);
                    diff += 1.0;
                }
                combinedLabels[best][id] = diffusedLabels[best][id];
                combinedProbas[best][id] = diffusedProbas[best][id];
            }
            System.out.println("diffusion step "+t+": "+(diff/ndata));
            if (diff/ndata<maxdiff) t=maxiter;
		}

		target = null;
	}
				
	private int[] atlasVolumeLabels(float[] pval, int[] lval) {
		// find appropriate threshold to have correct volume; should use a fast marching approach!
		BinaryHeap2D	heap = new BinaryHeap2D(nax*nay+nay*naz+naz*nax, BinaryHeap2D.MAXTREE);
		int[] labels = new int[naxyz];
        int[] start = new int[nobj];
        float[] bestscore = new float[nobj];
        for (int obj=nbg;obj<nobj;obj++) bestscore[obj] = -INF;
        heap.reset();
		// important: skip first label as background (allows for unbounded growth)
        for (int obj=nbg;obj<nobj;obj++) {
		    // find highest scoring voxel as starting point
            for (int b=0;b<nbest-1;b++) {
                for (int x=1;x<nax-1;x++) for (int y=1;y<nay-1;y++) for (int z=1;z<naz-1;z++) {
                    int xyz=x+nax*y+nax*nay*z;
                    if (Numerics.floor(lval[xyz+b*naxyz]/100.0f)==obj+1) {
                        if (pval[xyz+b*naxyz]>bestscore[obj]) {
                            bestscore[obj] = pval[xyz+b*naxyz];
                            start[obj] = xyz;
                        }
                    }
                }
                if (bestscore[obj]>-INF) b = nbest;
            }
            heap.addValue(bestscore[obj],start[obj],(byte)obj);
        }
        double[] vol = new double[nobj];
        double[] volmean = new double[nobj];
        for (int obj=nbg;obj<nobj;obj++) volmean[obj] = FastMath.exp(logVolMean[obj]);
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int xyz = heap.getFirstId();
            byte obj = heap.getFirstState();
            heap.removeFirst();
            if (labels[xyz]==0) {
                 if (vol[obj]<volmean[obj]) {
                    // update the values
                    vol[obj]+=rax*ray*raz;
                    labels[xyz] = obj;
                
                    // add neighbors
                    for (byte k = 0; k<26; k++) {
                        int ngb = Ngb.neighborIndex(k, xyz, nax, nay, naz);
                        if (ngb>0 && ngb<naxyz) {
                           if (labels[ngb]==0) {
                                for (int best=0;best<nbest;best++) {
                                    if (Numerics.floor(lval[xyz+best*naxyz]/100.0f)==obj+1) {
                                       heap.addValue(pval[ngb+best*naxyz],ngb,obj);
                                       best=nbest;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return labels;            
	}
			
	private int[] atlasBoundaryLabels(float[] pval, int[] lval) {
		// find appropriate threshold to have correct volume; should use a fast marching approach!
		BinaryHeapPair	heap = new BinaryHeapPair(nax*nay+nay*naz+naz*nax, BinaryHeapPair.MAXTREE);
		int[] labels = new int[naxyz];
        int[] start = new int[nobj];
        float[] bestscore = new float[nobj];
        for (int obj=nbg;obj<nobj;obj++) bestscore[obj] = -INF;
        heap.reset();
		// important: skip first label as background (allows for unbounded growth)
        for (int obj=nbg;obj<nobj;obj++) {
		    // find highest scoring voxel as starting point
            for (int b=0;b<nbest-1;b++) {
                for (int x=1;x<nax-1;x++) for (int y=1;y<nay-1;y++) for (int z=1;z<naz-1;z++) {
                    int xyz=x+nax*y+nax*nay*z;
                    if (Numerics.floor(lval[xyz+b*naxyz]/100.0f)==obj+1) {
                        if (pval[xyz+b*naxyz]>bestscore[obj]) {
                            bestscore[obj] = pval[xyz+b*naxyz];
                            start[obj] = xyz;
                        }
                    }
                }
                if (bestscore[obj]>-INF) b = nbest;
            }
            heap.addValue(bestscore[obj],start[obj],101*(obj+1));
        }
        double[] vol = new double[nobj];
        double[] volmean = new double[nobj];
        for (int obj=nbg;obj<nobj;obj++) volmean[obj] = FastMath.exp(logVolMean[obj]);
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int xyz = heap.getFirstId1();
            int obj1obj2 = heap.getFirstId2();
            int obj1 = Numerics.floor(obj1obj2/100.0f)-1;
            int obj2 = Numerics.round(obj1obj2-100*(obj1+1)-1);
            heap.removeFirst();
            if (labels[xyz]==0) {
                 if (vol[obj1]<volmean[obj1]) {
                    // update the values
                    vol[obj1]+=rax*ray*raz;
                    labels[xyz] = obj1obj2;
                
                    // add neighbors
                    for (byte k = 0; k<26; k++) {
                        int ngb = Ngb.neighborIndex(k, xyz, nax, nay, naz);
                        if (ngb>0 && ngb<naxyz) {
                           if (labels[ngb]==0) {
                                for (int best=0;best<nbest;best++) {
                                    if (Numerics.floor(lval[xyz+best*naxyz]/100.0f)==obj1+1) {
                                       heap.addValue(pval[ngb+best*naxyz],ngb,lval[xyz+best*naxyz]);
                                       best=nbest;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return labels;            
	}
	
	public void topologyBoundaryDefinition(String connectType, String lutdir) {

	    // load topology LUT
	    CriticalPointLUT lut;
             if (connectType.equals("26/6")) lut = new CriticalPointLUT(lutdir, "critical266LUT.raw.gz",200);
        else if (connectType.equals("6/26")) lut = new CriticalPointLUT(lutdir, "critical626LUT.raw.gz",200);
        else if (connectType.equals("18/6")) lut = new CriticalPointLUT(lutdir, "critical186LUT.raw.gz",200);
        else if (connectType.equals("6/18")) lut = new CriticalPointLUT(lutdir, "critical618LUT.raw.gz",200);
        else if (connectType.equals("6/6")) lut = new CriticalPointLUT(lutdir, "critical66LUT.raw.gz",200);
        else                                lut = new CriticalPointLUT(lutdir, "criticalWCLUT.raw.gz",200);
		if (!lut.loadCompressedPattern()) {
			System.out.println("Problem loading the algorithm's LUT from: "+lutdir);
			return;
        } else {
			System.out.println("LUT loaded from: "+lutdir);
		}
		
		// use a 6-neighborhood for maximum regularity
		int ngbdist = 2;
		
		// here we assume the maps have been collapsed into objects
		
		// initialize each boundary from bounding box per structure
		float minproba = 0.0f;
		float mindist = 0.000001f;
		
        int[] x0 = new int[nobj];
		int[] y0 = new int[nobj];
		int[] z0 = new int[nobj];
		
		int[] xN = new int[nobj];
		int[] yN = new int[nobj];
		int[] zN = new int[nobj];
		
		for (int obj=0;obj<nobj;obj++) {
		    x0[obj] = nx;
		    y0[obj] = ny;
		    z0[obj] = nz;
		    xN[obj] = -1;
		    yN[obj] = -1;
		    zN[obj] = -1;
		}
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
            int xyz = x + nx*y + nx*ny*z;
            if (mask[xyz]){
                int id = idmap[xyz];
                for (int obj=nbg;obj<nobj;obj++) {
                    for (int best=0;best<nbest;best++) {
                        if (combinedLabels[best][id]>100*(obj+1) && combinedLabels[best][id]<100*(obj+2) && combinedProbas[best][id]>minproba) {
                            x0[obj] = Numerics.min(x,x0[obj]);
                            y0[obj] = Numerics.min(y,y0[obj]);
                            z0[obj] = Numerics.min(z,z0[obj]);
                            xN[obj] = Numerics.max(x,xN[obj]);
                            yN[obj] = Numerics.max(y,yN[obj]);
                            zN[obj] = Numerics.max(z,zN[obj]); 
                            best = nbest;
                        }
                    }
                }
            }
        }
        // initialize the topology of the bounding box
        BinaryHeapPair	heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeap2D.MINTREE);
		// important: skip first label as background (allows for unbounded growth)
        for (byte obj=nbg;obj<nobj;obj++) {
            System.out.print("Structure "+obj+": topology correction");
            
		    byte[] topology = new byte[nxyz];
		    int[] label = new int[nxyz];
		    float[] score = new float[nxyz];
		    heap.reset();
		    // start with the bounding box
		    for (int x=x0[obj];x<=xN[obj];x++) for (int y=y0[obj];y<=yN[obj];y++) for (int z=z0[obj];z<=zN[obj];z++) {
		        if (x==x0[obj] || x==xN[obj] || y==y0[obj] || y==yN[obj] || z==z0[obj] || z==zN[obj]) {
		            int xyz = x+nx*y+nx*ny*z;
		            topology[xyz] = 1;
		            score[xyz] = minproba;
		            label[xyz] = 100*(obj+1);
		            if (mask[xyz]) {
		                int id = idmap[xyz];
		                for (int best=0;best<nbest;best++) {
		                    if (combinedLabels[best][id]>100*(obj+1) && combinedLabels[best][id]<100*(obj+2)) {
		                        score[xyz] = combinedProbas[best][id];
		                        label[xyz] = combinedLabels[best][id];
		                        best = nbest;
		                    }
		                }
		            }
		        }
		    }
		    // now add the neighbors
		    for (int x=x0[obj];x<=xN[obj];x++) for (int y=y0[obj];y<=yN[obj];y++) for (int z=z0[obj];z<=zN[obj];z++) {
		        if (x==x0[obj]+1 || x==xN[obj]-1 || y==y0[obj]+1 || y==yN[obj]-1 || z==z0[obj]+1 || z==zN[obj]-1) {
		            int xyz = x+nx*y+nx*ny*z;
		            float val = minproba + mindist;
		            int lbl = 100*(obj+1);
		            if (mask[xyz]) {
		                int id = idmap[xyz];
		                for (int best=0;best<nbest;best++) {
		                    if (combinedLabels[best][id]>100*(obj+1) && combinedLabels[best][id]<100*(obj+2)) {
		                        if (combinedProbas[best][id]>val) {
                                    val = combinedProbas[best][id];
                                    lbl = combinedLabels[best][id];
                                }
		                        best = nbest;
		                    }
		                }
		            }
		            heap.addValue(val,xyz,lbl);
		        }
		    }
		    // propagate
		    int ncritical = 0;
		    while (heap.isNotEmpty()) {
                float val = heap.getFirst();
                int xyz = heap.getFirstId1();
                int lbl = heap.getFirstId2();
                heap.removeFirst();
                
                // check if already set; if so skip
                if (topology[xyz]<1) {
                    // check for higher neighbor
                    boolean higher = false;
                    for (int i=-1;i<=1;i++) for (int j=-1;j<=1;j++) for (int l=-1;l<=1;l++) {
                        int ngb = xyz+i+nx*j+nx*ny*l;
                        if (topology[ngb]==1 && score[ngb]>val) {
                            val = score[ngb];
                            higher = true;
                        }
                    }
                    if (higher) {
                        heap.addValue(val + mindist,xyz,lbl);
                    } else {
                        // check the topology
                        if (lut.get(lut.keyFromPattern(topology,xyz,1,nx,nx*ny))) {
                            // regular point
                            topology[xyz] = 1;
                            score[xyz] = val;
                            label[xyz] = lbl;
                            
                            // search for neighbors
                            for (int i=-1;i<=1;i++) for (int j=-1;j<=1;j++) for (int l=-1;l<=1;l++) {
                                int ngb = xyz+i+nx*j+nx*ny*l;
                                if (topology[ngb]==0 && i*i+j*j+l*l<ngbdist) {
                                    // new neighbors
                                    float nextval = val + mindist;
                                    int nextlbl = lbl;
                                    if (mask[ngb]) {
                                        int id = idmap[ngb];
                                        for (int best=0;best<nbest;best++) {
                                            if (combinedLabels[best][id]>100*(obj+1) && combinedLabels[best][id]<100*(obj+2)) {
                                                if (combinedProbas[best][id]>nextval) {
                                                    nextval = combinedProbas[best][id];
                                                    nextlbl = combinedLabels[best][id];
                                                }
                                                best = nbest;
                                           }
                                        }
                                    }
                                    heap.addValue(nextval,ngb,nextlbl);
                                } else if (topology[ngb]==-1) {
                                    // missing critical points
                                    float nextval = val + mindist;
                                    int nextlbl = lbl;
                                    heap.addValue(nextval,ngb,nextlbl);
                                }
                            }
                        } else {
                            // mark as critical, but don't add it to the heap just yet
                            topology[xyz] = -1;
                            ncritical++;
                        }
		            }
		        }
		    }
		    System.out.println(" critical points: "+ncritical);
		    // update the combined probas with new values
            for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                int xyz = x + nx*y + nx*ny*z;
                if (mask[xyz] && score[xyz]>0) {
                    int id = idmap[xyz];
                    for (int best=0;best<nbest;best++) {
                        if (combinedProbas[best][id]<score[xyz]) {
                            if (combinedLabels[best][id]==label[xyz]) {
                                // same rank: just increase the value
                                combinedProbas[best][id] = score[xyz];
                                combinedLabels[best][id] = label[xyz];
                                best = nbest;
                            } else {
                                // increased rank: swap labels until obj is reached
                                int rank = nbest;
                                for (int next=best+1;next<nbest;next++) {
                                    if (combinedLabels[next][id]==label[xyz]) {
                                        rank = next;
                                        next = nbest;
                                    }
                                }
                                for (int prev=rank;prev>best;prev--) {
                                    if (prev<nbest) {
                                        combinedLabels[prev][id] = combinedLabels[prev-1][id];
                                        combinedProbas[prev][id] = combinedProbas[prev-1][id];
                                    }
                                }
                                combinedLabels[best][id] = label[xyz];
                                combinedProbas[best][id] = score[xyz];
                                best = nbest;
                            }
                        }
                    }
                }
            }
                                    
        }
        return;            
	}

	public void precomputeStoppingStatistics(float spread) {
	    // main idea: region growing from inside, until within volume prior
	    // and a big enough difference in "certainty" score?
	    
		// find appropriate threshold to have correct volume; should use a fast marching approach!
        int[] start = new int[nobj];
        float[] bestscore = new float[nobj];
		for (int obj=nbg;obj<nobj;obj++) bestscore[obj] = -INF;
		double[] voldata = new double[nobj];
		double[] avgbound = new double[nobj];
        double[] devbound = new double[nobj];
        double[] devdiff = new double[nobj];
        int[] nbound = new int[nobj];

		// first, find where then next objects probability is in the stack
        int[][] nextbest = new int[nobj][ndata];
		for (int obj=nbg;obj<nobj;obj++) {
		    for (int n=0;n<ndata;n++) {
		        nextbest[obj][n] = 0;
		        if (combinedLabels[0][n]>100*(obj+1) && combinedLabels[0][n]<100*(obj+2)) {
		            // find the highest proba for a different structure
                    for (int b=1;b<nbest;b++) {
                        if (combinedLabels[b][n]<100*(obj+1) || combinedLabels[b][n]>100*(obj+2)) {
                            nextbest[obj][n] = b;
                            b = nbest;
                        }
                    }
                } else {
                    // find the highest proba for current structure
                    for (int b=1;b<nbest;b++) {
                        if (combinedLabels[b][n]>100*(obj+1) || combinedLabels[b][n]<100*(obj+2)) {
                            nextbest[obj][n] = -b;
                            b = nbest;
                        }
                    }
                }
            }
        }
		// important: skip first label as background (allows for unbounded growth)
        for (int obj=nbg;obj<nobj;obj++) {
           float spatialvol = 0.0f;
		    // find highest scoring voxel as starting point
           for (int s=0;s<nskel;s++) {
           //for (int b=0;b<nbest;b++) {
               for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                    int xyz=x+nx*y+nx*ny*z;
                    if (mask[xyz]) {
                        int id = idmap[xyz];
                        if (skeletonLabels[s][id]==101*(obj+1)) {
                        //if (spatialLabels[b][id]>100*(obj+1) && spatialLabels[b][id]<100*(obj+2)) {
                        //if (combinedLabels[b][id]>100*(obj+1) && combinedLabels[b][id]<100*(obj+2)) {
                        //if (combinedLabels[b][idmap[xyz]]==101*(obj+1)) {
                            float score = skeletonProbas[s][id];
                            //float score = spatialProbas[b][id];
                            //if (b==0) score = combinedProbas[0][id]-combinedProbas[nextbest[obj][id]][id];
                            //else score = combinedProbas[b][id]-combinedProbas[0][id];
                            if (score>bestscore[obj]) {
                                bestscore[obj] = score;
                                start[obj] = xyz;
                                //System.out.println("("+obj+":"+x+","+y+","+z+"="+score+")");
                            }
                        }
                    }
                }
                if (bestscore[obj]>-INF) s = nskel;
                //if (bestscore[obj]>-INF) b = nbest;
            }
            //System.out.println("start ("+obj+":"+start[obj]+"="+bestscore[obj]+")");
		    // compute label volumes
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz=x+nx*y+nx*ny*z;
                if (mask[xyz]) {
                    int id = idmap[xyz];
                    if (combinedLabels[0][id]>100*(obj+1) && combinedLabels[0][id]<100*(obj+2)) {
                    //if (combinedLabels[b][idmap[xyz]]==101*(obj+1)) {
                        voldata[obj] += rx*ry*rz;
                    }
                    if (spatialLabels[0][id]>100*(obj+1) && spatialLabels[0][id]<100*(obj+2)) {
                        spatialvol += rx*ry*rz;
                    }
                }
            }
            // revert to spatial prior volume if nothing is left when including the intensities
            System.out.println("volumes ("+obj+":"+voldata[obj]+"|"+spatialvol+")");
            if (voldata[obj]==0) voldata[obj] = spatialvol;
        }
        // not needed? yes needed
        // Posterior volumes:
        System.out.println("posterior volumes: ");
        for (int obj=nbg;obj<nobj;obj++) {
            //double pvol = 0.5;
            double pvol = FastMath.exp( -0.5*volumeImportance*Numerics.square((FastMath.log(Numerics.max(1.0,voldata[obj]))-logVolMean[obj])/logVolStdv[obj]));
            double logvolmean = (1.0-pvol)*logVolMean[obj]+pvol*FastMath.log(Numerics.max(1.0,voldata[obj]));
            double logvolstdv = FastMath.sqrt( (1.0-pvol)*( Numerics.square(logVolStdv[obj]) )
                                + pvol*Numerics.square(logVolMean[obj]-FastMath.log(Numerics.max(1.0,voldata[obj]))) );

            System.out.print("Label "+obj+": atlas volume = "+FastMath.exp(logVolMean[obj])+" ["+FastMath.exp(logVolMean[obj]-spread*logVolStdv[obj])+", "+FastMath.exp(logVolMean[obj]+spread*logVolStdv[obj])+"]");
            System.out.print(", data volume: "+voldata[obj]+" -> posterior volume = "+FastMath.exp(logvolmean)+" ["+FastMath.exp(logvolmean-spread*logvolstdv)+", "+FastMath.exp(logvolmean+spread*logvolstdv)+"]\n");
            logVolMean[obj] = (float)logvolmean;
            logVolStdv[obj] = (float)logvolstdv;
        }   
        
        for (int obj=nbg;obj<nobj;obj++) {
            System.out.print("Label "+obj+": log vol = "+logVolMean[obj]+" log stdv = "+logVolStdv[obj]+" -> "+FastMath.exp(logVolMean[obj])+" ["+FastMath.exp(logVolMean[obj]-spread*logVolStdv[obj])+", "+FastMath.exp(logVolMean[obj]+spread*logVolStdv[obj])+"]\n");
        }
        // Starting points
        for (int obj=nbg;obj<nobj;obj++) {
            System.out.print("Label "+obj+": start = "+start[obj]+" (score: "+bestscore[obj]+")\n");
        }   
    }
    
	public void conditionalPrecomputedDirectVolumeGrowth(float spread) {
	    // main idea: region growing from inside, until within volume prior
	    // and a big enough difference in "certainty" score?
	    
		// find appropriate threshold to have correct volume; should use a fast marching approach!
		BinaryHeapPair	heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeapPair.MAXTREE);
		int[] labels = new int[ndata];
        int[] start = new int[nobj];
        double[] vol = new double[nobj];
        float[] bestscore = new float[nobj];
		for (int obj=nbg;obj<nobj;obj++) bestscore[obj] = -INF;
		int[][] nextbest = new int[nobj][ndata];
		heap.reset();
		
		// first, find where then next objects probability is in the stack
		for (int obj=nbg;obj<nobj;obj++) {
		    for (int n=0;n<ndata;n++) {
		        nextbest[obj][n] = 0;
		        if (combinedLabels[0][n]>100*(obj+1) && combinedLabels[0][n]<100*(obj+2)) {
		            // find the highest proba for a different structure
                    for (int b=1;b<nbest;b++) {
                        if (combinedLabels[b][n]<100*(obj+1) || combinedLabels[b][n]>100*(obj+2)) {
                            nextbest[obj][n] = b;
                            b = nbest;
                        }
                    }
                } else {
                    // find the highest proba for current structure
                    for (int b=1;b<nbest;b++) {
                        if (combinedLabels[b][n]>100*(obj+1) || combinedLabels[b][n]<100*(obj+2)) {
                            nextbest[obj][n] = -b;
                            b = nbest;
                        }
                    }
                }
            }
        }
		// important: skip first labels as background (allows for unbounded growth)
        for (int obj=nbg;obj<nobj;obj++) {
		    // find highest scoring voxel as starting point
           for (int s=0;s<nskel;s++) {
           //for (int b=0;b<nbest;b++) {
               for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                    int xyz=x+nx*y+nx*ny*z;
                    if (mask[xyz]) {
                        int id = idmap[xyz];
                        if (skeletonLabels[s][id]==101*(obj+1)) {
                            // use a combination of skeleton prior and combined spatial and intensity for starting point (to test!)
                            for (int b=0;b<nbest;b++) {
                                if (combinedLabels[b][id]>100*(obj+1) && combinedLabels[b][id]<100*(obj+2)) {
                                    float score = skeletonProbas[s][id]*combinedProbas[b][id];
                                    if (score>bestscore[obj]) {
                                        bestscore[obj] = score;
                                        start[obj] = xyz;
                                        //System.out.println("("+obj+":"+x+","+y+","+z+"="+score+")");
                                    }
                                }
                            }
                            
                        //if (spatialLabels[b][id]>100*(obj+1) && spatialLabels[b][id]<100*(obj+2)) {
                        //if (combinedLabels[b][id]>100*(obj+1) && combinedLabels[b][id]<100*(obj+2)) {
                        //if (combinedLabels[b][idmap[xyz]]==101*(obj+1)) {
                            /*
                            float score = skeletonProbas[s][id];
                            //float score = spatialProbas[b][id];
                            //if (b==0) score = combinedProbas[0][id]-combinedProbas[nextbest[obj][id]][id];
                            //else score = combinedProbas[b][id]-combinedProbas[0][id];
                            if (score>bestscore[obj]) {
                                bestscore[obj] = score;
                                start[obj] = xyz;
                                //System.out.println("("+obj+":"+x+","+y+","+z+"="+score+")");
                            }*/
                        }
                    }
                }
                if (bestscore[obj]>-INF) s = nskel;
                //if (bestscore[obj]>-INF) b = nbest;
            }
            if (start[obj]==0) System.out.println("\n!! Missing structure: "+obj);    
            // hardcode the starting points?
        }
                
        float[] prev = new float[nobj];
        double[] bestvol = new double[nobj];
        for (int obj=0;obj<nobj;obj++) {
            bestvol[obj] = FastMath.exp(logVolMean[obj]);
        }
        System.out.println("\nOptimized volumes: ");
        for (int obj=nbg;obj<nobj;obj++) System.out.println(obj+": "+bestvol[obj]);
        // re-run one last time to get the segmentation
        heap.reset();
        for (int obj=0;obj<nobj;obj++) {
            vol[obj] = 0.0;
        }
        for(int id=0;id<ndata;id++) labels[id] = 0;
        // init all starting points before everything else (avoid writing over them)
        for (int obj=nbg;obj<nobj;obj++) {
            // hardcode the starting points?
            //heap.addValue(bestscore[obj],start[obj],101*(obj+1));
            vol[obj]+= rx*ry*rz;
            labels[idmap[start[obj]]] = obj;
        }
        // look for neighbors
        for (int obj=nbg;obj<nobj;obj++) {
            for (byte k = 0; k<connectivity; k++) {
                int ngb = Ngb.neighborIndex(k, start[obj], nx, ny, nz);
                if (ngb>0 && ngb<nxyz && mask[ngb]) {
                    if (labels[idmap[ngb]]==0) {
                        for (int best=0;best<nbest;best++) {
                            if (combinedLabels[best][idmap[ngb]]>100*(obj+1) && combinedLabels[best][idmap[ngb]]<100*(obj+2)) {
                                float score = combinedProbas[best][idmap[ngb]]-combinedProbas[Numerics.max(0,nextbest[obj][idmap[ngb]])][idmap[ngb]];
                                float offset = 0.0f;
                                for (int s=0;s<nskel;s++) {
                                    if (skeletonLabels[s][idmap[ngb]]==101*(obj+1)) {
                                        offset = -Numerics.abs(skeletonProbas[s][idmap[ngb]]-skeletonProbas[s][idmap[start[obj]]]);
                                        s = nskel;
                                    }
                                }
                                score += offset;
                                //score *= (1.0f+offset);
                                if (k>=18) if (score>0) score *= ISQRT3; else score /= ISQRT3;
                                else if (k>=6) if (score>0) score *= ISQRT2; else score /= ISQRT2;
                                heap.addValue(score,ngb,combinedLabels[best][idmap[ngb]]);
                                best=nbest;
                            }
                        }
                    }
                }
            }
        }
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int xyz = heap.getFirstId1();
            int obj1obj2 = heap.getFirstId2();
            heap.removeFirst();
            if (labels[idmap[xyz]]==0) {
                int obj = Numerics.floor(obj1obj2/100)-1;
                if (vol[obj]<bestvol[obj]) {
                    // update the values
                    vol[obj]+=rx*ry*rz;
                    labels[idmap[xyz]] = obj;
                
                    // add neighbors
                    //for (byte k = 0; k<6; k++) {
                    for (byte k = 0; k<connectivity; k++) {
                        int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                        if (ngb>0 && ngb<nxyz && idmap[ngb]>-1) {
                            if (mask[ngb]) {
                                if (labels[idmap[ngb]]==0) {
                                    for (int best=0;best<nbest;best++) {
                                        if (combinedLabels[best][idmap[ngb]]>100*(obj+1) && combinedLabels[best][idmap[ngb]]<100*(obj+2)) {
                                            float newscore = combinedProbas[best][idmap[ngb]]-combinedProbas[Numerics.max(0,nextbest[obj][idmap[ngb]])][idmap[ngb]];
                                            float offset = 0.0f;
                                            for (int s=0;s<nskel;s++) {
                                                if (skeletonLabels[s][idmap[ngb]]==101*(obj+1)) {
                                                    offset = -Numerics.abs(skeletonProbas[s][idmap[ngb]]-skeletonProbas[s][idmap[xyz]]);
                                                    s = nskel;
                                                }
                                            }
                                            newscore += offset;
                                            //newscore *= (1.0f+offset);
                                            if (k>=18) if (newscore>0) newscore *= ISQRT3; else newscore /= ISQRT3;
                                            else if (k>=6) if (newscore>0) newscore *= ISQRT2; else newscore /= ISQRT2;
                                            // weight by relative size, so smaller structures are prioritized? not useful
                                            //newscore *= (1.0f-vol[obj]/bestvol[obj]);
                                            heap.addValue(newscore,ngb,combinedLabels[best][idmap[ngb]]);
                                            best=nbest;
                                        }

                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // final segmentation: collapse onto result images
        finalLabel = new int[nxyz];
        finalProba = new float[nxyz];
        float[] kept = new float[nobj];
        for (int x=1;x<ntx-1;x++) for (int y=1;y<nty-1;y++) for (int z=1;z<ntz-1;z++) {
            int xyz = x+ntx*y+ntx*nty*z;
            if (mask[xyz]) {
                int obj = labels[idmap[xyz]];
                if (obj==0) {
                    // check for all background classes
                    for (int best=0;best<nbest;best++) {
                        if (combinedLabels[best][idmap[xyz]]>100 && combinedLabels[best][idmap[xyz]]<100*(nbg+1)) {
                            finalProba[xyz] = Numerics.max(finalProba[xyz],combinedProbas[best][idmap[xyz]]);
                            best=nbest;
                        }
                    }
                } else {
                    for (int best=0;best<nbest;best++) {
                        if (combinedLabels[best][idmap[xyz]]>100*(obj+1) && combinedLabels[best][idmap[xyz]]<100*(obj+2)) {
                            finalProba[xyz] = Numerics.max(finalProba[xyz],combinedProbas[best][idmap[xyz]]);
                            best=nbest;
                        }
                    }
                }
                finalLabel[xyz] = obj;
            }
        }
        return;            
	}

	public void conditionalPrecomputedRelativeVolumeGrowth(float spread) {
	    // main idea: region growing from inside, until within volume prior
	    // and a big enough difference in "certainty" score?
	    
	    // use starting point probability as a reweighting factor, 
	    // to keep small, poorly defined regions from disappearing
	    
		// find appropriate threshold to have correct volume; should use a fast marching approach!
		BinaryHeapPair	heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeapPair.MAXTREE);
		int[] labels = new int[ndata];
        int[] start = new int[nobj];
        double[] vol = new double[nobj];
        float[] bestscore = new float[nobj];
		for (int obj=nbg;obj<nobj;obj++) bestscore[obj] = -INF;
		int[][] nextbest = new int[nobj][ndata];
		heap.reset();
		
		// first, find where then next objects probability is in the stack
		for (int obj=nbg;obj<nobj;obj++) {
		    for (int n=0;n<ndata;n++) {
		        nextbest[obj][n] = 0;
		        if (combinedLabels[0][n]>100*(obj+1) && combinedLabels[0][n]<100*(obj+2)) {
		            // find the highest proba for a different structure
                    for (int b=1;b<nbest;b++) {
                        if (combinedLabels[b][n]<100*(obj+1) || combinedLabels[b][n]>100*(obj+2)) {
                            nextbest[obj][n] = b;
                            b = nbest;
                        }
                    }
                } else {
                    // find the highest proba for current structure
                    for (int b=1;b<nbest;b++) {
                        if (combinedLabels[b][n]>100*(obj+1) || combinedLabels[b][n]<100*(obj+2)) {
                            nextbest[obj][n] = -b;
                            b = nbest;
                        }
                    }
                }
            }
        }
		// important: skip first labels as background (allows for unbounded growth)
        for (int obj=nbg;obj<nobj;obj++) {
		    // find highest scoring voxel as starting point
           for (int s=0;s<nskel;s++) {
           //for (int b=0;b<nbest;b++) {
               for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                    int xyz=x+nx*y+nx*ny*z;
                    if (mask[xyz]) {
                        int id = idmap[xyz];
                        if (skeletonLabels[s][id]==101*(obj+1)) {
                            // use a combination of skeleton prior and combined spatial and intensity for starting point (to test!)
                            for (int b=0;b<nbest;b++) {
                                if (combinedLabels[b][id]>100*(obj+1) && combinedLabels[b][id]<100*(obj+2)) {
                                    float score = skeletonProbas[s][id]*combinedProbas[b][id];
                                    if (score>bestscore[obj]) {
                                        bestscore[obj] = score;
                                        start[obj] = xyz;
                                        //System.out.println("("+obj+":"+x+","+y+","+z+"="+score+")");
                                    }
                                }
                            }
                            
                        //if (spatialLabels[b][id]>100*(obj+1) && spatialLabels[b][id]<100*(obj+2)) {
                        //if (combinedLabels[b][id]>100*(obj+1) && combinedLabels[b][id]<100*(obj+2)) {
                        //if (combinedLabels[b][idmap[xyz]]==101*(obj+1)) {
                            /*
                            float score = skeletonProbas[s][id];
                            //float score = spatialProbas[b][id];
                            //if (b==0) score = combinedProbas[0][id]-combinedProbas[nextbest[obj][id]][id];
                            //else score = combinedProbas[b][id]-combinedProbas[0][id];
                            if (score>bestscore[obj]) {
                                bestscore[obj] = score;
                                start[obj] = xyz;
                                //System.out.println("("+obj+":"+x+","+y+","+z+"="+score+")");
                            }*/
                        }
                    }
                }
                if (bestscore[obj]>-INF) s = nskel;
                //if (bestscore[obj]>-INF) b = nbest;
            }
            if (start[obj]==0) System.out.println("\n!! Missing structure: "+obj);    
            // hardcode the starting points?
        }
                
        float[] prev = new float[nobj];
        double[] bestvol = new double[nobj];
        for (int obj=0;obj<nobj;obj++) {
            bestvol[obj] = FastMath.exp(logVolMean[obj]);
        }
        System.out.println("\nOptimized volumes: ");
        for (int obj=nbg;obj<nobj;obj++) System.out.println(obj+": "+bestvol[obj]);
        // re-run one last time to get the segmentation
        heap.reset();
        for (int obj=0;obj<nobj;obj++) {
            vol[obj] = 0.0;
        }
        for(int id=0;id<ndata;id++) labels[id] = 0;
        // init all starting points before everything else (avoid writing over them)
        for (int obj=nbg;obj<nobj;obj++) {
            // hardcode the starting points?
            //heap.addValue(bestscore[obj],start[obj],101*(obj+1));
            vol[obj]+= rx*ry*rz;
            labels[idmap[start[obj]]] = obj;
        }
        // look for neighbors
        for (int obj=nbg;obj<nobj;obj++) {
            for (byte k = 0; k<connectivity; k++) {
                int ngb = Ngb.neighborIndex(k, start[obj], nx, ny, nz);
                if (ngb>0 && ngb<nxyz && mask[ngb]) {
                    if (labels[idmap[ngb]]==0) {
                        for (int best=0;best<nbest;best++) {
                            if (combinedLabels[best][idmap[ngb]]>100*(obj+1) && combinedLabels[best][idmap[ngb]]<100*(obj+2)) {
                                float score = combinedProbas[best][idmap[ngb]]-combinedProbas[Numerics.max(0,nextbest[obj][idmap[ngb]])][idmap[ngb]];
                                float offset = 0.0f;
                                for (int s=0;s<nskel;s++) {
                                    if (skeletonLabels[s][idmap[ngb]]==101*(obj+1)) {
                                        offset = -Numerics.abs(skeletonProbas[s][idmap[ngb]]-skeletonProbas[s][idmap[start[obj]]]);
                                        s = nskel;
                                    }
                                }
                                score += offset;
                                //score *= (1.0f+offset);
                                if (k>=18) if (score>0) score *= ISQRT3; else score /= ISQRT3;
                                else if (k>=6) if (score>0) score *= ISQRT2; else score /= ISQRT2;
                                score = score/bestscore[obj];
                                heap.addValue(score,ngb,combinedLabels[best][idmap[ngb]]);
                                best=nbest;
                            }
                        }
                    }
                }
            }
        }
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int xyz = heap.getFirstId1();
            int obj1obj2 = heap.getFirstId2();
            heap.removeFirst();
            if (labels[idmap[xyz]]==0) {
                int obj = Numerics.floor(obj1obj2/100)-1;
                if (vol[obj]<bestvol[obj]) {
                    // update the values
                    vol[obj]+=rx*ry*rz;
                    labels[idmap[xyz]] = obj;
                
                    // add neighbors
                    //for (byte k = 0; k<6; k++) {
                    for (byte k = 0; k<connectivity; k++) {
                        int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                        if (ngb>0 && ngb<nxyz && idmap[ngb]>-1) {
                            if (mask[ngb]) {
                                if (labels[idmap[ngb]]==0) {
                                    for (int best=0;best<nbest;best++) {
                                        if (combinedLabels[best][idmap[ngb]]>100*(obj+1) && combinedLabels[best][idmap[ngb]]<100*(obj+2)) {
                                            float newscore = combinedProbas[best][idmap[ngb]]-combinedProbas[Numerics.max(0,nextbest[obj][idmap[ngb]])][idmap[ngb]];
                                            float offset = 0.0f;
                                            for (int s=0;s<nskel;s++) {
                                                if (skeletonLabels[s][idmap[ngb]]==101*(obj+1)) {
                                                    offset = -Numerics.abs(skeletonProbas[s][idmap[ngb]]-skeletonProbas[s][idmap[xyz]]);
                                                    s = nskel;
                                                }
                                            }
                                            newscore += offset;
                                            //newscore *= (1.0f+offset);
                                            if (k>=18) if (newscore>0) newscore *= ISQRT3; else newscore /= ISQRT3;
                                            else if (k>=6) if (newscore>0) newscore *= ISQRT2; else newscore /= ISQRT2;
                                            // weight by relative size, so smaller structures are prioritized? not useful
                                            //newscore *= (1.0f-vol[obj]/bestvol[obj]);
                                            newscore = newscore/bestscore[obj];
                                            heap.addValue(newscore,ngb,combinedLabels[best][idmap[ngb]]);
                                            best=nbest;
                                        }

                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // final segmentation: collapse onto result images
        finalLabel = new int[nxyz];
        finalProba = new float[nxyz];
        float[] kept = new float[nobj];
        for (int x=1;x<ntx-1;x++) for (int y=1;y<nty-1;y++) for (int z=1;z<ntz-1;z++) {
            int xyz = x+ntx*y+ntx*nty*z;
            if (mask[xyz]) {
                int obj = labels[idmap[xyz]];
                if (obj==0) {
                    // check for all background classes
                    for (int best=0;best<nbest;best++) {
                        if (combinedLabels[best][idmap[xyz]]>100 && combinedLabels[best][idmap[xyz]]<100*(nbg+1)) {
                            finalProba[xyz] = Numerics.max(finalProba[xyz],combinedProbas[best][idmap[xyz]]);
                            best=nbest;
                        }
                    }
                } else {
                    for (int best=0;best<nbest;best++) {
                        if (combinedLabels[best][idmap[xyz]]>100*(obj+1) && combinedLabels[best][idmap[xyz]]<100*(obj+2)) {
                            finalProba[xyz] = Numerics.max(finalProba[xyz],combinedProbas[best][idmap[xyz]]);
                            best=nbest;
                        }
                    }
                }
                finalLabel[xyz] = obj;
            }
        }
        return;            
	}
	
	public void conditionalPrecomputedOrderedVolumeGrowth(float spread) {
	    // main idea: region growing from inside, until within volume prior
	    // and a big enough difference in "certainty" score?
	    
	    // this version starts with smallest structures to ensure they 
	    // have the opportunity to reach their expected volume, even if other 
	    // structure have higher probability there
	    
		// find appropriate threshold to have correct volume; should use a fast marching approach!
		BinaryHeapPair	heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeapPair.MAXTREE);
		int[] labels = new int[ndata];
        int[] start = new int[nobj];
        double[] vol = new double[nobj];
        float[] bestscore = new float[nobj];
		for (int obj=nbg;obj<nobj;obj++) bestscore[obj] = -INF;
		int[][] nextbest = new int[nobj][ndata];
		heap.reset();
		
		// first, find where then next objects probability is in the stack
		for (int obj=nbg;obj<nobj;obj++) {
		    for (int n=0;n<ndata;n++) {
		        nextbest[obj][n] = 0;
		        if (combinedLabels[0][n]>100*(obj+1) && combinedLabels[0][n]<100*(obj+2)) {
		            // find the highest proba for a different structure
                    for (int b=1;b<nbest;b++) {
                        if (combinedLabels[b][n]<100*(obj+1) || combinedLabels[b][n]>100*(obj+2)) {
                            nextbest[obj][n] = b;
                            b = nbest;
                        }
                    }
                } else {
                    // find the highest proba for current structure
                    for (int b=1;b<nbest;b++) {
                        if (combinedLabels[b][n]>100*(obj+1) || combinedLabels[b][n]<100*(obj+2)) {
                            nextbest[obj][n] = -b;
                            b = nbest;
                        }
                    }
                }
            }
        }
		// important: skip first labels as background (allows for unbounded growth)
        for (int obj=nbg;obj<nobj;obj++) {
		    // find highest scoring voxel as starting point
           for (int s=0;s<nskel;s++) {
           //for (int b=0;b<nbest;b++) {
               for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                    int xyz=x+nx*y+nx*ny*z;
                    if (mask[xyz]) {
                        int id = idmap[xyz];
                        if (skeletonLabels[s][id]==101*(obj+1)) {
                            // use a combination of skeleton prior and combined spatial and intensity for starting point (to test!)
                            for (int b=0;b<nbest;b++) {
                                if (combinedLabels[b][id]>100*(obj+1) && combinedLabels[b][id]<100*(obj+2)) {
                                    float score = skeletonProbas[s][id]*combinedProbas[b][id];
                                    if (score>bestscore[obj]) {
                                        bestscore[obj] = score;
                                        start[obj] = xyz;
                                        //System.out.println("("+obj+":"+x+","+y+","+z+"="+score+")");
                                    }
                                }
                            }
                            
                        //if (spatialLabels[b][id]>100*(obj+1) && spatialLabels[b][id]<100*(obj+2)) {
                        //if (combinedLabels[b][id]>100*(obj+1) && combinedLabels[b][id]<100*(obj+2)) {
                        //if (combinedLabels[b][idmap[xyz]]==101*(obj+1)) {
                            /*
                            float score = skeletonProbas[s][id];
                            //float score = spatialProbas[b][id];
                            //if (b==0) score = combinedProbas[0][id]-combinedProbas[nextbest[obj][id]][id];
                            //else score = combinedProbas[b][id]-combinedProbas[0][id];
                            if (score>bestscore[obj]) {
                                bestscore[obj] = score;
                                start[obj] = xyz;
                                //System.out.println("("+obj+":"+x+","+y+","+z+"="+score+")");
                            }*/
                        }
                    }
                }
                if (bestscore[obj]>-INF) s = nskel;
                //if (bestscore[obj]>-INF) b = nbest;
            }
            if (start[obj]==0) System.out.println("\n!! Missing structure: "+obj);    
            // hardcode the starting points?
        }
                
        float[] prev = new float[nobj];
        double[] bestvol = new double[nobj];
        for (int obj=0;obj<nobj;obj++) {
            bestvol[obj] = FastMath.exp(logVolMean[obj]);
        }
        //System.out.println("\nOptimized volumes: ");
        //for (int obj=nbg;obj<nobj;obj++) System.out.println(obj+": "+bestvol[obj]);
        // re-run one last time to get the segmentation
        heap.reset();
        for (int obj=0;obj<nobj;obj++) {
            vol[obj] = 0.0;
        }
        for(int id=0;id<ndata;id++) labels[id] = 0;
        // init all starting points before everything else (avoid writing over them)
        for (int obj=nbg;obj<nobj;obj++) {
            // hardcode the starting points?
            //heap.addValue(bestscore[obj],start[obj],101*(obj+1));
            vol[obj]+= rx*ry*rz;
            labels[idmap[start[obj]]] = obj;
        }
        // order objects by increasing volume
        int[] ordering = new int[nobj];
        for (int obj=nbg;obj<nobj;obj++) {
            ordering[obj] = obj;
        }
        for (int obj0=nbg;obj0<nobj;obj0++) {
            for (int obj=nbg;obj<nobj;obj++) {
				if (bestvol[ordering[obj]]>bestvol[ordering[obj0]]) {
					// switch place
					int tmp = ordering[obj0];
					ordering[obj0] = ordering[obj];
					ordering[obj] = tmp;
				}
			}
		}
		System.out.println("\nOrdered volumes: ");
        for (int obj=nbg;obj<nobj;obj++) System.out.println(ordering[obj]+": "+bestvol[ordering[obj]]);
        
        // look for neighbors
        for (int ord=nbg;ord<nobj;ord++) {
            int obj = ordering[ord];
            for (byte k = 0; k<connectivity; k++) {
                int ngb = Ngb.neighborIndex(k, start[obj], nx, ny, nz);
                if (ngb>0 && ngb<nxyz && mask[ngb]) {
                    if (labels[idmap[ngb]]==0) {
                        for (int best=0;best<nbest;best++) {
                            if (combinedLabels[best][idmap[ngb]]>100*(obj+1) && combinedLabels[best][idmap[ngb]]<100*(obj+2)) {
                                float score = combinedProbas[best][idmap[ngb]]-combinedProbas[Numerics.max(0,nextbest[obj][idmap[ngb]])][idmap[ngb]];
                                float offset = 0.0f;
                                for (int s=0;s<nskel;s++) {
                                    if (skeletonLabels[s][idmap[ngb]]==101*(obj+1)) {
                                        offset = -Numerics.abs(skeletonProbas[s][idmap[ngb]]-skeletonProbas[s][idmap[start[obj]]]);
                                        s = nskel;
                                    }
                                }
                                score += offset;
                                //score *= (1.0f+offset);
                                if (k>=18) if (score>0) score *= ISQRT3; else score /= ISQRT3;
                                else if (k>=6) if (score>0) score *= ISQRT2; else score /= ISQRT2;
                                heap.addValue(score,ngb,combinedLabels[best][idmap[ngb]]);
                                best=nbest;
                            }
                        }
                    }
                }
            }
            while (heap.isNotEmpty()) {
                float score = heap.getFirst();
                int xyz = heap.getFirstId1();
                int obj1obj2 = heap.getFirstId2();
                heap.removeFirst();
                if (labels[idmap[xyz]]==0) {
                    // here we always use the same object, we could skip the elaborate labeling...
                    //int obj = Numerics.floor(obj1obj2/100)-1;
                    if (vol[obj]<bestvol[obj]) {
                        // update the values
                        vol[obj]+=rx*ry*rz;
                        labels[idmap[xyz]] = obj;
                    
                        // add neighbors
                        //for (byte k = 0; k<6; k++) {
                        for (byte k = 0; k<connectivity; k++) {
                            int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                            if (ngb>0 && ngb<nxyz && idmap[ngb]>-1) {
                                if (mask[ngb]) {
                                    if (labels[idmap[ngb]]==0) {
                                        for (int best=0;best<nbest;best++) {
                                            if (combinedLabels[best][idmap[ngb]]>100*(obj+1) && combinedLabels[best][idmap[ngb]]<100*(obj+2)) {
                                                float newscore = combinedProbas[best][idmap[ngb]]-combinedProbas[Numerics.max(0,nextbest[obj][idmap[ngb]])][idmap[ngb]];
                                                float offset = 0.0f;
                                                for (int s=0;s<nskel;s++) {
                                                    if (skeletonLabels[s][idmap[ngb]]==101*(obj+1)) {
                                                        offset = -Numerics.abs(skeletonProbas[s][idmap[ngb]]-skeletonProbas[s][idmap[xyz]]);
                                                        s = nskel;
                                                    }
                                                }
                                                newscore += offset;
                                                //newscore *= (1.0f+offset);
                                                if (k>=18) if (newscore>0) newscore *= ISQRT3; else newscore /= ISQRT3;
                                                else if (k>=6) if (newscore>0) newscore *= ISQRT2; else newscore /= ISQRT2;
                                                // weight by relative size, so smaller structures are prioritized? not useful
                                                //newscore *= (1.0f-vol[obj]/bestvol[obj]);
                                                heap.addValue(newscore,ngb,combinedLabels[best][idmap[ngb]]);
                                                best=nbest;
                                            }
    
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // final segmentation: collapse onto result images
        finalLabel = new int[nxyz];
        finalProba = new float[nxyz];
        float[] kept = new float[nobj];
        for (int x=1;x<ntx-1;x++) for (int y=1;y<nty-1;y++) for (int z=1;z<ntz-1;z++) {
            int xyz = x+ntx*y+ntx*nty*z;
            if (mask[xyz]) {
                int obj = labels[idmap[xyz]];
                if (obj==0) {
                    // check for all background classes
                    for (int best=0;best<nbest;best++) {
                        if (combinedLabels[best][idmap[xyz]]>100 && combinedLabels[best][idmap[xyz]]<100*(nbg+1)) {
                            finalProba[xyz] = Numerics.max(finalProba[xyz],combinedProbas[best][idmap[xyz]]);
                            best=nbest;
                        }
                    }
                } else {
                    for (int best=0;best<nbest;best++) {
                        if (combinedLabels[best][idmap[xyz]]>100*(obj+1) && combinedLabels[best][idmap[xyz]]<100*(obj+2)) {
                            finalProba[xyz] = Numerics.max(finalProba[xyz],combinedProbas[best][idmap[xyz]]);
                            best=nbest;
                        }
                    }
                }
                finalLabel[xyz] = obj;
            }
        }
        return;            
	}

}
