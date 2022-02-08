package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import Jama.*;

/*
 * @author Pierre-Louis bazin (pilou.bazin@uva.nl)
 *
 */
public class NonlocalLinearFiberInterpolation {
	float[] image = null;
	float[][] reference = null;
	float[][] mappedProba = null;
	float[][] mappedTheta = null;
	float[][] mappedLambda = null;
	float[] weights = null;
	
	int nref = 0;
	
	int nx, ny, nz, nxyz;
	float rx, ry, rz;
	
	int nk = 0;
	
	// distance for patch and search windows
	int patch = 2;
	int search = 3;
	boolean useMedian = true;
	
	float[] resultProba;
	float[] resultTheta;
	float[] resultLambda;
	
	// set inputs
	public final void setInputImage(float[] val) { image = val; }
	
	public final void setReferenceNumber(int val) { 
	    nref = val;
	    reference = new float[nref][];
	    mappedProba = new float[nref][];
	    mappedTheta = new float[nref][];
	    mappedLambda = new float[nref][];
	    weights = new float[nref];
	    for (int r=0;r<nref;r++) weights[r] = 1.0f;
	}
	public final void setReferenceImageAt(int num, float[] val) { reference[num] = val; }
	public final void setMappedProbaAt(int num, float[] val) { mappedProba[num] = val; }
	public final void setMappedThetaAt(int num, float[] val) { mappedTheta[num] = val; }
	public final void setMappedLambdaAt(int num, float[] val) { mappedLambda[num] = val; }
	public final void setWeightAt(int num, float val) { weights[num] = val; }
	
	public final void setPatchDistance(int val) { patch = val; }
	public final void setSearchDistance(int val) { search = val; }

	public final void setLineNumber(int val) { nk = val; }
	
	public final void setUseMedian(boolean val) { useMedian = val; }

	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }

	// outputs
	public final float[] getMappedProba() { return resultProba; }
	public final float[] getMappedTheta() { return resultTheta; }
	public final float[] getMappedLambda() { return resultLambda; }

	public void execute2D() {
	    
	    // look for neighbor patches following Coupe 2011
        int nw = (2*search+1)*(2*search+1)*nref;
        float[] distance = new float[nw];

        resultProba = new float[nxyz*nk];
        resultTheta = new float[nxyz*nk];
        resultLambda = new float[nxyz*nk];
        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) if (image[x+nx*y]!=0) {
            float mindist = 1e9f;
            int w=0;
            // window of search
            for (int xs=x-search;xs<=x+search;xs++) for (int ys=y-search;ys<=y+search;ys++) for (int r=0;r<nref;r++) { 
                if (xs>=0 && xs<nx && ys>=0 && ys<ny && reference[r][xs+nx*ys]!=0) {
                    // compute patch distances
                    distance[w] = 0.0f;
                    float ndist = 0.0f;
                    for (int dx=-patch;dx<=patch;dx++) for (int dy=-patch;dy<=patch;dy++) {
                        if (x+dx>=0 && x+dx<nx && y+dy>=0 && y+dy<ny && image[x+dx+nx*(y+dy)]!=0
                            && xs+dx>=0 && xs+dx<nx && ys+dy>=0 && ys+dy<ny && reference[r][xs+dx+nx*(ys+dy)]!=0) {
                            distance[w] += Numerics.square(image[x+dx+nx*(y+dy)]-reference[r][xs+dx+nx*(ys+dy)]);
                            ndist++;
                        }
                    }
                    if (ndist>0) {
                        distance[w] /= ndist;
                        // no update if exactly zero (in case the target in included in reference)
                        if (distance[w] < mindist && distance[w]>0) mindist = distance[w];
                    } else {
                        distance[w] = -1.0f;
                    }
                    w++;
                } else {
                    distance[w] = -1.0f;
                    w++;
                }
            }
            //if (mindist==0) mindist = 0.001f;
            if (useMedian) {
                float[] weight = new float[nw];
                float[] probas = new float[nw];
                float[] thetas = new float[nw];
                float[] lambdas = new float[nw];
                for (int k=0;k<nk;k++) {
                    w=0;
                    for (int xs=x-search;xs<=x+search;xs++) for (int ys=y-search;ys<=y+search;ys++) for (int r=0;r<nref;r++) {
                        
                        if (distance[w]>=0.0f) {
                            weight[w] = weights[r]*(float)FastMath.exp(-distance[w]/mindist);
                            probas[w] = mappedProba[r][xs+nx*ys+nx*ny*k];
                            thetas[w] = mappedTheta[r][xs+nx*ys+nx*ny*k];
                            lambdas[w] = mappedLambda[r][xs+nx*ys+nx*ny*k];
                        } else {
                            weight[w] = -1.0f;
                            probas[w] = 0.0f;
                            thetas[w] = 0.0f;
                            lambdas[w] = 0.0f;
                        }
                        w++;
                    }
                    resultProba[x+nx*y+nx*ny*k] = weightedMedian(probas, weight);
                    resultTheta[x+nx*y+nx*ny*k] = weightedMedian(thetas, weight);
                    resultLambda[x+nx*y+nx*ny*k] = weightedMedian(lambdas, weight);
                }
            } else {
                for (int k=0;k<nk;k++) {
                    // weighted average
                    double sumP = 0.0f;
                    double sumT = 0.0f;
                    double sumL = 0.0f;
                    double den = 0.0f;
                    
                    w=0;
                    for (int xs=x-search;xs<=x+search;xs++) for (int ys=y-search;ys<=y+search;ys++) for (int r=0;r<nref;r++) { 
                        if (distance[w]>=0.0f) {
                            double weight = weights[r]*FastMath.exp(-distance[w]/mindist);
                            sumP += weight*mappedProba[r][xs+nx*ys+nx*ny*k];
                            sumT += weight*mappedTheta[r][xs+nx*ys+nx*ny*k];
                            sumL += weight*mappedLambda[r][xs+nx*ys+nx*ny*k];
                            den += weight;
                        }
                        w++;
                    }
                    resultProba[x+nx*y+nx*ny*k] = (float)(sumP/den);
                    resultTheta[x+nx*y+nx*ny*k] = (float)(sumT/den);
                    resultLambda[x+nx*y+nx*ny*k] = (float)(sumL/den);
                }
            }
	    }
		System.out.print("Done\n");
	}
	
    private static final float weightedMedian(float[] val, float[] wgh) {
	    float tmp;
	    for (int n=0;n<val.length;n++) if (wgh[n]!=-1) {
			for (int m=n+1;m<val.length;m++) if (wgh[m]!=-1) {
			    // here we ignore the -1 values entirely
				if (val[m]<val[n]) {
					// switch place
					tmp = val[n];
					val[n] = val[m];
					val[m] = tmp;
					// switch weights
					tmp = wgh[n];
					wgh[n] = wgh[m];
					wgh[m] = tmp;
				}
			}
		}
		float sum = 0.0f;
		for (int n=0;n<val.length;n++) if (wgh[n]!=-1) {
		    sum += wgh[n];
		}
		float half = 0.0f;
		float med = 0.0f;
		for (int n=0;n<val.length;n++) if (wgh[n]!=-1) {
		    half += wgh[n];
		    if (half>=0.5f*sum) {
                if (n>0) {
                    // weighted sum?
                    med = ( (0.5f*sum-half+wgh[n])*val[n] + (half-0.5f*sum)*val[n-1] )/wgh[n];
                } else {
                    med = val[n];
                }
                n=val.length;
            }
        }
        return med;
	}	
}
