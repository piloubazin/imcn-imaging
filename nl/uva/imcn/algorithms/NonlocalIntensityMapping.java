package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import Jama.*;

/*
 * @author Pierre-Louis bazin (pilou.bazin@uva.nl)
 *
 */
public class NonlocalIntensityMapping {
	float[] image = null;
	float[][] reference = null;
	float[][] mapped = null;
	float[][] correl = null;
	float[] weights = null;
	
	int nref = 0;
	
	int nx, ny, nz, nxyz;
	float rx, ry, rz;
	
	// distance for patch and search windows
	int patch = 2;
	int search = 3;
	boolean useMedian = false;
	boolean useCorrelations = false;
	
	float[] result;
	
	// set inputs
	public final void setInputImage(float[] val) { image = val; }
	
	public final void setReferenceNumber(int val) { 
	    nref = val;
	    reference = new float[nref][];
	    mapped = new float[nref][];
	    weights = new float[nref];
	    for (int r=0;r<nref;r++) weights[r] = 1.0f;
	}
	public final void setReferenceImageAt(int num, float[] val) { reference[num] = val; }
	public final void setMappedImageAt(int num, float[] val) { mapped[num] = val; }
	public final void setWeightAt(int num, float val) { weights[num] = val; }
	
	public final void setPatchDistance(int val) { patch = val; }
	public final void setSearchDistance(int val) { search = val; }

	public final void setUseMedian(boolean val) { useMedian = val; }

	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }

	// outputs
	public final float[] getMappedImage() { return result; }

	public void execute2D() {

	    // optionally, pre-compute patch correlations between reference and mapped contrast
	    if (useCorrelations) precomputeReferenceMappedCorrelation();
	    
	    // look for neighbor patches following Coupe 2011
        int nw = (2*search+1)*(2*search+1)*nref;
        float[] distance = new float[nw];
        float[] corrdist = new float[nw];

        result = new float[nxyz];
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
                            && xs+dx>=0 && xs+dx<nx && ys+dy>=0 && ys<ny && reference[r][xs+dx+nx*(ys+dy)]!=0) {
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
                    if (useCorrelations) corrdist[w] = Numerics.abs(correl[r][xs+nx*ys]);
                    else corrdist[w] = 1.0f;
                    w++;
                } else {
                    distance[w] = -1.0f;
                    w++;
                }
            }
            //if (mindist==0) mindist = 0.001f;
            if (useMedian) {
                float[] weight = new float[nw];
                float[] intens = new float[nw];
                w=0;
                for (int xs=x-search;xs<=x+search;xs++) for (int ys=y-search;ys<=y+search;ys++) for (int r=0;r<nref;r++) { 
                    if (distance[w]>=0.0f) {
                        weight[w] = weights[r]*corrdist[w]*(float)FastMath.exp(-distance[w]/mindist);
                        intens[w] = mapped[r][xs+nx*ys];
                    } else {
                        weight[w] = -1.0f;
                        intens[w] = 0.0f;
                    }
                    w++;
                }
                result[x+nx*y] = weightedMedian(intens, weight);
            } else {
                // weighted average
                double sum = 0.0f;
                double den = 0.0f;
                
                w=0;
                for (int xs=x-search;xs<=x+search;xs++) for (int ys=y-search;ys<=y+search;ys++) for (int r=0;r<nref;r++) { 
                    if (distance[w]>=0.0f) {
                        double weight = weights[r]*corrdist[w]*FastMath.exp(-distance[w]/mindist);
                        sum += weight*mapped[r][xs+nx*ys];
                        den += weight;
                    }
                    w++;
                }
                result[x+nx*y] = (float)(sum/den);
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

	private final void precomputeReferenceMappedCorrelation() {
	    
        correl = new float[nref][nxyz];
        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int r=0;r<nref;r++) if (reference[r][x+nx*y]!=0) {
            correl[r][x+nx*y] = 0.0f;
            
            double meanref = 0.0;
            double meanmap = 0.0;
            double npatch = 0.0;
            
            for (int dx=-patch;dx<=patch;dx++) for (int dy=-patch;dy<=patch;dy++) {
                if (x+dx>=0 && x+dx<nx && y+dy>=0 && y+dy<ny && reference[r][x+dx+nx*(y+dy)]!=0 && mapped[r][x+dx+nx*(y+dy)]!=0) {
                    meanref += reference[r][x+dx+nx*(y+dy)];
                    meanmap += mapped[r][x+dx+nx*(y+dy)];
                    npatch++;
                }
            }
            if (npatch>0) {
                meanref /= npatch;
                meanmap /= npatch;
                
                double prod = 0.0;
                double varref = 0.0;
                double varmap = 0.0;
                for (int dx=-patch;dx<=patch;dx++) for (int dy=-patch;dy<=patch;dy++) {
                    if (x+dx>=0 && x+dx<nx && y+dy>=0 && y+dy<ny && reference[r][x+dx+nx*(y+dy)]!=0 && mapped[r][x+dx+nx*(y+dy)]!=0) {
                        prod += (reference[r][x+dx+nx*(y+dy)]-meanref)*(mapped[r][x+dx+nx*(y+dy)]-meanmap);
                        varref += (reference[r][x+dx+nx*(y+dy)]-meanref)*(reference[r][x+dx+nx*(y+dy)]-meanref);
                        varmap += (mapped[r][x+dx+nx*(y+dy)]-meanmap)*(mapped[r][x+dx+nx*(y+dy)]-meanmap);
                    }
                }
                
                correl[r][x+nx*y] = (float)(prod/FastMath.sqrt(varref*varmap));
            }
	    }
	    return;
	}
}
