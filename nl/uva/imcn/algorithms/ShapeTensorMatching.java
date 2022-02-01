package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import Jama.*;

/*
 * @author Pierre-Louis bazin (pilou.bazin@uva.nl)
 *
 */
public class ShapeTensorMatching {
	float[] ani = null;
	float[] theta = null;
	float[][] images = null;
	float[][] reference = null;
	boolean[] isAngular = null;
	
	int nimg = 0;
	
	int nx, ny, nz, nxyz;
	float rx, ry, rz;
	
	// distance for search pencil
	float angle = (float)(5.0*FastMath.PI/180.0);
	float distance = 10.0f;
	
	// exclude regions with low anisotropy
	float threshold = 0.2f;
	
	// define a patch size for similarity computation
	int patch = 3;
	
	boolean usePencil=false;
	
	int nk = 4;
	
	float[] result;
	
	// set inputs
	public final void setInputTheta(float[] val) { theta = val; }
	public final void setInputAni(float[] val) { ani = val; }
	
	public final void setImageNumber(int val) { 
	    nimg = val;
	    images = new float[nimg][];
	    reference = new float[nimg][];
	    isAngular = new boolean[nimg];
	}
	public final void setInputImageAt(int num, float[] val) { images[num] = val; }
	public final void setReferenceImageAt(int num, float[] val) { reference[num] = val; }
	public final void setImageTypeAt(int num, boolean val) { isAngular[num] = val; }

	public final void setPatchSize(int val) { patch = val; }
	
	public final void setSearchAngle(float val) { angle = (float)(val*FastMath.PI/180.0); }
	public final void setSearchDistance(float val) { distance = val; }

	public final void setAniThreshold(float val) { threshold = val; }

	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }

	// outputs
	public final float[] getMatchingImage() { return result; }

	public void execute2D() {
	    if (usePencil) executeSearchPencil();
	    else executeGlobalSearch();
	}
	
	public void executeSearchPencil() {

	    // rescale angles into radians
	    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
	        theta[x+nx*y] *= (float)(FastMath.PI/180.0);
	    }
	    
	    // rescale images and references so that they have similar ranges?
	    // or use correlations, rather?
	    
        result = new float[nxyz];
        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) if (ani[x+nx*y]>threshold) {
            
            float mindiff = 1e9f;
            int xmin = 0;
            int ymin = 0;
            
            // define search pencil
            for (int d=0;d<distance;d++) {
                
                // compute the max half angle
                int maxth = (int)FastMath.ceil(d*FastMath.sin(angle/2.0));
                
                for (int th=-maxth;th<=maxth;th++) {
                    
                    int dx = d*Numerics.round(FastMath.cos(theta[x+nx*y]+angle*th/maxth));
                    int dy = d*Numerics.round(FastMath.sin(theta[x+nx*y]+angle*th/maxth));
                
                    // positive direction
                    int xs = x+dx;
                    int ys = y+dy;
                    
                    // compute patch distances
                    float diff = 0.0f;
                    float ndiff = 0.0f;
                    for (int px=-patch;px<=patch;px++) for (int py=-patch;py<=patch;py++) {
                        if (x+px>=0 && x+px<nx && y+py>=0 && y+py<ny && images[0][x+px+nx*(y+py)]!=0
                            && xs+px>=0 && xs+px<nx && ys+py>=0 && ys+py<ny && reference[0][xs+px+nx*(ys+py)]!=0) {
                            for (int i=0;i<nimg;i++) {
                                diff += Numerics.square(images[i][x+px+nx*(y+py)]-reference[i][xs+px+nx*(ys+py)]);
                            }
                            ndiff++;
                        }
                    }
                    if (ndiff>0) diff /= ndiff;
                        
                    if (diff < mindiff) {
                        mindiff = diff;
                        xmin = xs;
                        ymin = ys;
                    }
                    
                    // negative direction
                    xs = x-dx;
                    ys = y-dy;
                    
                    // compute patch distances
                    diff = 0.0f;
                    ndiff = 0.0f;
                    for (int px=-patch;px<=patch;px++) for (int py=-patch;py<=patch;py++) {
                        if (x+px>=0 && x+px<nx && y+py>=0 && y+py<ny && images[0][x+px+nx*(y+py)]!=0
                            && xs+px>=0 && xs+px<nx && ys+py>=0 && ys+py<ny && reference[0][xs+px+nx*(ys+py)]!=0) {
                            for (int i=0;i<nimg;i++) {
                                diff += Numerics.square(images[i][x+px+nx*(y+py)]-reference[i][xs+px+nx*(ys+py)]);
                            }
                            ndiff++;
                        }
                    }
                    if (ndiff>0) diff /= ndiff;
                        
                    if (diff < mindiff) {
                        mindiff = diff;
                        xmin = xs;
                        ymin = ys;
                    }
                    
                }
            }
            // return the signed distance projected on direction
            double lx = (xmin-x)*FastMath.cos(theta[x+nx*y]);
            double ly = (ymin-y)*FastMath.sin(theta[x+nx*y]);
            result[x+nx*y] = (float)(lx+ly);
	    }
		System.out.print("Done\n");
	}
	
	public void executeGlobalSearch() {

	    // rescale angles into radians
	    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
	        theta[x+nx*y] *= (float)(FastMath.PI/180.0);
	    }
	    
	    int search = Numerics.floor(distance);
        int nw = (2*search+1)*(2*search+1);

	    // rescale images and references so that they have similar ranges?
	    // or use correlations, rather?
	    double[] meandiff = new double[nimg];
	    double[] stdvdiff = new double[nimg];
	    for (int i=0;i<nimg;i++) {
	        meandiff[i] = 0.0;
	        double den=0.0;
            for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) if (images[i][x+nx*y]!=0 && reference[i][x+nx*y]!=0) {
                float val = images[i][x+nx*y]-reference[i][x+nx*y];
                if (isAngular[i]) meandiff[i] += Numerics.minmag(val,val+180.0f,val-180.0f);
                else meandiff[i] += val;
                den++;
            }
            if (den>0) meandiff[i] /= den;
            
            stdvdiff[i] = 0.0;
            for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) if (images[i][x+nx*y]!=0 && reference[i][x+nx*y]!=0) {
                float val = images[i][x+nx*y]-reference[i][x+nx*y];
                if (isAngular[i]) val = Numerics.minmag(val,val+180.0f,val-180.0f);
                stdvdiff[i] += Numerics.square(val-meandiff[i]);
            }
            if (den>0) stdvdiff[i] = FastMath.sqrt(stdvdiff[i]/den);
        }    
                
        result = new float[nxyz];
        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) if (ani[x+nx*y]>threshold) {
            
            float mindiff = 1e9f;
            float xmin = 0;
            float ymin = 0;

            float[] weights = new float[nw];
            int[] xloc = new int[nw];
            int[] yloc = new int[nw];

            int np=0;
            for (int xs=x-search;xs<=x+search;xs++) for (int ys=y-search;ys<=y+search;ys++) { 
                if (xs>=0 && xs<nx && ys>=0 && ys<ny 
                    && images[0][xs+nx*ys]!=0 && reference[0][xs+nx*ys]!=0) {
                    // compute patch distances
                    float diff = 0.0f;
                    float ndiff = 0.0f;
                    for (int px=-patch;px<=patch;px++) for (int py=-patch;py<=patch;py++) {
                        if (x+px>=0 && x+px<nx && y+py>=0 && y+py<ny && images[0][x+px+nx*(y+py)]!=0
                            && xs+px>=0 && xs+px<nx && ys+py>=0 && ys+py<ny && reference[0][xs+px+nx*(ys+py)]!=0) {
                            for (int i=0;i<nimg;i++) {
                                float val = images[i][x+px+nx*(y+py)]-reference[i][xs+px+nx*(ys+py)];
                                if (isAngular[i]) val = Numerics.minmag(val,val+180.0f,val-180.0f);
                                
                                diff += Numerics.square((val-meandiff[i])/stdvdiff[i]);
                            }
                            ndiff++;
                        }
                    }
                    if (ndiff>0) diff /= ndiff;
                    weights[np] = (float)FastMath.exp(-0.5*diff);
                    xloc[np] = xs;
                    yloc[np] = ys;
                    np++;
                    
                    if (diff < mindiff) {
                        mindiff = diff;
                        xmin = xs;
                        ymin = ys;
                    }
                    
                }
            }
            // weighted average of K lowest distances?
            short[] ids = new short[nk];
            float[] bws = new float[nk];
            Numerics.argmax(ids, bws, weights, nk);
            xmin = 0.0f;
            ymin = 0.0f;
            float den = 0.0f;
            for (int k=0;k<nk;k++) {
                xmin += bws[k]*xloc[ids[k]];
                ymin += bws[k]*yloc[ids[k]];
                den += bws[k];
            }
            if (den>0) {
                xmin /= den;
                ymin /= den;
            }
            
            // return the signed distance projected on direction
            double lx = (xmin-x)*FastMath.cos(theta[x+nx*y]);
            double ly = (ymin-y)*FastMath.sin(theta[x+nx*y]);
            //result[x+nx*y] = (float)(lx+ly);
            result[x+nx*y] = (float)FastMath.sqrt(Numerics.square(xmin-x)+Numerics.square(ymin-y));
	    }
		System.out.print("Done\n");
	}
	

}
