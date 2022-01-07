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
	
	float[] result;
	
	// set inputs
	public final void setInputTheta(float[] val) { theta = val; }
	public final void setInputAni(float[] val) { ani = val; }
	
	public final void setImageNumber(int val) { 
	    nimg = val;
	    images = new float[nimg][];
	    reference = new float[nimg][];
	}
	public final void setInputImageAt(int num, float[] val) { images[num] = val; }
	public final void setReferenceImageAt(int num, float[] val) { reference[num] = val; }
	
	public final void setSearchAngle(int val) { angle = val; }
	public final void setSearchDistance(int val) { distance = val; }

	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }

	// outputs
	public final float[] getMatchingImage() { return result; }

	public void execute2D() {

	    
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
	

}
