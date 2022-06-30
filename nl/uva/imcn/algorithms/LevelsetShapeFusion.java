package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import nl.uva.imcn.structures.*;
import nl.uva.imcn.libraries.*;
import nl.uva.imcn.methods.*;

import org.apache.commons.math3.util.FastMath;

//import Jama.*;

/*
 * @author Pierre-Louis Bazin
 */
public class LevelsetShapeFusion {

	// data buffers
	private float[][] lvlImages;
	
	private int nsubj;
	private boolean devParam = false;
	private boolean topoParam = true;
	private     String	            lutdir = null;
	private float    smooth = 0.1f;
	private float    sharpen = 0.0f;
	private float    distance = 10.0f;
	
	private float[] 			meanImage;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;

	public final void setNumberOfImages(int val) {
	    nsubj = val;
	    lvlImages = new float[nsubj][];
	}
	public final void setLevelsetImageAt(int n, float[] val) { lvlImages[n] = val; }
	
	public final void setIncludeVariance(boolean val) { devParam=val; }
	public final void setCorrectSkeletonTopology(boolean val) { topoParam=val; }
	public final void setTopologyLUTdirectory(String val) { lutdir = val; }

	public final void setLevelsetDistance(float val) { distance = val; }

	public final void setCurvatureSmoothing(float val) { smooth = val; }

	public final void setSlopeSharpening(float val) { sharpen = val; }

	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
	
	// to be used for JIST definitions, generic info / help
	public final String getPackage() { return "IMCN Toolkit"; }
	public final String getCategory() { return "Shape"; }
	public final String getLabel() { return "Levelset Fusion"; }
	public final String getName() { return "LevelsetFusion"; }

	public final String[] getAlgorithmAuthors() { return new String[]{"Pierre-Louis Bazin"}; }
	public final String getAffiliation() { return "Integrative Model-basec Cognitve Neuroscience unit, Universiteit van Amsterdam | Max Planck Institute for Human Cognitive and Brain Sciences"; }
	public final String getDescription() { return "Combines a collection of levelset surfaces into an average with average volume and spherical topology"; }
	public final String getLongDescription() { return getDescription(); }
		
	public final String getVersion() { return "1.0"; };

	// create outputs
	public final float[] getLevelsetAverage() { return meanImage; }

	public void execute() {
		
		float[][] levelsets = lvlImages;
		
		System.out.println("compute average");
		float[] average = null;
        average = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) {	
            average[xyz] = 0.0f;
            for (int n=0;n<nsubj;n++) {
                levelsets[n][xyz] = Numerics.bounded(levelsets[n][xyz],-distance,distance);
                average[xyz] += levelsets[n][xyz]/(float)nsubj;
            }
		}
		if (devParam) {
            System.out.println("compute stdev");
            float[] stdev = new float[nxyz];
            for (int xyz=0;xyz<nxyz;xyz++) {	
                stdev[xyz] = 0.0f;
                for (int n=0;n<nsubj;n++) stdev[xyz] += (levelsets[n][xyz]-average[xyz])*(levelsets[n][xyz]-average[xyz])/(float)nsubj;
                stdev[xyz] = (float)FastMath.sqrt(stdev[xyz]);
            }
            for (int xyz=0;xyz<nxyz;xyz++) {	
                average[xyz] = average[xyz]-stdev[xyz];
            }
        }
        if (sharpen>0.0f && sharpen<1.0f) {
            System.out.println("sharpen with Laplacian");
            float[] lap = new float[nxyz];
            for (int xyz=0;xyz<nxyz;xyz++) lap[xyz] = average[xyz];
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz=x+nx*y+nx*ny*z;
                lap[xyz] = 6.0f*average[xyz] - average[xyz-1] - average[xyz+1]
                            - average[xyz-nx] - average[xyz+nx] - average[xyz-nx*ny] - average[xyz+nx*ny];
            }
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz=x+nx*y+nx*ny*z;
                average[xyz] = (1.0f-sharpen)*average[xyz] + sharpen*lap[xyz];
            }
        }            
		
		// compute mean volume to normalize the average to the same size
		double meanvol = 0.0;
		float mindist = 1e6f;
		for (int xyz=0;xyz<nxyz;xyz++) {
			for (int n=0;n<nsubj;n++) if (levelsets[n][xyz]<0) meanvol += 1.0/nsubj;
			if (average[xyz]<mindist) mindist = average[xyz];
		}
		System.out.println("mean volume (voxels): "+meanvol);
		System.out.println("minimum avg. distance: "+mindist);
		
		// find appropriate threshold to have correct volume; should use a fast marching approach!
		BinaryHeap2D	heap = new BinaryHeap2D(nx*ny+ny*nz+nz*nx, BinaryHeap2D.MINTREE);
		double vol = 0.0;
        boolean[] label = new boolean[nxyz];
        float initdist = Numerics.min(0.0f, 0.5f*mindist);
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
            int xyz=x+nx*y+nx*ny*z;
            if (average[xyz]<initdist || average[xyz]==mindist) {
                vol++;
                label[xyz] = true;
                
                // check for neighbors outside
                for (byte k = 0; k<6 ; k++) {
                    int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                    if (average[ngb]>=initdist) {
                        // add to the heap (multiple instances are OK)
                        heap.addValue(average[ngb],ngb,(byte)1);
                    }
                }
            } else {
                label[xyz] = false;
            }
        }
		System.out.println("init volume (voxels): "+vol);
		
        // run until the volume exceeds the mean volume
        float threshold = 0.0f;
        while (heap.isNotEmpty() && vol<meanvol) {
            //threshold = heap.getFirst();
            int xyz = heap.getFirstId();
            threshold = average[xyz];
            heap.removeFirst();	
            if (label[xyz]==false) {
                vol++;
                label[xyz] = true;
                // add neighbors
                for (byte k = 0; k<6; k++) {
                    int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                    if (label[ngb]==false) {
                        heap.addValue(average[ngb],ngb,(byte)1);
                    }
                }
            }
        }
        System.out.println("final volume (voxels): "+vol);
		System.out.println("Distance offset ~ "+threshold);
        // result
        for (int xyz=0;xyz<nxyz;xyz++) {	
            if (label[xyz]) {
                average[xyz] = Numerics.min(-0.001f, average[xyz] - threshold);
            } else {
                average[xyz] = Numerics.max(0.001f, average[xyz] - threshold);
            }
        }            
        meanImage = ObjectTransforms.fastMarchingDistanceFunction(average, nx, ny, nz);
        
        // optional: smooth levelset
        if (smooth>0.0f && smooth<1.0f) {
            SmoothGdm gdm = new SmoothGdm(meanImage, nx,ny,nz, rx,ry,rz, null, (1.0f-smooth), smooth, "no", null);
			gdm.evolveNarrowBand(500,0.0005f);
			meanImage = gdm.getLevelSet();
        }
               
		// optional: correct topology here
		if (topoParam==true) {
            FastMarchingTopologyCorrection topocorrect = new FastMarchingTopologyCorrection();
		
            topocorrect.setDimensions(nx,ny,nz);
            topocorrect.setResolutions(rx,ry,rz);
		
            topocorrect.setShapeImage(meanImage);
            topocorrect.setShapeImageType("signed_distance_function");
		
            topocorrect.setPropagationDirection("background->object");
            topocorrect.setTopology("18/6");
            topocorrect.setTopologyLUTdirectory(lutdir);
            topocorrect.setMinimumDistance(0.00001f);
		
            topocorrect.execute();
		
            meanImage = topocorrect.getCorrectedImage();
		}

		// threshold the resulting distance function
		for (int xyz=0;xyz<nxyz;xyz++) {	
            meanImage[xyz] = Numerics.bounded(meanImage[xyz],-distance,distance);
 		}

        return;
	}


}
