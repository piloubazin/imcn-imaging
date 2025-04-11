package nl.uva.imcn.methods;

import nl.uva.imcn.utilities.*;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.analysis.*;
import org.apache.commons.math3.special.*;

/**
 *
 *  This function gives the diffwerence in appearance for a local model of vessels 
 *  (tubular with fixed orientation, variable center and diameter)  	
 *
 *	@version    Summer 2015
 *	@author     Victoire Plessis
 *		
 *
 */

 public class VesselDiameterCostFunction2D implements MultivariateFunction{
	
	
	int nx,ny;
	float[][] probaInImg;
	
	int rIn;
	int x,y;
	float[] finalDir;
	float[] tan1;
	float[] tan2;
	int nbrPoint;	
	
	int[][] coordinates;
	float[] intensity;
	
	float dev=1/(2.0f*(float)(FastMath.sqrt(2.0*(float)(FastMath.log(2.0)))));
	
	// direction labeling		
	public	static	final	byte	X = 0;
	public	static	final	byte	Y = 1;
		
	public double value(double[] point){
		double alpha=point[0];
		double radius=point[1];
				
		double fa=0.0;
		
		for(int n=0;n<nbrPoint;n++){
			float xi= (float)coordinates[n][X]-((float)alpha*tan1[X]);
			float yj= (float)coordinates[n][Y]-((float)alpha*tan1[Y]);
			float newDist= (float)FastMath.sqrt( xi*xi+yj*yj );
			float c1=2.0f*dev/3.0f*(float)FastMath.sqrt(dev*dev+newDist*newDist)/(2.0f*dev*dev+newDist*newDist);
			float c2=(float)FastMath.pow(radius*radius/(2.0f*dev*dev+newDist*newDist),1.0/3.0);
			float x=( (c2-1)/c1 + c1  ) / (float)FastMath.sqrt(2.0f)  ;
			float po=( 1.0f + (float)Erf.erf( x ) )/2.0f ;
			po-=intensity[n];
			fa+=(double)(po*po);
		}
		
		return fa;
	}
	
	public void setImagesData(float[][] probaInImg_, int nx_, int ny_){
		nx=nx_;
		ny=ny_;
		probaInImg=probaInImg_;
	}
	
	public void setPointData(int x_,int y_, int rIn_,float[] finalDir_, float[] tan1_){
		x=x_;
		y=y_;
		finalDir=finalDir_;
		rIn=rIn_;
		tan1=tan1_;
		
		float maxDist= (float)rIn+0.5f;
		int pointNbr=0;
		for (int i=-rIn;i<=rIn;i++) for (int j=-rIn;j<=rIn;j++){				
					float dist= (float)FastMath.sqrt((float)i*i +(float)j*j);					
					if(dist<=maxDist){
						if(x+i<nx &&  x+i>=0 && y+j<ny &&  y+j>=0 ){
							float orthTest=(float)i*finalDir[X]+(float)j*finalDir[Y];
							orthTest=orthTest*orthTest;
							if(orthTest<1.0e-9f){
								pointNbr++;
							}
						}
					}
		}
		nbrPoint=pointNbr;
		int[][] coordinates_=new int[pointNbr][2];
		float[] intensity_=new float[pointNbr];
		pointNbr=0;
		for (int i=-rIn;i<=rIn;i++) for (int j=-rIn;j<=rIn;j++){				
					float dist= (float)FastMath.sqrt((float)i*i +(float)j*j);					
					if(dist<=maxDist){
						if(x+i<nx &&  x+i>=0 && y+j<ny &&  y+j>=0 ){
							float orthTest=(float)i*finalDir[X]+(float)j*finalDir[Y];
							orthTest=orthTest*orthTest;
							if(orthTest<1.0e-9f){
								coordinates_[pointNbr][X]=i;
								coordinates_[pointNbr][Y]=j;
								intensity_[pointNbr]=probaInImg[x+i][y+j];
								pointNbr++;
							}
						}
					}
		}
		coordinates=coordinates_;
		intensity=intensity_;
		
	}	
	
	public float[] getTan1(){
		return tan1;
	}
}
