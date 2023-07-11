package nl.uva.imcn.structures;

import java.io.*;
import java.util.*;

/**
 *
 *  Binary sorting trees, either min-trees or max-trees.
 *	<p>
 *  Values are sorted in a binary tree which require that each parent node is lower (resp. higher) than its children.
 *  The root of the tree is the lowest (resp. highest) value in the tree, and all operations (adding or removing a point)
 *  have <i>O( N log N )<i> complexity.
 *	<p>
 *	These trees are used principally in fast marching methods. This specific tree has four indices for use in Toads,
 *	other versions may be needed (@see BinaryTree for a more generic implementation) 
 *
 *	@version    July 2004
 *	@author     Pierre-Louis Bazin
 *		
 *
 */

public class BinaryHeap3DPair {
	
	private float[] val;
	private int[] 	x;
	private int[] 	y;
	private int[] 	z;
	private byte[] 	idx;
	private byte[] 	idy;
	private byte[] 	idz;
	
	private int 		currentSize;
	private int			capacity;
	private int 		minormax;
	
	public static final	int MINTREE = -1;
	public static final	int MAXTREE = 1;
	
	public BinaryHeap3DPair(int Nsize, int type) {
		currentSize = 0;
		capacity = Nsize;
		minormax = type;
		val = new float[capacity+1];
		x = new int[capacity+1];
		y = new int[capacity+1];
		z = new int[capacity+1];
		idx = new byte[capacity+1];
		idy = new byte[capacity+1];
		idz = new byte[capacity+1];
		if (minormax==MINTREE)
			val[0] = -1e12f;
		else if (minormax==MAXTREE)
			val[0] = 1e12f;
	}
	
	public BinaryHeap3DPair(int Nsize, int Nincrease, int type) {
		currentSize = 0;
		capacity = Nsize;
		minormax = type;
		val = new float[capacity+1];
		x = new int[capacity+1];
		y = new int[capacity+1];
		z = new int[capacity+1];
		idx = new byte[capacity+1];
		idy = new byte[capacity+1];
		idz = new byte[capacity+1];
		if (minormax==MINTREE)
			val[0] = -1e12f;
		else if (minormax==MAXTREE)
			val[0] = 1e12f;
		
		capacity = Nincrease;
	}
	
	/**
	 *  to reset the binary tree
	 */
	public final void reset() {
		currentSize = 0;
	}
	
	/**
	 *  to set the binary tree type
	 */
	public final void setMaxTree() {
		minormax = MAXTREE;
		val[0] = 1e12f;
	}
	public final void setMinTree() {
		minormax = MINTREE;
		val[0] = -1e12f;
	}
	
	/**
	 *  add a new value into the binary tree
	 */
	public final void addValue(float val_, int x_, byte idx_, int y_, byte idy_, int z_, byte idz_) {
		// check for size
		if  (currentSize == val.length - 1) {
			float[] oldVal = val;
			int[] oldX = x;
			int[] oldY = y;
			int[] oldZ = z;
			byte[] oldIdx = idx;
			byte[] oldIdy = idy;
			byte[] oldIdz = idz;
			val = new float[currentSize+capacity];
			x = new int[currentSize+capacity];
			y = new int[currentSize+capacity];
			z = new int[currentSize+capacity];
			idx = new byte[currentSize+capacity];
			idy = new byte[currentSize+capacity];
			idz = new byte[currentSize+capacity];
			for (int i=0;i<oldVal.length;i++) {
				val[i] = oldVal[i];
				x[i] = oldX[i];
				y[i] = oldY[i];
				z[i] = oldZ[i];
				idx[i] = oldIdx[i];
				idy[i] = oldIdy[i];
				idz[i] = oldIdz[i];
			}
		}
		// insert new  point into the proper location		
		int hole = ++currentSize;
		
		if (minormax==MINTREE) {
			for ( ; val_ < val[ hole/2 ]; hole /= 2 ) {
				val[hole] = val[hole/2];
				x[hole] = x[hole/2];
				y[hole] = y[hole/2];
				z[hole] = z[hole/2];
				idx[hole] = idx[hole/2];
				idy[hole] = idy[hole/2];
				idz[hole] = idz[hole/2];
			}
			val[hole] = val_;
			x[hole] = x_;
			y[hole] = y_;
			z[hole] = z_;
			idx[hole] = idx_;
			idy[hole] = idy_;
			idz[hole] = idz_;
		} else if (minormax==MAXTREE) {
			for ( ; val_ > val[ hole/2 ]; hole /= 2 ) {
				val[hole] = val[hole/2];
				x[hole] = x[hole/2];
				y[hole] = y[hole/2];
				z[hole] = z[hole/2];
				idx[hole] = idx[hole/2];
				idy[hole] = idy[hole/2];
				idz[hole] = idz[hole/2];
			}
			val[hole] = val_;
			x[hole] = x_;
			y[hole] = y_;
			z[hole] = z_;
			idx[hole] = idx_;
			idy[hole] = idy_;
			idz[hole] = idz_;
		}
		
		return;
	}//addValue
	
	/**
	 *  remove the first value from the tree
	 */
	public final void removeFirst() {
		int hole = 1;
		
		val[hole] = val[currentSize];
		x[hole] = x[currentSize];
		y[hole] = y[currentSize];
		z[hole] = z[currentSize];
		idx[hole] = idx[currentSize];
		idy[hole] = idy[currentSize];
		idz[hole] = idz[currentSize];
		currentSize--;
		
		int child;
		float tmp = val[hole];
		int tmpX = x[hole];
		int tmpY = y[hole];
		int tmpZ = z[hole];
		byte tmpIdx = idx[hole];
		byte tmpIdy = idy[hole];
		byte tmpIdz = idz[hole];
		
		if (minormax==MINTREE) {
			for ( ; hole*2 <= currentSize; hole = child ) {
				child = hole*2;
				if (child != currentSize && val[child+1]<val[child])
					child++;
				if ( val[child]<tmp ) {
					val[ hole ] = val[ child ];
					x[ hole ] = x[ child ];
					y[ hole ] = y[ child ];
					z[ hole ] = z[ child ];
					idx[ hole ] = idx[ child ];
					idy[ hole ] = idy[ child ];
					idz[ hole ] = idz[ child ];
				} else
					break;
			}
			val[ hole ] = tmp;
			x[ hole ] = tmpX;
			y[ hole ] = tmpY;
			z[ hole ] = tmpZ;
			idx[ hole ] = tmpIdx;
			idy[ hole ] = tmpIdy;
			idz[ hole ] = tmpIdz;
		} else if (minormax==MAXTREE) {
			for ( ; hole*2 <= currentSize; hole = child ) {
				child = hole*2;
				if (child != currentSize && val[child+1]>val[child])
					child++;
				if ( val[child]>tmp ) {
					val[ hole ] = val[ child ];
					x[ hole ] = x[ child ];
					y[ hole ] = y[ child ];
					z[ hole ] = z[ child ];
					idx[ hole ] = idx[ child ];
					idy[ hole ] = idy[ child ];
					idz[ hole ] = idz[ child ];
				} else
					break;
			}
			val[ hole ] = tmp;
			x[ hole ] = tmpX;
			y[ hole ] = tmpY;
			z[ hole ] = tmpZ;
			idx[ hole ] = tmpIdx;
			idy[ hole ] = tmpIdy;
			idz[ hole ] = tmpIdz;
		} 

		return;
	}// removeFirstValue

	/**
	 * return the first value and its coordinates
	 */
	public final float getFirst() {
		return val[1];
	}
	public final int getFirstX() {
		return x[1];
	}
	public final int getFirstY() {
		return y[1];
	}
	public final int getFirstZ() {
		return z[1];
	}
	public final byte getFirstIdX() {
		return idx[1];
	}
	public final byte getFirstIdY() {
		return idy[1];
	}
	public final byte getFirstIdZ() {
		return idz[1];
	}
	
	/**
	 * various utilities
	 */
	public final void print() {
		int i;
		int n;
			
		n = 2;
		for (i=1;i<currentSize;i++) {
			System.out.print("  "+(int)(100*val[i]));
			if ( ( i%n ) == n-1) {
				// odd number
				System.out.print("\n");
				n = 2*n;
			}
		}
		return;
	}//print
	
	/**
	 *  check the binary tree property
	 */
	public final boolean isNotEmpty() {
		return (currentSize > 0);
	}
	
	/**
	 *  check the binary tree property
	 */
	public final int getCurrentSize() {
		return currentSize;
	}
	
	/**
	 *  check the binary tree property
	 */
	public final boolean isBinaryTree() {
		int i;
		int n;
		boolean isBinary=true;
				
		if (minormax==MINTREE) {
			for (i=2;i<currentSize;i++) {
				n = i/2;
				if ( val[n] > val[i] ) {
					// odd number
					isBinary = false;
					break;
				}
			}
		} else if (minormax==MAXTREE) {
			for (i=2;i<currentSize;i++) {
				n = i/2;
				if ( val[n] < val[i] ) {
					// odd number
					isBinary = false;
					break;
				}
			}
		}
		return isBinary;
	}//isBinaryTree

}

