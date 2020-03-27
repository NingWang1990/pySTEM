#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "calculate_descriptor.h"


float correlation_coefficient(float *image, int num_cols,int patch_x, int patch_y,int i,int j,int k,int l)
{       float sum_ab = 0.0;
	float sum_aa = 0.0;
	float sum_bb = 0.0;
	float sum_a = 0.0;
	float sum_b = 0.0;
	float a_mean,b_mean,corr_ef;
	int m,n;
	int length = (2*patch_x+1)*(2*patch_y+1);
	
	for (m=-patch_x;m<=patch_x;m++)
	{
	  for (n=-patch_y;n<=patch_y;n++)
	  {
          /*
	  sum_ab += image[i+m][j+n]*image[i+k+m][j+l+n];
	  sum_aa += image[i+m][j+n]*image[i+m][j+n];
	  sum_bb += image[i+k+m][j+l+n]*image[i+k+m][j+l+n];
	  sum_a += image[i+m][j+n];
	  sum_b += image[i+k+m][j+l+n];
	  */
          sum_ab += *(image+(i+m)*num_cols+j+n) * (*(image+(i+k+m)*num_cols+j+l+n));
	  sum_aa += *(image+(i+m)*num_cols+j+n) * (*(image+(i+m)*num_cols+j+n) );
	  sum_bb +=  (*(image+(i+k+m)*num_cols+j+l+n))* (*(image+(i+k+m)*num_cols+j+l+n));
	  sum_a += *(image+(i+m)*num_cols+j+n);
	  sum_b += *(image+(i+k+m)*num_cols+j+l+n);
	  }
	}
	a_mean = sum_a / length;
	b_mean = sum_b / length;
	corr_ef = (sum_ab - length*a_mean*b_mean)/(sqrtf(sum_aa-length*a_mean*a_mean)*sqrtf(sum_bb-length*b_mean*b_mean));
        return corr_ef; 	
}	




int calc_descriptor(float *image,float*descriptor, int num_rows, int num_cols, int patch_x, int patch_y, int region_x, int region_y, int region_grid_x, int region_grid_y,int n_descriptors)
{
        #pragma omp parallel
	{
	int i,j;
	int k, l, index, index_l;
	#pragma omp for
        for (i=patch_x+region_x;i<num_rows-patch_x-region_x;i++)
	{
	  for (j=patch_y+region_y;j<num_cols-patch_y-region_y;j++)
	  {
           index_l = 0;
	   for (k=-region_x;k<=region_x;k+=region_grid_x)
	   {
	     for (l=-region_y;l<=region_y;l+=region_grid_y)
	     {
             index = (i-patch_x-region_x)*(num_cols-2*(patch_y+region_y))*n_descriptors + (j-patch_y-region_y)*n_descriptors + index_l;
	     *(descriptor+index) = correlation_coefficient(image, num_cols,patch_x,patch_y,i,j,k,l);
	     index_l += 1;
	     }
	   }
	 }	
	}
	}
	return 0;
}
