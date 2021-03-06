#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "preselected_translations.h"


float correlation_coefficient(float *image, int num_cols,int patch_x, int patch_y,int i,int j,int k,int l, int removing_mean)
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
	if (removing_mean == 1){
	corr_ef = (sum_ab - length*a_mean*b_mean)/(sqrtf(sum_aa-length*a_mean*a_mean)*sqrtf(sum_bb-length*b_mean*b_mean));}
	else {
	corr_ef = (sum_ab)/(sqrtf(sum_aa)*sqrtf(sum_bb));}
        return corr_ef; 	
}	




int calc_descriptors_preselected(float *image, float *descriptor,int *translations, int num_rows, int num_cols, int patch_x, int patch_y, int region_x, int region_y,int n_descriptors,int step,int num_rows_desp, int num_cols_desp, int removing_mean)
{
        #pragma omp parallel
	{
	int i,j;
	int i_d, j_d;
	int m, k, l, index, index_l;
	#pragma omp for
        for (i=patch_x+region_x;i<num_rows-patch_x-region_x;i+=step)
	{
	  for (j=patch_y+region_y;j<num_cols-patch_y-region_y;j+=step)
	  {
	   i_d = (i-patch_x-region_x)/step;
	   j_d = (j-patch_y-region_y)/step;
	   for (m=0;m<n_descriptors;m++)
	     {  
             index = i_d*num_cols_desp*n_descriptors + j_d*n_descriptors + m;
	     k = *(translations + 2*m);
             l = *(translations + 2*m +1);
	     *(descriptor+index) = correlation_coefficient(image, num_cols,patch_x,patch_y,i,j,k,l, removing_mean);
	     }
	 }	
	}
	}
	return 0;
}
