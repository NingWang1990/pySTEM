float correlation_coefficient(float *image, int num_cols,int patch_x, int patch_y,int i,int j,int k,int l, int removing_mean);
int calc_descriptors_preselected(float *image, float *descriptor,int *translations, int num_rows, int num_cols, int patch_x, int patch_y, int region_x, int region_y,int n_descriptors,int step,int num_rows_desp, int num_cols_desp, int removing_mean);
