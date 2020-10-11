/* Refer to https://dfm.io/posts/python-c-extensions/
  https://docs.scipy.org/doc/numpy-1.13.0/user/c-info.how-to-extend.html
*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include "preselected_translations.h"


/* DocStrings */
static char module_docstring[] = 
    "This module provides an interface for calculating local-correlation-map descriptor using C.";
static char func_docstring[] = 
    "This function calculates the local-correlation-map descriptor";

/* Available functions */
static PyObject *preselectedtranslations_calc(PyObject *self, PyObject *args);

/* Module specification */

static PyMethodDef module_methods[] = {
    {"calc", preselectedtranslations_calc, METH_VARARGS,func_docstring},
    {NULL,NULL,0,NULL}
};

static struct PyModuleDef _preselectedtranslations =
{
    PyModuleDef_HEAD_INIT,
    "_preselectedtranslations", /* name of module */
    module_docstring,          /* module documentation */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    module_methods
};
PyMODINIT_FUNC PyInit__preselectedtranslations(void)
{
    import_array(); 
    return PyModule_Create(&_preselectedtranslations);
}

/*  define the function */
static PyObject *preselectedtranslations_calc(PyObject *self, PyObject *args)
{
      PyObject *image_obj=NULL;
      PyObject *descriptor_obj=NULL;
      PyObject *translations_obj=NULL;
      int num_rows,num_cols, patch_x, patch_y, region_x, region_y, n_descriptors, step,num_rows_desp,num_cols_desp,removing_mean;
      /* Parse the input tuple */
      if (!PyArg_ParseTuple(args,"OOOiiiiiiiiiii",&image_obj,&descriptor_obj,&translations_obj,&num_rows,&num_cols,&patch_x,&patch_y,
      		                             &region_x,&region_y,&n_descriptors,&step,&num_rows_desp,&num_cols_desp,&removing_mean))
       	  return NULL;
      /* Interpret the input objects as numpy arrays. */
      PyObject * image_array=NULL;
      image_array = PyArray_FROM_OTF(image_obj,NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

      /* throw an exception if that didn't work */
      if (image_array == NULL){
          Py_XDECREF(image_array);
          return NULL;
      }
      /* Interpret translations array */
      PyObject * translations_array=NULL;
      translations_array = PyArray_FROM_OTF(translations_obj,NPY_INT32, NPY_ARRAY_IN_ARRAY);

      /* throw an exception if that didn't work */
      if (translations_array == NULL){
          Py_XDECREF(translations_array);
          return NULL;
      }
      
      /* Interpret the output objects as numpy arrays. */
      PyObject * descriptor_array=NULL;
#if NPY_API_VERSION >= 0x0000000c
      descriptor_array = PyArray_FROM_OTF(descriptor_obj,NPY_FLOAT32,NPY_ARRAY_INOUT_ARRAY2);
#else
      descriptor_array = PyArray_FROM_OTF(descriptor_obj,NPY_FLOAT32,NPY_ARRAY_INOUT_ARRAY);
#endif
      /* throw an exception if that didn't work */
      if (descriptor_array == NULL){
	  Py_XDECREF(image_array);
#if NPY_API_VERSION >= 0x0000000c
          PyArray_DiscardWritebackIfCopy(descriptor_array);
#endif
          Py_XDECREF(descriptor_array);
	  return NULL;
      }
      float *image = (float*) PyArray_DATA(image_array);
      float *descriptor = (float*) PyArray_DATA(descriptor_array);
      int *translations = (int*)PyArray_DATA(translations_array);
      /* call the external C function to compute the local-correlation-map descriptor */
      int value = calc_descriptors_preselected(image,descriptor,translations,num_rows, num_cols,patch_x,patch_y,region_x,region_y,
                                  n_descriptors,step,num_rows_desp,num_cols_desp, removing_mean);

      Py_DECREF(image_array);
      Py_DECREF(translations_array);
#if NPY_API_VERSION >= 0x0000000c
      PyArray_ResolveWritebackIfCopy(descriptor_array);
#endif
      Py_DECREF(descriptor_array);
      Py_INCREF(Py_None);

      return Py_None;

}

