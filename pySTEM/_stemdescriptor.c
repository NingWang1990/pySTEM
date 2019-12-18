/* Refer to https://dfm.io/posts/python-c-extensions/
  https://docs.scipy.org/doc/numpy-1.13.0/user/c-info.how-to-extend.html
*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include "calculate_descriptor.h"


/* DocStrings */
static char module_docstring[] = 
    "This module provides an interface for calculating local-correlation-map descriptor using C.";
static char func_docstring[] = 
    "This function calculates the local-correlation-map descriptor";

/* Available functions */
static PyObject *stemdescriptor_calc(PyObject *self, PyObject *args);

/* Module specification */

static PyMethodDef module_methods[] = {
    {"calc", stemdescriptor_calc, METH_VARARGS,func_docstring},
    {NULL,NULL,0,NULL}
};

static struct PyModuleDef _stemdescriptor =
{
    PyModuleDef_HEAD_INIT,
    "_stemdescriptor", /* name of module */
    module_docstring,          /* module documentation */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    module_methods
};
PyMODINIT_FUNC PyInit__stemdescriptor(void)
{
    import_array(); 
    return PyModule_Create(&_stemdescriptor);
}

/* Initialize the module */
//PyMODINIT_FUNC init_stemdescriptor(void)
//{
//    PyObject *m = Py_InitModule("_stemdescriptor", module_methods,module_docstring);
//    if (m == NULL)
//	    return;
    /* load 'numpy' functionality.  */
//    import_array();
//}

/*  define the function */
static PyObject *stemdescriptor_calc(PyObject *self, PyObject *args)
{
	
      PyObject *image_obj;
      PyObject *descriptor_obj;
      printf("here now");
      int num_rows,num_cols, patch_x, patch_y, region_x, region_y,region_grid_x,region_grid_y,n_descriptors;
      /* Parse the input tuple */
      if (!PyArg_ParseTuple(args,"OOiiiiiiiii",&image_obj,&descriptor_obj,&num_rows,&num_cols,&patch_x,&patch_y,
		                             &region_x,&region_y,&region_grid_x,&region_grid_y,&n_descriptors))
	  return NULL;

      /* Interpret the input objects as numpy arrays. */
      PyObject * image_array = PyArray_FROM_OTF(image_obj,NPY_FLOAT32, NPY_IN_ARRAY);

      /* throw an exception if that didn't work */
      if (image_array == NULL){
          Py_XDECREF(image_array);
          return NULL;
      }
      /* Interpret the output objects as numpy arrays. */
      PyObject * descriptor_array = PyArray_FROM_OTF(descriptor_obj,NPY_FLOAT32,NPY_IN_ARRAY);

      /* throw an exception if that didn't work */
      if (descriptor_array == NULL){
          PyArray_XDECREF_ERR(descriptor_array);
      }
      float *image = (float*) PyArray_DATA(image_array);
      float *descriptor = (float*) PyArray_DATA(descriptor_array);
      /* call the external C function to compute the local-correlation-map descriptor */
      int value = calc_descriptor(image,descriptor,num_rows, num_cols,patch_x,patch_y,region_x,region_y,
                                          region_grid_x,region_grid_y,n_descriptors);

     

      Py_INCREF(Py_None);


      return Py_None;

}

