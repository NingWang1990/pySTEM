/* Refer to https://dfm.io/posts/python-c-extensions/
  https://docs.scipy.org/doc/numpy-1.13.0/user/c-info.how-to-extend.html
  https://docs.python.org/3/extending/newtypes_tutorial.html
*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include "FftCorr.h"
#include <string.h>
#include <new>
//#include <iostream>

extern "C" {

// --- WindowFFT boilerplate
// Python wrapper type for WindowFFT
typedef struct {
    PyObject_HEAD
    WindowFFT windowFFT;
} Pywrap_WindowFFT;

// Deallocator/destructor for python wrapper object
static void WindowFFT_dealloc (Pywrap_WindowFFT *self)
{
   // call C++ destructor
   self->windowFFT.~WindowFFT ();
   // deallocate python's memory
   Py_TYPE(self)->tp_free((PyObject *) self);
}

// Allocator for python wrapper object
static PyObject *
WindowFFT_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   // allocate memory from python
   Pywrap_WindowFFT *self = (Pywrap_WindowFFT*) type->tp_alloc(type, 0);
   if (self != NULL) {
      // initialize to zero (to be on the safe side)
      memset (&(self->windowFFT), 0, sizeof(WindowFFT));
   }
   return (PyObject *) self;
}

// Initializer for python wrapper object (calls C++ constructor)
static int
WindowFFT_init(Pywrap_WindowFFT *self, PyObject *args, PyObject *kwds)
{
    //static char *kwlist[] = { ... , NULL } ;
    int Nx, Ny;

    if (!PyArg_ParseTuple (args, "ii", &Nx, &Ny))
        return -1;
    // check that FFT dimensions are OK
    if (Nx <= 2 || Ny <= 2)
    {
       PyErr_SetString(PyExc_ValueError, "FFT dimensions must be >2");
       return -1;
    }

    // call constructor
    new (&self->windowFFT) WindowFFT(Nx, Ny);

    // check that this worked
    if (!self->windowFFT.ok ())  {
       // call destructor (something went wrong...)
       self->windowFFT.~WindowFFT ();
       // initialize to zero (to be on the safe side)
       memset (&(self->windowFFT), 0, sizeof(WindowFFT));
       // set exception
       PyErr_NoMemory ();
       return -1;
    }

    return 0;
}


/* Available functions */
static PyObject *stemdescriptor2_calc(PyObject *self, PyObject *args);

/* Module specification */

PyMODINIT_FUNC PyInit_stemdescriptor2(void)
{
   // --- Python wrapper for WindowFFT class: python type descriptor
   static PyTypeObject WindowFftType = { PyVarObject_HEAD_INIT(NULL, 0) };
   // initialize selected named entries afterwards
   WindowFftType.tp_name = "stemdescriptor2.WindowFFT";
   WindowFftType.tp_doc = "Perform 2D FFTs of sub-images (windows) in an image";
   WindowFftType.tp_basicsize = sizeof(Pywrap_WindowFFT);
   WindowFftType.tp_itemsize = 0;
   WindowFftType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
   WindowFftType.tp_new = WindowFFT_new;
   WindowFftType.tp_init = (initproc) WindowFFT_init;
   WindowFftType.tp_dealloc = (destructor) WindowFFT_dealloc;
   //WindowFftType.tp_members = Custom_members,
   static PyMethodDef WindowFFT_methods[] = {
      // declare "calc" to call "stemdescriptor_calc"
      {"calc", stemdescriptor2_calc, METH_VARARGS,
       "This function calculates the local-correlation-map descriptor" },

      {NULL}  /* Sentinel */
   };
   WindowFftType.tp_methods = WindowFFT_methods;

   // create python type descriptor object
   if (PyType_Ready(&WindowFftType) < 0) return NULL;

   // import ???
   import_array();

   // --- create new module object
   static char module_docstring[] =
       "This module provides an interface for calculating local-correlation-map descriptor using C, using an FFT-based algorithm.";

   static struct PyModuleDef _stemdescriptor =
   {
      PyModuleDef_HEAD_INIT,
      "stemdescriptor2", /* name of module */
      module_docstring,  /* module documentation */
      -1,                /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
      NULL //module_methods
   };

   PyObject* m = PyModule_Create(&_stemdescriptor);
   if (m == NULL) return NULL;

   // --- try to add WindowFFT type to this module
   Py_INCREF(&WindowFftType);
   if (PyModule_AddObject(m, "WindowFFT", (PyObject *) &WindowFftType) < 0) {
      Py_DECREF(&WindowFftType);
      Py_DECREF(m);
      return NULL;
   }

   return m;
}

/*  define the function */
static PyObject *stemdescriptor2_calc(PyObject *self_, PyObject *args)
{
   Pywrap_WindowFFT *self = (Pywrap_WindowFFT *)self_;
   PyObject *image_obj=NULL;
   PyObject *descriptor_obj=NULL;
   int num_rows,num_cols, patch_x, patch_y, region_x, region_y,region_grid_x,region_grid_y,n_descriptors, step,num_rows_desp,num_cols_desp;
   /* Parse the input tuple */
   if (!PyArg_ParseTuple(args,"OOiiiiiiiiiiii",&image_obj,&descriptor_obj,&num_rows,&num_cols,&patch_x,&patch_y,
            &region_x,&region_y,&region_grid_x,&region_grid_y,&n_descriptors,&step,&num_rows_desp,&num_cols_desp))
      return NULL;
   int Rx = self->windowFFT.Nx - (2*patch_y + 1), Rx2 = Rx/2;
   int Ry = self->windowFFT.Ny - (2*patch_x + 1), Ry2 = Ry/2;
   if (Rx < region_y || Ry < region_x)  {
      PyErr_SetString(PyExc_ValueError, "Requested region is to large for FFT dimensions");
      return NULL;
   }
   /* Interpret the input objects as numpy arrays. */
   PyObject * image_array=NULL;
   image_array = PyArray_FROM_OTF(image_obj,NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

   /* throw an exception if that didn't work */
   if (image_array == NULL){
      Py_XDECREF(image_array);
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

/*#if NPY_API_VERSION >= 0x0000000c
      PyArray_DiscardWritebackIfCopy(descriptor_array);
#endif */
      Py_XDECREF(descriptor_array);
      return NULL;
   }
   float *image = (float*) PyArray_DATA(image_array);
   float *descriptor = (float*) PyArray_DATA(descriptor_array);

   //using namespace std;
   //cout << "Rx=" << Rx << "Ry=" << Ry << endl;
   //cout << "num_cols_desp=" << num_cols_desp << endl;
   //cout << "n_descriptors=" << n_descriptors << endl;

   // --- the actual core of functionality
#pragma omp parallel
   {
      // fftCorr has fast-running index first
      FftCorr fftCorr(self->windowFFT, 2*patch_y + 1, 2*patch_x + 1);
      double *res = new double[Rx * Ry];
#pragma omp for collapse(2)
      for (int i=patch_x+region_x;i<num_rows-patch_x-region_x;i+=step) {
         for (int j=patch_y+region_y;j<num_cols-patch_y-region_y;j+=step) {
            int i_d = (i-patch_x-region_x)/step;
            int j_d = (j-patch_y-region_y)/step;

            // fftCorr has fast-running index first
            fftCorr.getCorrMap (image, num_cols, j, i, self->windowFFT,
                                res);
            int offsetDesc = n_descriptors * (i_d * num_cols_desp + j_d);
            for (int k = -region_x ; k <= region_x; k+=region_grid_x)
               for (int l = -region_y ; l <= region_y; l+=region_grid_y)
                  descriptor[offsetDesc++] = res[(k + Ry2)*Rx + l + Rx2];
         }	
      }
      delete [] res;
   }

   Py_DECREF(image_array);
/*#if NPY_API_VERSION >= 0x0000000c
   PyArray_ResolveWritebackIfCopy(descriptor_array);
#endif */
   Py_DECREF(descriptor_array);
   Py_INCREF(Py_None);

   return Py_None;

}

// closing bracket of extern "C"
}

