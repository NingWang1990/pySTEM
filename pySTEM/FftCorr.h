//#define NDEBUG

#include <fftw3.h>
#include <cassert>
#include <math.h>
#include <string.h>
#ifndef T_COMPLEX
typedef double T_COMPLEX[2];
#endif

class WindowFFT {
   protected:
      // FFTW plans
      fftw_plan planFor, planRev;

      /// Workspace
      T_COMPLEX* work;
   public:

      // size of FFT in x,y
      const int Nx, Ny;

      inline bool ok () const
      {
         return (bool)planFor && (bool)planRev;
      }

      WindowFFT (int Nx_, int Ny_);
      ~WindowFFT ();

      int getSize () const { return Nx * Ny; }

      class Linear {
         public:
            static inline double apply (double x) { return x; }
      };

      class Square {
         public:
            static inline double apply (double x) { return x * x; }
      };

      class SquareRoot {
         public:
            static inline double apply (double x) { return ::sqrt(x); }
      };

      class SquareRootInv {
         public:
            static inline double apply (double x) { return ::sqrt(1./x); }
      };

      /// Reverse FFT from real-space to reciprocal space, not normalized
      // nx, ny   patch to copy from in, padded symmetrically to Nx, Ny
      // in, strideIn input data, column stride for input data (contiguous columns)
      // out: output data, Nx x Ny complex numbers
      template <class T, class f = Linear>
      void toRecSpace (int nx, int ny, T* in, int strideIn, T_COMPLEX* out);

      /// Reverse FFT from real-space to reciprocal space
      /// central nx x ny patch filled with 1, rest filled with 0
      void toRecSpace1 (int nx, int ny, T_COMPLEX* out);

      template <class T>
      inline void toRecSpace2 (int nx, int ny, T* in, int strideIn, T_COMPLEX* out)
      {
         toRecSpace<T,Square> (nx, ny, in, strideIn, out);
      }

      /// Forward FFT from reciprocal space to real-space, not normalized
      // nx, ny   patch to deliver, trimming symmetrically from Nx, Ny
      // in: input data, Nx x Ny complex numbers
      // out, strideOut output data, column stride for output data (contiguous columns)
      template <class T, class f = Linear>
      void toRealSpace (int nx, int ny, T_COMPLEX* in, T* out, int strideOut);
};

template<class T, class f>
void WindowFFT::toRecSpace (int nx, int ny, T* in, int strideIn, T_COMPLEX* out)
{
   assert(nx > 0 && nx <= Nx);
   assert(ny > 0 && ny <= Ny);
   assert(in);
   assert(out);
   int plx = Nx/2 - nx/2;
   int ply = Ny/2 - ny/2;
   bool useWork = (fftw_alignment_of((double*)out) != fftw_alignment_of((double*)work));
   double (*data)[2] = (double(*)[2])(useWork ? work : out);
   for (int iy = 0; iy < ply; iy++)  {
      for (int ix = 0; ix < Nx; ix++)  {
         data[ix + iy * Nx][0] = 0.;
         data[ix + iy * Nx][1] = 0.;
      }
   }
   for (int iy = 0; iy < ny; iy++)  {
      int wy = (iy + ply) * Nx;
      for (int ix = 0; ix < plx; ix++)  {
         data[ix + wy][0] = 0.;
         data[ix + wy][1] = 0.;
      }
      int wyp = wy + plx;
      for (int ix = 0; ix < nx; ix++)  {
         data[ix + wyp][0] = f::apply (in[ix + iy * strideIn]);
         data[ix + wyp][1] = 0.;
      }
      for (int ix = plx + nx; ix < Nx; ix++)  {
         data[ix + wy][0] = 0.;
         data[ix + wy][1] = 0.;
      }
   }
   for (int iy = ply + ny; iy < Ny; iy++)  {
      for (int ix = 0; ix < Nx; ix++)  {
         data[ix + iy * Nx][0] = 0.;
         data[ix + iy * Nx][1] = 0.;
      }
   }
   fftw_execute_dft (planFor, (fftw_complex*)data, (fftw_complex*)data);
   if (useWork) memcpy(out, work, sizeof(T_COMPLEX) * Nx * Ny);
}

template<class T, class f>
void WindowFFT::toRealSpace (int nx, int ny, T_COMPLEX* in, T* out, int strideOut)
{
   assert(nx > 0 && nx <= Nx);
   assert(ny > 0 && ny <= Ny);
   assert(in);
   assert(out);
   bool useWork = (fftw_alignment_of((double*)in) != fftw_alignment_of((double*)work));
   double (*data)[2] = (double(*)[2])(useWork ? work : in);
   if (useWork) memcpy(work, in, sizeof(T_COMPLEX) * Nx * Ny);
   fftw_execute_dft (planRev, data, data);
   int plx = nx - nx/2;
   int ply = ny - ny/2;
   for (int iy = 0; iy < ply; iy++)  {
      int wp = nx - plx + (iy + ny - ply)*strideOut;
      for (int ix = 0; ix < plx; ix++)  {
         out[ix + wp] = T(f::apply(data[ix + iy * Nx][0]));
      }
      wp -= Nx;
      for (int ix = plx + Nx - nx; ix < Nx; ix++)  {
         out[ix + wp] = T(f::apply(data[ix + iy * Nx][0]));
      }
   }
   for (int iy = ply + Ny - ny; iy < Ny; iy++)  {
      int wp = nx - plx + (iy + ny - ply - Ny)*strideOut;
      for (int ix = 0; ix < plx; ix++)  {
         out[ix + wp] = T(f::apply(data[ix + iy * Nx][0]));
      }
      wp -= Nx;
      for (int ix = plx + Nx - nx; ix < Nx; ix++)  {
         out[ix + wp] = T(f::apply(data[ix + iy * Nx][0]));
      }
   }
}

class FftCorr {
   protected:
      T_COMPLEX *filter1, *work1, *work2;
      double *workNorm;
      int px, py;
   public:
      FftCorr (WindowFFT &, int patchSizeX, int patchSizeY);
      ~FftCorr ();

      template<class T>
      void getCorrMap (const T* image, int width,
                       int cx, int cy,
                       WindowFFT &,
                       double *out);
};

static void mulConj (int N, fftw_complex *inOut, const fftw_complex *in2)
{
   for (int i = 0; i < N; ++i)  {
      double re = inOut[i][0] * in2[i][0] + inOut[i][1] * in2[i][1];
      double im = inOut[i][0] * in2[i][1] - inOut[i][1] * in2[i][0];
      inOut[i][0] = re;
      inOut[i][1] = -im;
   }
}


template<class T>
void FftCorr::getCorrMap (const T* image, int width,
                          int cx, int cy,
                          WindowFFT &wFFT,
                          double *out)
{
   int offset = cx - wFFT.Nx/2 + width * (cy - wFFT.Ny/2);
   int resX = wFFT.Nx - px, resY = wFFT.Ny - py;
   // compute norm
   wFFT.toRecSpace2 (wFFT.Nx, wFFT.Ny, image + offset, width, work1);
   mulConj (wFFT.getSize (), work1, filter1);
   wFFT.toRealSpace<double,WindowFFT::SquareRootInv> (resX, resY, work1, workNorm, resX);
   // compute non-normalized correlation
   wFFT.toRecSpace (wFFT.Nx, wFFT.Ny, image + offset, width, work1);
   wFFT.toRecSpace (px, py, image + cx - px/2 + width * (cy - py/2), width, work2);
   mulConj (wFFT.getSize (), work1, work2);
   wFFT.toRealSpace (resX, resY, work1, out, resX);
   // normalize
   double n0 = workNorm[resX/2 + resX * (resY/2)];
   for (int i = 0; i < resX * resY; ++i)
      out[i] *= n0 * workNorm[i];
}
