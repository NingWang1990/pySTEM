#include "FftCorr.h"
//#include <iostream>
//using namespace std;

WindowFFT::WindowFFT (int Nx_, int Ny_)
   : Nx(Nx_), Ny(Ny_)
{
   assert (Nx > 0);
   assert (Ny > 0);
   work = fftw_alloc_complex (Nx * Ny);
   if (work)  {
      unsigned flags = FFTW_MEASURE | FFTW_DESTROY_INPUT;
      planFor = fftw_plan_dft_2d(Ny, Nx, work, work, FFTW_FORWARD, flags);
      planRev = fftw_plan_dft_2d(Ny, Nx, work, work, FFTW_BACKWARD, flags);
   } else {
      planFor = planRev = NULL;
   }
   //cout << "WindowFFT(Nx="<<Nx<<",Ny="<<Ny<<")"<<endl;
}

WindowFFT::~WindowFFT ()
{
   if (planRev) fftw_destroy_plan (planRev);
   if (planFor) fftw_destroy_plan (planFor);
   if (work) fftw_free(work);
   //cout << "~WindowFFT(Nx="<<Nx<<",Ny="<<Ny<<")"<<endl;
}

void WindowFFT::toRecSpace1 (int nx, int ny, T_COMPLEX* out)
{
   assert(nx > 0 && nx <= Nx);
   assert(ny > 0 && ny <= Ny);
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
   for (int iy = ply ; iy < ply + ny; iy++)  {
      for (int ix = 0; ix < plx; ix++)  {
         data[ix + iy * Nx][0] = 0.;
         data[ix + iy * Nx][1] = 0.;
      }
      for (int ix = plx; ix < plx+nx; ix++)  {
         data[ix + iy * Nx][0] = 1.;
         data[ix + iy * Nx][1] = 0.;
      }
      for (int ix = plx + nx; ix < Nx; ix++)  {
         data[ix + iy * Nx][0] = 0.;
         data[ix + iy * Nx][1] = 0.;
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

class One {
   public:
      static inline double apply (double x) { return 1.; }
};

FftCorr::FftCorr (WindowFFT &wFFT, int patchSizeX, int patchSizeY)
   : px(patchSizeX), py(patchSizeY)
{
   assert(px>0 && px < wFFT.Nx);
   assert(py>0 && py < wFFT.Ny);
   filter1 = (T_COMPLEX*)fftw_alloc_complex(wFFT.getSize ());
   work1 = (T_COMPLEX*)fftw_alloc_complex(wFFT.getSize ());
   work2 = (T_COMPLEX*)fftw_alloc_complex(wFFT.getSize ());
   workNorm = new double[(wFFT.Nx - px) * (wFFT.Ny - py)];
   wFFT.toRecSpace1 (px, py, filter1);
   //wFFT.toRecSpace<double,One> (px, py, (double*)filter1, wFFT.Nx, filter1);
}

FftCorr::~FftCorr ()
{
   delete [] workNorm;
   fftw_free (work2);
   fftw_free (work1);
   fftw_free (filter1);
}

