// Declaration of Motion model for nonlinear planar model (unicycle system)
// This is the conditional probability density function which relates the actual state to the next state

#ifndef __NON_LINEAR_SYSTEM_MOBILE__
#define __NON_LINEAR_SYSTEM_MOBILE__

#include <pdf/conditionalpdf.h>
#include <pdf/gaussian.h>

namespace BFL
{
    // Nonlinear conditional gaussian pdf
    class NonlinearSystemPdf : public ConditionalPdf<MatrixWrapper::ColumnVector, MatrixWrapper::ColumnVector>
        {
            public:
                // Constructor
                NonlinearSystemPdf( const Gaussian& additiveNoise);

                // Destructor
                virtual ~NonlinearSystemPdf();

                // Method for sampling from the stochastic model
                virtual bool SampleFrom ( Sample<MatrixWrapper::ColumnVector>& one_sample, int method=DEFAULT, void * args=NULL) const;

            private:
                Gaussian _additiveNoise;
        };
}

#endif