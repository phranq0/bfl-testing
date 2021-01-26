// Header file which defines nonlinear probability density function (aka Measurement Model)
// related to a pont-shaped mobile robot, which is a conditional probability density

#ifndef __NON_LINEAR_MEAS_MOBILE__
#define __NON_LINEAR_MEAS_MOBILE__

#include <pdf/conditionalpdf.h>
#include <pdf/gaussian.h>

namespace BFL
{
    // Non linear conditional gaussian, inherited from conditional
    class NonlinearMeasurementPdf : public ConditionalPdf<MatrixWrapper::ColumnVector, MatrixWrapper::ColumnVector>
        {
            public:
                // Constructor, takes gaussian pdf as measurement noise
                NonlinearMeasurementPdf(const Gaussian& measNoise);

                // Destructor 
                virtual ~NonlinearMeasurementPdf();

                // Method for getting the measurement model
                virtual Probability ProbabilityGet(const MatrixWrapper::ColumnVector& measurement) const;

            private:
                Gaussian _measNoise;

        };
}

#endif