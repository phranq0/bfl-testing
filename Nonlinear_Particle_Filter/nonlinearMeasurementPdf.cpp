// Definition of the probabilistic Motion Model (Unicycle)
// This model will be used from sampling the next pose (state) of each particle
// The sensor here is a unique laser reading, and the model is applied to each of the particles

#include "nonlinearMeasurementPdf.h"
#include <wrappers/rng/rng.h>          // Random number generator

#define MEASMODEL_NUMCONDARGUMENTS_MOBILE 1       // Sensor array size
#define MEASMODEL_DIMENSION_MOBILE 3              // Measured state size

namespace BFL
{
    using namespace MatrixWrapper;

    NonlinearMeasurementPdf::NonlinearMeasurementPdf(const Gaussian& measNoise)
        : ConditionalPdf<ColumnVector,ColumnVector>(MEASMODEL_DIMENSION_MOBILE,MEASMODEL_NUMCONDARGUMENTS_MOBILE)
    {
        _measNoise = measNoise;
    }

    NonlinearMeasurementPdf::~NonlinearMeasurementPdf(){};

    // Get conditional probability
    Probability NonlinearMeasurementPdf::ProbabilityGet(const ColumnVector& measurement) const
    {
        ColumnVector state = ConditionalArgumentGet(0);

        // Compute expected measurement for the current particle
        ColumnVector expected_measurement(1);
        expected_measurement(1) = 2 * state(2);

        // Computes the likelihood of the measurement
        return _measNoise.ProbabilityGet(expected_measurement - measurement);
    }
}