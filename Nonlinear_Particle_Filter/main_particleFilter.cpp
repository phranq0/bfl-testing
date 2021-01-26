// Complete test for running the particle filter example of Bfl

#include <filter/bootstrapfilter.h>

#include <model/systemmodel.h>
#include <model/measurementmodel.h>

#include "nonlinearSystemPdf.h"
#include "nonlinearMeasurementPdf.h"

#include "mobile_robot.h"

#include <iostream>
#include <fstream>

// Include file with properties
#include "mobile_robot_wall_cts.h"

using namespace MatrixWrapper;
using namespace BFL;
using namespace std;

int main(int argc, char** argv)
{
  cerr << "Basic Particle Filter testing" << endl;

  // ---------------------------------------------------- Motion Model
  // Nonlinear model of unicycle robot (x,y,theta)

  // -------------- Gaussian noise for the motion model
  // Mean values
  ColumnVector sys_noise_Mu(STATE_SIZE);
  sys_noise_Mu(1) = MU_SYSTEM_NOISE_X;
  sys_noise_Mu(2) = MU_SYSTEM_NOISE_Y;
  sys_noise_Mu(3) = MU_SYSTEM_NOISE_THETA;

  // Covariance matrix
  SymmetricMatrix sys_noise_Cov(STATE_SIZE);
  sys_noise_Cov = 0.0;
  sys_noise_Cov(1,1) = SIGMA_SYSTEM_NOISE_X;
  sys_noise_Cov(1,2) = 0.0;
  sys_noise_Cov(1,3) = 0.0;
  sys_noise_Cov(2,1) = 0.0;
  sys_noise_Cov(2,2) = SIGMA_SYSTEM_NOISE_Y;
  sys_noise_Cov(2,3) = 0.0;
  sys_noise_Cov(3,1) = 0.0;
  sys_noise_Cov(3,2) = 0.0;
  sys_noise_Cov(3,3) = SIGMA_SYSTEM_NOISE_THETA;

  // Create the gaussian distribution
  Gaussian system_Uncertainty(sys_noise_Mu, sys_noise_Cov);

  // Create the nonlinear motion model 
  // P(x(k+1)|x(k),u(k)) (in this case u is constant)
  NonlinearSystemPdf sys_pdf(system_Uncertainty);
  SystemModel<ColumnVector> sys_model(&sys_pdf);

  // ----------------------------------------------------- Measurement Model

  // Measurement model defined as distance from the wall
  double wall_ct = 2/(sqrt(pow(RICO_WALL,2.0) + 1));
  Matrix H(MEAS_SIZE,STATE_SIZE);
  H = 0.0;
  H(1,1) = wall_ct * RICO_WALL;
  H(1,2) = 0 - wall_ct;
  H(1,3) = 0.0;

  // Construct the measurement noise (a scalar in this case)
  ColumnVector meas_noise_Mu(MEAS_SIZE);
  meas_noise_Mu(1) = MU_MEAS_NOISE;
  SymmetricMatrix meas_noise_Cov(MEAS_SIZE);
  meas_noise_Cov(1,1) = SIGMA_MEAS_NOISE;

  // Create the gaussian distribution
  Gaussian measurement_Uncertainty(meas_noise_Mu, meas_noise_Cov);

  // Create the measurement model
  // P(z(k)|x(k)) 
  LinearAnalyticConditionalGaussian meas_pdf(H, measurement_Uncertainty);
  LinearAnalyticMeasurementModelGaussianUncertainty meas_model(&meas_pdf);

  // ------------------------------------------- Prior Distribution 
  // This represents the discrete set of particles at initialization step
  
  // Continuous Gaussian prior (try even with uniform particles)
  ColumnVector prior_Mu(STATE_SIZE);
  prior_Mu(1) = PRIOR_MU_X;
  prior_Mu(2) = PRIOR_MU_Y;
  prior_Mu(3) = PRIOR_MU_THETA;
  SymmetricMatrix prior_Cov(STATE_SIZE);
  prior_Cov(1,1) = PRIOR_COV_X;
  prior_Cov(1,2) = 0.0;
  prior_Cov(1,3) = 0.0;
  prior_Cov(2,1) = 0.0;
  prior_Cov(2,2) = PRIOR_COV_Y;
  prior_Cov(2,3) = 0.0;
  prior_Cov(3,1) = 0.0;
  prior_Cov(3,2) = 0.0;
  prior_Cov(3,3) = PRIOR_COV_THETA;
  Gaussian prior_cont(prior_Mu,prior_Cov);

  // Discrete prior for Particle filter (using the continuous Gaussian prior)
  vector<Sample<ColumnVector> > prior_samples(NUM_SAMPLES);
  MCPdf<ColumnVector> prior_discr(NUM_SAMPLES,STATE_SIZE);
  prior_cont.SampleFrom(prior_samples,NUM_SAMPLES,CHOLESKY,NULL);
  prior_discr.ListOfSamplesSet(prior_samples);

  // --------------------------------------------------------- Instance of the filter
  BootstrapFilter<ColumnVector,ColumnVector> filter(&prior_discr, 0, NUM_SAMPLES/4.0);

  // --------------------------------------------------------- Initialize mobile robot simulation
  // Model of mobile robot in world with one wall
  // The model is used to simulate the distance measurements
  MobileRobot mobile_robot;
  ColumnVector input(2);
  input(1) = 0.1;
  input(2) = 0.0;

  // For plots
  std::ofstream file_state;
  file_state.open ("gtruth_vs_estimate.csv");

  // -------------------------------------------------------- Main Estimation Loop
  cout << "MAIN: Starting estimation" << endl;
  unsigned int time_step;
  for (time_step = 0; time_step < NUM_TIME_STEPS-1; time_step++)
    {
      // Move the simulated robot
      mobile_robot.Move(input);

      // Take a measurement
      ColumnVector measurement = mobile_robot.Measure();

      // Update the filter (NB this performs both Prediction and Correction step)
      // Call this without meas_model and measurement for only Predict particles for motion model
      filter.Update(&sys_model,input,&meas_model,measurement);

      // Get and save estimate for each step
      Pdf<ColumnVector> * posterior = filter.PostGet();
      file_state << posterior->ExpectedValueGet()[0] << "," << posterior->ExpectedValueGet()[1] << "," << posterior->ExpectedValueGet()[2]
          << "," << mobile_robot.GetState()[0] << "," << mobile_robot.GetState()[1] << "," << mobile_robot.GetState()[2] << endl;

    } 

  file_state.close();

  Pdf<ColumnVector> * posterior_final = filter.PostGet();
  cout << "After " << time_step+1 << " timesteps " << endl;
  cout << " Posterior Mean = " << endl << posterior_final->ExpectedValueGet() << endl
       << " Covariance = " << endl << posterior_final->CovarianceGet() << "" << endl;


  cout << "Finished simulation" << endl;


  return 0;
}