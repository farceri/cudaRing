//
// Author: Francesco Arceri
// Date:   10-25-2021
//
// Include C++ header files

#include "include/DPM2D.h"
#include "include/FileIO.h"
#include "include/FIRE.h"
#include "include/defs.h"
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <functional>
#include <utility>
#include <thrust/host_vector.h>
#include <experimental/filesystem>

using namespace std;

int main(int argc, char **argv) {
  // variables
  long numParticles = 32, nDim = 2, numVertexPerParticle = 32;
  long numVertices = numParticles * numVertexPerParticle, skiplines = 1;
  long iteration = 0, maxIterations = 1e06, printFreq = 100, savePosFreq = 1e07;
  long minStep = 20, numStep = 0, updateFreq = 100;
  double polydispersity = 0.1, phi0 = 0.4;
  double cutDistance = 2., forceTollerance = 1e-10, forceCheck;
  double ea = 1e03, el = 1, eb = 1e-02, ec = 1, timeStep, dt0 = 5e-02;
  double calA0 = 1.01, thetaA = 1., thetaK = 0., cutoff, maxDelta;
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, dt0, 10*dt0, 0.2};
  std::string inFile, outDir = argv[1];
	// initialize dpm object
	DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  dpm.setEnergyCosts(ea, el, eb, ec);
  ioDPMFile ioDPM(&dpm);
  std::experimental::filesystem::create_directory(outDir);
  // initialize polydisperse packing
  dpm.setPolySizeDistribution(calA0, polydispersity);
  dpm.setSinusoidalRestAngles(thetaA, thetaK);
  dpm.setRandomParticles(phi0);
  // minimize soft sphere packing
  dpm.initFIRE(particleFIREparams, minStep, numStep, numParticles);
  dpm.setParticleMassFIRE();
  dpm.calcParticleNeighborList(cutDistance);
  cutoff = (1 + cutDistance) * dpm.getMinParticleSigma();
  dpm.resetParticleLastPositions();
  while((dpm.getParticleMaxUnbalancedForce() > forceTollerance) && (iteration != maxIterations)) {
    dpm.particleFIRELoop();
    maxDelta = dpm.getParticleMaxDisplacement();
    if(3*maxDelta > cutoff) {
      dpm.calcParticleNeighborList(cutDistance);
      dpm.resetParticleLastPositions();
    }
    iteration += 1;
  }
  cout << "\nFIRE: iteration: " << iteration;
  cout << " maxUnbalancedForce: " << setprecision(precision) << dpm.getParticleMaxUnbalancedForce();
  cout << " energy: " << setprecision(precision) << dpm.getParticlePotentialEnergy() << endl;
  // put vertices on particle perimeters
  dpm.initVerticesOnParticles();
  cout << "current packing fraction: " << setprecision(precision) << dpm.getPhi() << endl;
  // minimize deformable particle packing
  dpm.calcNeighborList(cutDistance);
  dpm.calcForceEnergy();
  iteration = 0;
  timeStep = dpm.setTimeScale(dt0);
  cout << "fire time step: " << timeStep << endl;
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> vertexFIREparams = {0.2, 0.5, 1.1, 0.99, timeStep, 10*timeStep, 0.2};
  dpm.initFIRE(vertexFIREparams, minStep, numStep, numVertices);
  dpm.resetLastPositions();
  cutoff = (1 + cutDistance) * dpm.getMeanVertexRadius();
  dpm.setDisplacementCutoff(cutoff, cutDistance);
  forceCheck = dpm.getTotalForceMagnitude();
  while((forceCheck > forceTollerance) && (iteration != maxIterations)) {
    dpm.vertexFIRELoop();
    forceCheck = dpm.getTotalForceMagnitude();
    if(iteration % printFreq == 0) {
      cout << "\nFIRE: iteration: " << iteration;
      cout << " maxUnbalancedForce: " << setprecision(precision) << forceCheck;
      cout << " energy: " << setprecision(precision) << dpm.getPotentialEnergy();
      cout << " timeStep: " << setprecision(precision) << dpm.getFIRETimeStep() << endl;
    }
    iteration += 1;
  }
  cout << "\nFIRE: iteration: " << iteration;
  cout << " maxUnbalancedForce: " << setprecision(precision) << dpm.getMaxUnbalancedForce();
  cout << " energy: " << setprecision(precision) << dpm.getPotentialEnergy() << endl;
  // save minimized configuration
  ioDPM.savePacking(outDir);

  return 0;
}
