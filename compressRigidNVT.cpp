//
// Author: Francesco Arceri
// Date:   10-25-2021
//
// Include C++ header files

#include "include/DPM2D.h"
#include "include/FileIO.h"
#include "include/Simulator.h"
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
  bool read = true, readState = false;
  long numParticles = atol(argv[4]), nDim = 2, numVertexPerParticle = 32, numVertices;
  long iteration = 0, maxIterations = 5e06, maxSearchStep = 1500, searchStep = 0;
  long step, maxStep = atof(argv[6]), printFreq = int(maxStep / 10), minStep = 20, numStep = 0;
  double sigma, cutDistance = 1, polydispersity = 0.16, previousPhi, currentPhi = 0.1, deltaPhi = 5e-03;
  double forceTollerance = 1e-12, waveQ, FIREStep = 1e-03, timeStep, timeUnit, dt = atof(argv[2]);
  double Tinject = atof(argv[3]), phiTh = 0.84, cutoff, scaleFactor, maxDelta;
  double ec = 1, calA0 = atof(argv[5]), damping, inertiaOverDamping = 100;
  thrust::host_vector<double> boxSize(nDim);
  std::string outDir = argv[1], currentDir, inDir, energyFile;
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, FIREStep, 10*FIREStep, 0.2};
	// initialize dpm object
	DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  ioDPMFile ioDPM(&dpm);
  std::experimental::filesystem::create_directory(outDir);
  // read initial configuration
  if(read == true) {
    inDir = outDir + argv[7];
    ioDPM.readRigidPackingFromDirectory(inDir, numParticles, nDim);
    if(readState == true) {
      ioDPM.readRigidState(inDir, numParticles, nDim);
    }
  } else {
    // initialize polydisperse packing and minimize soft particle packing with harmonic potential
    dpm.setPolySizeDistribution(calA0, polydispersity);
    dpm.setScaledRandomParticles(currentPhi, 1.1); //extraRad
    dpm.scaleParticlePacking();
    dpm.setEnergyCosts(0, 0, 0, ec);
    // minimize soft sphere packing
    dpm.initFIRE(particleFIREparams, minStep, numStep, numParticles);
    dpm.setParticleMassFIRE();
    dpm.calcParticleNeighborList(cutDistance);
    dpm.calcParticleForceEnergy();
    cutoff = (1 + cutDistance) * dpm.getMinParticleSigma();
    dpm.resetParticleLastPositions();
    while((dpm.getParticleMaxUnbalancedForce() > forceTollerance) && (iteration != maxIterations)) {
      dpm.particleFIRELoop();
      if(iteration % printFreq == 0 && iteration != 0) {
      cout << "FIRE: iteration: " << iteration;
      cout << " maxUnbalancedForce: " << setprecision(precision) << dpm.getParticleMaxUnbalancedForce();
      cout << " energy: " << dpm.getParticlePotentialEnergy() << endl;
      }
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
    currentDir = outDir + "sp/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.saveParticlePacking(currentDir);
    // put vertices on particle perimeters
    dpm.initVerticesOnParticles();
    dpm.scalePacking(dpm.getMeanParticleSize());
    currentDir = outDir + "dpm/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.savePacking(currentDir);
  }
  dpm.setPotentialType(simControlStruct::potentialEnum::wca);
  dpm.setEnergyCosts(0, 0, 0, ec);
  currentPhi = dpm.getPreferredPhi();
  cout << "Start compression: current packing fraction: " << currentPhi << endl;
  previousPhi = currentPhi;
  // isotropic isothermal compression
  while (searchStep < maxSearchStep) {
    sigma = dpm.getMeanParticleSize();
    damping = sqrt(inertiaOverDamping) / sigma;
    timeUnit = sigma / sqrt(ec);
    timeStep = dpm.setTimeStep(dt * timeUnit);
    cout << "Units - time: " << timeUnit << " space: " << sigma << " time step: " << timeStep << endl;
    cout << "Thermostat - damping: " << damping << " Tinject: " << Tinject << endl;
    dpm.initRigidLangevin(Tinject, damping, readState);
    dpm.calcNeighborList(cutDistance);
    dpm.calcRigidForceEnergy();
    waveQ = dpm.getRigidWaveNumber();
    // equilibrate rigid particles
    step = 0;
    dpm.resetLastPositions();
    cutoff = (1 + cutDistance) * dpm.getMeanVertexRadius();
    dpm.setDisplacementCutoff(cutoff);
    while(step != maxStep) {
      dpm.rigidLangevinLoop();
      if(step % printFreq == 0) {
        cout << "Rigid Langevin: current step: " << step;
        cout << " E/N: " << dpm.getParticlePotentialEnergy() / numParticles;
        cout << " T: " << dpm.getParticleTemperature();
        cout << " ISF: " << dpm.getParticleISF(waveQ) << endl;
      }
      step += 1;
    }
    // save minimized configuration
    std::string currentDir = outDir + std::to_string(dpm.getPreferredPhi()).substr(0,6) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.saveRigidPacking(currentDir);
    // check if target density is met
    if(currentPhi >= phiTh) {
      cout << " target density met, current phi: " << currentPhi << endl;
      searchStep = maxSearchStep; // exit condition
    } else {
      scaleFactor = sqrt((currentPhi + deltaPhi) / currentPhi);
      //dpm.scaleRigidVertices(scaleFactor);
      dpm.scaleBoxSize(scaleFactor);
      //dpm.scaleRigidPacking(dpm.getMeanParticleSize());
      boxSize = dpm.getBoxSize();
      currentPhi = dpm.getPreferredPhi();
      cout << "\nNew packing fraction: " << currentPhi << endl;
      cout << "New boxSize: Lx: " << boxSize[0] << " Ly: " << boxSize[1] << " scale: " << scaleFactor << endl;
      searchStep += 1;
    }
    ioDPM.saveEnergy(step, timeStep);
  }
  ioDPM.closeEnergyFile();

  return 0;
}
