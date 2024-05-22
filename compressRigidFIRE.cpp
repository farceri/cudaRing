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
  bool read = false, savePos = false;
  long numParticles = 32, nDim = 2, numVertexPerParticle = 32, numVertices;
  long iteration = 0, maxIterations = 5e06, printFreq = 1e03;
  long minStep = 20, numStep = 0, updateFreq = 10, savePosFreq = 1e05;
  long maxSearchStep = 500, searchStep = 0, posIndex = 0;
  double polydispersity = 0.2, previousPhi, currentPhi = 0.4, deltaPhi = 2e-03, scaleFactor;
  double cutDistance = 2., forceTollerance = 1e-12, pressureTollerance = 1e-10, pressure;
  double ec = 240, calA0 = 1., timeStep = 5e-03, forceCheck, cutoff, maxDelta;
  bool jamCheck = 0, overJamCheck = 0, underJamCheck = 0;
  std::string outDir = argv[1], currentDir, inDir;
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, 1e-01, 1., 0.2};
	// initialize dpm object
	DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  dpm.setSpringConstants(0, 0, 0, ec);
  ioDPMFile ioDPM(&dpm);
  std::experimental::filesystem::create_directory(outDir);
  // read initial configuration
  if(read == true) {
    inDir = outDir + argv[2];
    ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
  } else {
    // initialize polydisperse packing
    dpm.setPolySizeDistribution(calA0, polydispersity);
    dpm.setRandomParticles(currentPhi, 1.2); //extraRad
    // minimize soft sphere packing
    dpm.initFIRE(particleFIREparams, minStep, numStep, numParticles);
    dpm.setParticleMassFIRE();
    dpm.calcParticleNeighborList(cutDistance);
    dpm.calcParticleForceEnergy();
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
  }
  dpm.calcParticleShape();
  cout << "current packing fraction: " << setprecision(precision) << dpm.getPhi() << " " << dpm.getPreferredPhi() << endl;
  numVertices = dpm.getNumVertices();
  thrust::host_vector<double> positions(numVertices * nDim);
  thrust::host_vector<double> radii(numVertices);
  thrust::host_vector<double> lengths(numVertices);
  thrust::host_vector<double> areas(numParticles);

  // quasistatic isotropic compression
  cout << "fire time step: " << timeStep << endl;
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> vertexFIREparams = {0.2, 0.5, 1.1, 0.99, timeStep, 10*timeStep, 0.2};
  dpm.initRigidFIRE(vertexFIREparams, minStep, numStep, numVertices, cutDistance);
  cutoff = (1 + cutDistance) * dpm.getMinVertexRadius();
  currentPhi = dpm.getPhi();
  previousPhi = currentPhi;
  while (searchStep < maxSearchStep) {
    dpm.calcNeighborList(cutDistance);
    dpm.calcRigidForceEnergy();
    dpm.resetLastPositions();
    cutoff = (1 + cutDistance) * dpm.getMeanVertexRadius();
    dpm.setDisplacementCutoff(cutoff, cutDistance);
    iteration = 0;
    forceCheck = dpm.getRigidMaxUnbalancedForce();
    while((forceCheck > forceTollerance) && (iteration != maxIterations)) {
      dpm.rigidFIRELoop();
      forceCheck = dpm.getRigidMaxUnbalancedForce();
      if(iteration % printFreq == 0) {
        cout << "\nFIRE: iteration: " << iteration;
        cout << " vertex MUF: " << setprecision(precision) << dpm.getMaxUnbalancedForce() << " particle MUF: " << dpm.getRigidMaxUnbalancedForce();
        cout << " energy: " << dpm.getPotentialEnergy() << endl;
      }
      if((savePos == true) && ((iteration * (searchStep + 1)) % savePosFreq == 0)) {
        ioDPM.save2DFile(outDir + "pos" + std::to_string(posIndex) + ".dat", dpm.getVertexPositions(), nDim);
        ioDPM.save1DFile(outDir + "rad" + std::to_string(posIndex) + ".dat", dpm.getVertexRadii());
        posIndex += 1;
      }
      iteration += 1;
    }
    pressure = dpm.getPressure();
    cout << "FIRE: iteration: " << iteration;
    cout << " maxUnbalancedForce: " << setprecision(precision) << forceCheck;
    cout << " energy: " << dpm.getPotentialEnergy();
    cout << " pressure: " << pressure << endl;
    // save minimized configuration
    currentDir = outDir + std::to_string(currentPhi).substr(0,7) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.saveRigidPacking(currentDir);
    // check configuration after energy minimization
    jamCheck = (pressure < 2.0 * pressureTollerance && pressure > pressureTollerance);
    overJamCheck = (pressure > 2.0 * pressureTollerance);
    underJamCheck = (pressure < pressureTollerance);
    if(jamCheck) {
      cout << "Compression step: " << searchStep;
      cout << " Found jammed configuration, pressure: " << setprecision(precision) << pressure;
      cout << " packing fraction: " << currentPhi << endl;
      if(pressureTollerance == 1e-06) {
        searchStep = maxSearchStep; // exit condition
      }
      pressureTollerance = 1e-06;
      deltaPhi = 2e-03;
    } else {
      // compute scaleFactor with binary search
      if(underJamCheck) {
        scaleFactor = sqrt((currentPhi + deltaPhi) / currentPhi);
        cout << "Compression step: " << searchStep;
        cout << " Found underjammed configuration, scaleFactor: " << scaleFactor;
        // save most recent underjammed configuration
        previousPhi = currentPhi;
        positions = dpm.getVertexPositions();
        radii = dpm.getVertexRadii();
        lengths = dpm.getRestLengths();
        areas = dpm.getRestAreas();
      } else if(overJamCheck) {
        deltaPhi *= 0.5;
        scaleFactor = sqrt((previousPhi + deltaPhi) / previousPhi);
        cout << "Compression step: " << searchStep;
        cout << " Found overjammed configuration, scaleFactor: " << scaleFactor;
        // copy back most recent underjammed configuration and compress half much
        dpm.setVertexPositions(positions);
        dpm.setVertexRadii(radii);
        dpm.setRestLengths(lengths);
        dpm.setRestAreas(areas);
      }
      dpm.scaleVertices(scaleFactor);
      currentPhi = dpm.getPhi();
      cout << " new packing fraction: " << currentPhi << " " << dpm.getPreferredPhi() << endl;
      searchStep += 1;
    }
  }

  return 0;
}
