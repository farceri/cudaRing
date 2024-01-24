//
// Author: Francesco Arceri
// Date:   12-16-2021
//
// Include C++ header files

#include "include/DPM2D.h"
#include "include/FileIO.h"
#include "include/Simulator.h"
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
  bool readAndMakeNewDir = false, readAndSaveSameDir = true, runDynamics = true;
  // readAndMakeNewDir reads the input dir and makes/saves a new output dir (cool or heat packing)
  // readAndSaveSameDir reads the input dir and saves in the same input dir (thermalize packing)
  // runDynamics works with readAndSaveSameDir and saves all the dynamics (run and save dynamics)
  bool readState, logSave, linSave = true;
  long numParticles = 128, nDim = 2, numVertexPerParticle = 32;
  long step = 0, maxStep = atof(argv[4]), saveEnergyFreq = 1e04;
  long initialStep = 0, checkPointFreq = 1e06, multiple = 1, saveFreq = 1;
  long linFreq = int(checkPointFreq/10), firstDecade = 0;
  double ec = 240, cutDistance = 2., timeStep, dt0 = atof(argv[3]); // relative to the k's
  // kb/ka should be bound by 1e-04 and 1 and both kc/kl should be 1
  double Dr = 1e-01, driving = atof(argv[2]), cutoff, maxDelta;
  std::string outDir, energyFile, corrFile, currentDir, readDir;
  std::string inDir = argv[1], dirSample = argv[2];
  dirSample = "ab/Dr1e-01-v0" + dirSample + "/";
  // initialize dpm object
  DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  dpm.setEnergyCosts(0, 0, 0, ec);
  ioDPMFile ioDPM(&dpm);
  // set input and output
  if (readAndSaveSameDir == true) {//keep running the same dynamics
    readState = true;
    inDir = inDir + dirSample;
    outDir = inDir;
    if(runDynamics == true) {
      logSave = true;
      outDir = outDir + "dynamics/";
      if(std::experimental::filesystem::exists(outDir) == true) {
        inDir = outDir;
        initialStep = atof(argv[5]);
      } else {
        std::experimental::filesystem::create_directory(outDir);
      }
    }
  } else {//start a new dyanmics
    if(readAndMakeNewDir == true) {
      readState = true;
      outDir = inDir + "../../" + dirSample;
    } else {
      if(std::experimental::filesystem::exists(inDir + "ab/") == false) {
        std::experimental::filesystem::create_directory(inDir + "ab");
      }
      outDir = inDir + dirSample;
    }
    std::experimental::filesystem::create_directory(outDir);
  }
  ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) {
    ioDPM.readRigidState(inDir, numParticles, dpm.getNumVertices(), nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  // initialization
  timeStep = dpm.setTimeStep(dt0);
  dpm.calcParticlesShape();
  cout << "Current phi: " << dpm.getPhi() << endl;
  cout << "Time scale: " << timeStep << endl;
  cout << "Peclet number: " << (driving/Dr) / dpm.getMeanParticleSize() << " v0: " << driving << " Dr: " << Dr << endl;
  // initialize simulation
  dpm.initRigidActiveBrownian(Dr, driving, cutDistance, readState);
  dpm.calcNeighborList(cutDistance);
  dpm.resetLastPositions();
  cutoff = (1 + cutDistance) * dpm.getMeanVertexRadius();
  dpm.setDisplacementCutoff(cutoff, cutDistance);
  // run AB integrator
  while(step != maxStep + 1) {
    dpm.rigidActiveBrownianLoop();
    if(step % saveEnergyFreq == 0) {
      ioDPM.saveRigidEnergy(step, timeStep);
      if(step % checkPointFreq == 0) {
        cout << "Rigid Active Brownian: current step: " << step;
        cout << " potential energy: " << dpm.getPotentialEnergy();
        cout << " Teff: " << dpm.getParticleTemperature();
        cout << " phi: " << dpm.getPhi() << endl;
      	ioDPM.saveRigidPacking(outDir);
        ioDPM.saveRigidState(outDir);
      }
    }
    if(logSave == true) {
      if(step > (multiple * checkPointFreq)) {
        saveFreq = 1;
        multiple += 1;
      }
      if((step - (multiple-1) * checkPointFreq) > saveFreq*10) {
        saveFreq *= 10;
      }
      if(((step - (multiple-1) * checkPointFreq) % saveFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioDPM.saveRigidTrajectory(currentDir);
        ioDPM.saveRigidPacking(currentDir);
      }
    }
    if(linSave == true) {
      if((step > (firstDecade * checkPointFreq)) && ((step % linFreq) == 0)) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioDPM.saveRigidTrajectory(currentDir);
        ioDPM.saveRigidPacking(currentDir);
      }
    }
    step += 1;
  }
  // save final configuration
  ioDPM.saveRigidPacking(outDir);
  ioDPM.closeEnergyFile();

  return 0;
}
