//
// Author: Francesco Arceri
// Date:   10-03-2021
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
#include <stdlib.h>
#include <math.h>
#include <functional>
#include <utility>
#include <thrust/host_vector.h>
#include <experimental/filesystem>

using namespace std;

int main(int argc, char **argv) {
  // variables
  bool readAndMakeNewDir = false, readAndSaveSameDir = false, runDynamics = false;
  // readAndMakeNewDir reads the input dir and makes/saves a new output dir (cool or heat packing)
  // readAndSaveSameDir reads the input dir and saves in the same input dir (thermalize packing)
  // runDynamics works with readAndSaveSameDir and saves all the dynamics (run and save dynamics)
  bool readState, logSave, linSave = true;
  long numParticles = 32, nDim = 2, numVertexPerParticle = 32;
  long step = 0, maxStep = atof(argv[4]), checkPointFreq = int(maxStep / 10), updateFreq = 1e01;
  long initialStep = 0, saveEnergyFreq = int(checkPointFreq / 10), multiple = 1, saveFreq = 1;
  long linFreq = saveEnergyFreq, firstDecade = 0;
  double cutDistance = 2., timeStep, dt0 = atof(argv[3]); // relative to the k's
  double ea = 1e03, el = 1, eb = 1e-02, ec = 1, cutoff, maxDelta;
  // kb/ka should be bound by 1e-04 and 1 and both kc/kl should be 1
  double Dr = 1e-01, driving = atof(argv[2]), gamma = 1e03;
  std::string outDir, energyFile, corrFile, currentDir, readDir;
  std::string inDir = argv[1], dirSample = argv[2];
  dirSample = "ab/Dr1e-01-v0" + dirSample + "/";
  // initialize dpm object
  DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  dpm.setEnergyCosts(ea, el, eb, ec);
  ioDPMFile ioDPM(&dpm);
  // set input and output
  if (readAndSaveSameDir == true) {//keep running the same dynamics
    readState = true;
    inDir = inDir + dirSample;
    outDir = inDir;
    if(runDynamics == true) {
      logSave = false;
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
        std::experimental::filesystem::create_directory(inDir + "ab/");
      }
      outDir = inDir + dirSample;
    }
    std::experimental::filesystem::create_directory(outDir);
  }
  ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
  if(readState == true) {
    ioDPM.readState(inDir, numParticles, dpm.getNumVertices(), nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  // initialization
  timeStep = dpm.setTimeScale(dt0);
  cout << "Current phi: " << dpm.getPhi() << endl;
  cout << "Time scale: " << timeStep << endl;
  cout << "Peclet number: " << (driving/Dr) / dpm.getMeanParticleSize() << " v0: " << driving << " Dr: " << Dr << " Dr * dt: " << Dr*timeStep << endl;
  // initialize simulation
  dpm.calcNeighborList(cutDistance);
  dpm.calcForceEnergy();
  dpm.initActiveBrownianPlastic(Dr, driving, gamma, readState);
  dpm.resetLastPositions();
  cutoff = (1 + cutDistance) * dpm.getMeanVertexRadius();
  dpm.setDisplacementCutoff(cutoff, cutDistance);
  // run AB integrator
  while(step != maxStep + 1) {
    dpm.activeBrownianPlasticLoop();
    if(step % saveEnergyFreq == 0) {
      ioDPM.saveEnergy(step, timeStep);
      if(step % checkPointFreq == 0) {
        cout << "Active Brownian DampedL0: current step: " << step;
        cout << " potential energy: " << dpm.getPotentialEnergy();
        cout << " Teff: " << dpm.getTemperature();
        cout << " phi: " << dpm.getPhi() << endl;
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
        ioDPM.savePacking(currentDir);
      }
    }
    if(linSave == true) {
      if((step > (firstDecade * checkPointFreq)) && ((step % linFreq) == 0)) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioDPM.savePacking(currentDir);
      }
    }
    step += 1;
  }
  // save final configuration
  ioDPM.saveConfiguration(outDir);
  ioDPM.closeEnergyFile();

  return 0;
}
