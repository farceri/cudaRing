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
  bool readAndMakeNewDir = true, readAndSaveSameDir = false, runDynamics = false;
  // readAndMakeNewDir reads the input dir and makes/saves a new output dir (cool or heat packing)
  // readAndSaveSameDir reads the input dir and saves in the same input dir (thermalize packing)
  // runDynamics works with readAndSaveSameDir and saves all the dynamics (run and save dynamics)
  bool readState = true, logSave, linSave = false, saveFinal = true;
  long numParticles = 512, nDim = 2, numVertexPerParticle = 32;
  long step = 0, maxStep = atof(argv[6]), checkPointFreq = int(maxStep / 10), updateFreq = 10;
  long initialStep = 0, saveEnergyFreq = int(checkPointFreq / 10), multiple = 1, saveFreq = 1;
  long linFreq = saveEnergyFreq, firstDecade = 0;
  double cutDistance = 1, waveQ, damping, inertiaOverDamping = atof(argv[8]), timeUnit, timeStep = atof(argv[2]);
  double sigma, Tinject = atof(argv[3]), Dr = atof(argv[4]), driving = atof(argv[5]);
  double ea = 100, el = 1, eb = 1e-01, ec = 1, cutoff, maxDelta;
  // kb/ka should be bound by 1e-04 and 1 and both kc/kl should be 1
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "active-langevin/";
  dirSample = whichDynamics + "Dr" + argv[4] + "-f0" + argv[5] + "/";
  // initialize dpm object
  DPM2D dpm(numParticles, nDim, numVertexPerParticle);
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
        initialStep = atof(argv[7]);
      } else {
        std::experimental::filesystem::create_directory(outDir);
      }
    }
  } else {//start a new dyanmics
    if(readAndMakeNewDir == true) {
      readState = true;
      outDir = inDir + "../../" + dirSample;
    } else {
      if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
        std::experimental::filesystem::create_directory(inDir + whichDynamics);
      }
      outDir = inDir + dirSample;
    }
    std::experimental::filesystem::create_directory(outDir);
  }
  ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
  sigma = dpm.getMeanParticleSize();
  dpm.setEnergyCosts(ea, el, eb, ec);
  if(readState == true) {
    ioDPM.readState(inDir, numParticles, dpm.getNumVertices(), nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  // initialization
  damping = sqrt(inertiaOverDamping * ec) / sigma;
  cout << "damping: " << damping << " with inertia over damping: " << inertiaOverDamping << endl;
  timeUnit = sigma * sigma * damping / ec;
  timeStep = dpm.setTimeStep(timeStep*timeUnit);
  cout << "Time step: " << timeStep << " Tinject: " << Tinject << " sigma: " << sigma << endl;
  cout << "Peclet number: " << ((driving/damping) / Dr) / sigma << " v0: " << driving / damping << ", " << driving << " Dr: " << Dr/timeUnit << ", " << Dr << endl;
  Dr = Dr/timeUnit;
  // initialize simulation
  dpm.calcNeighborList(cutDistance);
  dpm.calcForceEnergy();
  dpm.initActiveLangevin(Tinject, Dr, driving, damping, readState);
  dpm.resetLastPositions();
  cutoff = (1 + cutDistance) * dpm.getMeanVertexRadius();
  dpm.setDisplacementCutoff(cutoff, cutDistance);
  waveQ = dpm.getDeformableWaveNumber();
  // run AB integrator
  while(step != maxStep) {
    dpm.activeLangevinLoop();
    if(step % saveEnergyFreq == 0) {
      ioDPM.saveEnergy(step, timeStep);
      if(step % checkPointFreq == 0) {
        cout << "Active Langevin: current step: " << step;
        cout << " E: " << dpm.getSmoothPotentialEnergy() + dpm.getKineticEnergy();
        cout << " T: " << dpm.getTemperature();
        cout << " ISF: " << dpm.getParticleISF(waveQ);
        cout << " phi: " << dpm.getPhi() << endl;
      	ioDPM.saveConfiguration(outDir);
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
        ioDPM.saveState(currentDir);
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioDPM.saveState(currentDir);
      }
    }
    step += 1;
  }
  // save final configuration
  if(saveFinal == true) {
    ioDPM.saveConfiguration(outDir);
  }
  ioDPM.closeEnergyFile();

  return 0;
}
