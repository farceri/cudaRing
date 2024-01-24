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
  bool readState = false, logSave, linSave = false, saveFinal = true;
  long numParticles = atof(argv[6]), nDim = 2, numVertexPerParticle = 32;
  long step = 0, maxStep = atof(argv[4]), initialStep = 0, multiple = 1, saveFreq = 1;
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), updateCount = 0;
  double cutDistance = 1, timeStep = atof(argv[2]), timeUnit = 0, sigma, waveQ;
  double ea = 1e04, el = 10, eb = 1, ec = 1, maxDelta, cutoff, Tinject = atof(argv[3]);
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "nve/";
  dirSample = whichDynamics + "T" + argv[3] + "/";
  // initialize dpm object
  DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  dpm.setPotentialType(simControlStruct::potentialEnum::wca);
  //dpm.setInteractionType(simControlStruct::interactionEnum::smooth);
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
      //outDir = inDir + "../../../" + dirSample;
    } else {
      if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
        std::experimental::filesystem::create_directory(inDir + whichDynamics);
      }
      outDir = inDir + dirSample;
    }
    std::experimental::filesystem::create_directory(outDir);
  }
  ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
  dpm.setEnergyCosts(ea, el, eb, ec);
  if(readState == true) {
    ioDPM.readState(inDir, numParticles, dpm.getNumVertices(), nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  sigma = dpm.getMeanParticleSize();
  timeUnit = sigma;//epsilon and mass are 1 sqrt(m sigma^2 / epsilon)
  timeStep = dpm.setTimeStep(timeStep * timeUnit);
  cout << "Time step: " << timeStep << " sigma: " << sigma << " Tinject: " << Tinject << endl;
  // initialize simulation
  dpm.calcNeighborList(cutDistance);
  dpm.calcForceEnergy();
  dpm.initNVE(Tinject, readState);
  dpm.resetLastPositions();
  cutoff = (1 + cutDistance) * dpm.getMeanVertexRadius();
  dpm.setDisplacementCutoff(cutoff, cutDistance);
  waveQ = dpm.getDeformableWaveNumber();
  // run NVE integrator
  while(step != maxStep) {
    dpm.NVELoop();
    if(step % linFreq == 0) {
      ioDPM.saveEnergy(step, timeStep);
      if(step % checkPointFreq == 0) {
        cout << "NVE: current step: " << step;
        cout << " E/N: " << (dpm.getSmoothPotentialEnergy() + dpm.getKineticEnergy()) / numParticles;
        cout << " T: " << dpm.getTemperature();
        cout << " ISF: " << dpm.getParticleISF(waveQ);
        cout << " phi: " << dpm.getPhi() << endl;
        if(saveFinal == true) {
      	  ioDPM.savePacking(outDir);
        }
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
    ioDPM.savePacking(outDir);
  }
  ioDPM.closeEnergyFile();

  return 0;
}
