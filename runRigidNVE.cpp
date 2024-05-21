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
  bool readAndMakeNewDir = false, readAndSaveSameDir = true, runDynamics = true;
  // readAndMakeNewDir reads the input dir and makes/saves a new output dir (cool or heat packing)
  // readAndSaveSameDir reads the input dir and saves in the same input dir (thermalize packing)
  // runDynamics works with readAndSaveSameDir and saves all the dynamics (run and save dynamics)
  bool readState = false, logSave = false, linSave = true, saveFinal = false;
  long numParticles = atof(argv[6]), nDim = 2, numVertexPerParticle = 32;
  long step = 0, maxStep = atof(argv[4]), initialStep = atof(argv[5]), multiple = 1, saveFreq = 1;
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), updateCount = 0;
  double ec = 1, cutDistance, cutoff = 0.5, sigma, waveQ, size; 
  double timeStep = atof(argv[2]), timeUnit,  Tinject = atof(argv[3]);
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "nve/";
  dirSample = whichDynamics + "T" + argv[3] + "/";
  // initialize dpm object
  DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  dpm.setParticleType(simControlStruct::particleEnum::rigid);
  dpm.setPotentialType(simControlStruct::potentialEnum::wca);
  dpm.setInteractionType(simControlStruct::interactionEnum::vertexSmooth);
  dpm.setNeighborType(simControlStruct::neighborEnum::neighbor);
  dpm.setConcavityType(simControlStruct::concavityEnum::on);
  ioDPMFile ioDPM(&dpm);
  // set input and output
  if (readAndSaveSameDir == true) {//keep running the same dynamics
    readState = true;
    inDir = inDir + dirSample;
    outDir = inDir;
    if(runDynamics == true) {
      outDir = outDir + "dynamics-smooth-test/";
      if(std::experimental::filesystem::exists(outDir) == true) {
        inDir = outDir;
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
  ioDPM.readRigidPackingFromDirectory(inDir, numParticles, nDim);
  dpm.setEnergyCosts(0, 0, 0, ec);
  if(readState == true) {
    ioDPM.readRigidState(inDir, numParticles, nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  sigma = dpm.getMeanParticleSize();
  timeUnit = sigma;//epsilon and mass are 1 sqrt(m sigma^2 / epsilon)
  timeStep = dpm.setTimeStep(timeStep * timeUnit);
  cout << "Time step: " << timeStep << " sigma: " << sigma << " Tinject: " << Tinject << endl;
  // initialize simulation
  dpm.initRigidNVE(Tinject, readState);
  size = 2 * dpm.getMeanVertexRadius();
  cutDistance = dpm.setDisplacementCutoff(cutoff, size);
  if(dpm.simControl.neighborType == simControlStruct::neighborEnum::neighbor) {
    dpm.calcNeighborList(cutDistance);
  } else if(dpm.simControl.neighborType == simControlStruct::neighborEnum::cell) {
    dpm.fillLinkedList();
  }
  dpm.calcRigidForceEnergy();
  dpm.resetUpdateCount();
  waveQ = dpm.getRigidWaveNumber();
  // record simulation time
  float elapsed_time_ms = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // run Rigid NVE integrator
  ioDPM.savePacking(outDir);
  ioDPM.saveNeighbors(outDir);
  while(step != maxStep) {
    dpm.rigidNVELoop();
    if(step % linFreq == 0) {
      ioDPM.saveRigidEnergy(step, timeStep, numParticles);
      if(step % checkPointFreq == 0) {
        cout << "Rigid NVE: current step: " << step;
        cout << " E/N: " << (dpm.getParticlePotentialEnergy() + dpm.getRigidKineticEnergy()) / numParticles;
        cout << " T: " << dpm.getParticleTemperature();
        cout << " ISF: " << dpm.getParticleISF(waveQ);
        updateCount = dpm.getUpdateCount();
        if(step != 0 && updateCount > 0) {
          cout << " number of updates: " << updateCount << " frequency " << checkPointFreq / updateCount << endl;
        } else {
          cout << " no updates" << endl;
        }
        dpm.resetUpdateCount();
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
  // instrument code to measure end time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);
  printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms); // exec. time
  // save final configuration
  if(saveFinal == true) {
    ioDPM.savePacking(outDir);
  }
  ioDPM.closeEnergyFile();

  return 0;
}
