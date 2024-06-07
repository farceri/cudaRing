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
  bool readState = true, logSave = false, linSave = true, saveFinal = true;
  long numParticles = atof(argv[6]), nDim = 2, numVertexPerParticle = 32, numVertices;
  long step = 0, maxStep = atof(argv[4]), initialStep = atof(argv[5]), multiple = 1, saveFreq = 1, updateCount = 0;
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10); 
  double cutDistance, cutoff = 0.5, timeStep = atof(argv[2]), timeUnit = 0, sigma, waveQ;
  double ea = 1e05, el = 20, eb = 10, ec = 1, LJcut = 1.5, Tinject = atof(argv[3]), size;
  std::string outDir, energyFile, currentDir, potType = argv[7], inDir = argv[1], dirSample, whichDynamics;
  // initialize dpm object
  DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  if(potType == "lj") {
    dpm.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    whichDynamics = "nve-lj/";
    dpm.setLJcutoff(LJcut);
  } else if(potType == "wca") {
    dpm.setPotentialType(simControlStruct::potentialEnum::wca);
    whichDynamics = "nve-wca/";
  } else {
    cout << "Setting default harmonic potential" << endl;
    whichDynamics = "nve/";
  }
  dirSample = whichDynamics + "T" + argv[3] + "/";
  dpm.setInteractionType(simControlStruct::interactionEnum::vertexSmooth);
  ioDPMFile ioDPM(&dpm);
  // set input and output
  if (readAndSaveSameDir == true) {//keep running the same dynamics
    readState = true;
    inDir = inDir + dirSample;
    outDir = inDir;
    if(runDynamics == true) {
      outDir = outDir + "dynamics/";
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
  ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
  dpm.setEnergyCosts(ea, el, eb, ec);
  if(readState == true) {
    ioDPM.readState(inDir, numParticles, dpm.getNumVertices(), nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  sigma = dpm.getMeanParticleSize();
  timeUnit = sigma / sqrt(ec);//epsilon and mass are 1 sqrt(m sigma^2 / epsilon)
  timeStep = dpm.setTimeStep(timeStep * timeUnit);
  cout << "Time step: " << timeStep << " sigma: " << sigma << " Tinject: " << Tinject << endl;
  numVertices = dpm.getNumVertices();
  // initialize simulation
  dpm.initNVE(Tinject, readState);
  if(readAndMakeNewDir == true) {
    dpm.adjustTemperature(Tinject);
  }
  size = 2 * dpm.getMeanVertexRadius();
  cutDistance = dpm.setDisplacementCutoff(cutoff, size);
  dpm.calcNeighbors(cutDistance);
  dpm.calcForceEnergy();
  dpm.resetUpdateCount();
  waveQ = dpm.getDeformableWaveNumber();
  // record simulation time
  float elapsed_time_ms = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // run NVE integrator
  ioDPM.savePacking(outDir);
  ioDPM.saveNeighbors(outDir);
  while(step != maxStep) {
    dpm.NVELoop();
    if(step % saveEnergyFreq == 0) {
      ioDPM.saveDeformableEnergy(step, timeStep, numVertices);
      if(step % checkPointFreq == 0) {
        cout << "NVE: current step: " << step;
        cout << " E/N: " << dpm.getEnergy() / numVertices;
        cout << " T: " << dpm.getTemperature();
        cout << " ISF: " << dpm.getParticleISF(waveQ);
        cout << " phi: " << dpm.getPhi();
        updateCount = dpm.getUpdateCount();
        if(step != 0 && updateCount > 0) {
          cout << " number of updates: " << updateCount << " frequency " << checkPointFreq / updateCount << endl;
        } else {
          cout << " no updates" << endl;
        }
        dpm.resetUpdateCount();
        if(saveFinal == true) {
      	  ioDPM.savePacking(outDir);
          ioDPM.saveNeighbors(outDir);
        }
      }
      if(readAndMakeNewDir == true) {
        dpm.adjustTemperature(Tinject);
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
        //ioDPM.saveNeighbors(outDir);
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioDPM.saveState(currentDir);
        //ioDPM.saveNeighbors(outDir);
      }
    }
    step += 1;
  }
  // instrument code to measure end time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);
  printf("Elapsed time: %f ms, %f s.\n", elapsed_time_ms, elapsed_time_ms / 1000); // exec. time
  // save final configuration
  if(saveFinal == true) {
    ioDPM.savePacking(outDir);
    ioDPM.saveNeighbors(outDir);
  }
  ioDPM.closeEnergyFile();

  return 0;
}
