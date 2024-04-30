//
// Author: Francesco Arceri
// Date:   10-03-2021
//
// Test script

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
  bool readState = true, readAndSaveSameDir = true, saveFinal = true;
  bool testStress = false, testNVE = true, testNVT = false, testActive = false;
  long numParticles = atol(argv[6]), nDim = 2, numVertexPerParticle = 32; // this is a default
  long numVertices = numParticles * numVertexPerParticle, updateCount = 0;
  long step = 0, maxStep = atof(argv[5]), checkPointFreq = int(maxStep/10), saveEnergyFreq = int(maxStep/100);
  double cutDistance = 1, Tinject = atof(argv[3]), Dr = 1, driving = 1e-02;
  double forceUnit, timeUnit, timeStep = atof(argv[2]), sigma, damping, iod = 100, cutoff, maxDelta;
  double ea = 1e02, el = 1e01, eb = atof(argv[4]), ec = 1, scaleFactor = 1.0001;
  std::string inDir = argv[1], outDir, currentDir, energyFile;
  // initialize dpm object
  DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  dpm.setEnergyCosts(ea, el, eb, ec);
  ioDPMFile ioDPM(&dpm);
  ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
  dpm.setPotentialType(simControlStruct::potentialEnum::wca);
  dpm.setInteractionType(simControlStruct::interactionEnum::smooth);
  if(readState == true) {
    ioDPM.readState(inDir, numParticles, dpm.getNumVertices(), nDim);
  }
  // set length and time scales
  sigma = dpm.getMeanParticleSize();
  dpm.setEnergyCosts(ea, el, eb, ec);
  // instrument code to measure start time
  float elapsed_time_ms = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // initialize
  dpm.calcNeighborList(cutDistance);
  dpm.calcForceEnergy();
  if(testStress == true) {
    cout << "Test stress" << endl;
    if(readAndSaveSameDir == false) {
      outDir = inDir + "/testStress/";
    } else {
      outDir = inDir;
    }
    std::experimental::filesystem::create_directory(outDir);
    dpm.calcForceEnergy();
    dpm.calcNeighborList(cutDistance);
    dpm.calcParticlesPositions();
    currentDir = outDir + "/step" + std::to_string(step) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.saveConfiguration(currentDir);
    for (step = 0; step < maxStep; step++) {
      dpm.scaleVertices(scaleFactor);
      currentDir = outDir + "/step" + std::to_string(step) + "/";
      std::experimental::filesystem::create_directory(currentDir);
      ioDPM.saveConfiguration(currentDir);
    }
  }
  else if(testNVE == true) {
    cout << "Test NVE" << endl;
    if(readAndSaveSameDir == false) {
      outDir = inDir + "/testNVE/";
    } else {
      outDir = inDir;
    }
    std::experimental::filesystem::create_directory(outDir);
    timeUnit = sigma/sqrt(ec);
    dpm.initNVE(Tinject, readState);
  }
  else if(testNVT == true) {
    cout << "Test NVT" << endl;
    if(readAndSaveSameDir == false) {
      outDir = inDir + "/testNVT/";
    } else {
      outDir = inDir;
    }
    std::experimental::filesystem::create_directory(outDir);
    damping = sqrt(iod * ec) / sigma;
    cout << "Tinject: " << Tinject << endl;
    timeUnit = 1 / damping;
    dpm.initLangevin2(Tinject, damping, readState);
  }
  else if(testActive == true) {
    cout << "Test Active" << endl;
    if(readAndSaveSameDir == false) {
      outDir = inDir + "/testActive/";
    } else {
      outDir = inDir;
    }
    std::experimental::filesystem::create_directory(outDir);
    damping = sqrt(iod * ec) / sigma;
    cout << "damping: " << damping << " with inertia over damping: " << iod << endl;
    timeUnit = 1 / damping;
    Dr = Dr/timeUnit;
    forceUnit = iod / sigma;
    driving = driving*forceUnit;
    cout << "Tinject: " << Tinject << " Dr: " << Dr << " f0: " << driving << endl;
    dpm.initActiveLangevin(Tinject, Dr, driving, damping, readState);
  }
  timeStep = dpm.setTimeStep(timeUnit * timeStep);
  cout << "Time step: " << timeStep << endl;
  dpm.calcNeighborList(cutDistance);
  dpm.calcForceEnergy();
  // output file
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  dpm.resetLastPositions();
  cutoff = (1 + cutDistance) * dpm.getMeanVertexRadius();
  dpm.setDisplacementCutoff(cutoff, cutDistance);
  if(testStress == false) {
    while(step != maxStep) {
      if(testNVE == true) {
        dpm.NVELoop();
      }
      else if(testNVT == true) {
        dpm.langevin2Loop();
      }
      else if(testActive == true) {
        dpm.activeLangevinLoop();
      }
      if(step % saveEnergyFreq == 0) {
        ioDPM.saveEnergy(step, timeStep, numParticles, numVertices);
        if(step % checkPointFreq == 0) {
          cout << "Test: current step: " << step;
          cout << " E: " << (dpm.getPotentialEnergy() + dpm.getKineticEnergy()) / numParticles;
          cout << " T: " << dpm.getTemperature() << endl;
          if(saveFinal == true) {
            ioDPM.saveConfiguration(outDir);
          }
        }
      }
      step += 1;
    }
  }
  // instrument code to measure end time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);
  printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms); // exec. time
  if(saveFinal == true) {
    ioDPM.saveConfiguration(outDir);
  }
  ioDPM.closeEnergyFile();

  return 0;
}
