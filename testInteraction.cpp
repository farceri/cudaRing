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
#include <math.h>
#include <functional>
#include <utility>
#include <experimental/filesystem>

using namespace std;
using std::cout;

int main(int argc, char **argv) {
  bool read = true, readState = true, saveFinal = false, linSave = true;
  bool lj = false, wca = false, alltoall = false, cell = false, rigid = false;
  long step = 0, numParticles = 2, nDim = 2, numVertexPerParticle = 20, maxStep = atof(argv[3]), updateCount = 0;
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10);
  double LJcutoff = 4, cutoff = 0.5, cutDistance, timeStep = atof(argv[2]), sigma, timeUnit, size;
  double sigma0 = 1, sigma1 = 3, lx = 10, ly = 10, vel1 = -0.1, y0 = 0.5, y1 = 0.7, epot, ekin;
  double ea = 1e05, el = 20, eb = 0, ec = 1;
  std::string outDir, energyFile, inDir = argv[1], currentDir, dirSample;
  // initialize sp object
	DPM2D dp(numParticles, nDim, numVertexPerParticle);
  dp.printDeviceProperties();
  long numVertices = dp.getNumVertices();
  dp.setSimulationType(simControlStruct::simulationEnum::gpu);
  if(rigid == true) {
    dp.setParticleType(simControlStruct::particleEnum::rigid);
  }
  dp.setPotentialType(simControlStruct::potentialEnum::wca);
  dp.setInteractionType(simControlStruct::interactionEnum::vertexSmooth);
  dp.setConcavityType(simControlStruct::concavityEnum::on);
  dp.setEnergyCosts(ea, el, eb, ec);
  if(lj == true) {
    dp.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    dirSample = "lj/";
    cout << "Setting Lennard-Jones potential" << endl;
    dp.setLJcutoff(LJcutoff);
  } else if(wca == true) {
    dp.setPotentialType(simControlStruct::potentialEnum::wca);
    dirSample = "wca/";
    cout << "Setting WCA potential" << endl;
  } else {
    cout << "Setting Harmonic potential" << endl;
    dirSample = "smooth/";
  }
  if(alltoall == true) {
    dp.setNeighborType(simControlStruct::neighborEnum::allToAll);
  } else if(cell == true) {
    dp.setNeighborType(simControlStruct::neighborEnum::cell);
    cutoff = 5;
  }
  ioDPMFile ioDPM(&dp);
  // set input and output
  if (read == true) {//keep running the same dynamics
    cout << "Read packing" << endl;
    inDir = inDir + dirSample;
    outDir = inDir;
    ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
    if(readState == true) {
      ioDPM.readState(inDir, numParticles, numVertices, nDim);
    }
  } else {//start a new dyanmics
    cout << "Initialize new packing" << endl;
    dp.setTwoParticleTest(lx, ly, y0, y1, vel1);
    //dp.printTwoParticles();
    if(std::experimental::filesystem::exists(inDir + dirSample) == false) {
      std::experimental::filesystem::create_directory(inDir + dirSample);
    }
    outDir = inDir + dirSample;
  }
  std::experimental::filesystem::create_directory(outDir);
  // output file
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  // initialization
  timeUnit = sigma0;//epsilon and mass are 1 sqrt(m sigma^2 / epsilon)
  timeStep = dp.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma0 << endl;
  cout << "initial velocity on particle 1: " << vel1 << " time step: " << timeStep << endl;
  // initialize simulation
  if(dp.getNeighborType() == simControlStruct::neighborEnum::neighbor || dp.getNeighborType() == simControlStruct::neighborEnum::cell) {
    size = dp.getVertexRadius();
    cutDistance = dp.setDisplacementCutoff(cutoff, size);
    dp.calcNeighbors(cutDistance);
  }
  dp.calcForceEnergy();
  dp.resetUpdateCount();
  dp.resetLastPositions();
  // record simulation time
  float elapsed_time_ms = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // run integrator
  ioDPM.savePacking(outDir);
  ioDPM.saveNeighbors(outDir);
  while(step != maxStep) {
    dp.testInteraction(timeStep);
    if(step % saveEnergyFreq == 0) {
      ioDPM.saveEnergy(step, timeStep, numParticles, numVertices);
      if(step % checkPointFreq == 0) {
        cout << "NVE: current step: " << step;
        if(rigid == true) {
          epot = dp.getParticlePotentialEnergy();
          ekin = dp.getRigidKineticEnergy();
        } else {
          epot = dp.getPotentialEnergy() / numVertices;
          ekin = dp.getKineticEnergy() / numVertices;
        }
        cout << " U: " << epot;
        cout << " K: " << ekin;
        cout << " Energy: " << epot + ekin;
        if(dp.simControl.neighborType == simControlStruct::neighborEnum::neighbor) {
          updateCount = dp.getUpdateCount();
          if(step != 0 && updateCount > 0) {
            cout << " number of updates: " << updateCount << " frequency " << checkPointFreq / updateCount << endl;
          } else {
            cout << " no updates" << endl;
          }
          dp.resetUpdateCount();
        } else {
          cout << endl;
        }
        if(saveFinal == true) {
          ioDPM.savePacking(outDir);
          ioDPM.saveNeighbors(outDir);
        }
      }
    }
    //sp.calcParticleNeighborList(cutDistance);
    //sp.checkParticleNeighbors();
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioDPM.savePacking(currentDir);
        ioDPM.saveNeighbors(currentDir);
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
    ioDPM.saveNeighbors(outDir);
  }
  ioDPM.closeEnergyFile();

  return 0;
}
