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
#include <thrust/host_vector.h>
#include <experimental/filesystem>

using namespace std;

int main(int argc, char **argv) {
  // variables
  bool readAndMakeNewDir = false, readAndSaveSameDir = false, runDynamics = false;
  // readAndMakeNewDir reads the input dir and makes/saves a new output dir (cool or heat packing)
  // readAndSaveSameDir reads the input dir and saves in the same input dir (thermalize packing)
  // runDynamics works with readAndSaveSameDir and saves all the dynamics (run and save dynamics)
  bool readState = false, logSave, linSave = true, saveFinal = true;
  long numParticles = atof(argv[7]), nDim = 2, numVertexPerParticle = 32;
  long maxStep = atof(argv[4]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10);
  long step = 0, initialStep = 0, multiple = 1, saveFreq = 1, firstDecade = 0;
  double cutDistance = 1, waveQ, sigma, damping, inertiaOverDamping = atof(argv[6]);
  double ec = 1, timeUnit, timeStep = atof(argv[2]), Tinject = atof(argv[3]), cutoff;
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "langevin/";
  dirSample = whichDynamics + "T" + argv[3] + "/";
  // initialize dpm object
  DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  dpm.setPotentialType(simControlStruct::potentialEnum::wca);
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
  ioDPM.readRigidPackingFromDirectory(inDir, numParticles, nDim);
  dpm.setEnergyCosts(0, 0, 0, ec);
  if(readState == true) {
    ioDPM.readRigidState(inDir, numParticles, nDim);
  }
  // output file
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  // initialization
  sigma = dpm.getMeanParticleSize();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = sigma / sqrt(ec);
  timeStep = dpm.setTimeStep(timeStep * timeUnit);
  cout << "Energy scales: interaction " << ec << endl;
  cout << "Units - time: " << timeUnit << " space: " << sigma << " time step: " << timeStep << endl;
  cout << "Thermostat - damping: " << damping << " Tinject: " << Tinject << endl;
  cout << "Current phi: " << dpm.getPreferredPhi() << endl;
  // initialize simulation
  dpm.initRigidLangevin(Tinject, damping, readState);
  dpm.calcNeighborList(cutDistance);
  dpm.calcRigidForceEnergy();
  dpm.resetLastPositions();
  cutoff = (1 + cutDistance) * dpm.getMeanVertexRadius();
  dpm.setDisplacementCutoff(cutoff, cutDistance);
  waveQ = dpm.getRigidWaveNumber();
  // run integrator
  while(step != maxStep) {
    dpm.rigidLangevinLoop();
    if(step % linFreq == 0) {
      ioDPM.saveRigidEnergy(step, timeStep, numParticles);
      if(step % checkPointFreq == 0) {
        cout << "Rigid Langevin: current step: " << step;
        cout << " E/N: " << dpm.getParticleEnergy() / numParticles;
        cout << " T: " << dpm.getParticleTemperature();
        cout << " ISF: " << dpm.getParticleISF(waveQ) << endl;
        if(saveFinal == true) {
          ioDPM.saveRigidPacking(outDir);
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
        ioDPM.saveRigidState(currentDir);
      }
    }
    if(linSave == true) {
      if((step % linFreq) == 0) {
        currentDir = outDir + "/t" + std::to_string(initialStep + step) + "/";
        std::experimental::filesystem::create_directory(currentDir);
        ioDPM.saveRigidState(currentDir);
      }
    }
    step += 1;
  }
  // save final configuration
  if(saveFinal == true) {
    ioDPM.saveRigidPacking(outDir);
  }
  ioDPM.closeEnergyFile();

  return 0;
}
