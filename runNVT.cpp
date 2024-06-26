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
  bool readState = true, logSave, linSave = false, saveFinal = true;
  long numParticles = atof(argv[7]), nDim = 2, numVertexPerParticle = 32, numVertices;
  long maxStep = atof(argv[4]), checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10);
  long step = 0, initialStep = 0, multiple = 1, saveFreq = 1, firstDecade = 0, updateCount = 0;
  double cutDistance = 1, waveQ, sigma, damping, inertiaOverDamping = atof(argv[6]);
  double timeUnit, timeStep = atof(argv[2]), Tinject = atof(argv[3]);
  double ea = 1e05, el = 1, eb = 1, ec = 1, cutoff, maxDelta;
  std::string outDir, energyFile, currentDir, inDir = argv[1], dirSample, whichDynamics = "langevin/";
  dirSample = whichDynamics + "T" + argv[3] + "/";
  // initialize dpm object
  DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  dpm.setPotentialType(simControlStruct::potentialEnum::wca);
  dpm.setInteractionType(simControlStruct::interactionEnum::smooth);
  dpm.setConcavityType(simControlStruct::concavityEnum::off);
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
  numVertices = dpm.getNumVertices();
  // output file
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  // initialization
  sigma = dpm.getMeanParticleSize();
  damping = sqrt(inertiaOverDamping) / sigma;
  timeUnit = sigma / sqrt(ec);
  timeStep = dpm.setTimeStep(timeStep * timeUnit);
  cout << "Energy scales: area " << ea << " segment " << el << " bending " << eb << " interaction " << ec << endl;
  cout << "Units - time: " << timeUnit << " space: " << sigma << " time step: " << timeStep << endl;
  cout << "Thermostat - damping: " << damping << " Tinject: " << Tinject << endl;
  cout << "Current phi: " << dpm.getPhi() << endl;
  // initialize simulation
  dpm.calcNeighborList(cutDistance);
  dpm.calcForceEnergy();
  dpm.initLangevin2(Tinject, damping, readState);
  cutoff = (1 + cutDistance) * 2*dpm.getMeanVertexRadius();
  dpm.setDisplacementCutoff(cutoff, cutDistance);
  dpm.resetUpdateCount();
  dpm.setParticleInitialPositions();
  waveQ = dpm.getDeformableWaveNumber();
  // run integrator
  while(step != maxStep) {
    dpm.langevin2Loop();
    if(step % linFreq == 0) {
      ioDPM.saveDeformableEnergy(step, timeStep, numVertices);
      if(step % checkPointFreq == 0) {
        cout << "Langevin: current step: " << step;
        cout << " E/N: " << (dpm.getPotentialEnergy() + dpm.getKineticEnergy()) / numParticles;
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
