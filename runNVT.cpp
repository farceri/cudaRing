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
  bool readState = true, logSave = false, linSave = true, saveFinal = true, smooth = true;
  // input variables
  std::string inDir = argv[1], noiseType = argv[8], potType = argv[9], mode = argv[10];
  double timeStep = atof(argv[2]), Tinject = atof(argv[3]), damping = atof(argv[4]);
  long maxStep = atof(argv[5]), initialStep = atof(argv[6]), numParticles = atol(argv[7]);
  // other variables
  long nDim = 2, numVertexPerParticle = 32, numVertices, step = 0, multiple = 1, saveFreq = 1, updateCount = 0;
  long checkPointFreq = int(maxStep / 10), linFreq = int(checkPointFreq / 10), saveEnergyFreq = int(linFreq / 10);
  double cutDistance, cutoff = 0.5, timeUnit = 0, sigma, size, waveQ;
  double ea = 1e05, el = 20, eb = 10, ec = 1, LJcut = 1.5;
  std::string outDir, energyFile, currentDir, dirSample, whichDynamics = "langevin";
  // set simulation mode
  if(mode == "change") {
    readAndMakeNewDir = true;
    cout << "Change mode: make new directory and run dynamics with different parameters" << endl;
  } else if(mode == "over") {
    readAndSaveSameDir = true;
    cout << "Over mode: run dynamics with same parameters and save in same directory" << endl;
  } else if(mode == "run") {
    readAndSaveSameDir = true;
    runDynamics = true;
    cout << "Run mode: make DYNAMICS directory and run dynamics with same parameters" << endl;
  } else if(mode == "runlog") {
    readAndSaveSameDir = true;
    runDynamics = true;
    logSave = true;
    linSave = false;
    cout << "Run Log mode: make DYNAMICS-LOG directory, run dynamics with same parameters and save log-spaced trajectories" << endl;
  } else {
    cout << "Default mode: make new directory path and run initial dynamics" << endl;
    readState = false;
  }
  // initialize dpm object
  DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  if(noiseType == "brownian") {
    dpm.setNoiseType(simControlStruct::noiseEnum::brownian);
    whichDynamics = "brownian";
  } else if(noiseType == "driven") {
    dpm.setNoiseType(simControlStruct::noiseEnum::drivenBrownian);
    whichDynamics = "driven";
  } else {
    cout << "Setting default langevin noise" << endl;
  }
  if(potType == "lj") {
    dpm.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    whichDynamics = whichDynamics + "-lj/";
    dpm.setLJcutoff(LJcut);
  } else if(potType == "wca") {
    dpm.setPotentialType(simControlStruct::potentialEnum::wca);
    whichDynamics = whichDynamics + "-wca/";
  } else {
    cout << "Setting default harmonic potential" << endl;
  }
  if(smooth) {
    dpm.setInteractionType(simControlStruct::interactionEnum::vertexSmooth);
    dpm.setNeighborType(simControlStruct::neighborEnum::neighbor);
  }
  
  // set input and output
  dirSample = whichDynamics + "T" + argv[3] + "-damping" + argv[4] + "/";
  ioDPMFile ioDPM(&dpm);
  if (readAndSaveSameDir == true) {//keep running the same dynamics
    readState = true;
    inDir = inDir + dirSample;
    outDir = inDir;
    if(runDynamics == true) {
      outDir = outDir + "dynamics";
      if(logSave == true) outDir = outDir + "-log/";
      else outDir = outDir + "/";
      if(std::experimental::filesystem::exists(outDir) == true) {
        inDir = outDir;
      } else {
        std::experimental::filesystem::create_directory(outDir);
      }
    }
  } else {//start a new dyanmics
    if(readAndMakeNewDir == true) {
      readState = true;
      outDir = inDir + "../" + dirSample;
    } else {
      if(std::experimental::filesystem::exists(inDir + whichDynamics) == false) {
        std::experimental::filesystem::create_directory(inDir + whichDynamics);
      }
      outDir = inDir + dirSample;
    }
    std::experimental::filesystem::create_directory(outDir);
  }
  cout << "inDir: " << inDir << endl << "outDir: " << outDir << endl;
  ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
  dpm.setEnergyCosts(ea, el, eb, ec);
  if(readState == true) {
    ioDPM.readState(inDir, numParticles, dpm.getNumVertices(), nDim);
  }
  numVertices = dpm.getNumVertices();
  // output file
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);

  // initialize simulation
  sigma = dpm.getMeanParticleSize();
  timeUnit = sigma / sqrt(ec);
  timeStep = dpm.setTimeStep(timeStep * timeUnit);
  cout << "Units - time: " << timeUnit << " space: " << sigma << " time step: " << timeStep << endl;
  cout << "Thermostat - damping: " << damping << " Tinject: " << Tinject << endl;
  damping /= timeUnit;
  ioDPM.saveDriveParams(outDir, damping);

  // initialize integration scheme
  dpm.initLangevin(Tinject, damping, readState);
  size = 2 * dpm.getMeanVertexRadius();
  cutDistance = dpm.setDisplacementCutoff(cutoff, size);
  dpm.calcNeighbors(cutDistance);
  dpm.calcForceEnergy();;
  dpm.resetUpdateCount();
  waveQ = dpm.getDeformableWaveNumber();
  // record simulation time
  float elapsed_time_ms = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // run integrator
  ioDPM.savePacking(outDir);
  ioDPM.saveInitialNeighbors(outDir);
  while(step != maxStep) {
    dpm.langevinLoop();
    if(step % saveEnergyFreq == 0) {
      ioDPM.saveDeformableEnergy(step, timeStep, numVertices);
      if(step % checkPointFreq == 0) {
        cout << "NVT: current step: " << step;
        cout << " E/N: " << dpm.getEnergy() / numVertices;
        cout << " T: " << dpm.getTemperature();
        cout << " Ks/Kc: " << dpm.getShapeCOMEnergyRatio();
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
  // instrument code to measure end time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);
  printf("Elapsed time: %f ms, %f s.\n", elapsed_time_ms, elapsed_time_ms / 1000); // exec. time
  // save final configuration
  if(saveFinal == true) {
    ioDPM.savePacking(outDir);
  }
  ioDPM.closeEnergyFile();

  return 0;
}
