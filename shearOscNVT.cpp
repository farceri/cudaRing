//
// Author: Francesco Arceri
// Date:   10-03-2021
//
// Include C++ header files

// Example usage
// ./runNVT /dirPath/0.920/ <timstep>=1e-04 <Tinject>=1e-02> <damping>=1e-02 <strainStep>=1e-03 <maxStrain>=0.2
// <maxStep>=1e07 <numParticles>=256 <noiseType>=langevin <potType=wca <tensionType>=off

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
  bool readState = true, saveFinal = true, smooth = true;
  // input variables
  std::string inDir = argv[1], noiseType = argv[9], potType = argv[10], tensionType = argv[11];
  double timeStep = atof(argv[2]), Tinject = atof(argv[3]), damping = atof(argv[4]);
  double omega = atof(argv[5]), maxStrain = atof(argv[6]);
  long maxStep, numCycles = atol(argv[7]), numParticles = atol(argv[8]);
  // other variables
  long nDim = 2, numVertexPerParticle = 32, numVertices, step = 0, checkPointFreq = int(maxStep / 10);
  double cutDistance, cutoff = 0.5, timeUnit = 0, sigma, size, strain, saveStrain = 0.002;
  double ea = 1e05, el = 20, eb = 10, ec = 1, LJcut = 1.5, g0 = 10;
  long strainIndex, saveStrainFreq = std::lround(saveStrain / strainStep);   // e.g. strainStep = 1e-4 -> 10
  std::string outDir, energyFile, shearFile, currentDir, dirSample, whichDynamics = "oshear";
  // simulation mode
  cout << "Oscillatory shear simulation - strain amplitude: " << maxStrain << " omega: " << omega << endl;
  // initialize dpm object
  DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  dpm.setGeometryType(simControlStruct::geometryEnum::leesEdwards);
  if(noiseType == "brownian") {
    dpm.setNoiseType(simControlStruct::noiseEnum::brownian);
  } else if(noiseType == "driven") {
    dpm.setNoiseType(simControlStruct::noiseEnum::drivenBrownian);
  } else {
    cout << "Setting default langevin noise" << endl;
  }
  if(potType == "lj") {
    dpm.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    dpm.setLJcutoff(LJcut);
  } else if(potType == "wca") {
    dpm.setPotentialType(simControlStruct::potentialEnum::wca);
  } else {
    cout << "Setting default harmonic potential" << endl;
  }
  if(tensionType == "tension") {
    dpm.setTensionType(simControlStruct::tensionEnum::on);
    cout << "Tension is ON" << endl;
  } else {
    cout << "Tension is OFF" << endl;
  }
  if(smooth) {
    dpm.setInteractionType(simControlStruct::interactionEnum::vertexSmooth);
    dpm.setNeighborType(simControlStruct::neighborEnum::neighbor);
  }
  
  // set input and output
  dirSample = whichDynamics + argv[5] + "/";
  ioDPMFile ioDPM(&dpm);
  if(std::experimental::filesystem::exists(inDir + dirSample) == false) {
    std::experimental::filesystem::create_directory(inDir + dirSample);
  }
  outDir = inDir + dirSample;
  std::experimental::filesystem::create_directory(outDir);
  cout << "inDir: " << inDir << endl << "outDir: " << outDir << endl;
  ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
  dpm.setEnergyCosts(ea, el, eb, ec);
  if(tensionType == "tension") dpm.setTension(g0);
  if(readState == true) ioDPM.readState(inDir, numParticles, dpm.getNumVertices(), nDim);
  numVertices = dpm.getNumVertices();
  // output files
  energyFile = outDir + "energy.dat";
  ioDPM.openEnergyFile(energyFile);
  shearFile = outDir + "shear.dat";
  ioDPM.openShearFile(shearFile);

  // initialize simulation
  sigma = dpm.getMeanParticleSize();
  timeUnit = sigma / sqrt(ec);
  cout << "Units - time: " << timeUnit << " space: " << sigma << " time step: " << timeStep << endl;
  cout << "Thermostat - damping: " << damping << " Tinject: " << Tinject << endl;
  timeStep *= timeUnit;
  timeStep = dpm.setTimeStep(timeStep);
  damping /= timeUnit;
  ioDPM.saveDriveParams(outDir, damping);
  maxStep = 2 * np.pi * numCycles / (omega * timeStep);
  cout << "Oscillatory shear - number of cycles: " << numCycles << " number of simulation steps: " << maxStep << endl;

  // initialize integration scheme
  dpm.initLangevin(Tinject, damping, readState);
  size = 2 * dpm.getMeanVertexRadius();
  cutDistance = dpm.setDisplacementCutoff(cutoff, size);
  dpm.calcNeighbors(cutDistance);
  dpm.calcForceEnergy();;
  dpm.resetUpdateCount();
  // record simulation time
  float elapsed_time_ms = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  ioDPM.savePacking(outDir);
  ioDPM.saveInitialNeighbors(outDir);
  prevStrain = 0;
  while(step != maxStep) {
    // apply oscillatory strain
    strain = maxStrain * sin(omega * step * timeStep);
    dpm.setLEstrain(strain);
    dpm.applyLEStrain(strain - prevStrain); // apply strain difference
    prevStrain = strain;
    //dpm.calcNeighbors(cutDistance);
    //dpm.calcForceEnergy();
    // run dynamics
    dpm.langevinLoop();
    if(step % checkPointFreq == 0) {
      ioDPM.saveDeformableEnergy(step, timeStep, numVertices);
      ioDPM.saveStressStrain(strain, numVertices);
    }
    step += 1;
    // save strain
    cycleIndex = std::lround(step / cycleStep);
    if (cycleIndex % cycleStep == 0) {
      cout << "Cycle: " << cycleIndex << " completion: " << cycleIndex * 100 / numCycles << "%" << endl;
      cout << "NVT: current step: " << step;
      cout << " E/N: " << dpm.getEnergy() / numVertices;
      cout << " T: " << dpm.getTemperature();
      cout << " phi: " << dpm.getPhi() << endl << endl;
    }
    if(linSave == true && (step % linFreq) == 0) {
      currentDir = outDir + "/t" + std::to_string(step) + "/";
      std::experimental::filesystem::create_directory(currentDir);
      ioDPM.saveState(currentDir);
      ioDPM.saveContacts(currentDir);
      ioDPM.saveStrainInfo(currentDir, strain, cycleIndex);
    }
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
