//
// Author: Francesco Arceri
// Date:   10-03-2021
//
// Include C++ header files

// Example usage
// ./shearoscNVT /dirPath/0.920/ <timstep>=1e-04 <Tinject>=1e-02> <damping>=1e-02 <magn>=10
// <omega>=1 <maxStrain>=0.2 <numCycles>=60 <numParticles>=256 <potType=wca <tensionType>=0

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
  std::string inDir = argv[1], potType = argv[10], tensionType = argv[11];
  double timeStep = atof(argv[2]), Tinject = atof(argv[3]), damping = atof(argv[4]);
  double magn = atof(argv[5]), omega = atof(argv[6]), maxStrain = atof(argv[7]);
  long maxStep, numCycles = atol(argv[8]), numParticles = atol(argv[9]);
  // other variables
  long nDim = 2, numVertexPerParticle = 32, numVertices, step = 1, cycleStep, cycleIndex, cycleFreq, saveFreq;
  double cutDistance, cutoff = 0.5, timeUnit, gammaUnit, sigma, size, strain, prevStrain, strainStep;
  double ea = 1e05, el = 20, eb = 10, ec = 1, LJcut = 1.5, g0 = 10, eAdh = 0.1, Adhcut = 0.4;
  std::string outDir, energyFile, shearFile, currentDir, dirSample, whichDynamics = "oshear";
  // simulation mode
  cout << "Oscillatory shear simulation - strain amplitude: " << maxStrain << " omega: " << omega << endl;
  // initialize dpm object
  DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  dpm.setGeometryType(simControlStruct::geometryEnum::leesEdwards);
  dpm.setNoiseType(simControlStruct::noiseEnum::langevin);
  if(potType == "lj") {
    dpm.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    dpm.setLJcutoff(LJcut);
  } else if(potType == "wca") {
    dpm.setPotentialType(simControlStruct::potentialEnum::wca);
  } else if(potType == "adhesivewca") {
    dpm.setPotentialType(simControlStruct::potentialEnum::adhesivewca);
    whichDynamics = whichDynamics + "-adh";
  } else {
    cout << "Setting default harmonic potential" << endl;
  }
  if(tensionType == "tension") {
    g0 = magn;
    dpm.setTensionType(simControlStruct::tensionEnum::on);
    cout << "Tension is ON, g0: " << g0 << endl;
  } else if(tensionType == "adhesion") {
    eAdh = magn;
    dpm.setAdhesionType(simControlStruct::adhesionEnum::on);
    cout << "Adhesion is ON, eAdh: " << eAdh << endl;
  } else {
    whichDynamics = whichDynamics + "/";
  }
  if(smooth) {
    dpm.setInteractionType(simControlStruct::interactionEnum::vertexSmooth);
    dpm.setNeighborType(simControlStruct::neighborEnum::neighbor);
  }
  
  // set input and output
  dirSample = whichDynamics + argv[6] + "/";
  ioDPMFile ioDPM(&dpm);
  if(std::experimental::filesystem::exists(inDir + dirSample) == false) {
    std::experimental::filesystem::create_directory(inDir + dirSample);
  }
  outDir = inDir + dirSample;
  std::experimental::filesystem::create_directory(outDir);
  cout << "inDir: " << inDir << endl << "outDir: " << outDir << endl;
  ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
  dpm.setEnergyCosts(ea, el, eb, ec);
  numVertices = dpm.getNumVertices();
  if(readState == true) ioDPM.readState(inDir, numParticles, dpm.getNumVertices(), nDim);
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
  if(tensionType == "tension") {
    gammaUnit = ec / sigma;
    g0 *= gammaUnit;
    dpm.setTension(g0);
  } else if(tensionType == "adhesion") {
    dpm.setAdhesionConstants(eAdh, Adhcut); // ec = sigma = 1
  }
  cycleStep = 2 * PI / (omega * timeStep); // number of steps per cycle
  maxStep = cycleStep * numCycles;
  cout << "Oscillatory shear - number of cycles: " << numCycles << ", number of simulation steps: " << maxStep;
  cout << ", number of simulation steps per cycle: " << cycleStep << endl;
  saveFreq = max(1L, cycleStep / 50);
  cycleFreq = max(1L, numCycles / 10) * cycleStep;

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
  std::experimental::filesystem::create_directory(outDir + "/first/");
  std::experimental::filesystem::create_directory(outDir + "/last/");
  prevStrain = 0;
  while(step != maxStep) {
    // apply oscillatory strain
    strain = maxStrain * sin(omega * step * timeStep);
    strainStep = strain - prevStrain;
    prevStrain = strain;
    dpm.setLEstrain(strain);
    dpm.applyLEStrain(strainStep); // apply strain difference
    //dpm.calcNeighbors(cutDistance);
    //dpm.calcForceEnergy();
    // run dynamics
    dpm.langevinLoop();
    if(step % saveFreq == 0) {
      ioDPM.saveDeformableEnergy(step, timeStep, numVertices);
      ioDPM.saveStressStrain(strain, numVertices);
    }
    step += 1;
    // save strain
    cycleIndex = std::lround((double)step / (double)cycleStep);
    if (step % cycleFreq == 0) {
      cout << "Cycle: " << cycleIndex << " completion: " << cycleIndex * 100 / numCycles << "%" << ", current strain: " << strain << endl;
      cout << "NVT: current step: " << step;
      cout << " E/N: " << dpm.getEnergy() / numVertices;
      cout << " T: " << dpm.getTemperature();
      cout << " phi: " << dpm.getPhi() << endl << endl;
    }
    if(cycleIndex == 0 && (step % saveFreq) == 0) {
      currentDir = outDir + "/first/t" + std::to_string(step) + "/";
      std::experimental::filesystem::create_directory(currentDir);
      ioDPM.saveState(currentDir);
      ioDPM.saveContacts(currentDir);
      ioDPM.saveStrainInfo(currentDir, strain, cycleIndex);
    }
    if(cycleIndex == (numCycles - 1) && (step % saveFreq) == 0) {
      currentDir = outDir + "/last/t" + std::to_string(step) + "/";
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
