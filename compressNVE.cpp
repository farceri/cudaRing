//
// Author: Francesco Arceri
// Date:   10-25-2021
//
// Include C++ header files

#include "include/DPM2D.h"
#include "include/FileIO.h"
#include "include/Simulator.h"
#include "include/FIRE.h"
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
  bool read = false, readState = false, cpu = false, wca = true, smooth = false, concavity = false;
  long numParticles = atol(argv[4]), nDim = 2, numVertexPerParticle = 32, numVertices;
  long step, iteration = 0, maxIterations = 5e06, maxSearchStep = 1500, searchStep = 0, updateCount;
  long maxStep = atof(argv[6]), printFreq = int(maxStep / 10), saveFreq = int(printFreq / 10), minStep = 20, numStep = 0;
  double sigma, polydispersity = 0.2, previousPhi, currentPhi = 0.2, deltaPhi = 5e-03, phiTh = 0.9;
  double cutDistance, cutoff = 2, forceTollerance = 1e-12, waveQ, FIREStep = 1e-02, timeUnit, prevEnergy = 0;
  double Tinject = atof(argv[3]), maxDelta, scaleFactor, timeStep = atof(argv[2]), size;
  double ea = 1e05, el = 20, eb = 10, ec = 1, calA0 = atof(argv[5]), thetaA = 1, thetaK = 0;
  thrust::host_vector<double> boxSize(nDim);
  std::string outDir = argv[1], currentDir, inDir, energyFile;
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, FIREStep, 10*FIREStep, 0.2};
	// initialize dpm object
	DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  ioDPMFile ioDPM(&dpm);
  std::experimental::filesystem::create_directory(outDir);
  // read initial configuration
  if(read == true) {
    inDir = outDir + argv[7];
    ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
    if(readState == true) {
      ioDPM.readState(inDir, numParticles, dpm.getNumVertices(), nDim);
    }
  } else {
    dpm.setPotentialType(simControlStruct::potentialEnum::harmonic);
    dpm.setInteractionType(simControlStruct::interactionEnum::vertexVertex);
    // initialize polydisperse packing and minimize soft particle packing with harmonic potential
    dpm.setPolySizeDistribution(calA0, polydispersity);
    dpm.setScaledRandomParticles(currentPhi, 1.2); //extraRad
    dpm.scaleParticlePacking();
    currentDir = outDir + "init/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.saveParticlePacking(currentDir);
    dpm.setEnergyCosts(0, 0, 0, ec);
    // minimize soft sphere packing
    dpm.initFIRE(particleFIREparams, minStep, numStep, numParticles);
    dpm.setParticleMassFIRE();
    size = 2 * dpm.getMinParticleSigma();
    cutDistance = dpm.setDisplacementCutoff(cutoff, size);
    dpm.calcParticleNeighborList(cutDistance);
    dpm.calcParticleForceEnergy();
    while((dpm.getParticleMaxUnbalancedForce() > forceTollerance) && (iteration != maxIterations)) {
      dpm.particleFIRELoop();
      if(iteration % printFreq == 0 && iteration != 0) {
      cout << "FIRE: iteration: " << iteration;
      cout << " maxUnbalancedForce: " << setprecision(precision) << dpm.getParticleMaxUnbalancedForce();
      cout << " energy: " << dpm.getParticleEnergy() << endl;
      }
      iteration += 1;
    }
    cout << "\nFIRE: iteration: " << iteration;
    cout << " maxUnbalancedForce: " << setprecision(precision) << dpm.getParticleMaxUnbalancedForce();
    cout << " energy: " << setprecision(precision) << dpm.getParticleEnergy() << endl;
    currentDir = outDir + "sp/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.saveParticlePacking(currentDir);
    // put vertices on particle perimeters
    dpm.initVerticesOnParticles();
    dpm.scalePacking(dpm.getMeanParticleSize());
    currentDir = outDir + "dpm/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.savePacking(currentDir);
  }
  if(cpu == true) {
    dpm.setSimulationType(simControlStruct::simulationEnum::cpu);
  }
  if(wca == true) {
    dpm.setPotentialType(simControlStruct::potentialEnum::wca);
  }
  if(smooth == true) {
    dpm.setInteractionType(simControlStruct::interactionEnum::vertexSmooth);
    dpm.setNeighborType(simControlStruct::neighborEnum::neighbor);
  }
  if(concavity == true) {
    dpm.setConcavityType(simControlStruct::concavityEnum::on);
  }
  dpm.setEnergyCosts(ea, el, eb, ec);
  cout << "Energy scales: area " << ea << " segment " << el << " bending " << eb << " interaction " << ec << endl;
  numVertices = dpm.getNumVertices();
  dpm.calcParticlesShape();
  currentPhi = dpm.getPhi();
  cout << "Start compression: current packing fraction: " << dpm.getPhi() << " preferred: " << dpm.getPreferredPhi() << endl;
  previousPhi = currentPhi;
  // isotropic isothermal compression
  if(searchStep != 0) {
      dpm.calcNeighbors(cutDistance);
      dpm.calcForceEnergy();
      cout << "Energy after compression - E/N: " << sp.getEnergy() / numVertices << endl;
      dpm.adjustKineticEnergy(prevEnergy);
      dpm.calcForceEnergy();
      cout << "Energy after adjustment - E/N: " << sp.getEnergy() / numVertices << endl;
    }
  while (searchStep < maxSearchStep) {
    currentDir = outDir + std::to_string(dpm.getPhi()).substr(0,6) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    energyFile = currentDir + "energy.dat";
    ioDPM.openEnergyFile(energyFile);
    sigma = dpm.getMeanParticleSize();
    timeUnit = sigma / sqrt(ec);//epsilon and mass are 1 sqrt(m sigma^2 / epsilon)
    timeStep = dpm.setTimeStep(timeStep * timeUnit);
    cout << "Time step: " << timeStep << " sigma: " << sigma << " Tinject: " << Tinject << endl;
    dpm.initNVE(Tinject, readState);
    size = 2 * dpm.getMeanVertexRadius();
    cutDistance = dpm.setDisplacementCutoff(cutoff, size);
    dpm.calcNeighbors(cutDistance);
    dpm.calcForceEnergy();
    dpm.resetLastPositions();
    step = 0;
    waveQ = dpm.getDeformableWaveNumber();
    // equilibrate deformable particles
    while(step != maxStep) {
      dpm.NVELoop();
      if(step % saveFreq == 0) {
        ioDPM.saveEnergy(step, timeStep, numParticles);
      }
      if(step % printFreq == 0) {
        cout << "NVE: current step: " << step;
        cout << " E/N: " << dpm.getEnergy() / numParticles;
        cout << " T: " << dpm.getTemperature();
        cout << " ISF: " << dpm.getParticleISF(waveQ);
        cout << " phi: " << dpm.getPhi() << endl;
      }
      step += 1;
    }
    prevEnergy = sp.getEnergy();
    cout << "Energy before compression - E/N: " << prevEnergy / numVertices << endl;
    // save minimized configuration
    ioDPM.savePacking(currentDir);
    ioDPM.saveNeighbors(currentDir);
    ioDPM.closeEnergyFile();
    // check if target density is met
    if(currentPhi >= phiTh) {
      cout << "Target density met, current phi: " << currentPhi << endl;
      searchStep = maxSearchStep; // exit condition
    } else {
      scaleFactor = sqrt((currentPhi + deltaPhi) / currentPhi);
      //dpm.scaleVertices(scaleFactor);
      //dpm.calcParticlesShape();
      //dpm.scalePacking(dpm.getMeanParticleSize());
      dpm.scaleBoxSize(scaleFactor);
      boxSize = dpm.getBoxSize();
      currentPhi = dpm.getPhi();
      cout << "\nNew packing fraction: " << currentPhi << " preferred: " << dpm.getPreferredPhi() << endl;
      cout << "New boxSize: Lx: " << boxSize[0] << " Ly: " << boxSize[1] << " scale: " << scaleFactor << endl;
      searchStep += 1;
    }
  }

  return 0;
}
