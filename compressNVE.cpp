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
  bool read = false, readState = false, smooth = true;
  // input variables
  long numParticles = atol(argv[6]), maxStep = atof(argv[7]);
  double timeStep = atof(argv[3]), Tinject = atof(argv[4]), shape0 = atof(argv[5]), lx = atof(argv[8]), ly = atof(argv[9]);
  std::string outDir = argv[1], inDir = argv[2], potType = argv[10], boxType = argv[11];
  // other variables
  long nDim = 2, numVertexPerParticle = 32, numVertices;
  long step, iteration = 0, maxIterations = 5e06, maxSearchStep = 1500, searchStep = 0, updateCount;
  long printFreq = int(maxStep / 10), saveFreq = int(printFreq / 10), minStep = 20, numStep = 0;
  double sigma, polydispersity = 0.1, previousPhi, currentPhi = 0.1, deltaPhi = 2e-03, phiTh = 0.9, boxRadius = 0;
  double cutDistance, cutoff = 0.5, LJcut = 1.5, forceTollerance = 1e-12, waveQ, FIREStep = 1e-02, timeUnit, prevEnergy = 0;
  double maxDelta, scaleFactor, size;
  double ea = 1e05, el = 20, eb = 10, ec = 1, ew = 10, thetaA = 1, thetaK = 0;
  thrust::host_vector<double> boxSize(nDim);
  std::string currentDir, energyFile, simType = "gpu", dynType = "nve";
  // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
  std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, FIREStep, 10*FIREStep, 0.2};
	// initialize dpm object
	DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  if(boxType == "square") {
    dpm.setGeometryType(simControlStruct::geometryEnum::fixedBox);
    dpm.setBoxEnergyCost(ew);
  } else if(boxType == "sides") {
    dpm.setGeometryType(simControlStruct::geometryEnum::fixedSides);
    dpm.setBoxEnergyCost(ew);
  } else if(boxType == "circle") {
    dpm.setGeometryType(simControlStruct::geometryEnum::roundBox);
    dpm.setBoxEnergyCost(ew);
  } else {
    dpm.setGeometryType(simControlStruct::geometryEnum::normal);
  }
  ioDPMFile ioDPM(&dpm);
  if(boxType == "circle") {
    outDir = outDir + "circle/" + argv[6] + "-A" + argv[5] + "/";
  } else {
    outDir = outDir + argv[6] + "-box" + argv[8] + argv[9] + "-A" + argv[5] + "/";
  }
  cout << outDir << endl;
  std::experimental::filesystem::create_directory(outDir);

  // initialization
  if (inDir != "0") {
    cout << "Reading initial configuration from " << inDir << endl;
    read = true;
    readState = true;
    inDir = outDir + inDir + "/";
    ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
    ioDPM.readState(inDir, numParticles, dpm.getNumVertices(), nDim);
  } else { // initialize polydisperse packing and minimize soft particle packing with harmonic potential
    dpm.setPotentialType(simControlStruct::potentialEnum::harmonic);
    dpm.setInteractionType(simControlStruct::interactionEnum::vertexVertex);
    dpm.setPolySizeDistribution(shape0, polydispersity);
    if(boxType == "circle") {
      dpm.setRoundScaledRandomParticles(currentPhi, 1.5, lx); // lx is box radius for round geometry
    } else {
      dpm.setScaledRandomParticles(currentPhi, 1.5, lx, ly); // 1.5 is extraRad
    }
    dpm.scaleParticlePacking();
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
      cout << " energy: " << dpm.getParticlePotentialEnergy() << endl;
      }
      iteration += 1;
    }
    cout << "\nFIRE: iteration: " << iteration;
    cout << " maxUnbalancedForce: " << setprecision(precision) << dpm.getParticleMaxUnbalancedForce();
    cout << " energy: " << setprecision(precision) << dpm.getParticlePotentialEnergy() << endl;
    currentDir = outDir + "fire/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.saveParticlePacking(currentDir);
    // put vertices on particle perimeters
    dpm.initVerticesOnParticles();
    dpm.scalePacking(dpm.getMeanParticleSize());
  }

  // simulation settings
  if(simType == "cpu") {
    dpm.setSimulationType(simControlStruct::simulationEnum::cpu);
  } else if(simType == "omp") {
    dpm.setSimulationType(simControlStruct::simulationEnum::omp);
  } else {
    dpm.setSimulationType(simControlStruct::simulationEnum::gpu);
  }
  if(potType == "wca") {
    dpm.setPotentialType(simControlStruct::potentialEnum::wca);
  } else if(potType == "lj") {
    dpm.setPotentialType(simControlStruct::potentialEnum::lennardJones);
    dpm.setLJcutoff(LJcut);
  } else {
    dpm.setPotentialType(simControlStruct::potentialEnum::harmonic);
  }
  if(smooth == true) {
    dpm.setInteractionType(simControlStruct::interactionEnum::vertexSmooth);
    dpm.setNeighborType(simControlStruct::neighborEnum::neighbor);
  }
  dpm.setEnergyCosts(ea, el, eb, ec);
  cout << "Energy scales: area " << ea << " segment " << el << " bending " << eb << " interaction " << ec << endl;
  numVertices = dpm.getNumVertices();

  // quasistatic isothermal compression
  dpm.calcParticleShape();
  currentPhi = dpm.getPhi();
  cout << "CURRENT PHI: " << currentPhi << " preferred phi: " << dpm.getPreferredPhi() << endl;
  previousPhi = currentPhi;
  sigma = dpm.getMeanParticleSize();
  size = 2 * dpm.getMeanVertexRadius();
  timeUnit = sigma / sqrt(ec); //epsilon and mass are 1 sqrt(m sigma^2 / epsilon)
  timeStep = dpm.setTimeStep(timeStep * timeUnit);
  cout << "Time step: " << timeStep << " sigma: " << sigma << " size: " << size << " Tinject: " << Tinject << endl;
  if (dynType == "scalevel") {
    dpm.initNVERescale(Tinject);
  } else {
    dpm.initNVE(Tinject, readState);
  }
  currentDir = outDir + "initial/";
  std::experimental::filesystem::create_directory(currentDir);
  ioDPM.savePacking(currentDir);
  while (searchStep < maxSearchStep) {
    currentDir = outDir + std::to_string(round(dpm.getPhi() * 1000) / 1000).substr(0,5) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    energyFile = currentDir + "energy.dat";
    ioDPM.openEnergyFile(energyFile);
    cutDistance = dpm.setDisplacementCutoff(cutoff, size);
    dpm.calcNeighbors(cutDistance);
    dpm.calcForceEnergy();
    dpm.resetUpdateCount();
    dpm.setParticleInitialPositions();
    waveQ = dpm.getDeformableWaveNumber();
    // equilibrate deformable particles
    step = 0;
    // remove energy injected by compression
    if(searchStep != 0 && dynType == "nve") {
      dpm.calcNeighbors(cutDistance);
      dpm.calcForceEnergy();
      cout << "Energy after compression - E/N: " << dpm.getEnergy() / numVertices << endl;
      dpm.adjustKineticEnergy(prevEnergy);
      dpm.calcForceEnergy();
      cout << "Energy after adjustment - E/N: " << dpm.getEnergy() / numVertices << endl;
    }
    while(step != maxStep) {
      if (dynType == "scalevel") {
        dpm.NVERescaleLoop();
      } else {
        dpm.NVELoop();
      }
      if(step % saveFreq == 0) {
        ioDPM.saveEnergy(step, timeStep, numParticles, numVertices);
      }
      if(step % printFreq == 0) {
        cout << "NVE: current step: " << step;
        cout << " E/N: " << dpm.getEnergy() / numVertices;
        cout << " T: " << dpm.getTemperature();
        cout << " ISF: " << dpm.getParticleISF(waveQ);
        cout << " phi: " << dpm.getPhi() << endl;
        ioDPM.savePacking(currentDir);
      }
      step += 1;
    }
    prevEnergy = dpm.getEnergy();
    cout << "Energy before compression - E/N: " << prevEnergy / numVertices << endl;
    // save minimized configuration
    ioDPM.savePacking(currentDir);
    //ioDPM.saveNeighbors(currentDir);
    ioDPM.closeEnergyFile();
    // check if target density is met
    if(currentPhi >= phiTh) {
      cout << "\nTarget density met, current phi: " << currentPhi << endl;
      searchStep = maxSearchStep; // exit condition
    } else {
      scaleFactor = sqrt((currentPhi + deltaPhi) / currentPhi);
      //dpm.scaleVertices(scaleFactor);
      //dpm.calcParticlesShape();
      //dpm.scalePacking(dpm.getMeanParticleSize());
      dpm.scaleBox(scaleFactor);
      currentPhi = dpm.getPhi();
      cout << "\nNEW PACKING FRACTION: " << currentPhi << " preferred: " << dpm.getPreferredPhi() << endl;
      if(boxType == "roundBox") {
        boxRadius = dpm.getBoxRadius();
        cout << "New boxSize: boxR: " << boxRadius << " scale: " << scaleFactor << endl;
      } else {
        boxSize = dpm.getBoxSize();
        cout << "New boxSize: Lx: " << boxSize[0] << " Ly: " << boxSize[1] << " scale: " << scaleFactor << endl;
      }
      searchStep += 1;
    }
  }
  return 0;
}
