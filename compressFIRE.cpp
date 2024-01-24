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
  bool read = false, readCellFormat = false, readState = false; // this is for thermalization
  long numParticles = 256, nDim = 2, numVertexPerParticle = 32, numVertices;
  long iteration = 0, maxIterations = 5e06, printFreq = 1e05;
  long minStep = 20, numStep = 0, step = 0, maxStep = 1e04, updateCount;
  long maxSearchStep = 500, searchStep = 0, saveEnergyFreq = 1e03;
  double polydispersity = 0.2, previousPhi, currentPhi = 0.2, phiTh = 0.2, deltaPhi = 2e-03, scaleFactor;
  double cutDistance = 1, forceTollerance0 = 1e-10, forceTollerance, energyTollerance = 1e-20, energyCheck;
  double ea = 1e03, el = 1, eb = 0, ec = 1, timeStep, timeUnit, forceCheck, sigma, damping, iod = 10, dt = 1e-02;
  double calA0 = 1.1, thetaA = 1, thetaK = 0, Tinject = 1e-02, cutoff, maxDelta, FIREStep;
  bool jamCheck = 0, overJamCheck = 0, underJamCheck = 0;
  std::string outDir = argv[1], currentDir, inDir, inFile;
	// initialize dpm object
	DPM2D dpm(numParticles, nDim, numVertexPerParticle);
  dpm.setEnergyCosts(ea, el, eb, ec);
  ioDPMFile ioDPM(&dpm);
  std::experimental::filesystem::create_directory(outDir);
  // read initial configuration
  if(read == true) {
    if(readCellFormat == true) {
      inFile = "/home/francesco/Documents/Data/isoCompression/poly32-compression.test";
      ioDPM.readPackingFromCellFormat(inFile, 1);
    } else {
      inDir = argv[2];
      inDir = outDir + inDir;
      ioDPM.readPackingFromDirectory(inDir, numParticles, nDim);
    }
  } else {
    // initialize polydisperse packing and minimize soft particle packing
    dpm.setPolySizeDistribution(calA0, polydispersity);
    //dpm.setSinusoidalRestAngles(thetaA, thetaK);
    dpm.setRandomParticles(currentPhi, 1.1); //extraRad
    // fire paramaters: a_start, f_dec, f_inc, f_a, dt, dt_max, a
    sigma = dpm.getMeanParticleSize();
    timeUnit = sigma;
    FIREStep = dpm.setTimeStep(dt*10 * timeUnit);
    cout << "FIRE: timeStep: " << FIREStep << endl;
    std::vector<double> particleFIREparams = {0.2, 0.5, 1.1, 0.99, FIREStep, 10*FIREStep, 0.2};
    dpm.initFIRE(particleFIREparams, minStep, numStep, numParticles);
    dpm.setParticleMassFIRE();
    dpm.calcParticleNeighborList(cutDistance);
    dpm.calcParticleForceEnergy();
    cutoff = (1 + cutDistance) * dpm.getMinParticleSigma();
    dpm.resetParticleLastPositions();
    forceTollerance = forceTollerance0 / dpm.getMeanParticleSigma();
    while((dpm.getParticleMaxUnbalancedForce() > forceTollerance) && (iteration != maxIterations)) {
      dpm.particleFIRELoop();
      maxDelta = dpm.getParticleMaxDisplacement();
      if(3*maxDelta > cutoff) {
        dpm.calcParticleNeighborList(cutDistance);
        dpm.resetParticleLastPositions();
      }
      iteration += 1;
    }
    cout << "FIRE: iteration: " << iteration;
    cout << " maxUnbalancedForce: " << setprecision(precision) << dpm.getParticleMaxUnbalancedForce();
    cout << " energy: " << setprecision(precision) << dpm.getParticleEnergy() << endl;
    currentDir = outDir + "/sp/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.saveParticlePacking(currentDir);
    // put vertices on particle perimeters
    dpm.initVerticesOnParticles();
    currentDir = outDir + "/dp/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.savePacking(currentDir);
  }
  dpm.calcParticlesShape();
  cout << "current packing fraction: " << setprecision(precision) << dpm.getPhi() << endl;
  numVertices = dpm.getNumVertices();
  thrust::host_vector<double> positions(numVertices * nDim);
  thrust::host_vector<double> radii(numVertices);
  thrust::host_vector<double> lengths(numVertices);
  thrust::host_vector<double> areas(numParticles);
  // quasistatic isotropic compression
  currentPhi = dpm.getPhi();
  previousPhi = currentPhi;
  while (searchStep < maxSearchStep) {
    sigma = dpm.getMeanParticleSize();
    timeUnit = sigma;
    FIREStep = dpm.setTimeStep(dt * timeUnit);
    cout << "\nFIRE: timeStep: " << FIREStep;
    std::vector<double> vertexFIREparams = {0.2, 0.5, 1.1, 0.99, FIREStep, 10*FIREStep, 0.2};
    dpm.initFIRE(vertexFIREparams, minStep, numStep, numVertices);
    dpm.calcNeighborList(cutDistance);
    dpm.calcForceEnergy();
    dpm.resetLastPositions();
    cutoff = (1 + cutDistance) * dpm.getMeanVertexRadius();
    dpm.setDisplacementCutoff(cutoff, cutDistance);
    iteration = 0;
    forceCheck = dpm.getMaxUnbalancedForce();
    forceTollerance = forceTollerance0 / dpm.getMeanParticleSize();
    cout << " force tollerance: " << forceTollerance << endl;
    cutoff = (1 + cutDistance) * dpm.getMinVertexRadius();
    updateCount = 0;
    energyCheck = dpm.getPotentialEnergy();
    //while((energyCheck > energyTollerance) && (iteration != maxIterations)) {
    while((forceCheck > forceTollerance) && (iteration != maxIterations)) {
      dpm.vertexFIRELoop();
      //energyCheck = dpm.getPotentialEnergy();
      forceCheck = dpm.getMaxUnbalancedForce();
      if(iteration % printFreq == 0 && iteration != 0) {
        cout << "FIRE: iteration: " << iteration;
        cout << " maxUnbalancedForce: " << setprecision(precision) << forceCheck;
        cout << " energy: " << dpm.getPotentialEnergy() << endl;
      }
      iteration += 1;
    }
    cout << "FIRE: iteration: " << iteration;
    cout << " maxUnbalancedForce: " << setprecision(precision) << forceCheck;
    cout << " energy: " << dpm.getPotentialEnergy();
    cout << " updates: " << updateCount << endl;
    // save minimized configuration
    std::string currentDir = outDir + std::to_string(dpm.getPhi()) + "/";
    std::experimental::filesystem::create_directory(currentDir);
    ioDPM.savePacking(currentDir);
    // check configuration after energy minimization
    jamCheck = (forceCheck < 2.0 * forceTollerance && forceCheck > forceTollerance);
    overJamCheck = (forceCheck > 2.0 * forceTollerance);
    underJamCheck = (forceCheck < forceTollerance);
    //jamCheck = (energyCheck < 2.0 * energyTollerance && energyCheck > energyTollerance);
    //overJamCheck = (energyCheck > 2.0 * energyTollerance);
    //underJamCheck = (energyCheck < energyTollerance);
    if(jamCheck) {
      cout << "Compression step: " << searchStep;
      cout << " Found jammed configuration, packing fraction: " << currentPhi << endl;
    } else {
      // compute scaleFactor with binary search
      if(underJamCheck) {
        scaleFactor = sqrt((currentPhi + deltaPhi) / currentPhi);
        cout << "Compression step: " << searchStep;
        cout << " Found underjammed configuration, scaleFactor: " << scaleFactor << endl;
        // save most recent underjammed configuration
        previousPhi = currentPhi;
        positions = dpm.getVertexPositions();
        radii = dpm.getVertexRadii();
        lengths = dpm.getRestLengths();
        areas = dpm.getRestAreas();
      } else if(overJamCheck) {
        deltaPhi *= 0.5;
        scaleFactor = sqrt((previousPhi + deltaPhi) / previousPhi);
        cout << "Compression step: " << searchStep;
        cout << " Found overjammed configuration, scaleFactor: " << scaleFactor << endl;
        // copy back most recent underjammed configuration and compress half much
        dpm.setVertexPositions(positions);
        dpm.setVertexRadii(radii);
        dpm.setRestLengths(lengths);
        dpm.setRestAreas(areas);
      }
      if(currentPhi > phiTh) {
        sigma = dpm.getMeanParticleSize();
        damping = sqrt(iod) / sigma;
        timeUnit = 1 / damping;
        timeStep = dpm.setTimeStep(dt * timeUnit);
        // run NVE integrator
        cout << "Thermalization, timeStep: " << timeStep << endl;
        // thermalize packing after each energy minimization
        dpm.initLangevin(Tinject, damping, true);
        updateCount = 0;
        while(step != maxStep) {
          dpm.langevinLoop();
          maxDelta = dpm.getMaxDisplacement();
          if(3*maxDelta > cutoff) {
            dpm.calcNeighborList(cutDistance);
            dpm.resetLastPositions();
            updateCount += 1;
          }
          step += 1;
        }
        step = 0;
        cout << "NVT: current step: " << step;
        cout << " E/N: " << (dpm.getPotentialEnergy() + dpm.getKineticEnergy()) / numVertices;
        cout << " T: " << dpm.getTemperature();
        cout << " phi: " << dpm.getPhi();
        cout << " updates: " << updateCount << endl;
      }
      dpm.scaleVertices(scaleFactor);
      currentPhi = dpm.getPhi();
      cout << "new packing fraction: " << currentPhi << endl;
      searchStep += 1;
    }
  }

  return 0;
}
