//
// Author: Yuxuan Cheng
// Date:   10-09-2021
//

#ifndef FILEIO_H
#define FILEIO_H

#include "DPM2D.h"
#include "defs.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>

using namespace std;

class ioDPMFile
{
public:
  ifstream inputFile;
  ofstream outputFile;
  ofstream energyFile;
  ofstream corrFile;
  DPM2D * dpm_;

  ioDPMFile() = default;
  ioDPMFile(DPM2D * dpmPtr) {
    this->dpm_ = dpmPtr;
  }

  void readPackingFromCellFormat(string fileName, long skiplines) {
    this->openInputFile(fileName);
    this->readFromCellFormat(skiplines);
  }

  // open file and check if it throws an error
  void openInputFile(string fileName) {
    inputFile = ifstream(fileName.c_str());
    if (!inputFile.is_open()) {
      cerr << "ioDPMFile::openInputFile: error: could not open input file " << fileName << endl;
      exit(1);
    }
  }

  void openOutputFile(string fileName) {
    outputFile = ofstream(fileName.c_str());
    if (!outputFile.is_open()) {
      cerr << "ioDPMFile::openOutputFile: error: could not open input file " << fileName << endl;
      exit(1);
    }
  }

  void openEnergyFile(string fileName) {
    energyFile = ofstream(fileName.c_str());
    if (!energyFile.is_open()) {
      cerr << "ioDPMFile::openEnergyFile: error: could not open input file " << fileName << endl;
      exit(1);
    }
  }

  void saveEnergy(long step, double timeStep, long numParticles, long numVertices) {
    if(dpm_->simControl.particleType == simControlStruct::particleEnum::deformable) {
      saveDeformableEnergy(step, timeStep, numVertices);
    } else if(dpm_->simControl.particleType == simControlStruct::particleEnum::rigid) {
      saveRigidEnergy(step, timeStep, numParticles);
    }
  }

  void saveDeformableEnergy(long step, double timeStep, long numVertices) {
    double epot = dpm_->getPotentialEnergy() / numVertices;
    double ekin = dpm_->getKineticEnergy() / numVertices;
    double etot = epot + ekin;
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << epot << "\t";
    energyFile << setprecision(precision) << ekin << "\t";
    energyFile << setprecision(precision) << etot << "\t";
    energyFile << setprecision(precision) << dpm_->getPhi() << endl;
  }

  void saveRigidEnergy(long step, double timeStep, long numParticles) {
    double epot = dpm_->getParticlePotentialEnergy();
    double ekin = dpm_->getRigidKineticEnergy();
    double etot = epot + ekin;
    energyFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    energyFile << setprecision(precision) << epot << "\t";
    energyFile << setprecision(precision) << ekin << "\t";
    energyFile << setprecision(precision) << etot << endl;
  }

  void openCorrFile(string fileName) {
    corrFile = ofstream(fileName.c_str());
    if (!corrFile.is_open()) {
      cerr << "ioDPMFile::openCorrFile: error: could not open input file " << fileName << endl;
      exit(1);
    }
  }

  void closeEnergyFile() {
    energyFile.close();
  }

  void closeCorrFile() {
    corrFile.close();
  }

  void saveCorr(long step, double timeStep) {
    double isf, visf, isfSq, visfSq, deltaA;
    isf = dpm_->getParticleISF(dpm_->getDeformableWaveNumber());
    visf = dpm_->getVertexISF();
    deltaA = dpm_->getAreaFluctuation();
    corrFile << step + 1 << "\t" << (step + 1) * timeStep << "\t";
    corrFile << setprecision(precision) << dpm_->getParticleMSD() << "\t";
    corrFile << setprecision(precision) << dpm_->getVertexMSD() << "\t";
    corrFile << setprecision(precision) << isf << "\t";
    corrFile << setprecision(precision) << visf << "\t";
    corrFile << setprecision(precision) << deltaA << endl;
  }

  thrust::host_vector<double> read1DIndexFile(string fileName, long numRows) {
    thrust::host_vector<long> data;
    this->openInputFile(fileName);
    string inputString;
    long tmp;
    for (long row = 0; row < numRows; row++) {
      getline(inputFile, inputString);
      sscanf(inputString.c_str(), "%ld", &tmp);
      data.push_back(tmp);
    }
    inputFile.close();
    return data;
  }

  void save1DIndexFile(string fileName, thrust::host_vector<long> data) {
    this->openOutputFile(fileName);
    long numRows = data.size();
    for (long row = 0; row < numRows; row++) {
      //sprintf(outputFile, "%ld \n", data[row]);
      outputFile << setprecision(precision) << data[row] << endl;
    }
    outputFile.close();
  }

 void save1DSTDFile(string fileName, std::vector<long> data) {
    this->openOutputFile(fileName);
    long numRows = data.size();
    for (long row = 0; row < numRows; row++) {
      //sprintf(outputFile, "%ld \n", data[row]);
      outputFile << setprecision(precision) << data[row] << endl;
    }
    outputFile.close();
  }

  thrust::host_vector<double> read1DFile(string fileName, long numRows) {
    thrust::host_vector<double> data;
    this->openInputFile(fileName);
    string inputString;
    double tmp;
    for (long row = 0; row < numRows; row++) {
      getline(inputFile, inputString);
      sscanf(inputString.c_str(), "%lf", &tmp);
      data.push_back(tmp);
    }
    inputFile.close();
    return data;
  }

  void save1DFile(string fileName, thrust::host_vector<double> data) {
    this->openOutputFile(fileName);
    long numRows = data.size();
    for (long row = 0; row < numRows; row++) {
      //sprintf(outputFile, "%lf \n", data[row]);
      outputFile << setprecision(precision) << data[row] << endl;
    }
    outputFile.close();
  }

  thrust::host_vector<double> read2DFile(string fileName, long numRows) {
    thrust::host_vector<double> data;
    this->openInputFile(fileName);
    string inputString;
    double data1, data2;
    for (long row = 0; row < numRows; row++) {
      getline(inputFile, inputString);
      sscanf(inputString.c_str(), "%lf %lf", &data1, &data2);
      data.push_back(data1);
      data.push_back(data2);
    }
    inputFile.close();
    return data;
  }

  void save2DFile(string fileName, thrust::host_vector<double> data, long numCols) {
    this->openOutputFile(fileName);
    long numRows = int(data.size()/numCols);
    for (long row = 0; row < numRows; row++) {
      for(long col = 0; col < numCols; col++) {
        outputFile << setprecision(precision) << data[row * numCols + col] << "\t";
      }
      outputFile << endl;
    }
    outputFile.close();
  }

  void save2DIndexFile(string fileName, thrust::host_vector<long> data, long numCols) {
    this->openOutputFile(fileName);
    if(numCols != 0) {
      long numRows = int(data.size()/numCols);
      for (long row = 0; row < numRows; row++) {
        for(long col = 0; col < numCols; col++) {
          outputFile << data[row * numCols + col] << "\t";
        }
        outputFile << endl;
      }
      outputFile.close();
    } else {
      cout << "FileIO::save2DIndexFile:: numCols is equal to zero - no data are stored" << endl;
    }
  }

  void save3DIndexFile(string fileName, thrust::host_vector<long> data, long numRowx, long numRowy, long numCols) {
    this->openOutputFile(fileName);
    for (long rowx = 0; rowx < numRowx; rowx++) {
      for (long rowy = 0; rowy < numRowy; rowy++) {
        for(long col = 0; col < numCols; col++) {
          outputFile << data[(rowx * numRowx + rowy) * numCols + col] << "\t";
        }
        outputFile << endl;
      }
    }
    outputFile.close();
  }

  void saveParticlePacking(string dirName) {
    // save scalars
    string fileParams = dirName + "params.dat";
    ofstream saveParams(fileParams.c_str());
    openOutputFile(fileParams);
    saveParams << "numParticles" << "\t" << dpm_->getNumParticles() << endl;
    saveParams << "dt" << "\t" << dpm_->dt << endl;
    saveParams << "phi" << "\t" << dpm_->getParticlePhi() << endl;
    saveParams << "energy" << "\t" << dpm_->getParticlePotentialEnergy() / dpm_->getNumParticles() << endl;
    saveParams << "temperature" << "\t" << dpm_->getParticleTemperature() << endl;
    saveParams.close();
    // save vectors
    save1DFile(dirName + "boxSize.dat", dpm_->getBoxSize());
    save2DFile(dirName + "particlePos.dat", dpm_->getParticlePositions(), dpm_->nDim);
    save1DFile(dirName + "particleRad.dat", dpm_->getParticleRadii());
  }

  void readPackingFromDirectory(string dirName, long numParticles_, long nDim_) {
    if(dpm_->simControl.particleType == simControlStruct::particleEnum::deformable) {
      readDeformablePackingFromDirectory(dirName, numParticles_, nDim_);
    } else if(dpm_->simControl.particleType == simControlStruct::particleEnum::rigid) {
      readRigidPackingFromDirectory(dirName, numParticles_, nDim_);
    }
  }

  void readDeformablePackingFromDirectory(string dirName, long numParticles_, long nDim_) {
    thrust::host_vector<long> numVertexInParticleList_(numParticles_);
    numVertexInParticleList_ = read1DIndexFile(dirName + "numVertexInParticleList.dat", numParticles_);
    dpm_->setNumVertexInParticleList(numVertexInParticleList_);
    long numVertices_ = thrust::reduce(numVertexInParticleList_.begin(), numVertexInParticleList_.end(), 0, thrust::plus<long>());
    dpm_->setNumVertices(numVertices_);
    cout << "readFromDirectory:: numVertices: " << numVertices_ << " on device: " << dpm_->getNumVertices() << endl;
    dpm_->initParticleIdList();
    dpm_->initShapeVariables(numVertices_, numParticles_);
    dpm_->initDynamicalVariables(numVertices_);
    dpm_->initNeighbors(numVertices_);
    thrust::host_vector<double> boxSize_(nDim_);
    thrust::host_vector<double> pos_(numVertices_ * nDim_);
    thrust::host_vector<double> rad_(numVertices_);
    thrust::host_vector<double> a0_(numParticles_);
    thrust::host_vector<double> l0_(numVertices_);
    thrust::host_vector<double> theta0_(numVertices_);

    boxSize_ = read1DFile(dirName + "boxSize.dat", nDim_);
    dpm_->setBoxSize(boxSize_);
    pos_ = read2DFile(dirName + "positions.dat", numVertices_);
    dpm_->setVertexPositions(pos_);
    rad_ = read1DFile(dirName + "radii.dat", numVertices_);
    dpm_->setVertexRadii(rad_);
    a0_ = read1DFile(dirName + "restAreas.dat", numParticles_);
    dpm_->setRestAreas(a0_);
    l0_ = read1DFile(dirName + "restLengths.dat", numVertices_);
    dpm_->setRestLengths(l0_);
    theta0_ = read1DFile(dirName + "restAngles.dat", numVertices_);
    dpm_->setRestAngles(theta0_);
    // set length scales
    dpm_->setLengthScale();
    cout << "FileIO::readPackingFromDirectory: preferred phi: " << dpm_->getPreferredPhi() << " box-Lx: " << boxSize_[0] << ", Ly: " << boxSize_[1] << endl;
    dpm_->calcParticleShape();
  }

  void readRigidPackingFromDirectory(string dirName, long numParticles_, long nDim_) {
    thrust::host_vector<long> numVertexInParticleList_(numParticles_);
    numVertexInParticleList_ = read1DIndexFile(dirName + "numVertexInParticleList.dat", numParticles_);
    dpm_->setNumVertexInParticleList(numVertexInParticleList_);
    long numVertices_ = thrust::reduce(numVertexInParticleList_.begin(), numVertexInParticleList_.end(), 0, thrust::plus<long>());
    dpm_->setNumVertices(numVertices_);
    cout << "readRigidPackingFromDirectory:: numVertices: " << numVertices_ << " on device: " << dpm_->getNumVertices() << endl;
    dpm_->initParticleIdList();
    dpm_->initShapeVariables(numVertices_, numParticles_);
    dpm_->initDynamicalVariables(numVertices_);
    dpm_->initNeighbors(numVertices_);
    dpm_->initParticleVariables(numParticles_);
    dpm_->initRotationalVariables(numVertices_, numParticles_);
    dpm_->initDeltaVariables(numVertices_, numParticles_);
    thrust::host_vector<double> boxSize_(nDim_);
    thrust::host_vector<double> pos_(numVertices_ * nDim_);
    thrust::host_vector<double> rad_(numVertices_);
    thrust::host_vector<double> a0_(numParticles_);
    thrust::host_vector<double> particleRad_(numParticles_);
    thrust::host_vector<double> particlePos_(numParticles_);
    thrust::host_vector<double> particleAngles_(numParticles_);

    boxSize_ = read1DFile(dirName + "boxSize.dat", nDim_);
    dpm_->setBoxSize(boxSize_);
    pos_ = read2DFile(dirName + "positions.dat", numVertices_);
    dpm_->setVertexPositions(pos_);
    rad_ = read1DFile(dirName + "radii.dat", numVertices_);
    dpm_->setVertexRadii(rad_);
    a0_ = read1DFile(dirName + "restAreas.dat", numParticles_);
    dpm_->setRestAreas(a0_);
    particleRad_ = read1DFile(dirName + "particleRad.dat", numParticles_);
    dpm_->setParticleRadii(particleRad_);
    particlePos_ = read2DFile(dirName + "particlePos.dat", numParticles_);
    dpm_->setParticlePositions(particlePos_);
    particleAngles_ = read1DFile(dirName + "particleAngles.dat", numParticles_);
    dpm_->setParticleAngles(particleAngles_);
    // set length scales
    dpm_->setLengthScaleToOne();
    cout << "FileIO::readRigidPackingFromDirectory: phi: " << dpm_->getPreferredPhi() << endl;
  }

  void savePacking(string dirName) {
    if(dpm_->simControl.particleType == simControlStruct::particleEnum::deformable) {
      saveDeformablePacking(dirName);
    } else if(dpm_->simControl.particleType == simControlStruct::particleEnum::rigid) {
      saveRigidPacking(dirName);
    }
  }

  void saveDeformablePacking(string dirName) {
    // save scalars
    string fileParams = dirName + "params.dat";
    ofstream saveParams(fileParams.c_str());
    openOutputFile(fileParams);
    saveParams << "numParticles" << "\t" << dpm_->getNumParticles() << endl;
    saveParams << "ea" << "\t" << dpm_->ea << endl;
    saveParams << "el" << "\t" << dpm_->el << endl;
    saveParams << "eb" << "\t" << dpm_->eb << endl;
    saveParams << "ec" << "\t" << dpm_->ec << endl;
    saveParams << "dt" << "\t" << dpm_->dt << endl;
    saveParams << "phi" << "\t" << dpm_->getPhi() << endl;
    saveParams << "phi0" << "\t" << dpm_->getPreferredPhi() << endl;
    saveParams << "epot" << "\t" << dpm_->getPotentialEnergy() / dpm_->getNumParticles() << endl;
    saveParams << "temperature" << "\t" << dpm_->getTemperature() << endl;
    saveParams.close();
    // save vectors
    save1DFile(dirName + "boxSize.dat", dpm_->getBoxSize());
    save1DIndexFile(dirName + "numVertexInParticleList.dat", dpm_->getNumVertexInParticleList());
    save1DFile(dirName + "radii.dat", dpm_->getVertexRadii());
    save2DFile(dirName + "positions.dat", dpm_->getVertexPositions(), dpm_->nDim);
    //save2DFile(dirName + "forces.dat", dpm_->getVertexForces(), dpm_->nDim);
    save2DFile(dirName + "forces.dat", dpm_->getInteractionForces(), dpm_->nDim);
    save1DFile(dirName + "restAreas.dat", dpm_->getRestAreas());
    //save1DFile(dirName + "segmentLengths.dat", dpm_->getSegmentLengths());
    save1DFile(dirName + "restLengths.dat", dpm_->getRestLengths());
    //save1DFile(dirName + "segmentAngles.dat", dpm_->getSegmentAngles());
    save1DFile(dirName + "restAngles.dat", dpm_->getRestAngles());
    save2DFile(dirName + "velocities.dat", dpm_->getVertexVelocities(), dpm_->nDim);
    //save2DFile(dirName + "particlePos.dat", dpm_->getParticlePositions(), dpm_->nDim);
    //save1DFile(dirName + "particleAngles.dat", dpm_->getParticleAngles());
    save1DFile(dirName + "particleRad.dat", dpm_->getParticleRadii());
  }

  void saveRigidPacking(string dirName) {
    // save scalars
    string fileParams = dirName + "params.dat";
    ofstream saveParams(fileParams.c_str());
    openOutputFile(fileParams);
    saveParams << "numParticles" << "\t" << dpm_->getNumParticles() << endl;
    saveParams << "phi" << "\t" << dpm_->getPhi() << endl;
    saveParams << "epot" << "\t" << dpm_->getPotentialEnergy() << endl;
    saveParams << "temperature" << "\t" << dpm_->getTemperature() << endl;
    saveParams.close();
    // save vectors
    save1DFile(dirName + "boxSize.dat", dpm_->getBoxSize());
    save1DIndexFile(dirName + "numVertexInParticleList.dat", dpm_->getNumVertexInParticleList());
    save1DFile(dirName + "radii.dat", dpm_->getVertexRadii());
    save2DFile(dirName + "forces.dat", dpm_->getVertexForces(), dpm_->nDim);
    save2DFile(dirName + "positions.dat", dpm_->getVertexPositions(), dpm_->nDim);
    save1DFile(dirName + "restAreas.dat", dpm_->getRestAreas());
    save1DFile(dirName + "particleRad.dat", dpm_->getParticleRadii());
    save2DFile(dirName + "particlePos.dat", dpm_->getParticlePositions(), dpm_->nDim);
    save2DFile(dirName + "particleVel.dat", dpm_->getParticleVelocities(), dpm_->nDim);
    save1DFile(dirName + "particleAngles.dat", dpm_->getParticleAngles());
    save1DFile(dirName + "particleAngvel.dat", dpm_->getParticleAngularVelocities());
  }

  void readState(string dirName, long numParticles_, long numVertices_, long nDim_) {
    if(dpm_->simControl.particleType == simControlStruct::particleEnum::deformable) {
      readDeformableState(dirName, numParticles_, numVertices_, nDim_);
    } else if(dpm_->simControl.particleType == simControlStruct::particleEnum::rigid) {
      readRigidState(dirName, numParticles_, nDim_);
    }
  }
  
  void readDeformableState(string dirName, long numParticles_, long numVertices_, long nDim_) {
    thrust::host_vector<double> vel_(numVertices_ * nDim_);
    thrust::host_vector<double> particlePos_(numParticles_ * nDim_);
    thrust::host_vector<double> particleAngle_(numParticles_);
    vel_ = read2DFile(dirName + "velocities.dat", numVertices_);
    dpm_->setVertexVelocities(vel_);
    particlePos_ = read2DFile(dirName + "particlePos.dat", numParticles_);
    dpm_->setParticlePositions(particlePos_);
    particleAngle_ = read1DFile(dirName + "particleAngles.dat", numParticles_);
    dpm_->setParticleAngles(particleAngle_);
  }

  void readRigidState(string dirName, long numParticles_, long nDim_) {;
    thrust::host_vector<double> particleVel_(numParticles_ * nDim_);
    thrust::host_vector<double> particleAngvel_(numParticles_);
    particleVel_ = read2DFile(dirName + "particleVel.dat", numParticles_);
    dpm_->setParticleVelocities(particleVel_);
    particleAngvel_ = read1DFile(dirName + "particleAngvel.dat", numParticles_);
    dpm_->setParticleAngularVelocities(particleAngvel_);

  }

  void saveState(string dirName) {
    if(dpm_->simControl.particleType == simControlStruct::particleEnum::deformable) {
      saveDeformableState(dirName);
    } else if(dpm_->simControl.particleType == simControlStruct::particleEnum::rigid) {
      saveRigidState(dirName);
    }
  }

  void saveDeformableState(string dirName) {
    save2DFile(dirName + "positions.dat", dpm_->getVertexPositions(), dpm_->nDim);
    save2DFile(dirName + "velocities.dat", dpm_->getVertexVelocities(), dpm_->nDim);
    save2DFile(dirName + "forces.dat", dpm_->getVertexForces(), dpm_->nDim);
    save2DFile(dirName + "particlePos.dat", dpm_->getParticlePositions(), dpm_->nDim);
    save1DFile(dirName + "particleAngles.dat", dpm_->getParticleAngles());
  }

  void saveRigidState(string dirName) {
    save2DFile(dirName + "positions.dat", dpm_->getVertexPositions(), dpm_->nDim);
    save2DFile(dirName + "velocities.dat", dpm_->getVertexVelocities(), dpm_->nDim);
    save2DFile(dirName + "forces.dat", dpm_->getVertexForces(), dpm_->nDim);
    save2DFile(dirName + "particlePos.dat", dpm_->getParticlePositions(), dpm_->nDim);
    save2DFile(dirName + "particleVel.dat", dpm_->getParticleVelocities(), dpm_->nDim);
    save1DFile(dirName + "particleAngles.dat", dpm_->getParticleAngles());
    save1DFile(dirName + "particleAngvel.dat", dpm_->getParticleAngularVelocities());
  }

  void saveContacts(string dirName) {
    dpm_->calcContacts(0);
    save2DFile(dirName + "contacts.dat", dpm_->getContacts(), dpm_->contactLimit);
  }

  void saveNeighbors(string dirName) {
    if(dpm_->simControl.neighborType == simControlStruct::neighborEnum::neighbor) {
      save2DIndexFile(dirName + "neighborList.dat", dpm_->getNeighbors(), dpm_->neighborListSize);
      //if(dpm_->simControl.interactionType == simControlStruct::interactionEnum::vertexSmooth) {
      //  save2DIndexFile(dirName + "smoothNeighborList.dat", dpm_->getSmoothNeighbors(), dpm_->smoothNeighborListSize);
      //}
    } else if(dpm_->simControl.neighborType == simControlStruct::neighborEnum::cell) {
      save1DIndexFile(dirName + "linkedList.dat", dpm_->getLinkedList());
      save1DIndexFile(dirName + "listHeader.dat", dpm_->getListHeader());
      save2DIndexFile(dirName + "cellIndexList.dat", dpm_->getCellIndexList(), dpm_->nDim);
      //save3DIndexFile(dirName + "cellNeighborList.dat", dpm_->getCellNeighborList(), dpm_->numCells, dpm_->numCells, dpm_->cellNeighborListSize);
    }
  }

  void saveConfiguration(string dirName) {
    savePacking(dirName);
    saveNeighbors(dirName);
  }

  void saveSPDPMPacking(string dirName) {
    // save scalars
    string fileParams = dirName + "params.dat";
    ofstream saveParams(fileParams.c_str());
    openOutputFile(fileParams);
    saveParams << "numParticles" << "\t" << dpm_->getNumParticles() << endl;
    saveParams << "ea" << "\t" << dpm_->ea << endl;
    saveParams << "el" << "\t" << dpm_->el << endl;
    saveParams << "eb" << "\t" << dpm_->eb << endl;
    saveParams << "ec" << "\t" << dpm_->ec << endl;
    saveParams << "dt" << "\t" << dpm_->dt << endl;
    saveParams << "phiSP" << "\t" << dpm_->getParticlePhi() << endl;
    saveParams << "phiDPM" << "\t" << dpm_->getPreferredPhi() << endl;
    saveParams << "epot" << "\t" << dpm_->getPotentialEnergy() << endl;
    saveParams << "temperature" << "\t" << dpm_->getTemperature() << endl;
    saveParams.close();
    // save vectors
    save1DFile(dirName + "boxSize.dat", dpm_->getBoxSize());
    save1DIndexFile(dirName + "numVertexInParticleList.dat", dpm_->getNumVertexInParticleList());
    save2DFile(dirName + "positions.dat", dpm_->getVertexPositions(), dpm_->nDim);
    save1DFile(dirName + "radii.dat", dpm_->getVertexRadii());
    save1DFile(dirName + "restAreas.dat", dpm_->getRestAreas());
    save1DFile(dirName + "restLengths.dat", dpm_->getRestLengths());
    save1DFile(dirName + "restAngles.dat", dpm_->getRestAngles());
    // these two functions are overidden in spdpm2d
    save2DFile(dirName + "softPos.dat", dpm_->getParticlePositions(), dpm_->nDim);
    save1DFile(dirName + "softRad.dat", dpm_->getParticleRadii());
  }

  void saveSPDPMState(string dirName) {
    save1DFile(dirName + "particleAngles.dat", dpm_->getParticleAngles());
    save2DFile(dirName + "particleVel.dat", dpm_->getParticleVelocities(), dpm_->nDim);
    save2DFile(dirName + "particleForces.dat", dpm_->getParticleForces(), dpm_->nDim);
  }

  void readFromCellFormat(long skiplines) {// read dpm packing from cell format
    thrust::host_vector<double> boxSize_;
    thrust::host_vector<double> pos_;
    thrust::host_vector<double> rad_;
    thrust::host_vector<double> a0_;
    thrust::host_vector<double> l0_;
    thrust::host_vector<double> theta0_;
    thrust::host_vector<long> numVertexInParticleList_;
    long numParticles_, numVertexInParticle_, numVertices_ = 0;
    double phi_, lx, ly, stress_[MAXDIM + 1];
    double a0tmp, area, p0tmp, xtmp, ytmp, radtmp, l0tmp, theta0tmp, fx, fy;

    string inputString;
    //get rid of first line
    for (long l = 0; l < skiplines; l++) {
      getline(inputFile, inputString);
    }

    // read in simulation information from header
    getline(inputFile, inputString);
    sscanf(inputString.c_str(), "NUMCL %ld", &numParticles_);
    //cout << inputString << "read: " << numParticles_ << endl;

    // verify input file
    if (numParticles_ < 1) {
      cerr << "FileIO::readFromCellFormat: error: numParticles = " << numParticles_ << ". Ending here." << endl;
      exit(1);
    }

    getline(inputFile, inputString);
    sscanf(inputString.c_str(), "PACKF %lf", &phi_);
    //cout << inputString << "read: " << phi_ << endl;

    // initialize box lengths
    getline(inputFile, inputString);
    sscanf(inputString.c_str(), "BOXSZ %lf %lf", &lx, &ly);
    //cout << inputString << "read: " << lx << " " << ly << endl;
    boxSize_.push_back(lx);
    boxSize_.push_back(ly);

    // initialize stress
    getline(inputFile, inputString);
    sscanf(inputString.c_str(), "STRSS %lf %lf %lf", &stress_[0], &stress_[1], &stress_[2]);

    // loop over cells, read in coordinates
    long start = 0;
    for (long particleId = 0; particleId < numParticles_; particleId++) {
      // first parse cell info
      getline(inputFile, inputString);
      sscanf(inputString.c_str(), "CINFO %ld %*d %*d %lf %lf %lf", &numVertexInParticle_, &a0tmp, &area, &p0tmp);
      //cout << "particleId: " << particleId << ", a: " << a0tmp << " " << area << ", p: " << p0tmp << endl;
      numVertexInParticleList_.push_back(numVertexInParticle_);
      numVertices_ += numVertexInParticle_;
      a0_.push_back(a0tmp);

      // loop over vertices and store coordinates
      for (long vertexId = 0; vertexId < numVertexInParticle_; vertexId++) {
        // parse vertex coordinate info
        getline(inputFile, inputString);
        sscanf(inputString.c_str(), "VINFO %*d %*d %lf %lf %lf %lf %lf %lf %lf", &xtmp, &ytmp, &radtmp, &l0tmp, &theta0tmp, &fx, &fy);
        // check pbc
        //xtmp -= floor(xtmp/lx) * lx;
        //ytmp -= floor(ytmp/ly) * ly;
        //cout << "read: vertexId: " << start + vertexId << " forces: " << setprecision(12) << fx << " " << fy << endl;
        //cout << "read: vertexId: " << start + vertexId << " pos: " << setprecision(12) << xtmp << " " << ytmp << endl;
        // push back
        pos_.push_back(xtmp); // push x then y
        pos_.push_back(ytmp);
        rad_.push_back(radtmp);
        l0_.push_back(l0tmp);
        theta0_.push_back(theta0tmp);
      }
      start += numVertexInParticle_;
    }
    inputFile.close();
    // transfer to dpm class
    dpm_->setNumVertices(numVertices_);
    cout << "FileIO::readFromCellFormat: numVertices: " << numVertices_ << " on device: " << dpm_->getNumVertices() << endl;
    // first initilize indexing for polydisperse packing
    dpm_->setNumVertexInParticleList(numVertexInParticleList_);
    dpm_->initParticleIdList();
    dpm_->initShapeVariables(numVertices_, numParticles_);
    dpm_->initDynamicalVariables(numVertices_);
    dpm_->initNeighbors(numVertices_);
    // set all the rest
    dpm_->setBoxSize(boxSize_);
    dpm_->setVertexPositions(pos_);
    dpm_->setVertexRadii(rad_);
    dpm_->setRestAreas(a0_);
    dpm_->setRestLengths(l0_);
    dpm_->setRestAngles(theta0_);
    dpm_->setLengthScale();
    cout << "FileIO::readFromCellFormat: preferred phi: " << dpm_->getPreferredPhi() << endl;
  }

};

#endif // FILEIO_H //
