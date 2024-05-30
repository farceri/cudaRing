//
// Author: Francesco Arceri
// Date:   10-03-2021
//
// FUNCTION DECLARATIONS

#include "../include/DPM2D.h"
#include "../include/cudaKernel.cuh"
#include "../include/cached_allocator.cuh"
#include "../include/Simulator.h"
#include "../include/FIRE.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <time.h>
#include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

using namespace std;
using std::cout;
using std::endl;
using std::fill;
using std::vector;
using std::log2;

//************************** dpm object definition ***************************//
DPM2D::DPM2D(long nParticles, long dim, long nVertexPerParticle) {
  // default values
  srand48(time(0));
  dimBlock = 256;
  nDim = dim;
  numParticles = nParticles;
  numVertexPerParticle = nVertexPerParticle;
  // the default is same number of vertices per particle
  numVertices = numParticles * numVertexPerParticle;
  setDimBlock(dimBlock);
  setNDim(nDim);
  setNumParticles(numParticles);
  setNumVertexPerParticle(numVertexPerParticle);
  setNumVertices(numVertices);
	simControl.simulationType = simControlStruct::simulationEnum::gpu;
	simControl.particleType = simControlStruct::particleEnum::deformable;
	simControl.potentialType = simControlStruct::potentialEnum::harmonic;
	simControl.interactionType = simControlStruct::interactionEnum::vertexVertex;
	simControl.neighborType = simControlStruct::neighborEnum::neighbor;
	simControl.concavityType = simControlStruct::concavityEnum::off;
	simControl.monomerType = simControlStruct::monomerEnum::harmonic;
  // default force constants
  dt = 1e-03;
  rho0 = 1;
  ea = 1e05;
	el = 20;
	eb = 10;
	ec = 1;
  cutDistance = 1;
  updateCount = 0;
  d_boxSize.resize(nDim);
  h_boxSize.resize(nDim); //HOST
  thrust::fill(d_boxSize.begin(), d_boxSize.end(), double(1));
  d_stress.resize(nDim * nDim);
  thrust::fill(d_stress.begin(), d_stress.end(), double(0));
  d_numVertexInParticleList.resize(numParticles);
  d_firstVertexInParticleId.resize(numParticles);
  // particle variables
  initParticleVariables(numParticles);
  // vertex variables
  initShapeVariables(numVertices, numParticles);
  initDynamicalVariables(numVertices);
  initNeighbors(numVertices);
  syncNeighborsToDevice();
  // initialize contacts and neighbors
  initContacts(numParticles);
  initParticleNeighbors(numParticles);
  syncParticleNeighborsToDevice();
  if(cudaGetLastError()) cout << "DPM2D():: cudaGetLastError(): " << cudaGetLastError() << endl;
}

DPM2D::~DPM2D() {}

void DPM2D::printDeviceProperties() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for(int i =0; i < deviceCount; ++i) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    cout << "Device " << i << ": " << deviceProp.name << endl;
    cout << " Maximum threads per block: " << deviceProp.maxThreadsPerBlock << endl;
    cout << " Maximum dimensions of block (x,y,z): " << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << endl;
    cout << " Maximum dimensions of grid (x,y,z): " << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << endl;
  }
}

void DPM2D::initShapeVariables(long numVertices_, long numParticles_) {
  d_rad.resize(numVertices_);
  h_rad.resize(numVertices_); //HOST
  d_l0.resize(numVertices_);
  d_length.resize(numVertices_);
  d_perimeter.resize(numParticles_);
  d_a0.resize(numParticles_);
  d_area.resize(numParticles_);
  d_theta.resize(numVertices_);
  d_theta0.resize(numVertices_);
  d_particleRad.resize(numParticles_);
  d_particlePos.resize(numParticles_ * nDim);
  thrust::fill(d_rad.begin(), d_rad.end(), double(0));
  thrust::fill(d_l0.begin(), d_l0.end(), double(0));
  thrust::fill(d_length.begin(), d_length.end(), double(0));
  thrust::fill(d_perimeter.begin(), d_perimeter.end(), double(0));
  thrust::fill(d_a0.begin(), d_a0.end(), double(0));
  thrust::fill(d_area.begin(), d_area.end(), double(0));
  thrust::fill(d_theta.begin(), d_theta.end(), double(0));
  thrust::fill(d_theta0.begin(), d_theta0.end(), double(0));
  thrust::fill(d_particleRad.begin(), d_particleRad.end(), double(0));
  thrust::fill(d_particlePos.begin(), d_particlePos.end(), double(0));
}

void DPM2D::initDynamicalVariables(long numVertices_) {
  d_pos.resize(numVertices_ * nDim);
  h_pos.resize(numVertices_ * nDim); //HOST
  d_vel.resize(numVertices_ * nDim);
  d_force.resize(numVertices_ * nDim);
  h_force.resize(numVertices_ * nDim); //HOST
  h_interaction.resize(numVertices_ * nDim); //HOST
  d_energy.resize(numVertices_);
  h_energy.resize(numVertices_); //HOST
  d_lastPos.resize(numVertices_ * nDim);
  d_disp.resize(numVertices_);
  thrust::fill(d_pos.begin(), d_pos.end(), double(0));
  thrust::fill(d_vel.begin(), d_vel.end(), double(0));
  thrust::fill(d_force.begin(), d_force.end(), double(0));
  thrust::fill(d_energy.begin(), d_energy.end(), double(0));
  thrust::fill(d_lastPos.begin(), d_lastPos.end(), double(0));
  thrust::fill(d_disp.begin(), d_disp.end(), double(0));
}

void DPM2D::initParticleVariables(long numParticles_) {
  d_particleVel.resize(numParticles_ * nDim);
  d_particleForce.resize(numParticles_ * nDim);
  d_particleEnergy.resize(numParticles_);
  h_particleEnergy.resize(numParticles_); //HOST
  d_particleDisp.resize(numParticles_);
  d_particleLastPos.resize(numParticles_ * nDim);
  thrust::fill(d_particleVel.begin(), d_particleVel.end(), double(0));
  thrust::fill(d_particleForce.begin(), d_particleForce.end(), double(0));
  thrust::fill(d_particleEnergy.begin(), d_particleEnergy.end(), double(0));
  thrust::fill(d_particleDisp.begin(), d_particleDisp.end(), double(0));
  thrust::fill(d_particleLastPos.begin(), d_particleLastPos.end(), double(0));
  d_perParticleStress.resize(numParticles_ * nDim * nDim);
  thrust::fill(d_perParticleStress.begin(), d_perParticleStress.end(), double(0));
  d_particleAngle.resize(numParticles_);
  thrust::fill(d_particleAngle.begin(), d_particleAngle.end(), double(0));
}

void DPM2D::initDeltaVariables(long numVertices_, long numParticles_) {
  d_delta.resize(numVertices_ * nDim);
  d_initialPos.resize(numVertices_ * nDim);
  d_particleLastAngle.resize(numParticles_);
  d_particleDelta.resize(numParticles_ * nDim);
  d_particleDeltaAngle.resize(numParticles_);
  thrust::fill(d_delta.begin(), d_delta.end(), double(0));
  thrust::fill(d_initialPos.begin(), d_initialPos.end(), double(0));
  thrust::fill(d_particleLastAngle.begin(), d_particleLastAngle.end(), double(0));
  thrust::fill(d_particleDelta.begin(), d_particleDelta.end(), double(0));
  thrust::fill(d_particleDeltaAngle.begin(), d_particleDeltaAngle.end(), double(0));
}

void DPM2D::initRotationalVariables(long numVertices_, long numParticles_) {
  d_torque.resize(numVertices_);
  d_particleAngvel.resize(numParticles_);
  d_particleTorque.resize(numParticles_);
  thrust::fill(d_torque.begin(), d_torque.end(), double(0));
  thrust::fill(d_particleAngvel.begin(), d_particleAngvel.end(), double(0));
  thrust::fill(d_particleTorque.begin(), d_particleTorque.end(), double(0));
}

void DPM2D::initContacts(long numParticles_) {
  long maxContacts = 8 * nDim; // guess
  d_numContacts.resize(numParticles_);
  d_contactList.resize(numParticles_ * maxContacts);
  d_numPartNeighbors.resize(numParticles_);
  d_partNeighborList.resize(numParticles_ * maxContacts);
  d_contactVectorList.resize(numParticles_ * nDim * maxContacts);
  thrust::fill(d_numContacts.begin(), d_numContacts.end(), -1L);
  thrust::fill(d_contactList.begin(), d_contactList.end(), double(0));
  thrust::fill(d_numPartNeighbors.begin(), d_numPartNeighbors.end(), -1L);
  thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), double(0));
  thrust::fill(d_contactVectorList.begin(), d_contactVectorList.end(), double(0));
}

void DPM2D::initNeighbors(long numVertices_) {
  neighborListSize = 0;
  maxNeighbors = 0;
  d_neighborList.resize(numVertices_);
  d_maxNeighborList.resize(numVertices_);
  h_maxNeighborList.resize(numVertices_); //HOST
  thrust::fill(d_neighborList.begin(), d_neighborList.end(), -1L);
  thrust::fill(d_maxNeighborList.begin(), d_maxNeighborList.end(), maxNeighbors);
}

void DPM2D::initSmoothNeighbors(long numVertices_) {
  smoothNeighborListSize = 0;
  h_smoothNeighborList.resize(numVertices_); //HOST
  h_maxSmoothNeighborList.resize(numVertices_); //HOST
  thrust::fill(h_smoothNeighborList.begin(), h_smoothNeighborList.end(), -1L);
  thrust::fill(h_maxSmoothNeighborList.begin(), h_maxSmoothNeighborList.end(), 0);
}

double DPM2D::initCells(long numVertices_, double cellSize_) {
  numCells = static_cast<long>(d_boxSize[0] / cellSize_);
  cellSize = d_boxSize[0] / numCells;
  setCellDimGridBlock();
  h_linkedList.resize(numVertices);
  h_header.resize(numCells * numCells);
  h_cellIndexList.resize(numVertices * nDim);
  thrust::fill(h_linkedList.begin(), h_linkedList.end(), -1L);
  thrust::fill(h_header.begin(), h_header.end(), -1L);
  thrust::fill(h_cellIndexList.begin(), h_cellIndexList.end(), -1L);
  d_linkedList.resize(numVertices);
  d_header.resize(numCells * numCells);
  d_cellIndexList.resize(numVertices * nDim);
  thrust::fill(d_linkedList.begin(), d_linkedList.end(), -1L);
  thrust::fill(d_header.begin(), d_header.end(), -1L);
  thrust::fill(d_cellIndexList.begin(), d_cellIndexList.end(), -1L);
  return cellSize;
}

void DPM2D::initParticleNeighbors(long numParticles_) {
  partNeighborListSize = 0;
  partMaxNeighbors = 0;
  d_partNeighborList.resize(numParticles_);
  d_partMaxNeighborList.resize(numParticles_);
  thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), -1L);
  thrust::fill(d_partMaxNeighborList.begin(), d_partMaxNeighborList.end(), partMaxNeighbors);
}

void DPM2D::initParticleIdList() {
  long countVertices = 0;
  d_particleIdList.resize(numVertices);
  for (long particleId = 0; particleId < numParticles; particleId++) {
    d_firstVertexInParticleId[particleId] = countVertices;
    for(long vertexInPartId = 0; vertexInPartId < d_numVertexInParticleList[particleId]; vertexInPartId++) {
      d_particleIdList[countVertices] = particleId;
			countVertices += 1;
		}
  }
  long* firstVertexInParticleId = thrust::raw_pointer_cast(&d_firstVertexInParticleId[0]);
  cudaMemcpyToSymbol(d_firstVertexInParticleIdPtr, &firstVertexInParticleId, sizeof(firstVertexInParticleId));

  long* particleIdList = thrust::raw_pointer_cast(&d_particleIdList[0]);
  cudaMemcpyToSymbol(d_particleIdListPtr, &particleIdList, sizeof(particleIdList));
  h_particleIdList = d_particleIdList; //HOST
}

//**************************** setters and getters ***************************//
//send simControl information to the gpu
void DPM2D::syncSimControlToDevice() {
	cudaMemcpyToSymbol(d_simControl, &simControl, sizeof(simControl));
}

//get simControl information from the gpu
void DPM2D::syncSimControlFromDevice() {
	cudaMemcpyFromSymbol(&simControl, d_simControl, sizeof(simControl));
}

void DPM2D::setSimulationType(simControlStruct::simulationEnum simulationType_) {
	simControl.simulationType = simulationType_;
  if(simControl.simulationType == simControlStruct::simulationEnum::gpu) {
    cout << "DPM2D::setSimulationType: simulationType: gpu" << endl;
  } else if(simControl.simulationType == simControlStruct::simulationEnum::cpu) {
    cout << "DPM2D::setSimulationType: simulationType: cpu" << endl;
  } else if(simControl.simulationType == simControlStruct::simulationEnum::omp) {
    cout << "DPM2D::setSimulationType: simulationType: omp" << endl;
  } else {
    cout << "DPM2D::setSimulationType: please specify valid simulationType: gpu, cpu or omp" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::simulationEnum DPM2D::getSimulationType() {
	syncSimControlFromDevice();
	return simControl.simulationType;
}

void DPM2D::setParticleType(simControlStruct::particleEnum particleType_) {
	simControl.particleType = particleType_;
  if(simControl.particleType == simControlStruct::particleEnum::deformable) {
    cout << "DPM2D::setParticleType: particleType: deformable" << endl;
  } else if(simControl.particleType == simControlStruct::particleEnum::rigid) {
    initRotationalVariables(getNumVertices(), getNumParticles());
    cout << "DPM2D::setParticleType: particleType: rigid" << endl;
  } else {
    cout << "DPM2D::setParticleType: please specify valid particleType: deformable or rigid" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::particleEnum DPM2D::getParticleType() {
	syncSimControlFromDevice();
	return simControl.particleType;
}

void DPM2D::setPotentialType(simControlStruct::potentialEnum potentialType_) {
	simControl.potentialType = potentialType_;
  if(simControl.potentialType == simControlStruct::potentialEnum::harmonic) {
    cout << "DPM2D::setPotentialType: potentialType: harmonic" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::lennardJones) {
    cout << "DPM2D::setPotentialType: potentialType: lennardJones" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::adhesive) {
    cout << "DPM2D::setPotentialType: potentialType: adhesive" << endl;
  } else if(simControl.potentialType == simControlStruct::potentialEnum::wca) {
    cout << "DPM2D::setPotentialType: potentialType: wca" << endl;
  } else {
    cout << "DPM2D::setPotentialType: please specify valid potentialType: normal or lennardJones" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::potentialEnum DPM2D::getPotentialType() {
	syncSimControlFromDevice();
	return simControl.potentialType;
}

void DPM2D::setInteractionType(simControlStruct::interactionEnum interactionType_) {
	simControl.interactionType = interactionType_;
  if(simControl.interactionType == simControlStruct::interactionEnum::vertexVertex) {
    cout << "DPM2D::setInteractionType: interactionType: vertexVertex" << endl;
  } else if(simControl.interactionType == simControlStruct::interactionEnum::cellSmooth) {
    cout << "DPM2D::setInteractionType: interactionType: cellSmooth" << endl;
  } else if(simControl.interactionType == simControlStruct::interactionEnum::vertexSmooth) {
    //initSmoothNeighbors(getNumVertices());
    //cout << "DPM2D::setInteractionType: vertexSmooth: initialized smooth neighbors" << endl;
    cout << "DPM2D::setInteractionType: interactionType: vertexSmooth" << endl;
  } else if(simControl.interactionType == simControlStruct::interactionEnum::all) {
    cout << "DPM2D::setInteractionType: interactionType: all" << endl;
  } else {
    cout << "DPM2D::setInteractionType: please specify valid interactionType: vertexVertex, cellSmooth, vertexSmooth or all" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::interactionEnum DPM2D::getInteractionType() {
	syncSimControlFromDevice();
	return simControl.interactionType;
}

void DPM2D::setNeighborType(simControlStruct::neighborEnum neighborType_) {
	simControl.neighborType = neighborType_;
  if(simControl.neighborType == simControlStruct::neighborEnum::neighbor) {
    cout << "DPM2D::setNeighborType: neighborType: neighbor" << endl;
  } else if(simControl.neighborType == simControlStruct::neighborEnum::cell) {
    cout << "DPM2D::setNeighborType: neighborType: cell" << endl;
  } else if(simControl.neighborType == simControlStruct::neighborEnum::allToAll) {
    cout << "DPM2D::setNeighborType: neighborType: allToAll" << endl;
  } else {
    cout << "DPM2D::setNeighborType: please specify valid neighborType: neighbor, cell or allToAll" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::neighborEnum DPM2D::getNeighborType() {
	syncSimControlFromDevice();
	return simControl.neighborType;
}

void DPM2D::setConcavityType(simControlStruct::concavityEnum concavityType_) {
	simControl.concavityType = concavityType_;
  if(simControl.concavityType == simControlStruct::concavityEnum::on) {
    cout << "DPM2D::setConcavityType: concavityType: on" << endl;
  } else if(simControl.concavityType == simControlStruct::concavityEnum::off) {
    cout << "DPM2D::setConcavityType: concavityType: off" << endl;
  } else {
    cout << "DPM2D::setConcavityType: please specify valid concavityType: on or off" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::concavityEnum DPM2D::getConcavityType() {
	syncSimControlFromDevice();
	return simControl.concavityType;
}

void DPM2D::setMonomerType(simControlStruct::monomerEnum monomerType_) {
	simControl.monomerType = monomerType_;
  if(simControl.monomerType == simControlStruct::monomerEnum::harmonic) {
    cout << "DPM2D::setMonomerType: monomerType: harmonic" << endl;
  } else if(simControl.monomerType == simControlStruct::monomerEnum::FENE) {
    cout << "DPM2D::setMonomerType: monomerType: FENE" << endl;
  } else {
    cout << "DPM2D::setMonomerType: please specify valid monomerType: harmonic or FENE" << endl;
  }
	syncSimControlToDevice();
}

simControlStruct::monomerEnum DPM2D::getMonomerType() {
	syncSimControlFromDevice();
	return simControl.monomerType;
}

// TODO: add error checks for all the getters and setters
void DPM2D::setDimBlock(long dimBlock_) {
	dimBlock = dimBlock_;
	dimGrid = (numVertices + dimBlock - 1) / dimBlock;
	partDimGrid = (numParticles + dimBlock - 1) / dimBlock;
}

long DPM2D::getDimBlock() {
	return dimBlock;
}

void DPM2D::setCellDimGridBlock() {
	cellDimBlock.x = 256;
  cellDimBlock.y = 256;
	cellDimGrid.x = (numCells + cellDimBlock.x - 1) / cellDimBlock.x;
  cellDimGrid.y = (numCells + cellDimBlock.y - 1) / cellDimBlock.y;
  cout << "cellDimBlock: " << cellDimBlock.x << " " << cellDimBlock.y << " " << cellDimBlock.z << endl;
  cout << "cellDimGrid: " << cellDimGrid.x << " " << cellDimGrid.y << " " << cellDimGrid.z << endl;
  myKernel<<<cellDimGrid, cellDimBlock>>>();

  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

void DPM2D::setNDim(long nDim_) {
  nDim = nDim_;
  cudaMemcpyToSymbol(d_nDim, &nDim, sizeof(nDim));
}

long DPM2D::getNDim() {
  long nDimFromDevice;
  cudaMemcpyFromSymbol(&nDimFromDevice, d_nDim, sizeof(d_nDim));
	return nDimFromDevice;
}

void DPM2D::setNumParticles(long numParticles_) {
  numParticles = numParticles_;
  cudaMemcpyToSymbol(d_numParticles, &numParticles, sizeof(numParticles));
}

long DPM2D::getNumParticles() {
  long numParticlesFromDevice;
  cudaMemcpyFromSymbol(&numParticlesFromDevice, d_numParticles, sizeof(d_numParticles));
	return numParticlesFromDevice;
}

void DPM2D::setNumVertices(long numVertices_) {
  numVertices = numVertices_;
  cudaMemcpyToSymbol(d_numVertices, &(numVertices), sizeof(numVertices));
  setDimBlock(dimBlock); // recalculate dimGrid
}

long DPM2D::getNumVertices() {
  long numVerticesFromDevice;
  cudaMemcpyFromSymbol(&numVerticesFromDevice, d_numVertices, sizeof(d_numVertices));
	return numVerticesFromDevice;
}

void DPM2D::setNumVertexPerParticle(long numVertexPerParticle_) {
  numVertexPerParticle = numVertexPerParticle_;
  cudaMemcpyToSymbol(d_numVertexPerParticle, &numVertexPerParticle, sizeof(numVertexPerParticle));
}

long DPM2D::getNumVertexPerParticle() {
  long numVertexPerParticleFromDevice;
  cudaMemcpyFromSymbol(&numVertexPerParticleFromDevice, d_numVertexPerParticle, sizeof(d_numVertexPerParticle));
  return numVertexPerParticleFromDevice;
}

void DPM2D::setNumVertexInParticleList(thrust::host_vector<long> &numVertexInParticleList_) {
  if(numVertexInParticleList_.size() == ulong(numParticles)) {
    d_numVertexInParticleList = numVertexInParticleList_;
    long* numVertexInParticleList = thrust::raw_pointer_cast(&(d_numVertexInParticleList[0]));
    cudaMemcpyToSymbol(d_numVertexInParticleListPtr, &numVertexInParticleList, sizeof(numVertexInParticleList));
  } else {
    cout << "DPM2D::setNumVertexInParticleList: size of numVertexInParticleList does not match numParticles" << endl;
  }
}

thrust::host_vector<long> DPM2D::getNumVertexInParticleList() {
  thrust::host_vector<long> numVertexInParticleListFromDevice;
  if(d_numVertexInParticleList.size() == ulong(numParticles)) {
    cudaMemcpyFromSymbol(&d_numVertexInParticleList, d_numVertexInParticleListPtr, sizeof(d_numVertexInParticleListPtr));
    numVertexInParticleListFromDevice = d_numVertexInParticleList;
  } else {
    cout << "DPM2D::getNumVertexInParticleList: size of numVertexInParticleList from device does not match numParticles" << endl;
  }
  return numVertexInParticleListFromDevice;
}

// the length scale is always set to be the sqrt of the first particle area
void DPM2D::setLengthScale() {
  rho0 = sqrt((thrust::reduce(d_a0.begin(), d_a0.end(), double(0), thrust::plus<double>()))/numParticles); // set dimensional factor
  //cout << " lengthscale: " << rho0 << endl;
  cudaMemcpyToSymbol(d_rho0, &rho0, sizeof(rho0));
}

void DPM2D::setLengthScaleToOne() {
  rho0 = 1.; // for soft particles
  cudaMemcpyToSymbol(d_rho0, &rho0, sizeof(rho0));
}

//TODO: error messages for all the vector getters and setters
void DPM2D::setBoxSize(thrust::host_vector<double> &boxSize_) {
  if(boxSize_.size() == ulong(nDim)) {
    d_boxSize = boxSize_;
    double* boxSize = thrust::raw_pointer_cast(&(d_boxSize[0]));
    cudaMemcpyToSymbol(d_boxSizePtr, &boxSize, sizeof(boxSize));
    h_boxSize = d_boxSize; //HOST
  } else {
    cout << "DPM2D::setBoxSize: size of boxSize does not match nDim" << endl;
  }
}

thrust::host_vector<double> DPM2D::getBoxSize() {
  thrust::host_vector<double> boxSizeFromDevice;
  if(d_boxSize.size() == ulong(nDim)) {
    cudaMemcpyFromSymbol(&d_boxSize, d_boxSizePtr, sizeof(d_boxSizePtr));
    boxSizeFromDevice = d_boxSize;
  } else {
    cout << "DPM2D::getBoxSize: size of boxSize from device does not match nDim" << endl;
  }
  return boxSizeFromDevice;
}

//**************************** shape variables *******************************//
void DPM2D::setVertexRadii(thrust::host_vector<double> &rad_) {
  d_rad = rad_;
  if(getSimulationType() == simControlStruct::simulationEnum::omp || getSimulationType() == simControlStruct::simulationEnum::cpu) {
    cout << "DPM2D::setVertexRadii:: initialized positions on host" << endl;
    h_rad = d_rad;
  }
}

thrust::host_vector<double> DPM2D::getVertexRadii() {
  thrust::host_vector<double> radFromDevice;
  radFromDevice = d_rad;
  return radFromDevice;
}

void DPM2D::setRestAreas(thrust::host_vector<double> &a0_) {
  d_a0 = a0_;
}

thrust::host_vector<double> DPM2D::getRestAreas() {
  thrust::host_vector<double> a0FromDevice;
  a0FromDevice = d_a0;
  return a0FromDevice;
}

void DPM2D::setRestLengths(thrust::host_vector<double> &l0_) {
  d_l0 = l0_;
}

thrust::host_vector<double> DPM2D::getRestLengths() {
  thrust::host_vector<double> l0FromDevice;
  l0FromDevice = d_l0;
  return l0FromDevice;
}

void DPM2D::setRestAngles(thrust::host_vector<double> &theta0_) {
  d_theta0 = theta0_;
}

thrust::host_vector<double> DPM2D::getRestAngles() {
  thrust::host_vector<double> theta0FromDevice;
  theta0FromDevice = d_theta0;
  return theta0FromDevice;
}

thrust::host_vector<double> DPM2D::getSegmentLengths() {
  thrust::host_vector<double> lengthFromDevice;
  lengthFromDevice = d_length;
  return lengthFromDevice;
}

thrust::host_vector<double> DPM2D::getSegmentAngles() {
  thrust::host_vector<double> thetaFromDevice;
  thetaFromDevice = d_theta;
  return thetaFromDevice;
}

void DPM2D::setAreas(thrust::host_vector<double> &area_) {
  d_area = area_;
}

thrust::host_vector<double> DPM2D::getAreas() {
  thrust::host_vector<double> areaFromDevice;
  areaFromDevice = d_area;
  return areaFromDevice;
}

thrust::host_vector<double> DPM2D::getPerimeters() {
  thrust::host_vector<double> perimeterFromDevice;
  perimeterFromDevice = d_perimeter;
  return perimeterFromDevice;
}

double DPM2D::getMeanParticleSize() {
  return sqrt(thrust::reduce(d_a0.begin(), d_a0.end(), double(0), thrust::plus<double>()) / (PI * numParticles));
}

double DPM2D::getMeanParticleSigma() {
  return thrust::reduce(d_particleRad.begin(), d_particleRad.end(), double(0), thrust::plus<double>()) / numParticles;
}

double DPM2D::getMinParticleSigma() {
  return thrust::reduce(d_particleRad.begin(), d_particleRad.end(), double(1), thrust::minimum<double>());
}

double DPM2D::getMeanVertexRadius() {
  return thrust::reduce(d_rad.begin(), d_rad.end(), double(0), thrust::plus<double>()) / numVertices;
}

double DPM2D::getMinVertexRadius() {
  return thrust::reduce(d_rad.begin(), d_rad.end(), double(1), thrust::minimum<double>());
}

double DPM2D::getVertexRadius() {
  return d_rad[0]; // all vertices have the same radius
}


//*************************** particle variables *****************************//
void DPM2D::calcParticleShape() {
  // area and perimeter pointers
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  double *length = thrust::raw_pointer_cast(&d_length[0]);
  double *theta = thrust::raw_pointer_cast(&d_theta[0]);
  double *area = thrust::raw_pointer_cast(&d_area[0]);
  double *perimeter = thrust::raw_pointer_cast(&d_perimeter[0]);

  kernelCalcParticleShape<<<dimGrid, dimBlock>>>(pos, length, theta, area, perimeter);
}

void DPM2D::calcParticlePositions() {
  // vertex and particle position pointers
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  double *particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);

  kernelCalcParticlePositions<<<dimGrid, dimBlock>>>(pos, particlePos);
}

void DPM2D::setDefaultParticleRadii() {
  for (long particleId = 0; particleId < numParticles; particleId++) {
    d_particleRad[particleId] = sqrt(d_a0[particleId]/PI);
  }
}

void DPM2D::setParticleRadii(thrust::host_vector<double> &particleRad_) {
  d_particleRad = particleRad_;
}

thrust::host_vector<double> DPM2D::getParticleRadii() {
  thrust::host_vector<double> particleRadFromDevice;
  particleRadFromDevice = d_particleRad;
  return particleRadFromDevice;
}

void DPM2D::setParticlePositions(thrust::host_vector<double> &particlePos_) {
  d_particlePos = particlePos_;
}

thrust::host_vector<double> DPM2D::getParticlePositions() {
  thrust::host_vector<double> particlePosFromDevice;
  particlePosFromDevice = d_particlePos;
  return particlePosFromDevice;
}

void DPM2D::setParticleInitialPositions() {
  d_particleInitPos = d_particlePos;
}

void DPM2D::resetParticleLastPositions() {
  d_particleLastPos = d_particlePos;
}

void DPM2D::resetParticleLastAngles() {
  d_particleLastAngle = d_particleAngle;
}

void DPM2D::setParticleVelocities(thrust::host_vector<double> &particleVel_) {
  d_particleVel = particleVel_;
}

thrust::host_vector<double> DPM2D::getParticleVelocities() {
  thrust::host_vector<double> particleVelFromDevice;
  particleVelFromDevice = d_particleVel;
  return particleVelFromDevice;
}

void DPM2D::setParticleForces(thrust::host_vector<double> &particleForce_) {
  d_particleForce = particleForce_;
}

thrust::host_vector<double> DPM2D::getParticleForces() {
  thrust::host_vector<double> particleForceFromDevice;
  particleForceFromDevice = d_particleForce;
  return particleForceFromDevice;
}

thrust::host_vector<double> DPM2D::getParticleEnergies() {
  thrust::host_vector<double> particleEnergyFromDevice;
  particleEnergyFromDevice = d_particleEnergy;
  return particleEnergyFromDevice;
}

void DPM2D::setParticleAngles(thrust::host_vector<double> &particleAngle_) {
  d_particleAngle = particleAngle_;
}

thrust::host_vector<double> DPM2D::getParticleAngles() {
  thrust::host_vector<double> particleAngleFromDevice;
  particleAngleFromDevice = d_particleAngle;
  return particleAngleFromDevice;
}

void DPM2D::setParticleAngularVelocities(thrust::host_vector<double> &particleAngvel_) {
  d_particleAngvel = particleAngvel_;
}

thrust::host_vector<double> DPM2D::getParticleAngularVelocities() {
  thrust::host_vector<double> particleAngvelFromDevice;
  particleAngvelFromDevice = d_particleAngvel;
  return particleAngvelFromDevice;
}

void DPM2D::setParticleTorques(thrust::host_vector<double> &particleTorque_) {
  d_particleTorque = particleTorque_;
}

thrust::host_vector<double> DPM2D::getParticleTorques() {
  thrust::host_vector<double> particleTorqueFromDevice;
  particleTorqueFromDevice = d_particleTorque;
  return particleTorqueFromDevice;
}

//************************** dynamical variables *****************************//
void DPM2D::setVertexPositions(thrust::host_vector<double> &pos_) {
  d_pos = pos_;
  if(getSimulationType() == simControlStruct::simulationEnum::omp || getSimulationType() == simControlStruct::simulationEnum::cpu) {
    cout << "DPM2D::setVertexPositions:: initialized positions on host" << endl;
    h_pos = d_pos;
  }
}

thrust::host_vector<double> DPM2D::getVertexPositions() {
  thrust::host_vector<double> posFromDevice;
  posFromDevice = d_pos;
  return posFromDevice;
}

void DPM2D::resetLastPositions() {
  d_lastPos = d_pos;
}

void DPM2D::setInitialPositions() {
  d_initialPos = d_pos;
}

void DPM2D::setVertexVelocities(thrust::host_vector<double> &vel_) {
  d_vel = vel_;
}

thrust::host_vector<double> DPM2D::getVertexVelocities() {
  thrust::host_vector<double> velFromDevice;
  velFromDevice = d_vel;
  return velFromDevice;
}

void DPM2D::setVertexForces(thrust::host_vector<double> &force_) {
  d_force = force_;
}

thrust::host_vector<double> DPM2D::getVertexForces() {
  thrust::host_vector<double> forceFromDevice;
  forceFromDevice = d_force;
  return forceFromDevice;
}

void DPM2D::setVertexTorques(thrust::host_vector<double> &torque_) {
  d_torque = torque_;
}

thrust::host_vector<double> DPM2D::getVertexTorques() {
  thrust::host_vector<double> torqueFromDevice;
  torqueFromDevice = d_torque;
  return torqueFromDevice;
}

thrust::host_vector<double> DPM2D::getPerParticleStressTensor() {
  calcPerParticleStressTensor();
  thrust::host_vector<double> perParticleStressFromDevice;
  perParticleStressFromDevice = d_perParticleStress;
  return perParticleStressFromDevice;
}

thrust::host_vector<double> DPM2D::getStressTensor() {
  calcStressTensor();
  thrust::host_vector<double> stressFromDevice;
  stressFromDevice = d_stress;
  return stressFromDevice;
}

double DPM2D::getPressure() {
  calcStressTensor();
  double pressure = 0;
  for (long dim = 0; dim < nDim; dim++) {
    pressure += d_stress[dim * nDim + dim];
  }
  return pressure / (nDim * numVertices);
}

// return the sum of force magnitudes
double DPM2D::getTotalForceMagnitude() {
  thrust::device_vector<double> forceSquared(d_force.size());
  // compute squared velocities
  thrust::transform(d_force.begin(), d_force.end(), forceSquared.begin(), square());
  // sum squares
  double totalForceMagnitude = sqrt(thrust::reduce(forceSquared.begin(), forceSquared.end(), double(0), thrust::plus<double>()) / (numVertices * nDim));
  forceSquared.clear();
  return totalForceMagnitude;
}

// return the maximum force magnitude
double DPM2D::getMaxUnbalancedForce() {
  thrust::device_vector<double> forceSquared(d_force.size());
  // compute squared velocities
  thrust::transform(d_force.begin(), d_force.end(), forceSquared.begin(), square());

  double maxUnbalancedForce = sqrt(thrust::reduce(forceSquared.begin(), forceSquared.end(), double(-1), thrust::maximum<double>()));
  forceSquared.clear();
  return maxUnbalancedForce;
}

thrust::host_vector<long> DPM2D::getMaxNeighborList() {
  thrust::host_vector<long> maxNeighborListFromDevice;
  maxNeighborListFromDevice = d_maxNeighborList;
  return maxNeighborListFromDevice;
}

thrust::host_vector<long> DPM2D::getNeighbors() {
  thrust::host_vector<long> neighborListFromDevice;
  neighborListFromDevice = d_neighborList;
  return neighborListFromDevice;
}

thrust::host_vector<long> DPM2D::getSmoothNeighbors() {
  return h_smoothNeighborList;
}

thrust::host_vector<long> DPM2D::getLinkedList() {
  return h_linkedList;
}

thrust::host_vector<long> DPM2D::getListHeader() {
  return h_header;
}

thrust::host_vector<long> DPM2D::getCellIndexList() {
  return h_cellIndexList;
}

thrust::host_vector<long> DPM2D::getContacts() {
  thrust::host_vector<long> contactListFromDevice;
  contactListFromDevice = d_contactList;
  return contactListFromDevice;
}

void DPM2D::printNeighbors() {
  for (long vertexId = 0; vertexId < numVertices; vertexId++) {
    cout << "vertexId: " << vertexId << " list of neighbors: ";
    for (long neighborId = 0; neighborId < d_maxNeighborList[vertexId]; neighborId++) {
      cout << d_neighborList[vertexId * neighborListSize + neighborId] << " ";
    }
    cout << endl;
  }
}

void DPM2D::printContacts() {
  for (long particleId = 0; particleId < numParticles; particleId++) {
    cout << "particleId: " << particleId << " list of contacts: ";
    for (long contactId = 0; contactId < d_numContacts[particleId]; contactId++) {
      cout << d_contactList[particleId * contactLimit + contactId] << " ";
    }
    cout << endl;
  }
}

double DPM2D::getPotentialEnergy() {
  double epot = thrust::reduce(d_energy.begin(), d_energy.end(), double(0), thrust::plus<double>());
  switch (simControl.interactionType) {
    case simControlStruct::interactionEnum::vertexVertex:
    return epot;
    break;
    case simControlStruct::interactionEnum::cellSmooth:
    epot += thrust::reduce(d_particleEnergy.begin(), d_particleEnergy.end(), double(0), thrust::plus<double>());
    return epot;
    break;
    case simControlStruct::interactionEnum::vertexSmooth:
    epot += thrust::reduce(d_particleEnergy.begin(), d_particleEnergy.end(), double(0), thrust::plus<double>());
    return epot;
    break;
    default:
    return 0;
    break;
  }
}

double DPM2D::getKineticEnergy() {
  thrust::device_vector<double> velSquared(d_vel.size());
  // compute squared velocities
  thrust::transform(d_vel.begin(), d_vel.end(), velSquared.begin(), square());
  // sum squares
  return 0.5 * thrust::reduce(velSquared.begin(), velSquared.end());
}

double DPM2D::getEnergy() {
  return (getPotentialEnergy() + getKineticEnergy());
}

double DPM2D::getTemperature() {
  return 2. * getKineticEnergy() / (nDim * numVertices);
}

double DPM2D::getTotalEnergy() {
  double etot = getPotentialEnergy();
  etot += getKineticEnergy();
  return etot;
}

void  DPM2D::adjustKineticEnergy(double prevEtot) {
  double scale, ekin = getKineticEnergy();
  double deltaEtot = getPotentialEnergy() + ekin;
  deltaEtot -= prevEtot;
  if(ekin > deltaEtot) {
    scale = sqrt((ekin - deltaEtot) / ekin);
    //cout << "deltaEtot: " << deltaEtot << " ekin - deltaEtot: " << ekin - deltaEtot << " scale: " << scale << endl;
    long s_nDim(nDim);
    auto r = thrust::counting_iterator<long>(0);
    double* vel = thrust::raw_pointer_cast(&d_vel[0]);

    auto adjustVel = [=] __device__ (long vId) {
      #pragma unroll (MAXDIM)
      for (long dim = 0; dim < s_nDim; dim++) {
        vel[vId * s_nDim + dim] *= scale;
      }
    };

    cout << "DPM2D::adjustKineticEnergy:: scale: " << scale << endl;
    thrust::for_each(r, r + numVertices, adjustVel);
  } else {
    cout << "DPM2D::adjustKineticEnergy:: kinetic energy is less then change in total energy - no adjustment is made" << endl;
  }
}

double DPM2D::getPhi() {
  double phi = double(thrust::reduce(d_area.begin(), d_area.end(), double(0), thrust::plus<double>()));
  //cout << "\n area: " << phi / (d_boxSize[0] * d_boxSize[1]);
  // add vertex areas
  thrust::device_vector<double> d_vertexArea(d_area.size());
  double *vertexArea = thrust::raw_pointer_cast(&d_vertexArea[0]);
  const double *rad = thrust::raw_pointer_cast(&d_rad[0]);
  kernelCalcVertexArea<<<dimGrid,dimBlock>>>(rad, vertexArea);
  phi += PI * thrust::reduce(d_vertexArea.begin(), d_vertexArea.end(), double(0), thrust::plus<double>());
  //cout << " vertex: " << phi / (d_boxSize[0] * d_boxSize[1]) << endl;
  return phi / (d_boxSize[0] * d_boxSize[1]);
}

double DPM2D::getPreferredPhi() {
  double phi = double(thrust::reduce(d_a0.begin(), d_a0.end(), double(0), thrust::plus<double>()));
  // add vertex areas
  thrust::device_vector<double> d_vertexArea(d_area.size());
  double *vertexArea = thrust::raw_pointer_cast(&d_vertexArea[0]);
  const double *rad = thrust::raw_pointer_cast(&d_rad[0]);
  kernelCalcVertexArea<<<dimGrid,dimBlock>>>(rad, vertexArea);
  phi += PI * thrust::reduce(d_vertexArea.begin(), d_vertexArea.end(), double(0), thrust::plus<double>());
  return phi / (d_boxSize[0] * d_boxSize[1]);
}

double DPM2D::getRigidPhi() {
  double phi = double(thrust::reduce(d_a0.begin(), d_a0.end(), double(0), thrust::plus<double>()));
  return phi / (d_boxSize[0] * d_boxSize[1]);
}

double DPM2D::getParticlePhi() {
  thrust::device_vector<double> d_radSquared(numParticles);
  thrust::transform(d_particleRad.begin(), d_particleRad.end(), d_radSquared.begin(), square());
  return thrust::reduce(d_radSquared.begin(), d_radSquared.end(), double(0), thrust::plus<double>()) * PI / (d_boxSize[0] * d_boxSize[1]);
}

double DPM2D::get3DParticlePhi() {
  thrust::device_vector<double> d_volume(numParticles);
  thrust::fill(d_volume.begin(), d_volume.end(), double(1));
  long p_nDim(nDim);
  auto r = thrust::counting_iterator<long>(0);
  double *volume = thrust::raw_pointer_cast(&d_volume[0]);
  const double *rad = thrust::raw_pointer_cast(&d_particleRad[0]);

  auto computeVolume = [=] __device__ (long particleId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < p_nDim; dim++) {
      volume[particleId] *= rad[particleId];
    }
  };

  thrust::for_each(r, r + numParticles, computeVolume);
  return thrust::reduce(d_volume.begin(), d_volume.end(), double(0), thrust::plus<double>()) * 3 * PI / (4 * d_boxSize[0] * d_boxSize[1] * d_boxSize[2]);
}

double DPM2D::getVertexMSD() {
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  const double *initialPos = thrust::raw_pointer_cast(&d_initialPos[0]);
  double *delta = thrust::raw_pointer_cast(&d_delta[0]);
  kernelCalcVertexDistanceSq<<<dimGrid,dimBlock>>>(pos, initialPos, delta);
  return thrust::reduce(d_delta.begin(), d_delta.end(), double(0), thrust::plus<double>()) / (numVertices * d_boxSize[0] * d_boxSize[1]);
}

double DPM2D::getParticleMSD() {
  const double *particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *particleInitPos = thrust::raw_pointer_cast(&d_particleInitPos[0]);
  double *particleDelta = thrust::raw_pointer_cast(&d_particleDelta[0]);
  kernelCalcParticleDistanceSq<<<partDimGrid,dimBlock>>>(particlePos, particleInitPos, particleDelta);
  return thrust::reduce(d_particleDelta.begin(), d_particleDelta.end(), double(0), thrust::plus<double>()) / (numParticles * d_boxSize[0] * d_boxSize[1]);
}

double DPM2D::setDisplacementCutoff(double cutoff_, double size_) {
  switch (simControl.potentialType) {
    case simControlStruct::potentialEnum::harmonic:
    cutDistance = 1;
    break;
    case simControlStruct::potentialEnum::lennardJones:
    cutDistance = LJcutoff;
    break;
    case simControlStruct::potentialEnum::wca:
    cutDistance = WCAcut;
    break;
    case simControlStruct::potentialEnum::adhesive:
    cutDistance = 1 + l2;
    break;
  }
  cutDistance += cutoff_;
  cutoff = cutoff_ * size_;
  switch (simControl.neighborType) {
    case simControlStruct::neighborEnum::neighbor:
    cout << "DPM2D::setDisplacementCutoff - cutDistance: " << cutDistance << " cutoff: " << cutoff << endl;
    break;
    case simControlStruct::neighborEnum::cell:
    cellSize = initCells(getNumVertices(), cutoff);
    cout << "DPM2D::setDisplacementCutoff - cellSize: " << cellSize << " numCells: " << numCells * numCells << endl;
    break;
    case simControlStruct::neighborEnum::allToAll:
    break;
  }
  return cutDistance;
}

void DPM2D::resetUpdateCount() {
  updateCount = double(0);
  //cout << "DPM2D::resetUpdateCount - updatCount " << updateCount << endl;
}

long DPM2D::getUpdateCount() {
  return updateCount;
}

double DPM2D::getMaxDisplacement() {
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  const double *lastPos = thrust::raw_pointer_cast(&d_lastPos[0]);
  double *disp = thrust::raw_pointer_cast(&d_disp[0]);
  kernelCalcVertexDisplacement<<<dimGrid,dimBlock>>>(pos, lastPos, disp);
  return thrust::reduce(d_disp.begin(), d_disp.end(), double(-1), thrust::maximum<double>());
}

void DPM2D::checkMaxDisplacement() {
  double maxDelta;
  maxDelta = getMaxDisplacement();
  if(3*maxDelta > cutoff) {
    calcNeighborList(cutDistance);
    resetLastPositions();
    updateCount += 1;
    //cout << "DPM2D::checkMaxDisplacement - updated neighbors, maxDelta: " << maxDelta << " cutoff: " << cutoff << endl;
  }
}

void DPM2D::checkDisplacement() {
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  const double *lastPos = thrust::raw_pointer_cast(&d_lastPos[0]);
  thrust::device_vector<int> recalcFlag(d_rad.size());
  thrust::fill(recalcFlag.begin(), recalcFlag.end(), int(0));
  int *flag = thrust::raw_pointer_cast(&recalcFlag[0]);
  kernelCheckVertexDisplacement<<<dimGrid,dimBlock>>>(pos, lastPos, flag, cutoff);
  int sumFlag = thrust::reduce(recalcFlag.begin(), recalcFlag.end(), int(0), thrust::plus<int>());
  if(sumFlag != 0) {
    calcNeighborList(cutDistance);
    resetLastPositions();
    updateCount += 1;
  }
}

void DPM2D::checkNeighbors() {
  switch (simControl.neighborType) {
    case simControlStruct::neighborEnum::neighbor:
    checkDisplacement();
    //checkMaxDisplacement();
    break;
    case simControlStruct::neighborEnum::cell:
    fillLinkedList();
    break;
    case simControlStruct::neighborEnum::allToAll:
    break;
  }
  //if(getSimulationType() == simControlStruct::simulationEnum::omp || getSimulationType() == simControlStruct::simulationEnum::cpu) {
  //  calcSmoothNeighbors();
  //}
}


double DPM2D::getParticleMaxDisplacement() {
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *pLastPos = thrust::raw_pointer_cast(&d_particleLastPos[0]);
  double *pDisp = thrust::raw_pointer_cast(&d_particleDisp[0]);
  kernelCalcParticleDisplacement<<<partDimGrid,dimBlock>>>(pPos, pLastPos, pDisp);
  return thrust::reduce(d_particleDisp.begin(), d_particleDisp.end(), double(-1), thrust::maximum<double>());
}

void DPM2D::checkParticleMaxDisplacement() {
  double maxDelta;
  maxDelta = getParticleMaxDisplacement();
  if(3*maxDelta > cutoff) {
    calcNeighborList(cutDistance);
    resetLastPositions();
    //cout << "DPM2D::checParticleMaxDisplacement - updated neighbors, maxDelta: " << maxDelta << " cutoff: " << cutoff << endl;
  }
}

void DPM2D::checkParticleNeighbors() {
  switch (simControl.neighborType) {
    case simControlStruct::neighborEnum::neighbor:
    checkParticleMaxDisplacement();
    break;
    default:
    calcParticleNeighborList(cutDistance);
  }
}

double DPM2D::getDeformableWaveNumber() {
  return PI / (2. * sqrt(d_boxSize[0] * d_boxSize[1] * getPreferredPhi() / (PI * numParticles)));
}

double DPM2D::getRigidWaveNumber() {
  return PI / (2. * sqrt(d_boxSize[0] * d_boxSize[1] * getPreferredPhi() / (PI * numParticles)));
}

double DPM2D::getSoftWaveNumber() {
  if(nDim == 2) {
    return PI / (2. * sqrt(d_boxSize[0] * d_boxSize[1] * getParticlePhi() / (PI * numParticles)));
  } else if(nDim == 3) {
    return PI / (2. * sqrt(d_boxSize[0] * d_boxSize[1] * get3DParticlePhi() / (PI * numParticles)));
  } else {
    cout << "DPM2D::getSoftWaveNumber: this function works only for dim = 2 and 3" << endl;
    return 0;
  }
}

double DPM2D::getVertexISF() {
  double vertexWaveNumber = PI / (2 * d_rad[0]);
  thrust::device_vector<double> d_vertexSF(numVertices);
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  const double *initialPos = thrust::raw_pointer_cast(&d_initialPos[0]);
  double *vertexSF = thrust::raw_pointer_cast(&d_vertexSF[0]);
  kernelCalcVertexScatteringFunction<<<dimGrid,dimBlock>>>(pos, initialPos, vertexSF, vertexWaveNumber);
  return thrust::reduce(d_vertexSF.begin(), d_vertexSF.end(), double(0), thrust::plus<double>()) / numVertices;
}

double DPM2D::getParticleISF(double waveNumber_) {
  thrust::device_vector<double> d_particleSF(numParticles);
  const double *particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *particleInitPos = thrust::raw_pointer_cast(&d_particleInitPos[0]);
  double *particleSF = thrust::raw_pointer_cast(&d_particleSF[0]);
  kernelCalcParticleScatteringFunction<<<partDimGrid,dimBlock>>>(particlePos, particleInitPos, particleSF, waveNumber_);
  return thrust::reduce(d_particleSF.begin(), d_particleSF.end(), double(0), thrust::plus<double>()) / numParticles;
}

double DPM2D::getAreaFluctuation() {
  thrust::device_vector<double> deltaA(d_area.size());
  thrust::device_vector<double> deltaASq(d_area.size());
  thrust::fill(deltaA.begin(), deltaA.end(), double(0));
  thrust::transform(d_area.begin(), d_area.end(), d_a0.begin(), deltaA.begin(), thrust::minus<double>());
  thrust::transform(deltaA.begin(), deltaA.end(), deltaASq.begin(), square());
  return sqrt(thrust::reduce(deltaASq.begin(), deltaASq.end(), double(0), thrust::plus<double>()) / numParticles);
}

//************************ initilization functions ***************************//
void DPM2D::setMonoSizeDistribution() {
  thrust::fill(d_numVertexInParticleList.begin(), d_numVertexInParticleList.end(), numVertexPerParticle);
  long* numVertexInParticleList = thrust::raw_pointer_cast(&d_numVertexInParticleList[0]);
  cudaMemcpyToSymbol(d_numVertexInParticleListPtr, &numVertexInParticleList, sizeof(numVertexInParticleList));
  initParticleIdList();
  for (long particleId = 0; particleId < numParticles; particleId++) {
    auto numVertexInParticle = d_numVertexInParticleList[particleId];
    d_a0[particleId] = 1;
    d_area[particleId] = d_a0[particleId];
    auto calA0 = numVertexInParticle * tan(PI / numVertexInParticle) / PI; //calA0 = 1
    for (long vertexId = 0; vertexId < numVertexInParticle; vertexId++) {
      d_l0[d_firstVertexInParticleId[particleId] + vertexId] = 2. * sqrt(PI * calA0 * d_a0[particleId]) / numVertexInParticle;
  		d_theta0[d_firstVertexInParticleId[particleId] + vertexId] = 2. * PI / numVertexInParticle;
  		d_rad[d_firstVertexInParticleId[particleId] + vertexId] = 0.5 * d_l0[d_firstVertexInParticleId[particleId] + vertexId];
    }
  }
  h_rad = d_rad; //HOST
}

//void DPM2D::setBiSizeDistribution();

void DPM2D::setPolySizeDistribution(double calA0_, double polyDispersity) {
  calA0 = calA0_;
  double r1, r2, randNum, calA0temp;
  double numVertexInParticle, minVertexInParticle = numVertexPerParticle; // default
  numVertices = 0;
  // generate polydisperse number of vertices per particle
  for (long particleId = 0; particleId < numParticles; particleId++) {
    r1 = drand48();
    r2 = drand48();
    randNum = sqrt(-2. * log(r1)) * cos(2. * PI * r2);
    numVertexInParticle = floor(polyDispersity * numVertexPerParticle * randNum + numVertexPerParticle);
    if(numVertexInParticle < minVertexInParticle) {
      numVertexInParticle = minVertexInParticle;
    }
    // each particle has at least numVertexPerParticle vertices
    d_numVertexInParticleList[particleId] = numVertexInParticle;
    numVertices += numVertexInParticle;
  }
  cout << "DPM2D::setPolySizeDistribution: numVertices: " << numVertices << endl;
  cudaMemcpyToSymbol(d_numVertices, &(numVertices), sizeof(numVertices));
  setDimBlock(dimBlock); // recalculate dimGrid
  long* numVertexInParticleList = thrust::raw_pointer_cast(&d_numVertexInParticleList[0]);
  cudaMemcpyToSymbol(d_numVertexInParticleListPtr, &numVertexInParticleList, sizeof(numVertexInParticleList));

  // initialize everything else
  initParticleIdList();
  // we changed numVertices so we need to resize variables
  initShapeVariables(numVertices, numParticles);
  initDynamicalVariables(numVertices);
  initNeighbors(numVertices);
  syncNeighborsToDevice();
  for (long particleId = 0; particleId < numParticles; particleId++) {
    numVertexInParticle = d_numVertexInParticleList[particleId];
    d_a0[particleId] = (numVertexInParticle / minVertexInParticle) * (numVertexInParticle / minVertexInParticle);
    d_area[particleId] = d_a0[particleId];
    calA0temp = calA0 * numVertexInParticle * tan(PI / numVertexInParticle) / PI;
    for (long vertexId = 0; vertexId < numVertexInParticle; vertexId++) {
      d_l0[d_firstVertexInParticleId[particleId] + vertexId] = 2. * sqrt(PI * calA0temp * d_a0[particleId]) / numVertexInParticle;
  		d_theta0[d_firstVertexInParticleId[particleId] + vertexId] = 2. * PI / numVertexInParticle;
  		d_rad[d_firstVertexInParticleId[particleId] + vertexId] = 0.5 * d_l0[d_firstVertexInParticleId[particleId] + vertexId];
      //cout << "vertexId: " << d_firstVertexInParticleId[particleId] + vertexId << " l0: " << d_l0[d_firstVertexInParticleId[particleId] + vertexId] << " rad: " << d_rad[d_firstVertexInParticleId[particleId] + vertexId] << endl;
    }
  }
  h_rad = d_rad; //HOST
}

void DPM2D::setSinusoidalRestAngles(double thetaA, double thetaK) {
  double thetaR;
  for (long particleId = 0; particleId < numParticles; particleId++) {
    thetaR = 2. * PI / d_numVertexInParticleList[particleId];
    for (long vertexId = 0; vertexId < d_numVertexInParticleList[particleId]; vertexId++) {
      d_theta0[d_firstVertexInParticleId[particleId] + vertexId] = thetaA * thetaR * cos(thetaR * thetaK * vertexId);
    }
  }
}

// this works only for a square box
void DPM2D::setRandomParticles(double phi0, double extraRad_) {
  double boxLength = 1., scale = sqrt(getPreferredPhi() / phi0), extraRad = extraRad_;
  for (long dim = 0; dim < nDim; dim++) {
    d_boxSize[dim] = boxLength; // sqrt for 2d
  }
  double* boxSize = thrust::raw_pointer_cast(&(d_boxSize[0]));
  cudaMemcpyToSymbol(d_boxSizePtr, &boxSize, sizeof(boxSize));
  // extract random positions and radii
  double areaSum = 0;
  for (long particleId = 0; particleId < numParticles; particleId++) {
    d_a0[particleId] /= (scale * scale);
    d_area[particleId] = d_a0[particleId];
    for(long dim = 0; dim < nDim; dim++) {
      d_particlePos[particleId * nDim + dim] = d_boxSize[dim] * drand48();
    }
    d_particleRad[particleId] = extraRad * sqrt((2. * d_a0[particleId]) / (d_numVertexInParticleList[particleId] * sin(2. * PI / d_numVertexInParticleList[particleId])));
    areaSum += PI * d_particleRad[particleId] * d_particleRad[particleId];
  }
  for(long vertexId = 0; vertexId < numVertices; vertexId++) {
    d_l0[vertexId] /= scale;
    d_rad[vertexId] /= scale;
  }
  h_rad = d_rad; //HOST
  // need to set this otherwise forces are zeros
  setLengthScaleToOne();
  cout << "DPM2D::setRandomParticles: particle packing fraction: " << getPreferredPhi() << " " << areaSum/(boxSize[0] * boxSize[1]) << endl;
}

void DPM2D::setScaledRandomParticles(double phi0, double extraRad_) {
  thrust::host_vector<double> boxSize(nDim);
  double scale, extraRad = extraRad_;
  scale = sqrt(getPreferredPhi() / phi0);
  boxSize[0] = scale;
  boxSize[1] = scale;
  setBoxSize(boxSize);
  // extract random positions and radii
  double areaSum = 0;
  for (long particleId = 0; particleId < numParticles; particleId++) {
    for(long dim = 0; dim < nDim; dim++) {
      d_particlePos[particleId * nDim + dim] = d_boxSize[dim] * drand48();
    }
    d_particleRad[particleId] = extraRad * sqrt((2. * d_a0[particleId]) / (d_numVertexInParticleList[particleId] * sin(2. * PI / d_numVertexInParticleList[particleId])));
    areaSum += PI * d_particleRad[particleId] * d_particleRad[particleId];
  }
  // need to set this otherwise forces are zeros
  setLengthScaleToOne();
  cout << "DPM2D::setScaledRandomParticles: packing fraction: " << getPreferredPhi() << " " << areaSum / (boxSize[0] * boxSize[1]) << endl;
}

void DPM2D::initVerticesOnParticles() {
  double rad;
  long particleId, numVertexInParticle;
  for (long vertexId = 0; vertexId < numVertices; vertexId++) {
    particleId = d_particleIdList[vertexId];
    numVertexInParticle = d_numVertexInParticleList[particleId];
    rad = sqrt((2. * d_a0[particleId]) / (numVertexInParticle * sin(2. * PI / numVertexInParticle)));
		d_pos[vertexId * nDim] = rad * cos((2. * PI * vertexId) / numVertexInParticle) + d_particlePos[particleId * nDim];// + 1e-02 * d_l0[vertexId] * drand48();
		d_pos[vertexId * nDim + 1] = rad * sin((2. * PI * vertexId) / numVertexInParticle) + d_particlePos[particleId * nDim + 1];// + 1e-02 * d_l0[vertexId] * drand48();
  }
}

void DPM2D::scaleBoxSize(double scale) {
  resetParticleLastPositions();
  thrust::transform(d_particlePos.begin(), d_particlePos.end(), thrust::make_constant_iterator(scale), d_particlePos.begin(), thrust::divides<double>());
  // shift vertex positions accordingly
  thrust::host_vector<double> boxSize_(nDim);
  boxSize_ = getBoxSize();
  for (long dim = 0; dim < nDim; dim++) {
    boxSize_[dim] /= scale;
  }
  setBoxSize(boxSize_);
  translateVertices();
}

void DPM2D::scaleVertexPositions(double scale) {
  //calcParticlePositions();
  double* pos = thrust::raw_pointer_cast(&d_pos[0]);
  const double* particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);
  kernelScaleVertexPositions<<<dimGrid, dimBlock>>>(particlePos, pos, scale);
}

void DPM2D::scalePacking(double scale) {
  // particle variables
  thrust::transform(d_particlePos.begin(), d_particlePos.end(), thrust::make_constant_iterator(scale), d_particlePos.begin(), thrust::divides<double>());
  thrust::transform(d_particleRad.begin(), d_particleRad.end(), thrust::make_constant_iterator(scale), d_particleRad.begin(), thrust::divides<double>());
  // vertex variables
  thrust::transform(d_pos.begin(), d_pos.end(), thrust::make_constant_iterator(scale), d_pos.begin(), thrust::divides<double>());
  thrust::transform(d_rad.begin(), d_rad.end(), thrust::make_constant_iterator(scale), d_rad.begin(), thrust::divides<double>());
  h_rad = d_rad; //HOST
  thrust::transform(d_l0.begin(), d_l0.end(), thrust::make_constant_iterator(scale), d_l0.begin(), thrust::divides<double>());
  thrust::transform(d_a0.begin(), d_a0.end(), thrust::make_constant_iterator(scale * scale), d_a0.begin(), thrust::divides<double>());
  // boxSize
  thrust::host_vector<double> boxSize_(nDim);
  boxSize_ = getBoxSize();
  for (long dim = 0; dim < nDim; dim++) {
    boxSize_[dim] /= scale;
  }
  setBoxSize(boxSize_);
  cout << "DPM2D::scalePacking: preferred packing fraction: " << getPreferredPhi() << " Lx: " << boxSize_[0] << " Ly: " << boxSize_[1] << endl;
}

void DPM2D::scaleParticlePacking() {
  double scale = getMeanParticleSigma();
  // particle variables
  thrust::transform(d_particleRad.begin(), d_particleRad.end(), thrust::make_constant_iterator(scale), d_particleRad.begin(), thrust::divides<double>());
  thrust::transform(d_particlePos.begin(), d_particlePos.end(), thrust::make_constant_iterator(scale), d_particlePos.begin(), thrust::divides<double>());
  // vertex variables
  thrust::transform(d_rad.begin(), d_rad.end(), thrust::make_constant_iterator(scale), d_rad.begin(), thrust::divides<double>());
  h_rad = d_rad; //HOST
  thrust::transform(d_l0.begin(), d_l0.end(), thrust::make_constant_iterator(scale), d_l0.begin(), thrust::divides<double>());
  thrust::transform(d_a0.begin(), d_a0.end(), thrust::make_constant_iterator(scale * scale), d_a0.begin(), thrust::divides<double>());
  // boxSize
  thrust::host_vector<double> boxSize_(nDim);
  boxSize_ = getBoxSize();
  for (long dim = 0; dim < nDim; dim++) {
    boxSize_[dim] /= scale;
  }
  setBoxSize(boxSize_);
  //cout << "DPM2D::scalePacking: preferred packing fraction: " << getPreferredPhi() << " Lx: " << boxSize_[0] << " Ly: " << boxSize_[1] << endl;
}

void DPM2D::scaleVertices(double scale) {
  scaleVertexPositions(scale);
  thrust::transform(d_a0.begin(), d_a0.end(), thrust::make_constant_iterator(scale * scale), d_a0.begin(), thrust::multiplies<double>());
  thrust::transform(d_area.begin(), d_area.end(), thrust::make_constant_iterator(scale * scale), d_area.begin(), thrust::multiplies<double>());
  thrust::transform(d_l0.begin(), d_l0.end(), thrust::make_constant_iterator(scale), d_l0.begin(), thrust::multiplies<double>());
  thrust::transform(d_rad.begin(), d_rad.end(), thrust::make_constant_iterator(scale), d_rad.begin(), thrust::multiplies<double>());
  h_rad = d_rad; //HOST
}

void DPM2D::scaleParticles(double scale) {
  thrust::transform(d_particleRad.begin(), d_particleRad.end(), thrust::make_constant_iterator(scale), d_particleRad.begin(), thrust::multiplies<double>());
  thrust::transform(d_a0.begin(), d_a0.end(), thrust::make_constant_iterator(scale * scale), d_a0.begin(), thrust::multiplies<double>());
}

void DPM2D::pressureScaleParticles(double pscale) {
  thrust::transform(d_particlePos.begin(), d_particlePos.end(), thrust::make_constant_iterator(pscale), d_particlePos.begin(), thrust::multiplies<double>());
  thrust::transform(d_boxSize.begin(), d_boxSize.end(), thrust::make_constant_iterator(pscale), d_boxSize.begin(), thrust::multiplies<double>());
}

void DPM2D::scaleSoftParticles(double scale) {
  thrust::transform(d_particleRad.begin(), d_particleRad.end(), thrust::make_constant_iterator(scale), d_particleRad.begin(), thrust::multiplies<double>());
  thrust::transform(d_a0.begin(), d_a0.end(), thrust::make_constant_iterator(scale * scale), d_a0.begin(), thrust::multiplies<double>());
  //setSphericalLengthScale();
}

void DPM2D::scaleParticleVelocity(double scale) {
  thrust::transform(d_particleVel.begin(), d_particleVel.end(), thrust::make_constant_iterator(scale), d_particleVel.begin(), thrust::multiplies<double>());
}

// translate vertices by particle displacement
void DPM2D::translateVertices() {
	double* pos = thrust::raw_pointer_cast(&d_pos[0]);
	const double* pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double* pLastPos = thrust::raw_pointer_cast(&d_particleLastPos[0]);
  kernelTranslateVertices<<<dimGrid, dimBlock>>>(pPos, pLastPos, pos);
}

// rotate vertices by particle angle change
void DPM2D::rotateVertices() {
  double* pos = thrust::raw_pointer_cast(&d_pos[0]);
	const double* pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double* pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
  const double* pLastAngle = thrust::raw_pointer_cast(&d_particleLastAngle[0]);
  kernelRotateVertices<<<dimGrid, dimBlock>>>(pPos, pAngle, pLastAngle, pos);
}

// translate vertices by particle displacement
void DPM2D::translateAndRotateVertices() {
	double* pos = thrust::raw_pointer_cast(&d_pos[0]);
	const double* pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double* pLastPos = thrust::raw_pointer_cast(&d_particleLastPos[0]);
  const double* pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
  const double* pLastAngle = thrust::raw_pointer_cast(&d_particleLastAngle[0]);
  kernelTranslateAndRotateVertices<<<dimGrid, dimBlock>>>(pPos, pLastPos, pAngle, pLastAngle, pos);
}

// compute particle angles from velocity
void DPM2D::computeParticleAngleFromVel() {
  long p_nDim(nDim);
  auto r = thrust::counting_iterator<long>(0);
  double* pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
  const double* pVel = thrust::raw_pointer_cast(&d_particleVel[0]);

  auto computeParticleAngle = [=] __device__ (long particleId) {
    pAngle[particleId] = atan(pVel[particleId * p_nDim + 1] / pVel[particleId * p_nDim]);
  };

  thrust::for_each(r, r + numParticles, computeParticleAngle);
}

//*************************** force and energy *******************************//
void DPM2D::setEnergyCosts(double ea_, double el_, double eb_, double ec_) {
  ea = ea_;
  el = el_;
  eb = eb_;
  ec = ec_;
  cudaMemcpyToSymbol(d_ea, &ea, sizeof(ea));
  cudaMemcpyToSymbol(d_el, &el, sizeof(el));
  cudaMemcpyToSymbol(d_eb, &eb, sizeof(eb));
  cudaMemcpyToSymbol(d_ec, &ec, sizeof(ec));
}

void DPM2D::setAttractionConstants(double l1_, double l2_) {
  l1 = l1_;
  l2 = l2_;
  cudaMemcpyToSymbol(d_l1, &l1, sizeof(l1));
  cudaMemcpyToSymbol(d_l2, &l2, sizeof(l2));
}

void DPM2D::setLJcutoff(double LJcutoff_) {
  LJcutoff = LJcutoff_;
  cudaMemcpyToSymbol(d_LJcutoff, &LJcutoff, sizeof(LJcutoff));
  LJecut = 4 * (1 / pow(LJcutoff, 12) - 1 / pow(LJcutoff, 6));
  cudaMemcpyToSymbol(d_LJecut, &LJecut, sizeof(LJecut));
  //cout << "DPM2D::setLJcutoff - LJcutoff: " << LJcutoff << " LJecut: " << LJecut << endl;
}

void DPM2D::setFENEconstants(double stiff_, double ext_) {
  stiff = stiff_;
  extSq = ext_ * ext_;
  cudaMemcpyToSymbol(d_stiff, &stiff, sizeof(stiff));
  cudaMemcpyToSymbol(d_extSq, &extSq, sizeof(extSq));
}

double DPM2D::setTimeScale(double dt_) {
  double ta, tl, tb, tmin = 1e08;
  // compute typical time scale
  ta = rho0 / sqrt(ea);
  tl = (rho0 * d_l0[0]) / sqrt(ea * el); // TODO: replace values at 0 with averages
  tb = (rho0 * d_l0[0]) / sqrt(ea * eb); // TODO: replace values at 0 with averages
  // compute global time scale
  if (ta < tmin) tmin = ta;
  if (tl < tmin) tmin = tl;
  if (tb < tmin) tmin = tb;
  dt = tmin * dt_;
  cudaMemcpyToSymbol(d_dt, &dt, sizeof(dt));
  return dt;
}

double DPM2D::setTimeStep(double dt_) {
  dt = dt_;
  cudaMemcpyToSymbol(d_dt, &dt, sizeof(dt));
  return dt;
}

void DPM2D::setTwoParticleTest(double lx, double ly, double y0, double y1, double vel1) {
  setMonoSizeDistribution();
  if(cudaGetLastError()) cout << "DPM2D():: cudaGetLastError(): " << cudaGetLastError() << endl;
  thrust::host_vector<double> boxSize(nDim);
  // set particle radii
  for (long pId = 0; pId < numParticles; pId++) {
    d_particleRad[pId] = sqrt((2. * d_a0[pId]) / (d_numVertexInParticleList[pId] * sin(2. * PI / d_numVertexInParticleList[pId])));
  }
  boxSize[0] = lx;
  boxSize[1] = ly;
  setBoxSize(boxSize);
  // assign positions
  d_particlePos[0 * nDim] = lx * 0.65;
  d_particlePos[0 * nDim + 1] = ly * y0;
  d_particlePos[1 * nDim] = lx * 0.35;
  d_particlePos[1 * nDim + 1] = ly * y1;
  // initialize vertices
  initVerticesOnParticles();
  setInitialPositions();
  initNeighbors(numVertices);
  syncNeighborsToDevice();
  // assign velocity
  auto firstVertex = d_firstVertexInParticleId[1];
  auto lastVertex = firstVertex + d_numVertexInParticleList[1];
  switch (simControl.particleType) {
    case simControlStruct::particleEnum::deformable:
    for(long vId = firstVertex; vId < lastVertex; vId++) {
      d_vel[vId * nDim] = vel1;
    }
    break;
    case simControlStruct::particleEnum::rigid:
    d_particleVel[1 * nDim] = vel1;
    break;
  }

  setLengthScaleToOne();
  if(cudaGetLastError()) cout << "DPM2D():: cudaGetLastError(): " << cudaGetLastError() << endl;
}

void DPM2D::firstUpdate(double timeStep) {
  int s_nDim(nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
	double* pos = thrust::raw_pointer_cast(&d_pos[0]);
	double* vel = thrust::raw_pointer_cast(&d_vel[0]);
	const double* force = thrust::raw_pointer_cast(&d_force[0]);
  auto firstUpdate = [=] __device__ (long vId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      vel[vId * s_nDim + dim] += 0.5 * s_dt * force[vId * s_nDim + dim];
      pos[vId * s_nDim + dim] += s_dt * vel[vId * s_nDim + dim];
    }
  };

  auto firstVertex = d_firstVertexInParticleId[1];
	auto lastVertex = firstVertex + d_numVertexInParticleList[1];
  thrust::for_each(r + firstVertex, r + lastVertex, firstUpdate);
  //thrust::for_each(r, r + numVertices, firstUpdate);
}

void DPM2D::secondUpdate(double timeStep) {
  int s_nDim(nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
	double* vel = thrust::raw_pointer_cast(&d_vel[0]);
	const double* force = thrust::raw_pointer_cast(&d_force[0]);

  auto firstUpdate = [=] __device__ (long vId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      vel[vId * s_nDim + dim] += 0.5 * s_dt * force[vId * s_nDim + dim];
    }
  };

  auto firstVertex = d_firstVertexInParticleId[1];
	auto lastVertex = firstVertex + d_numVertexInParticleList[1];
  thrust::for_each(r + firstVertex, r + lastVertex, firstUpdate);
  //thrust::for_each(r, r + numVertices, firstUpdate);
}

void DPM2D::pinDeformableParticle(long pId) {
  // pin particle to initial vertex positions with a strong attractive spring
  long s_nDim(nDim);
  auto r = thrust::counting_iterator<long>(0);
	const double* boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);
	const double* initialPos = thrust::raw_pointer_cast(&d_initialPos[0]);
	const double* pos = thrust::raw_pointer_cast(&d_pos[0]);
	double* force = thrust::raw_pointer_cast(&d_force[0]);
	double* energy = thrust::raw_pointer_cast(&d_energy[0]);

  auto pinVertex = [=] __device__ (long vId) {
    auto distanceSq = 0.0;
    double delta[MAXDIM];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      delta[dim] = pos[vId * s_nDim + dim] - initialPos[vId * s_nDim + dim];
	    delta[dim] -= boxSize[dim] * round(delta[dim] / boxSize[dim]);
      distanceSq += delta[dim] * delta[dim];
    }
    auto distance = sqrt(distanceSq);
    if(distance > 0.0) {
      auto gradMultiple = d_ec * distance;
      #pragma unroll (MAXDIM)
      for (long dim = 0; dim < s_nDim; dim++) {
        force[vId * s_nDim + dim] += gradMultiple * delta[dim] / distance;
      }
      energy[vId] -= 0.5 * d_ec * distanceSq;
    }
  };

  auto firstVertex = d_firstVertexInParticleId[pId];
	auto lastVertex = firstVertex + d_numVertexInParticleList[pId];
  thrust::for_each(r + firstVertex, r + lastVertex, pinVertex);
}

void DPM2D::addTangentialForce(long pId) {
  // compute center of mass of particle 1
  double com[MAXDIM] = {0.0,0.0};
  for (long vId = d_firstVertexInParticleId[1]; vId < d_numVertexInParticleList[1]; vId++) {
    for (long dim = 0; dim < nDim; dim++) {
      com[dim] += d_pos[vId * nDim + dim];
    }
  }
  for (long dim = 0; dim < nDim; dim++) {
    com[dim] /= d_numVertexInParticleList[1];
  }
  // add velocity to particle 0 towards center of mass of particle 1
  long s_nDim(nDim);
  auto r = thrust::counting_iterator<long>(0);
	const double* boxSize = thrust::raw_pointer_cast(&d_boxSize[0]);
	const double* pos = thrust::raw_pointer_cast(&d_pos[0]);
	double* force = thrust::raw_pointer_cast(&d_force[0]);
	double* energy = thrust::raw_pointer_cast(&d_energy[0]);

  auto addTangentialForceToVertex = [=] __device__ (long vId) {
    auto distanceSq = 0.0;
    double delta[MAXDIM];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      delta[dim] = com[dim] - pos[vId * s_nDim + dim];
	    delta[dim] -= boxSize[dim] * round(delta[dim] / boxSize[dim]);
      distanceSq += delta[dim] * delta[dim];
    }
    auto distance = sqrt(distanceSq);
    for (long dim = 0; dim < s_nDim; dim++) {
      delta[dim] /= distance;
    }
    double tangent[MAXDIM];
    tangent[0] = -delta[1];
    tangent[1] = delta[0];
    auto ratio = 1 / distance;
    auto gradMultiple = 24 * d_el * (-pow(ratio,6)) / distance;
    #pragma unroll (MAXDIM)
    for (long dim = 0; dim < s_nDim; dim++) {
      force[vId * s_nDim + dim] += d_el * tangent[dim];
      force[vId * s_nDim + dim] += gradMultiple * delta[dim];
    }
    energy[vId] += 4 * d_el * (-pow(ratio,6));
  };

  auto firstVertex = d_firstVertexInParticleId[pId];
	auto lastVertex = firstVertex + d_numVertexInParticleList[pId];
  thrust::for_each(r + firstVertex, r + lastVertex, addTangentialForceToVertex);
}

void DPM2D::addForce(long pId) {
  long s_nDim(nDim);
  auto r = thrust::counting_iterator<long>(0);
	double* force = thrust::raw_pointer_cast(&d_force[0]);

  auto addForceToVertex = [=] __device__ (long vId) {
    force[vId * s_nDim] += 0.1;
  };

  auto firstVertex = d_firstVertexInParticleId[pId];
	auto lastVertex = firstVertex + d_numVertexInParticleList[pId];
  thrust::for_each(r + firstVertex, r + lastVertex, addForceToVertex);
}

void DPM2D::testDeformableInteraction(double timeStep) {
  firstUpdate(timeStep);
  checkNeighbors();
  calcForceEnergy();
  //addTangentialForce(0);
  //pinDeformableParticle(1);
  secondUpdate(timeStep);
}

void DPM2D::firstRigidUpdate(double timeStep) {
  int s_nDim(nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
  // translational variables
  double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  const double* pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
  // rotational variables
  double* pAngle = thrust::raw_pointer_cast(&d_particleAngle[0]);
	double* pAngvel = thrust::raw_pointer_cast(&d_particleAngvel[0]);
	const double *pTorque = thrust::raw_pointer_cast(&d_particleTorque[0]);
  auto firstRigidUpdate = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += 0.5 * s_dt * pForce[pId * s_nDim + dim];
      pPos[pId * s_nDim + dim] += s_dt * pVel[pId * s_nDim + dim];
    }
		pAngvel[pId] += 0.5 * s_dt * pTorque[pId];
		pAngle[pId] += s_dt * pAngvel[pId];
  };

  thrust::for_each(r, r + numParticles, firstRigidUpdate);
}

void DPM2D::secondRigidUpdate(double timeStep) {
  int s_nDim(nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
  // translational variables
	double *pVel = thrust::raw_pointer_cast(&d_particleVel[0]);
  const double* pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
  // rotational variables
	double* pAngvel = thrust::raw_pointer_cast(&d_particleAngvel[0]);
	const double *pTorque = thrust::raw_pointer_cast(&d_particleTorque[0]);
  auto secondRigidUpdate = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += 0.5 * s_dt * pForce[pId * s_nDim + dim];
    }
		pAngvel[pId] += 0.5 * s_dt * pTorque[pId];
  };

  thrust::for_each(r, r + numParticles, secondRigidUpdate);
}

void DPM2D::testRigidInteraction(double timeStep) {
  resetParticleLastPositions();
  resetParticleLastAngles();
  firstRigidUpdate(timeStep);
  translateVertices();
	rotateVertices();
  checkNeighbors();
  calcRigidForceEnergy();
  secondRigidUpdate(timeStep);
}

void DPM2D::testInteraction(double timeStep) {
  switch (simControl.particleType) {
    case simControlStruct::particleEnum::deformable:
    testDeformableInteraction(timeStep);
    break;
    case simControlStruct::particleEnum::rigid:
    testRigidInteraction(timeStep);
    break;
  }
}

void DPM2D::printTwoParticles() {
  cudaError err = cudaGetLastError();
  if(err != cudaSuccess) cout << "DPM2D::calcForceEnergyGPU::cudaGetLastError(): " << err << endl;
  thrust::host_vector<double> force(d_force.size(), 0.0);
  thrust::host_vector<double> vel(d_force.size(), 0.0);
  thrust::host_vector<double> pos(d_force.size(), 0.0);
  force = d_force;
  vel = d_vel;
  pos = d_pos;
  for (long pId = 0; pId < numParticles; pId++) {
    cout << "Particle " << pId << endl;
		auto firstVertex = d_firstVertexInParticleId[pId];
		auto lastVertex = firstVertex + d_numVertexInParticleList[pId];
    for(long vId = firstVertex; vId < lastVertex; vId++) {
      cout << "vertex " << vId << " fx: " << force[vId * nDim] << " fy: " << force[vId * nDim + 1] << endl;
      cout << "vx: " << vel[vId * nDim] << " vy: " << vel[vId * nDim + 1] << endl;
      cout << "x: " << pos[vId * nDim] << " y: " << pos[vId * nDim + 1] << endl;
    }
  }
}

void DPM2D::calcForceEnergy() {
  switch (simControl.simulationType) {
    case simControlStruct::simulationEnum::gpu:
    calcForceEnergyGPU();
    break;
    case simControlStruct::simulationEnum::cpu:
    calcForceEnergyCPU();
    break;
    case simControlStruct::simulationEnum::omp:
    calcForceEnergyOMP();
    break;
  }
}

void DPM2D::calcShapeForceEnergy() {
  calcParticleShape();
  calcParticlePositions();
  // shape variables
	const double *a0 = thrust::raw_pointer_cast(&d_a0[0]);
	const double *l0 = thrust::raw_pointer_cast(&d_l0[0]);
  const double *rad = thrust::raw_pointer_cast(&d_rad[0]);
	const double *theta0 = thrust::raw_pointer_cast(&d_theta0[0]);
  // dynamical variables
  const double *area = thrust::raw_pointer_cast(&d_area[0]);
	const double *theta = thrust::raw_pointer_cast(&d_theta[0]);
  const double *particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
	double *force = thrust::raw_pointer_cast(&d_force[0]);
	double *energy = thrust::raw_pointer_cast(&d_energy[0]);
  // compute shape force
  kernelCalcShapeForceEnergy<<<dimGrid, dimBlock>>>(a0, area, particlePos, l0, theta0, theta, pos, force, energy);
}

void DPM2D::calcForceEnergyGPU() {
  calcShapeForceEnergy();
  const double *rad = thrust::raw_pointer_cast(&d_rad[0]);
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
	double *force = thrust::raw_pointer_cast(&d_force[0]);
	double *energy = thrust::raw_pointer_cast(&d_energy[0]);
  thrust::fill(d_particleEnergy.begin(), d_particleEnergy.end(), double(0));
  double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  // compute interaction
  switch (simControl.interactionType) {
    case simControlStruct::interactionEnum::vertexVertex:
    kernelCalcVertexInteraction<<<dimGrid, dimBlock>>>(rad, pos, force, energy);
    break;
    case simControlStruct::interactionEnum::vertexSmooth:
    kernelCalcSmoothInteraction<<<dimGrid, dimBlock>>>(rad, pos, force, pEnergy);
    break;
    case simControlStruct::interactionEnum::cellSmooth:
    kernelCalcCellListSmoothInteraction<<<cellDimGrid, cellDimBlock>>>(rad, pos, force, pEnergy);
    break;
    case simControlStruct::interactionEnum::all:
    kernelCalcAllToAllVertexInteraction<<<dimGrid, dimBlock>>>(rad, pos, force, energy);
    break;
  }
}

thrust::host_vector<double> DPM2D::getInteractionForces() {
  switch (simControl.simulationType) {
    case simControlStruct::simulationEnum::gpu:
    return getInteractionForcesGPU();
    break;
    default:
    return h_interaction;
    break;
  }
}

thrust::host_vector<double> DPM2D::getInteractionForcesGPU() {
  thrust::device_vector<double> d_interaction(d_force.size());
  thrust::device_vector<double> d_interEnergy(d_energy.size());
  thrust::device_vector<double> d_interParticleEnergy(d_particleEnergy.size());
  thrust::fill(d_interaction.begin(), d_interaction.end(), double(0));
  thrust::fill(d_interEnergy.begin(), d_interEnergy.end(), double(0));
  thrust::fill(d_interParticleEnergy.begin(), d_interParticleEnergy.end(), double(0));
  const double *rad = thrust::raw_pointer_cast(&d_rad[0]);
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
	double *interaction = thrust::raw_pointer_cast(&d_interaction[0]);
	double *interEnergy = thrust::raw_pointer_cast(&d_interEnergy[0]);
  double *interPEnergy = thrust::raw_pointer_cast(&d_interParticleEnergy[0]);
  // compute interaction
  switch (simControl.interactionType) {
    case simControlStruct::interactionEnum::vertexVertex:
    kernelCalcVertexInteraction<<<dimGrid, dimBlock>>>(rad, pos, interaction, interPEnergy);
    break;
    case simControlStruct::interactionEnum::vertexSmooth:
    kernelCalcSmoothInteraction<<<dimGrid, dimBlock>>>(rad, pos, interaction, interPEnergy);
    break;
    case simControlStruct::interactionEnum::cellSmooth:
    kernelCalcCellListSmoothInteraction<<<cellDimGrid, cellDimBlock>>>(rad, pos, interaction, interPEnergy);
    break;
    case simControlStruct::interactionEnum::all:
    kernelCalcAllToAllVertexInteraction<<<dimGrid, dimBlock>>>(rad, pos, interaction, interPEnergy);
    break;
  }
  thrust::host_vector<double> interactionFromDevice;
  interactionFromDevice = d_interaction;
  return interactionFromDevice;
}

void DPM2D::calcForceEnergyCPU() {
  calcShapeForceEnergy();
  switch (simControl.interactionType) {
    case simControlStruct::interactionEnum::vertexVertex:
    calcVertexVertexInteraction();
    break;
    case simControlStruct::interactionEnum::vertexSmooth:
    thrust::fill(d_particleEnergy.begin(), d_particleEnergy.end(), double(0));
    calcSmoothInteraction();
    break;
    case simControlStruct::interactionEnum::cellSmooth:
    calcCellListSmoothInteraction();
    break;
  }
}

void DPM2D::calcForceEnergyOMP() {
  calcShapeForceEnergy();
  switch (simControl.interactionType) {
    case simControlStruct::interactionEnum::vertexVertex:
    calcVertexVertexInteractionOMP();
    break;
    case simControlStruct::interactionEnum::vertexSmooth:
    thrust::fill(d_particleEnergy.begin(), d_particleEnergy.end(), double(0));
    calcSmoothInteractionOMP();
    break;
    default:
    calcVertexVertexInteractionOMP();
    break;
  }
}

double DPM2D::pbcDistance(double x1, double x2, double size) {
	double delta = x1 - x2;
	return delta - size * round(delta / size); //round for distance, floor for position
}

void DPM2D::calcVertexVertexInteraction() {
  h_pos = d_pos;
  h_force = d_force;
  h_energy = d_energy;
  std::vector<double> thisPos(MAXDIM), otherPos(MAXDIM), delta(MAXDIM);
  double overlap, ratio, ratio6, ratio12, gradMultiple, epot;
  for (long particleId = 0; particleId < numParticles; particleId++) {
    auto firstVertex = d_firstVertexInParticleId[particleId];
    auto lastVertex = firstVertex + d_numVertexInParticleList[particleId];
    for (long vertexId = firstVertex; vertexId < lastVertex; vertexId++) {
      for (long dim = 0; dim < nDim; dim++) {
        thisPos[dim] = h_pos[vertexId * nDim + dim];
        h_interaction[vertexId * nDim + dim] = 0.0;
      }
      double thisRad = h_rad[vertexId];
      //long particleId = h_particleIdList[vertexId];
      for (long nListId = 0; nListId < h_maxNeighborList[vertexId]; nListId++) {
        long otherId = h_neighborList[vertexId * neighborListSize + nListId];
        if ((vertexId != otherId) && (otherId != -1)) {
          for (long dim = 0; dim < nDim; dim++) {
            otherPos[dim] = h_pos[otherId * nDim + dim];
          }
          double otherRad = h_rad[otherId];
          double radSum = thisRad + otherRad;
          double distanceSq = 0.0;
          for (long dim = 0; dim < nDim; dim++) {
            delta[dim] = pbcDistance(thisPos[dim], otherPos[dim], h_boxSize[dim]);
            distanceSq += delta[dim] * delta[dim];
          }
          double distance = sqrt(distanceSq);
          bool addForce = false;
          switch (simControl.potentialType) {
            case simControlStruct::potentialEnum::harmonic:
            overlap = 1 - distance / radSum;
            if(overlap > 0) {
              addForce = true;
              gradMultiple = ec * overlap / radSum;
              epot = (0.5 * ec * overlap * overlap) * 0.5;
            }
            break;
            case simControlStruct::potentialEnum::wca:
            if(distance <= (WCAcut * radSum)) {
              addForce = true;
              ratio = radSum / distance;
              ratio6 = pow(ratio, 6);
              ratio12 = ratio6 * ratio6;
              gradMultiple = 4 * ec * (12 * ratio12 - 6 * ratio6) / distance;
              epot = 0.5 * ec * (4 * (ratio12 - ratio6) + 1);
            }
            break;
          }
          if(addForce == true) {
            //cout << "checking vertexId: " << vertexId << " and otherId: " << otherId << endl;
            for (long dim = 0; dim < nDim; dim++) {
                h_force[vertexId * nDim + dim] += 0.5 * gradMultiple * delta[dim] / distance;
                h_force[otherId * nDim + dim] -= 0.5 * gradMultiple * delta[dim] / distance;
                h_interaction[vertexId * nDim + dim] += 0.5 * gradMultiple * delta[dim] / distance;
                h_interaction[otherId * nDim + dim] -= 0.5 * gradMultiple * delta[dim] / distance;
              }
            h_energy[vertexId] += epot * 0.5;
            h_energy[otherId] += epot * 0.5;
          }
        }
      }
    }
  }
  d_force = h_force;
  d_energy = h_energy;
}

void DPM2D::calcVertexVertexInteractionOMP() {
  h_pos = d_pos;
  h_force = d_force;
  h_energy = d_energy;
  //#pragma omp parallel for default(none) shared(h_boxSize, h_pos, h_force, h_energy, h_particleIdList, h_neighborList, h_maxNeighborList, simControl, ec, WCAcut, numVertices, neighborListSize, nDim)
  #pragma omp parallel for// reduction(+,h_force) reduction(+,h_energy) reduction(+,h_interaction)
  {
    for (long vertexId = 0; vertexId < numVertices; vertexId++) {
      std::vector<double> thisPos(MAXDIM), otherPos(MAXDIM), delta(MAXDIM);
      double overlap, ratio, ratio6, ratio12, gradMultiple, epot;
      for (long dim = 0; dim < nDim; dim++) {
        thisPos[dim] = h_pos[vertexId * nDim + dim];
        h_interaction[vertexId * nDim + dim] = 0.0;
      }
      double thisRad = h_rad[vertexId];
      long particleId = h_particleIdList[vertexId];
      for (long nListId = 0; nListId < h_maxNeighborList[vertexId]; nListId++) {
        long otherId = h_neighborList[vertexId * neighborListSize + nListId];
        if ((vertexId != otherId) && (otherId != -1)) {
          for (long dim = 0; dim < nDim; dim++) {
            otherPos[dim] = h_pos[otherId * nDim + dim];
          }
          double otherRad = h_rad[otherId];
          double radSum = thisRad + otherRad;
          double distanceSq = 0;
          for (long dim = 0; dim < nDim; dim++) {
            delta[dim] = pbcDistance(thisPos[dim], otherPos[dim], h_boxSize[dim]);
            distanceSq += delta[dim] * delta[dim];
          }
          double distance = sqrt(distanceSq);
          bool addForce = false;
          switch (simControl.potentialType) {
            case simControlStruct::potentialEnum::harmonic:
            overlap = 1 - distance / radSum;
            if(overlap > 0) {
              addForce = true;
              gradMultiple = ec * overlap / radSum;
              epot = (0.5 * ec * overlap * overlap) * 0.5;
            }
            break;
            case simControlStruct::potentialEnum::wca:
            if(distance <= (WCAcut * radSum)) {
              addForce = true;
              ratio = radSum / distance;
              ratio6 = pow(ratio, 6);
              ratio12 = ratio6 * ratio6;
              gradMultiple = 4 * ec * (12 * ratio12 - 6 * ratio6) / distance;
              epot = 0.5 * ec * (4 * (ratio12 - ratio6) + 1);
            }
            break;
          }
          if(addForce == true) {
            //cout << "checking vertexId: " << vertexId << " and otherId: " << otherId << endl;
            for (long dim = 0; dim < nDim; dim++) {
                h_force[vertexId * nDim + dim] += gradMultiple * delta[dim] / distance;
                h_interaction[vertexId * nDim + dim] += gradMultiple * delta[dim] / distance;
              }
            h_energy[vertexId] += epot;
          }
        }
      }
    }
  }
  d_force = h_force;
  d_energy = h_energy;
}

// get index of previous vertex of the same particle
long DPM2D::getPreviousId(long vertexId, long particleId) {
	if(vertexId == 0) {
		return d_numVertexInParticleList[particleId] - 1;
	}
  long previousId = vertexId - 1;
  long whichParticle = d_particleIdList[previousId];
  if( whichParticle == particleId ) {
    return previousId;
  } else {
    return previousId + d_numVertexInParticleList[particleId];// return last vertex in the particle
  }
}

// get index of next vertex of the same particle
long DPM2D::getNextId(long vertexId, long particleId) {
	if(vertexId == numVertices - 1) {
		return vertexId - d_numVertexInParticleList[particleId] + 1;
	}
  long nextId = vertexId + 1;
  long whichParticle = d_particleIdList[nextId];
  if( whichParticle == particleId ) {
    return nextId;
  } else {
    return nextId - d_numVertexInParticleList[particleId];// return first vertex in the particle
  }
}

double DPM2D::getProjection(double* thisPos, double* otherPos, double* previousPos, double length) {
	return (pbcDistance(thisPos[0], previousPos[0], h_boxSize[0]) * pbcDistance(otherPos[0], previousPos[0], h_boxSize[0]) + pbcDistance(thisPos[1], previousPos[1], h_boxSize[1]) * pbcDistance(otherPos[1], previousPos[1], h_boxSize[1])) / (length * length);
}

double DPM2D::calcCross(double* thisPos, double* otherPos, double* previousPos) {
  return pbcDistance(previousPos[0], otherPos[0], h_boxSize[0]) * pbcDistance(otherPos[1], thisPos[1], h_boxSize[1]) - pbcDistance(otherPos[0], thisPos[0], h_boxSize[0]) * pbcDistance(previousPos[1], otherPos[1], h_boxSize[1]);
}

bool DPM2D::checkSmoothInteraction(double* thisPos, double* otherPos, double* previousPos, double radSum) {
  double segment[MAXDIM], relSegment[MAXDIM], relPos[MAXDIM], projPos[MAXDIM], delta[MAXDIM];
  double distance, overlap, distanceSq = 0.0;
  bool isSmooth = false;
  for (long dim = 0; dim < nDim; dim++) {
    segment[dim] = pbcDistance(otherPos[dim], previousPos[dim], h_boxSize[dim]);
    distanceSq += segment[dim] * segment[dim];
  }
  double length = sqrt(distanceSq);
  // this takes into account periodic boundaries with respect to the segment
  for (long dim = 0; dim < nDim; dim++) {
    relSegment[dim] = pbcDistance(thisPos[dim], previousPos[dim], h_boxSize[dim]);
    relPos[dim] = previousPos[dim] + relSegment[dim];
  }
  // compute projection on the line between other and previous
  double projection = getProjection(relPos, otherPos, previousPos, length);
  if(projection > 0 && projection <= 1) {
    double reducedProj = max(0.0, min(1.0, projection));
    for (long dim = 0; dim < nDim; dim++) {
      projPos[dim] = previousPos[dim] + reducedProj * segment[dim];
    }
    distanceSq = 0;
    for (long dim = 0; dim < nDim; dim++) {
      delta[dim] = pbcDistance(thisPos[dim], projPos[dim], h_boxSize[dim]);
      distanceSq += delta[dim] * delta[dim];
    }
    distance = sqrt(distanceSq);
    switch (simControl.potentialType) {
      case simControlStruct::potentialEnum::harmonic:
      overlap = 1 - distance / radSum;
      if(overlap > 0) {
        isSmooth = true;
      }
      break;
      case simControlStruct::potentialEnum::wca:
      if(distance <= (WCAcut * radSum)) {
        isSmooth = true;
      }
      break;
    }
  }
  return isSmooth;
}

void DPM2D::calcSmoothNeighbors() {
  thrust::fill(h_maxSmoothNeighborList.begin(), h_maxSmoothNeighborList.end(), 0);
  smoothNeighborListSize = 2 * neighborListSize;
  h_smoothNeighborList.resize(numVertices * smoothNeighborListSize);
  thrust::fill(h_smoothNeighborList.begin(), h_smoothNeighborList.end(), -1L);
  h_pos = d_pos;
  double thisPos[MAXDIM], otherPos[MAXDIM], previousPos[MAXDIM];
  // identify neighbors interact with a smooth force
  for (long vertexId = 0; vertexId < numVertices; vertexId++) {
    long particleId = d_particleIdList[vertexId];
    for (long dim = 0; dim < nDim; dim++) {
      thisPos[dim] = h_pos[vertexId * nDim + dim];
    }
    double thisRad = h_rad[vertexId];
    for (long nListId = 0; nListId < h_maxNeighborList[vertexId]; nListId++) {
      long otherId = h_neighborList[vertexId * neighborListSize + nListId];
      if (otherId != -1) {
        for (long dim = 0; dim < nDim; dim++) {
          otherPos[dim] = h_pos[otherId * nDim + dim];
        }
        double radSum = thisRad + h_rad[otherId];
        // get previous vertex
        long otherParticleId = h_particleIdList[otherId];
        long previousId = getPreviousId(otherId, otherParticleId);
        for (long dim = 0; dim < nDim; dim++) {
          previousPos[dim] = d_pos[previousId * nDim + dim];
        }
        bool isSmoothNeighbor = false;
        isSmoothNeighbor = checkSmoothInteraction(thisPos, otherPos, previousPos, radSum);
        // add smooth neighbors to vertexId
        if(h_maxSmoothNeighborList[vertexId] < smoothNeighborListSize) {
          h_smoothNeighborList[vertexId * smoothNeighborListSize + h_maxSmoothNeighborList[vertexId]] = otherId*isSmoothNeighbor -1*(!isSmoothNeighbor);
          h_smoothNeighborList[vertexId * smoothNeighborListSize + h_maxSmoothNeighborList[vertexId] + 1] = previousId*isSmoothNeighbor -1*(!isSmoothNeighbor);
        }
        h_maxSmoothNeighborList[vertexId] += 2 * isSmoothNeighbor;
        // add smooth neighbor to otherId
        if(h_maxSmoothNeighborList[otherId] < smoothNeighborListSize) {
          h_smoothNeighborList[otherId * smoothNeighborListSize + h_maxSmoothNeighborList[otherId]] = vertexId*isSmoothNeighbor -1*(!isSmoothNeighbor);
        }
        h_maxSmoothNeighborList[otherId] += isSmoothNeighbor;
        // add smooth neighbor to previousId
        if(h_maxSmoothNeighborList[previousId] < smoothNeighborListSize) {
          h_smoothNeighborList[previousId * smoothNeighborListSize + h_maxSmoothNeighborList[previousId]] = vertexId*isSmoothNeighbor -1*(!isSmoothNeighbor);
        }
        h_maxSmoothNeighborList[previousId] += isSmoothNeighbor;
      }
    }
  }
}

double DPM2D::checkAngle(double angle, double limit) {
	if(angle < 0) {
		angle += 2*PI;
	}
	return angle - limit;
}

void DPM2D::calcSmoothInteraction() {
  h_pos = d_pos;
  h_force = d_force;
  h_particleEnergy = d_particleEnergy;
  thrust::fill(h_interaction.begin(), h_interaction.end(), double(0));
  for (long vertexId = 0; vertexId < numVertices; vertexId++) {
    double thisPos[MAXDIM], otherPos[MAXDIM], previousPos[MAXDIM], secondPreviousPos[MAXDIM];
    double delta[MAXDIM], segment[MAXDIM], projPos[MAXDIM];//, interSegment[MAXDIM], previousSegment[MAXDIM];// relSegment[MAXDIM], relPos[MAXDIM];
    double distance, gradMultiple, overlap, ratio, ratio6, ratio12, epot;
    long particleId = d_particleIdList[vertexId];
    for (long dim = 0; dim < nDim; dim++) {
      thisPos[dim] = h_pos[vertexId * nDim + dim];
    }
    double thisRad = h_rad[vertexId];
    //for (long otherId = 0; otherId < numVertices; otherId++) {
    //  long otherParticleId = h_particleIdList[otherId];
    //  if ((vertexId != otherId) && (otherParticleId != particleId)) {
    for (long nListId = 0; nListId < h_maxNeighborList[vertexId]; nListId++) {
      long otherId = h_neighborList[vertexId * neighborListSize + nListId];
      if ((vertexId != otherId) && (otherId != -1)) {
        for (long dim = 0; dim < nDim; dim++) {
          otherPos[dim] = h_pos[otherId * nDim + dim];
        }
        double otherRad = h_rad[otherId];
        double radSum = thisRad + otherRad;
        // get previous vertex
        long otherParticleId = h_particleIdList[otherId];
        long previousId = getPreviousId(otherId, otherParticleId);
        double distanceSq = 0;
        for (long dim = 0; dim < nDim; dim++) {
          previousPos[dim] = d_pos[previousId * nDim + dim];
          segment[dim] = pbcDistance(otherPos[dim], previousPos[dim], h_boxSize[dim]);
          distanceSq += segment[dim] * segment[dim];
        }
        double length = sqrt(distanceSq);
        // this takes into account periodic boundaries with respect to the segment
        //for (long dim = 0; dim < nDim; dim++) {
        //  relSegment[dim] = pbcDistance(thisPos[dim], previousPos[dim], h_boxSize[dim]);
        //  thisPos[dim] = previousPos[dim] + relSegment[dim];
        //}
        // compute projection on the line between other and previous
        double projection = (pbcDistance(thisPos[0], previousPos[0], h_boxSize[0]) * pbcDistance(otherPos[0], previousPos[0], h_boxSize[0]) + pbcDistance(thisPos[1], previousPos[1], h_boxSize[1]) * pbcDistance(otherPos[1], previousPos[1], h_boxSize[1])) / (length * length);
        //double projection = getProjection(thisPos, otherPos, previousPos, length);
        //cout << "projection: " << projection << " length: " << length << endl;
        if(projection >= 0 && projection < 1) {
          double reducedProj = max(0.0, min(1.0, projection));
          for (long dim = 0; dim < nDim; dim++) {
            projPos[dim] = previousPos[dim] + reducedProj * segment[dim];
          }
          distanceSq = 0;
          for (long dim = 0; dim < nDim; dim++) {
            delta[dim] = pbcDistance(thisPos[dim], projPos[dim], h_boxSize[dim]);
            distanceSq += delta[dim] * delta[dim];
          }
          distance = sqrt(distanceSq);
          bool addForce = false;
          switch (simControl.potentialType) {
            case simControlStruct::potentialEnum::harmonic:
            overlap = 1 - distance / radSum;
            if(overlap > 0) {
              addForce = true;
              gradMultiple = ec * overlap / radSum;
              epot = (0.5 * ec * overlap * overlap) * 0.5;
            }
            break;
            case simControlStruct::potentialEnum::wca:
            if(distance <= (WCAcut * radSum)) {
              addForce = true;
              ratio = radSum / distance;
              ratio6 = pow(ratio, 6);
              ratio12 = ratio6 * ratio6;
              gradMultiple = 4 * ec * (12 * ratio12 - 6 * ratio6) / distance;
              epot = 0.5 * ec * (4 * (ratio12 - ratio6) + 1);
            }
            break;
          }
          if(addForce == true) {
            //cout << vertexId << " vertex-segment interacton with " << otherId << " " << previousId << " projection: " << projection << " force: " << gradMultiple << endl;
            double cross = pbcDistance(previousPos[0], otherPos[0], h_boxSize[0]) * pbcDistance(otherPos[1], thisPos[1], h_boxSize[1]) - pbcDistance(otherPos[0], thisPos[0], h_boxSize[0]) * pbcDistance(previousPos[1], otherPos[1], h_boxSize[1]);
            //double cross = calcCross(thisPos, otherPos, previousPos);
            double absCross = fabs(cross);
            double sign = absCross / cross;
            // this vertex
            h_force[vertexId * nDim] += gradMultiple * sign * pbcDistance(previousPos[1], otherPos[1], h_boxSize[1]) / length;
            h_force[vertexId * nDim + 1] += gradMultiple * sign * pbcDistance(otherPos[0], previousPos[0], h_boxSize[0]) / length;
            // other vertex
            h_force[otherId * nDim] += gradMultiple * (sign * pbcDistance(thisPos[1], previousPos[1], h_boxSize[1]) + absCross * pbcDistance(previousPos[0], otherPos[0], h_boxSize[0]) / (length * length)) / length;
            h_force[otherId * nDim + 1] += gradMultiple * (sign * pbcDistance(previousPos[0], thisPos[0], h_boxSize[0]) + absCross * pbcDistance(previousPos[1], otherPos[1], h_boxSize[1]) / (length * length)) / length;
            // previous vertex
            h_force[previousId * nDim] += gradMultiple * (sign * pbcDistance(otherPos[1], thisPos[1], h_boxSize[1]) - absCross * pbcDistance(previousPos[0], otherPos[0], h_boxSize[0]) / (length * length)) / length;
            h_force[previousId * nDim + 1] += gradMultiple * (sign * pbcDistance(thisPos[0], otherPos[0], h_boxSize[0]) - absCross * pbcDistance(previousPos[1], otherPos[1], h_boxSize[1]) / (length * length)) / length; 
            // add energy
            h_particleEnergy[particleId] += epot;
            h_particleEnergy[otherParticleId] += epot;
            // this vertex
            h_interaction[vertexId * nDim] += gradMultiple * sign * pbcDistance(previousPos[1], otherPos[1], h_boxSize[1]) / length;
            h_interaction[vertexId * nDim + 1] += gradMultiple * sign * pbcDistance(otherPos[0], previousPos[0], h_boxSize[0]) / length;
            // other vertex
            h_interaction[otherId * nDim] += gradMultiple * (sign * pbcDistance(thisPos[1], previousPos[1], h_boxSize[1]) + absCross * pbcDistance(previousPos[0], otherPos[0], h_boxSize[0]) / (length * length)) / length;
            h_interaction[otherId * nDim + 1] += gradMultiple * (sign * pbcDistance(previousPos[0], thisPos[0], h_boxSize[0]) + absCross * pbcDistance(previousPos[1], otherPos[1], h_boxSize[1]) / (length * length)) / length;
            // previous vertex
            h_interaction[previousId * nDim] += gradMultiple * (sign * pbcDistance(otherPos[1], thisPos[1], h_boxSize[1]) - absCross * pbcDistance(previousPos[0], otherPos[0], h_boxSize[0]) / (length * length)) / length;
            h_interaction[previousId * nDim + 1] += gradMultiple * (sign * pbcDistance(thisPos[0], otherPos[0], h_boxSize[0]) - absCross * pbcDistance(previousPos[1], otherPos[1], h_boxSize[1]) / (length * length)) / length;
          }
        } else if(projection < 0) {
          // check that previous vertex is not already interacting with this vertex through a segment
          long secondPreviousId = getPreviousId(previousId, otherParticleId);
          distanceSq = 0;
          for (long dim = 0; dim < nDim; dim++) {
            secondPreviousPos[dim] = d_pos[secondPreviousId * nDim + dim];
            delta[dim] = pbcDistance(secondPreviousPos[dim], previousPos[dim], h_boxSize[dim]);
            distanceSq += delta[dim] * delta[dim];
          }
          length = sqrt(distanceSq);
          //for (long dim = 0; dim < nDim; dim++) {
          //  relSegment[dim] = pbcDistance(thisPos[dim], secondPreviousPos[dim], h_boxSize[dim]);
          //  relPos[dim] = secondPreviousPos[dim] + relSegment[dim];
          //}
          double previousProj = (pbcDistance(thisPos[0], secondPreviousPos[0], h_boxSize[0]) * pbcDistance(previousPos[0], secondPreviousPos[0], h_boxSize[0]) + pbcDistance(thisPos[1], secondPreviousPos[1], h_boxSize[1]) * pbcDistance(previousPos[1], secondPreviousPos[1], h_boxSize[1])) / (length * length);
          //double previousProj = getProjection(thisPos, previousPos, secondPreviousPos, length);
          if(previousProj >= 1) { // no for concave - yes for convex
            distanceSq = 0;
            for (long dim = 0; dim < nDim; dim++) {
              delta[dim] = pbcDistance(thisPos[dim], previousPos[dim], h_boxSize[dim]);
              distanceSq += delta[dim] * delta[dim];
            }
            distance = sqrt(distanceSq);
            bool addForce = false;
            switch (simControl.potentialType) {
              case simControlStruct::potentialEnum::harmonic:
              overlap = 1 - distance / radSum;
              if(overlap > 0) {
                addForce = true;
                gradMultiple = ec * overlap / radSum;
                epot = (0.5 * ec * overlap * overlap) * 0.5;
              }
              break;
              case simControlStruct::potentialEnum::wca:
              if(distance <= (WCAcut * radSum)) {
                addForce = true;
                ratio = radSum / distance;
                ratio6 = pow(ratio, 6);
                ratio12 = ratio6 * ratio6;
                gradMultiple = 4 * ec * (12 * ratio12 - 6 * ratio6) / distance;
                epot = 0.5 * ec * (4 * (ratio12 - ratio6) + 1);
              }
              break;
            }
            if(addForce == true) {
            //cout << vertexId << " vertex-vertex interacton with " << previousId << " projection: " << projection << " force: " << gradMultiple << endl;
            //cout << "PREVIOUS: checking vertexId: " << vertexId << " and previousId: " << previousId << endl;
              for (long dim = 0; dim < nDim; dim++) {
                h_force[vertexId * nDim + dim] += gradMultiple * delta[dim] / distance;
                h_force[previousId * nDim + dim] -= gradMultiple * delta[dim] / distance;
                h_interaction[vertexId * nDim + dim] += gradMultiple * delta[dim] / distance;
                h_interaction[previousId * nDim + dim] -= gradMultiple * delta[dim] / distance;
              }
              h_particleEnergy[particleId] += epot;
              h_particleEnergy[otherParticleId] += epot;
            }
            /*// check inverse interaction
            for (long dim = 0; dim < nDim; dim++) {
              interSegment[dim] = pbcDistance(thisPos[dim], previousPos[dim], h_boxSize[dim]);
              previousSegment[dim] = pbcDistance(previousPos[dim], secondPreviousPos[dim], h_boxSize[dim]);
            }
						auto endEndAngle = atan2(interSegment[0]*segment[1] - interSegment[1]*segment[0], interSegment[0]*segment[0] + interSegment[1]*segment[1]);
						checkAngle(endEndAngle, PI/2);
						auto endCapAngle = atan2(previousSegment[0]*segment[1] - previousSegment[1]*segment[0], previousSegment[0]*segment[0] + previousSegment[1]*segment[1]);
						checkAngle(endCapAngle, PI);
						auto isCapConvexInteraction = (endEndAngle >= 0 && endEndAngle <= endCapAngle);
						auto isCapConcaveInteraction = (endCapAngle < 0 && endEndAngle > (PI - fabs(endCapAngle)) && endEndAngle < PI);
						//isConcaveInteraction = false;
						auto isCapInteraction = (isCapConvexInteraction || isCapConcaveInteraction);
						// check if the interaction is inverse
						auto inverseEndEndAngle = (endEndAngle - 2*PI * (endEndAngle > PI));
						auto isConcaveInteraction = (endCapAngle < 0 && inverseEndEndAngle < 0 && inverseEndEndAngle >= endCapAngle);
						// endEndAngle for other end of the segment
						endEndAngle = PI - endEndAngle + fabs(endCapAngle);
						endEndAngle -= 2*PI * (endEndAngle > 2*PI);
						endEndAngle += 2*PI * (endEndAngle < 0);
						auto isInverseInteraction = (isConcaveInteraction || (endCapAngle > 0 && (endEndAngle < endCapAngle)));
						if((projection < 0 && isCapInteraction) || (projection > 0 && isInverseInteraction)) {
							if(isInverseInteraction) {
                distanceSq = 0;
                for (long dim = 0; dim < nDim; dim++) {
                  delta[dim] = pbcDistance(previousPos[dim], thisPos[dim], h_boxSize[dim]);
                  distanceSq += delta[dim] * delta[dim];
                }
                distance = sqrt(distanceSq);
                bool addForce = false;
                switch (simControl.potentialType) {
                  case simControlStruct::potentialEnum::harmonic:
                  overlap = 1 - distance / radSum;
                  if(overlap > 0) {
                    addForce = true;
                    gradMultiple = ec * overlap / radSum;
                    epot = (0.5 * ec * overlap * overlap) * 0.5;
                  }
                  break;
                  case simControlStruct::potentialEnum::wca:
                  if(distance <= (WCAcut * radSum)) {
                    addForce = true;
                    ratio = radSum / distance;
                    ratio6 = pow(ratio, 6);
                    ratio12 = ratio6 * ratio6;
                    gradMultiple = 4 * ec * (12 * ratio12 - 6 * ratio6) / distance;
                    epot = 0.5 * ec * (4 * (ratio12 - ratio6) + 1);
                  }
                  break;
                }
                if(addForce == true) {
                cout << vertexId << " inverse vertex-vertex interacton with " << previousId << " projection: " << projection << " force: " << gradMultiple << endl;
                //cout << "PREVIOUS: checking vertexId: " << vertexId << " and previousId: " << previousId << endl;
                  for (long dim = 0; dim < nDim; dim++) {
                    h_force[vertexId * nDim + dim] += gradMultiple * delta[dim] / distance;
                    h_force[previousId * nDim + dim] -= gradMultiple * delta[dim] / distance;
                    h_interaction[vertexId * nDim + dim] += gradMultiple * delta[dim] / distance;
                    h_interaction[previousId * nDim + dim] -= gradMultiple * delta[dim] / distance;
                  }
                  h_particleEnergy[particleId] += epot;
                  h_particleEnergy[otherParticleId] += epot;
                }
							}
            }*/
          }
        }
      }
    }
  }
  d_force = h_force;
  d_particleEnergy = h_particleEnergy;
}

void DPM2D::calcSmoothInteractionOMP() {
  h_pos = d_pos;
  h_force = d_force;
  h_particleEnergy = d_particleEnergy;
  thrust::fill(h_interaction.begin(), h_interaction.end(), double(0));
  //#pragma omp parallel for default(none) shared(d_pos, d_rad, d_particleIdList, d_neighborList, d_maxNeighborList, d_boxSize, d_force, d_energy, d_particleEnergy, simControl, ec, WCAcut, numVertices, neighborListSize, nDim)
  #pragma omp parallel for shared(h_force, h_particleEnergy, h_interaction)//reduction(+,h_force) reduction(+,h_particleEnergy) reduction(+,h_interaction)
  {
    for (long vertexId = 0; vertexId < numVertices; vertexId++) {
      double thisPos[MAXDIM], otherPos[MAXDIM], previousPos[MAXDIM], secondPreviousPos[MAXDIM];
      double delta[MAXDIM], segment[MAXDIM], projPos[MAXDIM];//, relSegment[MAXDIM], relPos[MAXDIM];
      double distance, gradMultiple, overlap, ratio, ratio6, ratio12, epot;
      long particleId = d_particleIdList[vertexId];
      for (long dim = 0; dim < nDim; dim++) {
        thisPos[dim] = h_pos[vertexId * nDim + dim];
      }
      double thisRad = h_rad[vertexId];
      for (long nListId = 0; nListId < h_maxNeighborList[vertexId]; nListId++) {
        long otherId = h_neighborList[vertexId * neighborListSize + nListId];
        if ((vertexId != otherId) && (otherId != -1)) {
          for (long dim = 0; dim < nDim; dim++) {
            otherPos[dim] = h_pos[otherId * nDim + dim];
          }
          double otherRad = h_rad[otherId];
          double radSum = thisRad + otherRad;
          // get previous vertex
          long otherParticleId = h_particleIdList[otherId];
          long previousId = getPreviousId(otherId, otherParticleId);
          double distanceSq = 0;
          for (long dim = 0; dim < nDim; dim++) {
            previousPos[dim] = d_pos[previousId * nDim + dim];
            segment[dim] = pbcDistance(otherPos[dim], previousPos[dim], h_boxSize[dim]);
            distanceSq += segment[dim] * segment[dim];
          }
          double length = sqrt(distanceSq);
          // this takes into account periodic boundaries with respect to the segment
          //for (long dim = 0; dim < nDim; dim++) {
          //  relSegment[dim] = pbcDistance(thisPos[dim], previousPos[dim], h_boxSize[dim]);
          //  relPos[dim] = previousPos[dim] + relSegment[dim];
          //}
          // compute projection on the line between other and previous
          double projection = (pbcDistance(thisPos[0], previousPos[0], h_boxSize[0]) * pbcDistance(otherPos[0], previousPos[0], h_boxSize[0]) + pbcDistance(thisPos[1], previousPos[1], h_boxSize[1]) * pbcDistance(otherPos[1], previousPos[1], h_boxSize[1])) / (length * length);
          //double projection = getProjection(relPos, otherPos, previousPos, length);
          //cout << "projection: " << projection << " length: " << length << endl;
          if(projection > 0 && projection <= 1) {
            double reducedProj = max(0.0, min(1.0, projection));
            for (long dim = 0; dim < nDim; dim++) {
              projPos[dim] = previousPos[dim] + reducedProj * segment[dim];
            }
            distanceSq = 0;
            for (long dim = 0; dim < nDim; dim++) {
              delta[dim] = pbcDistance(thisPos[dim], projPos[dim], h_boxSize[dim]);
              distanceSq += delta[dim] * delta[dim];
            }
            distance = sqrt(distanceSq);
            bool addForce = false;
            switch (simControl.potentialType) {
              case simControlStruct::potentialEnum::harmonic:
              overlap = 1 - distance / radSum;
              if(overlap > 0) {
                addForce = true;
                gradMultiple = ec * overlap / radSum;
                epot = (0.5 * ec * overlap * overlap) * 0.5;
              }
              break;
              case simControlStruct::potentialEnum::wca:
              if(distance <= (WCAcut * radSum)) {
                addForce = true;
                ratio = radSum / distance;
                ratio6 = pow(ratio, 6);
                ratio12 = ratio6 * ratio6;
                gradMultiple = 4 * ec * (12 * ratio12 - 6 * ratio6) / distance;
                epot = 0.5 * ec * (4 * (ratio12 - ratio6) + 1);
              }
              break;
            }
            if(addForce == true) {
              double cross = pbcDistance(previousPos[0], otherPos[0], h_boxSize[0]) * pbcDistance(otherPos[1], thisPos[1], h_boxSize[1]) - pbcDistance(otherPos[0], thisPos[0], h_boxSize[0]) * pbcDistance(previousPos[1], otherPos[1], h_boxSize[1]);
              //double cross = calcCross(thisPos, otherPos, previousPos);
              double absCross = fabs(cross);
              double sign = absCross / cross;
              //#pragma omp atomic seq_cst
              //{
                // this vertex
                h_force[vertexId * nDim] += gradMultiple * sign * pbcDistance(previousPos[1], otherPos[1], h_boxSize[1]) / length;
                h_force[vertexId * nDim + 1] += gradMultiple * sign * pbcDistance(otherPos[0], previousPos[0], h_boxSize[0]) / length;
                // other vertex
                h_force[otherId * nDim] += gradMultiple * (sign * pbcDistance(thisPos[1], previousPos[1], h_boxSize[1]) + absCross * pbcDistance(previousPos[0], otherPos[0], h_boxSize[0]) / (length * length)) / length;
                h_force[otherId * nDim + 1] += gradMultiple * (sign * pbcDistance(previousPos[0], thisPos[0], h_boxSize[0]) + absCross * pbcDistance(previousPos[1], otherPos[1], h_boxSize[1]) / (length * length)) / length;
                // previous vertex
                h_force[previousId * nDim] += gradMultiple * (sign * pbcDistance(otherPos[1], thisPos[1], h_boxSize[1]) - absCross * pbcDistance(previousPos[0], otherPos[0], h_boxSize[0]) / (length * length)) / length;
                h_force[previousId * nDim + 1] += gradMultiple * (sign * pbcDistance(thisPos[0], otherPos[0], h_boxSize[0]) - absCross * pbcDistance(previousPos[1], otherPos[1], h_boxSize[1]) / (length * length)) / length; 
                // add energy
                h_particleEnergy[particleId] += epot;
                h_particleEnergy[otherParticleId] += epot;
                // this vertex
                h_interaction[vertexId * nDim] += gradMultiple * sign * pbcDistance(previousPos[1], otherPos[1], h_boxSize[1]) / length;
                h_interaction[vertexId * nDim + 1] += gradMultiple * sign * pbcDistance(otherPos[0], previousPos[0], h_boxSize[0]) / length;
                // other vertex
                h_interaction[otherId * nDim] += gradMultiple * (sign * pbcDistance(thisPos[1], previousPos[1], h_boxSize[1]) + absCross * pbcDistance(previousPos[0], otherPos[0], h_boxSize[0]) / (length * length)) / length;
                h_interaction[otherId * nDim + 1] += gradMultiple * (sign * pbcDistance(previousPos[0], thisPos[0], h_boxSize[0]) + absCross * pbcDistance(previousPos[1], otherPos[1], h_boxSize[1]) / (length * length)) / length;
                // previous vertex
                h_interaction[previousId * nDim] += gradMultiple * (sign * pbcDistance(otherPos[1], thisPos[1], h_boxSize[1]) - absCross * pbcDistance(previousPos[0], otherPos[0], h_boxSize[0]) / (length * length)) / length;
                h_interaction[previousId * nDim + 1] += gradMultiple * (sign * pbcDistance(thisPos[0], otherPos[0], h_boxSize[0]) - absCross * pbcDistance(previousPos[1], otherPos[1], h_boxSize[1]) / (length * length)) / length;
              //}
            }
          } else if(projection <= 0) {
            // check that previous vertex is not already interacting with this vertex through a segment
            long secondPreviousId = getPreviousId(previousId, otherParticleId);
            distanceSq = 0;
            for (long dim = 0; dim < nDim; dim++) {
              secondPreviousPos[dim] = d_pos[secondPreviousId * nDim + dim];
              delta[dim] = pbcDistance(secondPreviousPos[dim], previousPos[dim], h_boxSize[dim]);
              distanceSq += delta[dim] * delta[dim];
            }
            length = sqrt(distanceSq);
            //for (long dim = 0; dim < nDim; dim++) {
            //  relSegment[dim] = pbcDistance(thisPos[dim], secondPreviousPos[dim], h_boxSize[dim]);
            //  relPos[dim] = secondPreviousPos[dim] + relSegment[dim];
            //}
            double previousProj = (pbcDistance(thisPos[0], secondPreviousPos[0], h_boxSize[0]) * pbcDistance(previousPos[0], secondPreviousPos[0], h_boxSize[0]) + pbcDistance(thisPos[1], secondPreviousPos[1], h_boxSize[1]) * pbcDistance(previousPos[1], secondPreviousPos[1], h_boxSize[1])) / (length * length);
            //double previousProj = getProjection(relPos, previousPos, secondPreviousPos, length);
            if(previousProj > 1) {
              distanceSq = 0;
              for (long dim = 0; dim < nDim; dim++) {
                delta[dim] = pbcDistance(thisPos[dim], previousPos[dim], h_boxSize[dim]);
                distanceSq += delta[dim] * delta[dim];
              }
              distance = sqrt(distanceSq);
              bool addForce = false;
              switch (simControl.potentialType) {
                case simControlStruct::potentialEnum::harmonic:
                overlap = 1 - distance / radSum;
                if(overlap > 0) {
                  addForce = true;
                  gradMultiple = ec * overlap / radSum;
                  epot = (0.5 * ec * overlap * overlap) * 0.5;
                }
                break;
                case simControlStruct::potentialEnum::wca:
                if(distance <= (WCAcut * radSum)) {
                  addForce = true;
                  ratio = radSum / distance;
                  ratio6 = pow(ratio, 6);
                  ratio12 = ratio6 * ratio6;
                  gradMultiple = 4 * ec * (12 * ratio12 - 6 * ratio6) / distance;
                  epot = 0.5 * ec * (4 * (ratio12 - ratio6) + 1);
                }
                break;
              }
              if(addForce == true) {
                //cout << "PREVIOUS: checking vertexId: " << vertexId << " and previousId: " << previousId << endl;
                //#pragma omp atomic seq_cst
                //{
                  for (long dim = 0; dim < nDim; dim++) {
                    h_force[vertexId * nDim + dim] += gradMultiple * delta[dim] / distance;
                    h_force[previousId * nDim + dim] -= gradMultiple * delta[dim] / distance;
                    h_interaction[vertexId * nDim + dim] += gradMultiple * delta[dim] / distance;
                    h_interaction[previousId * nDim + dim] -= gradMultiple * delta[dim] / distance;
                  }
                  h_particleEnergy[particleId] += epot;
                  h_particleEnergy[otherParticleId] += epot;
                //}
              }
            }
          }
        }
      }
    }
  }
  d_force = h_force;
  d_particleEnergy = h_particleEnergy;
}

long DPM2D::getNeighborCellId(long cellIdx, long cellIdy, long dx, long dy) {
	// check boundary conditions
	long cIdx = cellIdx + dx;
	if(cIdx >= numCells) {
		cIdx -= numCells;
	} else if(cIdx < 0) {
		cIdx += numCells;
	}
	long cIdy = cellIdy + dy;
	if(cIdy >= numCells) {
		cIdy -= numCells;
	} else if(cIdy < 0) {
		cIdy += numCells;
	}
	return cIdx * numCells + cIdy;
}

void DPM2D::calcCellListSmoothInteraction() {
  thrust::fill(d_particleEnergy.begin(), d_particleEnergy.end(), double(0));
  long cellIdx, cellIdy, dx, dy, otherCellId;
  long vertexId, otherId, particleId, otherParticleId, previousId, secondPreviousId;
  double distance, distanceSq, thisRad, otherRad, radSum, projection, length, cross, absCross, sign, previousProj;
  double thisPos[MAXDIM], otherPos[MAXDIM], previousPos[MAXDIM], secondPreviousPos[MAXDIM];
  double delta[MAXDIM], segment[MAXDIM], projPos[MAXDIM], relSegment[MAXDIM]; //relPos[MAXDIM]
  double gradMultiple, epot, overlap, ratio, ratio6, ratio12;
  //for (vertexId = 0; vertexId < numVertices; vertexId++) {
  for (long cellId = 0; cellId < numCells * numCells; cellId++) {
    for (vertexId = h_header[cellId]; vertexId != -1L; vertexId = h_linkedList[vertexId]) {
      particleId = d_particleIdList[vertexId];
      for (long dim = 0; dim < nDim; dim++) {
        thisPos[dim] = d_pos[vertexId * nDim + dim];
      }
      thisRad = d_rad[vertexId];
      //cellIdx = h_cellIndexList[vertexId * nDim];
      //cellIdy = h_cellIndexList[vertexId * nDim + 1];
      cellIdx = static_cast<long>(thisPos[0] / cellSize);
      cellIdy = static_cast<long>(thisPos[1] / cellSize);
      // loop over neighboring cells
      for (dx = -1; dx <= 1; dx++) {
        for (dy = -1; dy <= 1; dy++) {
          otherCellId = getNeighborCellId(cellIdx, cellIdy, dx, dy);
          for (otherId = h_header[otherCellId]; otherId != -1L; otherId = h_linkedList[otherId]) {
            otherParticleId = d_particleIdList[otherId];
            if ((vertexId != otherId) && (particleId != otherParticleId)) {
              //cout << "vertexId " << vertexId << " cellId " << cellIdx * numCells + cellIdy << " otherId " << otherId << " otherCellId " << otherCellId << endl;
              for (long dim = 0; dim < nDim; dim++) {
                otherPos[dim] = d_pos[otherId * nDim + dim];
              }
              otherRad = d_rad[otherId];
              radSum = thisRad + otherRad;
              // get previous vertex
              previousId = getPreviousId(otherId, otherParticleId);
              distanceSq = 0;
              for (long dim = 0; dim < nDim; dim++) {
                previousPos[dim] = d_pos[previousId * nDim + dim];
                segment[dim] = pbcDistance(otherPos[dim], previousPos[dim], d_boxSize[dim]);
                distanceSq += segment[dim] * segment[dim];
              }
              length = sqrt(distanceSq);
              for (long dim = 0; dim < nDim; dim++) {
                relSegment[dim] = pbcDistance(thisPos[dim], previousPos[dim], d_boxSize[dim]);
                thisPos[dim] = previousPos[dim] + relSegment[dim];
              }
              // compute projection on the line between other and previous
              //projection = (pbcDistance(thisPos[0], previousPos[0], d_boxSize[0]) * pbcDistance(otherPos[0], previousPos[0], d_boxSize[0]) + pbcDistance(thisPos[1], previousPos[1], d_boxSize[1]) * pbcDistance(otherPos[1], previousPos[1], d_boxSize[1])) / (length * length);
              projection = getProjection(thisPos, otherPos, previousPos, length);
              //cout << "projection: " << projection << " length: " << length << endl;
              if(projection > 0 && projection <= 1) {
                double reducedProj = max(0.0, min(1.0, projection));
                for (long dim = 0; dim < nDim; dim++) {
                  projPos[dim] = previousPos[dim] + reducedProj * segment[dim];
                }
                distanceSq = 0;
                for (long dim = 0; dim < nDim; dim++) {
                  delta[dim] = pbcDistance(thisPos[dim], projPos[dim], d_boxSize[dim]);
                  distanceSq += delta[dim] * delta[dim];
                }
                distance = sqrt(distanceSq);
                bool addForce = false;
                switch (simControl.potentialType) {
                  case simControlStruct::potentialEnum::harmonic:
                  overlap = 1 - distance / radSum;
                  if(overlap > 0) {
                    addForce = true;
                    gradMultiple = ec * overlap / radSum;
                    epot = (0.5 * ec * overlap * overlap);
                  }
                  break;
                  case simControlStruct::potentialEnum::wca:
                  if(distance <= (WCAcut * radSum)) {
                    addForce = true;
                    ratio = radSum / distance;
                    ratio6 = pow(ratio, 6);
                    ratio12 = ratio6 * ratio6;
                    gradMultiple = 4 * ec * (12 * ratio12 - 6 * ratio6) / distance;
                    epot = 0.5 * ec * (4 * (ratio12 - ratio6) + 1);
                  }
                  break;
                }
                if(addForce == true) {
                  //cross = pbcDistance(previousPos[0], otherPos[0], d_boxSize[0]) * pbcDistance(otherPos[1], thisPos[1], d_boxSize[1]) - pbcDistance(otherPos[0], thisPos[0], d_boxSize[0]) * pbcDistance(previousPos[1], otherPos[1], d_boxSize[1]);
                  cross = calcCross(thisPos, otherPos, previousPos);
                  absCross = fabs(cross);
                  sign = absCross / cross;
                  d_force[vertexId * nDim] += gradMultiple * sign * pbcDistance(previousPos[1], otherPos[1], d_boxSize[1]) / length;
                  d_force[vertexId * nDim + 1] += gradMultiple * sign * pbcDistance(otherPos[0], previousPos[0], d_boxSize[0]) / length;
                  // other vertex
                  d_force[otherId * nDim] += gradMultiple * (sign * pbcDistance(thisPos[1], previousPos[1], d_boxSize[1]) + absCross * pbcDistance(previousPos[0], otherPos[0], d_boxSize[0]) / (length * length)) / length;
                  d_force[otherId * nDim + 1] += gradMultiple * (sign * pbcDistance(previousPos[0], thisPos[0], d_boxSize[0]) + absCross * pbcDistance(previousPos[1], otherPos[1], d_boxSize[1]) / (length * length)) / length;
                  // previous vertex
                  d_force[previousId * nDim] += gradMultiple * (sign * pbcDistance(otherPos[1], thisPos[1], d_boxSize[1]) - absCross * pbcDistance(previousPos[0], otherPos[0], d_boxSize[0]) / (length * length)) / length;
                  d_force[previousId * nDim + 1] += gradMultiple * (sign * pbcDistance(thisPos[0], otherPos[0], d_boxSize[0]) - absCross * pbcDistance(previousPos[1], otherPos[1], d_boxSize[1]) / (length * length)) / length;
                  d_particleEnergy[particleId] += epot * 0.5;
                  d_particleEnergy[otherParticleId] += epot * 0.5;
                }
              } else if(projection <= 0) {
                secondPreviousId = getPreviousId(previousId, otherParticleId);
                distanceSq = 0;
                for (long dim = 0; dim < nDim; dim++) {
                  secondPreviousPos[dim] = d_pos[secondPreviousId * nDim + dim];
                  delta[dim] = pbcDistance(previousPos[dim], secondPreviousPos[dim], d_boxSize[dim]);
                  distanceSq += delta[dim] * delta[dim];
                }
                length = sqrt(distanceSq);
                previousProj = getProjection(thisPos, previousPos, secondPreviousPos, length);
                if(previousProj > 1) {
                  distanceSq = 0;
                  for (long dim = 0; dim < nDim; dim++) {
                    delta[dim] = pbcDistance(thisPos[dim], previousPos[dim], d_boxSize[dim]);
                    distanceSq += delta[dim] * delta[dim];
                  }
                  distance = sqrt(distanceSq);
                  bool addForce = false;
                  switch (simControl.potentialType) {
                    case simControlStruct::potentialEnum::harmonic:
                    overlap = 1 - distance / radSum;
                    if(overlap > 0) {
                      addForce = true;
                      gradMultiple = ec * overlap / radSum;
                      epot = (0.5 * ec * overlap * overlap);
                    }
                    break;
                    case simControlStruct::potentialEnum::wca:
                    if(distance <= (WCAcut * radSum)) {
                      addForce = true;
                      ratio = radSum / distance;
                      ratio6 = pow(ratio, 6);
                      ratio12 = ratio6 * ratio6;
                      gradMultiple = 4 * ec * (12 * ratio12 - 6 * ratio6) / distance;
                      epot = 0.5 * ec * (4 * (ratio12 - ratio6) + 1);
                    }
                    break;
                  }
                  if(addForce == true) {
                    //cout << "PREVIOUS: checking vertexId: " << vertexId << " and previousId: " << previousId << endl;
                    for (long dim = 0; dim < nDim; dim++) {
                      d_force[vertexId * nDim + dim] += 0.5 * gradMultiple * delta[dim] / distance;
                      d_force[previousId * nDim + dim] -= 0.5 * gradMultiple * delta[dim] / distance;
                    }
                    d_particleEnergy[particleId] += epot * 0.5;
                    d_particleEnergy[otherParticleId] += epot * 0.5;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  fillLinkedList();
}

void DPM2D::calcVertexForceTorque() {
  const double *rad = thrust::raw_pointer_cast(&d_rad[0]);
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  const double *particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *force = thrust::raw_pointer_cast(&d_force[0]);
  double *torque = thrust::raw_pointer_cast(&d_torque[0]);
	double *energy = thrust::raw_pointer_cast(&d_energy[0]);
  // torque here is used for angular acceleration
  kernelCalcVertexForceTorque<<<dimGrid, dimBlock>>>(rad, pos, particlePos, force, torque, energy);
  //cout << "vertex force 0: " << d_force[0] << " " << d_force[1] << " total: " << getTotalForceMagnitude() << endl;
}

void DPM2D::calcVertexSmoothForceTorque() {
  thrust::fill(d_particleEnergy.begin(), d_particleEnergy.end(), double(0));
  const double *rad = thrust::raw_pointer_cast(&d_rad[0]);
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  const double *particlePos = thrust::raw_pointer_cast(&d_particlePos[0]);
  double *force = thrust::raw_pointer_cast(&d_force[0]);
  double *torque = thrust::raw_pointer_cast(&d_torque[0]);
	double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  // torque here is used for angular acceleration
  kernelCalcVertexSmoothForceTorque<<<dimGrid, dimBlock>>>(rad, pos, particlePos, force, torque, pEnergy);
  //cout << "vertex force 0: " << d_force[0] << " " << d_force[1] << " total: " << getTotalForceMagnitude() << endl;
}

void DPM2D::calcRigidForceEnergy() {
  calcParticlePositions();
  switch (simControl.interactionType) {
    case simControlStruct::interactionEnum::vertexVertex:
    calcVertexForceTorque();
    transferForceToParticles();
    break;
    case simControlStruct::interactionEnum::vertexSmooth:
    calcVertexSmoothForceTorque();
    transferSmoothForceToParticles();
    break;
    default:
    break;
  }
}

void DPM2D::transferForceToParticles() {
  // vertex variables
	const double *force = thrust::raw_pointer_cast(&d_force[0]);
  const double *torque = thrust::raw_pointer_cast(&d_torque[0]);
	const double *energy = thrust::raw_pointer_cast(&d_energy[0]);
  // particle variables
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
  double *pTorque = thrust::raw_pointer_cast(&d_particleTorque[0]);
  double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  // sum force and torque over vertices of particle
  kernelCalcParticleRigidForceEnergy<<<dimGrid, dimBlock>>>(force, torque, energy, pForce, pTorque, pEnergy);
}

void DPM2D::transferSmoothForceToParticles() {
  // vertex variables
	const double *force = thrust::raw_pointer_cast(&d_force[0]);
  const double *torque = thrust::raw_pointer_cast(&d_torque[0]);
  // particle variables
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
  double *pTorque = thrust::raw_pointer_cast(&d_particleTorque[0]);
  // sum force and torque over vertices of particle
  kernelCalcParticleSmoothRigidForceEnergy<<<dimGrid, dimBlock>>>(force, torque, pForce, pTorque);
}

void DPM2D::calcStressTensor() {
  calcPerParticleStressTensor();
  thrust::fill(d_stress.begin(), d_stress.end(), double(0));
  const double *perPStress = thrust::raw_pointer_cast(&d_perParticleStress[0]);
	double *stress = thrust::raw_pointer_cast(&d_stress[0]);
  kernelCalcStressTensor<<<partDimGrid, dimBlock>>>(perPStress, stress);
}

void DPM2D::calcPerParticleStressTensor() {
  thrust::fill(d_stress.begin(), d_stress.end(), double(0));
  const double *rad = thrust::raw_pointer_cast(&d_rad[0]);
	const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *perPStress = thrust::raw_pointer_cast(&d_perParticleStress[0]);
  kernelCalcPerParticleStressTensor<<<partDimGrid, dimBlock>>>(rad, pos, pPos, perPStress);
}

void DPM2D::calcNeighborForces() {
  thrust::host_vector<double> neighborForce;
  neighborForce.resize(numVertices * neighborListSize * nDim);
  thrust::fill(neighborForce.begin(), neighborForce.end(), 0);
	const double *rad = thrust::raw_pointer_cast(&d_rad[0]);
  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
  double *neighforce = thrust::raw_pointer_cast(&neighborForce[0]);
  kernelCalcNeighborForces<<<dimGrid, dimBlock>>>(pos, rad, neighforce);
}

//************************* contacts and neighbors ***************************//
void DPM2D::calcParticleNeighbors() {
  long largestNeighbor = 8*nDim; // Guess
	do {
		//Make a contactList that is the right size
		neighborLimit = largestNeighbor;
		d_partNeighborList = thrust::device_vector<long>(numParticles * neighborLimit);
		//Prefill the contactList with -1
		thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), -1L);
		thrust::fill(d_numPartNeighbors.begin(), d_numPartNeighbors.end(), -1L);
		//Create device_pointers from thrust arrays
		double* pos = thrust::raw_pointer_cast(&d_pos[0]);
		double* rad = thrust::raw_pointer_cast(&d_rad[0]);
		long* pNeighborList = thrust::raw_pointer_cast(&d_partNeighborList[0]);
		long* numPNeighbors = thrust::raw_pointer_cast(&d_numPartNeighbors[0]);
		kernelCalcParticleNeighbors<<<dimGrid, dimBlock>>>(pos, rad, neighborLimit, pNeighborList, numPNeighbors);
		//Calculate the maximum number of contacts
		largestNeighbor = thrust::reduce(d_numPartNeighbors.begin(), d_numPartNeighbors.end(), -1L, thrust::maximum<long>());
    //cout << "DPM2D::calcParticleNeighbors: largestNeighbor = " << largestNeighbor << endl;
	} while(neighborLimit < largestNeighbor); // If the guess was not good, do it again
}

void DPM2D::calcContacts(double gapSize) {
  long largestContact = 8*nDim; // Guess
	do {
		//Make a contactList that is the right size
		contactLimit = largestContact;
		d_contactList = thrust::device_vector<long>(numParticles * contactLimit);
		//Prefill the contactList with -1
		thrust::fill(d_contactList.begin(), d_contactList.end(), -1L);
		thrust::fill(d_numContacts.begin(), d_numContacts.end(), -1L);
		//Create device_pointers from thrust arrays
		const double* pos = thrust::raw_pointer_cast(&d_pos[0]);
		const double* rad = thrust::raw_pointer_cast(&d_rad[0]);
		long* contactList = thrust::raw_pointer_cast(&d_contactList[0]);
		long* numContacts = thrust::raw_pointer_cast(&d_numContacts[0]);
		kernelCalcContacts<<<dimGrid, dimBlock>>>(pos, rad, gapSize, contactLimit, contactList, numContacts);
		//Calculate the maximum number of contacts
		largestContact = thrust::reduce(d_numContacts.begin(), d_numContacts.end(), -1L, thrust::maximum<long>());
    //cout << "DPM2D::calcContacts: largestContact = " << largestContact << endl;
	} while(contactLimit < largestContact); // If the guess was not good, do it again
}

//Return normalized contact vectors between every pair of particles in contact
thrust::host_vector<long> DPM2D::getContactVectors(double gapSize) {
	//Calculate the set of contacts
	calcContacts(gapSize);
	//Calculate the maximum number of contacts
	maxContacts = thrust::reduce(d_numContacts.begin(), d_numContacts.end(), -1L, thrust::maximum<long>());
	//Create the array to hold the contactVectors
	d_contactVectorList.resize(numParticles * nDim * maxContacts);
	thrust::fill(d_contactVectorList.begin(), d_contactVectorList.end(), double(0));
	double* pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	long* contactList = thrust::raw_pointer_cast(&d_contactList[0]);
	double* contactVectorList = thrust::raw_pointer_cast(&d_contactVectorList[0]);
	kernelCalcContactVectorList<<<dimGrid, dimBlock>>>(pPos, contactList, d_contactList.size()/numParticles, maxContacts, contactVectorList);
  // convert to host and return
  thrust::host_vector<long> contactVectorListFromDevice;
  contactVectorListFromDevice = d_contactVectorList;
  return contactVectorListFromDevice;
}

//*************************** vertex neighbors *******************************//
void DPM2D::calcNeighbors(double cutDistance) {
  switch (simControl.neighborType) {
    case simControlStruct::neighborEnum::neighbor:
    calcNeighborList(cutDistance);
    break;
    case simControlStruct::neighborEnum::cell:
    //fillLinkedList();
    break;
    case simControlStruct::neighborEnum::allToAll:
    break;
    default:
    break;
  }
}

void DPM2D::calcNeighborList(double cutDistance) {
  thrust::fill(d_maxNeighborList.begin(), d_maxNeighborList.end(), 0);
	thrust::fill(d_neighborList.begin(), d_neighborList.end(), -1L);
  syncNeighborsToDevice();

  const double *pos = thrust::raw_pointer_cast(&d_pos[0]);
	const double *rad = thrust::raw_pointer_cast(&d_rad[0]);

  kernelCalcNeighborList<<<dimGrid, dimBlock>>>(pos, rad, cutDistance);
  // compute maximum number of neighbors per particle
  maxNeighbors = thrust::reduce(d_maxNeighborList.begin(), d_maxNeighborList.end(), -1L, thrust::maximum<long>());
  syncNeighborsToDevice();
  //cout << "\n DPM2D::calcNeighborList: maxNeighbors: " << maxNeighbors << endl;

  // if the neighbors don't fit, resize the neighbor list
  if ( maxNeighbors > neighborListSize ) {
		neighborListSize = pow(2, ceil(std::log2(maxNeighbors)));
    //cout << "neighborListSize: " << neighborListSize << endl;
		// now create the actual storage and then put the neighbors in it
		d_neighborList.resize(numVertices * neighborListSize);
		// pre-fill the neighborList with -1
		thrust::fill(d_neighborList.begin(), d_neighborList.end(), -1L);
		syncNeighborsToDevice();
		kernelCalcNeighborList<<<dimGrid, dimBlock>>>(pos, rad, cutDistance);
	}
  h_neighborList.resize(d_neighborList.size());
  h_neighborList = d_neighborList;
  h_maxNeighborList = d_maxNeighborList;
}

void DPM2D::syncNeighborsToDevice() {
	//Copy the pointers and information about neighbors to the gpu
	cudaMemcpyToSymbol(d_neighborListSize, &neighborListSize, sizeof(neighborListSize));
	cudaMemcpyToSymbol(d_maxNeighbors, &maxNeighbors, sizeof(maxNeighbors));

	long* maxNeighborList = thrust::raw_pointer_cast(&d_maxNeighborList[0]);
	cudaMemcpyToSymbol(d_maxNeighborListPtr, &maxNeighborList, sizeof(maxNeighborList));

	long* neighborList = thrust::raw_pointer_cast(&d_neighborList[0]);
	cudaMemcpyToSymbol(d_neighborListPtr, &neighborList, sizeof(neighborList));
}

//*************************** vertex cell list *****************************//
void DPM2D::fillLinkedList() {
  // reset cell headers
  thrust::fill(h_header.begin(), h_header.end(), -1L);
  thrust::fill(h_cellIndexList.begin(), h_cellIndexList.end(), -1L);
  double thisPos[MAXDIM];
  for (long vertexId = 0; vertexId < numVertices; vertexId++) {
    for (long dim = 0; dim < nDim; dim++) {
      thisPos[dim] = d_pos[vertexId * nDim + dim];
      thisPos[dim] -= floor(thisPos[dim] / d_boxSize[dim]) * d_boxSize[dim];
    }
    long cIdx = static_cast<long>(thisPos[0] / cellSize);
    long cIdy = static_cast<long>(thisPos[1] / cellSize);
    long cId = cIdx * numCells + cIdy;
    //printf("vertexId: %ld cellId %ld \n", vertexId, cId);
    // link to the previous occupant of the list or -1 if it is the first in the cell
    h_linkedList[vertexId] = h_header[cId];
    // the last vertex (at the end of the loop) is the header
    h_header[cId] = vertexId;
    h_cellIndexList[vertexId * nDim] = cIdx;
    h_cellIndexList[vertexId * nDim + 1] = cIdy;
  }
  d_header = h_header;
  d_linkedList = h_linkedList;
  d_cellIndexList = h_cellIndexList;
  syncLinkedListToDevice();
  //for (long cellId = 0; cellId < numCells * numCells; cellId++) {
  //  cout << "cellId " << cellId << " header: " << header[cellId] << endl;
  //}
}

void DPM2D::syncLinkedListToDevice() {
  long* header = thrust::raw_pointer_cast(&d_header[0]);
	cudaMemcpyToSymbol(d_headerPtr, &header, sizeof(header));
  long* linkedList = thrust::raw_pointer_cast(&d_linkedList[0]);
	cudaMemcpyToSymbol(d_linkedListPtr, &linkedList, sizeof(linkedList));
  long* cellIndexList = thrust::raw_pointer_cast(&d_cellIndexList[0]);
	cudaMemcpyToSymbol(d_cellIndexListPtr, &cellIndexList, sizeof(cellIndexList));
}

//************************* particle neighbors *******************************//
void DPM2D::calcParticleNeighborList(double cutDistance) {
  thrust::fill(d_partMaxNeighborList.begin(), d_partMaxNeighborList.end(), 0);
	thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), -1L);
  syncParticleNeighborsToDevice();
  const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);

  kernelCalcParticleNeighborList<<<dimGrid, dimBlock>>>(pPos, pRad, cutDistance);
  // compute maximum number of neighbors per particle
  partMaxNeighbors = thrust::reduce(d_partMaxNeighborList.begin(), d_partMaxNeighborList.end(), -1L, thrust::maximum<long>());
  syncParticleNeighborsToDevice();
  //cout << "DPM2D::calcParticleNeighborList: maxNeighbors: " << partMaxNeighbors << endl;

  // if the neighbors don't fit, resize the neighbor list
  if ( partMaxNeighbors > partNeighborListSize ) {
		partNeighborListSize = pow(2, ceil(std::log2(partMaxNeighbors)));
    //cout << "DPM2D::calcParticleNeighborList: neighborListSize: " << neighborListSize << endl;
		// now create the actual storage and then put the neighbors in it
		d_partNeighborList.resize(numParticles * partNeighborListSize);
		// pre-fill the neighborList with -1
		thrust::fill(d_partNeighborList.begin(), d_partNeighborList.end(), -1L);
		syncParticleNeighborsToDevice();
		kernelCalcParticleNeighborList<<<dimGrid, dimBlock>>>(pPos, pRad, cutDistance);
	}
}

void DPM2D::syncParticleNeighborsToDevice() {
	//Copy the pointers and information about neighbors to the gpu
	cudaMemcpyToSymbol(d_partNeighborListSize, &partNeighborListSize, sizeof(partNeighborListSize));
	cudaMemcpyToSymbol(d_partMaxNeighbors, &partMaxNeighbors, sizeof(partMaxNeighbors));

	long* partMaxNeighborList = thrust::raw_pointer_cast(&d_partMaxNeighborList[0]);
	cudaMemcpyToSymbol(d_partMaxNeighborListPtr, &partMaxNeighborList, sizeof(partMaxNeighborList));

	long* partNeighborList = thrust::raw_pointer_cast(&d_partNeighborList[0]);
	cudaMemcpyToSymbol(d_partNeighborListPtr, &partNeighborList, sizeof(partNeighborList));
}

//************************* particle functions *******************************//
void DPM2D::calcParticleForceEnergy() {
	const double *pRad = thrust::raw_pointer_cast(&d_particleRad[0]);
	const double *pPos = thrust::raw_pointer_cast(&d_particlePos[0]);
	double *pForce = thrust::raw_pointer_cast(&d_particleForce[0]);
	double *pEnergy = thrust::raw_pointer_cast(&d_particleEnergy[0]);
  // compute particle interaction
  kernelCalcParticleInteraction<<<partDimGrid, dimBlock>>>(pRad, pPos, pForce, pEnergy);
}

// return the sum of force magnitudes
double DPM2D::getParticleTotalForceMagnitude() {
  thrust::device_vector<double> forceSquared(d_force.size());
  // compute squared velocities
  thrust::transform(d_particleForce.begin(), d_particleForce.end(), forceSquared.begin(), square());
  // sum squares
  double totalForceMagnitude = sqrt(thrust::reduce(forceSquared.begin(), forceSquared.end(), double(0), thrust::plus<double>()) / (numParticles * nDim));
  forceSquared.clear();
  return totalForceMagnitude;
}

double DPM2D::getParticleMaxUnbalancedForce() {
  thrust::device_vector<double> forceSquared(d_particleForce.size());
  thrust::transform(d_particleForce.begin(), d_particleForce.end(), forceSquared.begin(), square());
  double maxUnbalancedForce = sqrt(thrust::reduce(forceSquared.begin(), forceSquared.end(), double(-1), thrust::maximum<double>()));
  forceSquared.clear();
  return maxUnbalancedForce;
}

double DPM2D::getRigidMaxUnbalancedForce() {
  //calcRigidForceEnergy();
  thrust::device_vector<double> forceSquared(d_particleForce.size());
  thrust::transform(d_particleForce.begin(), d_particleForce.end(), forceSquared.begin(), square());
  double particleMaxUnbalancedForce = sqrt(thrust::reduce(forceSquared.begin(), forceSquared.end(), double(-1), thrust::maximum<double>()));
	forceSquared.resize(d_particleTorque.size());
	thrust::transform(d_particleTorque.begin(), d_particleTorque.end(), forceSquared.begin(), square());
	double particleMaxUnbalancedTorque = sqrt(thrust::reduce(forceSquared.begin(), forceSquared.end(), double(-1), thrust::maximum<double>()));
  forceSquared.clear();
	return std::max(particleMaxUnbalancedForce, particleMaxUnbalancedTorque);
}

double DPM2D::getParticlePotentialEnergy() {
  return thrust::reduce(d_particleEnergy.begin(), d_particleEnergy.end(), double(0), thrust::plus<double>());
}

double DPM2D::getParticleKineticEnergy() {
  thrust::device_vector<double> velSquared(d_particleVel.size());
  // compute squared velocities
  thrust::transform(d_particleVel.begin(), d_particleVel.end(), velSquared.begin(), square());
  // sum squares
  return 0.5 * thrust::reduce(velSquared.begin(), velSquared.end(), double(0), thrust::plus<double>());
}

double DPM2D::getRigidKineticEnergy() {
  double ekin = getParticleKineticEnergy();
  thrust::device_vector<double> angMomentum(d_particleAngvel.size());
  // multiple angular velocity by distance from axis of rotation
  thrust::transform(d_particleAngvel.begin(), d_particleAngvel.end(), d_particleRad.begin(), angMomentum.begin(), thrust::multiplies<double>());
  // compute squared momentum
  thrust::device_vector<double> rotEnergy(d_particleAngvel.size());
  thrust::transform(angMomentum.begin(), angMomentum.end(), rotEnergy.begin(), square());
  ekin += 0.5 * thrust::reduce(rotEnergy.begin(), rotEnergy.end(), double(0), thrust::plus<double>());
  return ekin;
}

double DPM2D::getParticleTemperature() {
  double ekin = getParticleKineticEnergy();
  return 2 * ekin / (numParticles * nDim);
}

double DPM2D::getParticleDrift() {
  return thrust::reduce(d_particlePos.begin(), d_particlePos.end(), double(0), thrust::plus<double>()) / (numParticles * nDim);
}

thrust::host_vector<long> DPM2D::getParticleNeighbors() {
  thrust::host_vector<long> partNeighborListFromDevice;
  partNeighborListFromDevice = d_partNeighborList;
  return partNeighborListFromDevice;
}

//********************************* minimizers *******************************//
void DPM2D::initFIRE(std::vector<double> &FIREparams, long minStep_, long numStep_, long numDOF_) {
  this->fire_ = new FIRE(this);
  if(FIREparams.size() == 7) {
    double a_start_ = FIREparams[0];
    double f_dec_ = FIREparams[1];
    double f_inc_ = FIREparams[2];
    double f_a_ = FIREparams[3];
    double fire_dt_ = FIREparams[4];
    double fire_dt_max_ = FIREparams[5];
    double a_ = FIREparams[6];
    this->fire_->initMinimizer(a_start_, f_dec_, f_inc_, f_a_, fire_dt_, fire_dt_max_, a_, minStep_, numStep_, numDOF_);
    resetParticleLastPositions();
  } else {
    cout << "DPM2D::initFIRE: wrong number of FIRE parameters, must be 7" << endl;
  }
}

void DPM2D::setParticleMassFIRE() {
  //this->fire_->setParticleMass();
  this->fire_->d_mass.resize(numParticles * nDim);
	for (long particleId = 0; particleId < numParticles; particleId++) {
		for (long dim = 0; dim < nDim; dim++) {
			this->fire_->d_mass[particleId * nDim + dim] = PI / (d_particleRad[particleId] * d_particleRad[particleId]);
		}
	}
}

void DPM2D::setTimeStepFIRE(double timeStep_) {
  this->fire_->setFIRETimeStep(timeStep_);
}


void DPM2D::particleFIRELoop() {
  this->fire_->minimizerParticleLoop();
}

void DPM2D::vertexFIRELoop() {
  this->fire_->minimizerVertexLoop();
}

void DPM2D::initRigidFIRE(std::vector<double> &FIREparams, long minStep_, long numStep_, long numDOF_, double cutDist_) {
  initFIRE(FIREparams, minStep_, numStep_, numDOF_);
  initDeltaVariables(getNumVertices(), getNumParticles());
  initRotationalVariables(getNumVertices(), getNumParticles());
  resetParticleLastPositions();
  this->fire_->cutDistance = cutDist_;
}

void DPM2D::rigidFIRELoop() {
  this->fire_->minimizerRigidLoop();
}

//******************* integrators for deformable particles *******************//
void DPM2D::initLangevin(double Temp, double gamma, bool readState) {
  this->sim_ = new Langevin(this, SimConfig(Temp, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(Temp);
  this->sim_->lcoeff1 = exp(-gamma * dt);
  this->sim_->lcoeff2 = sqrt(1 - exp(-2*gamma*dt));
  this->sim_->d_thermalVel.resize(d_vel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  resetLastPositions();
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  calcParticlePositions();
  d_particleInitPos = d_particlePos;
  if(readState == false) {
    //this->sim_->injectKineticEnergy();
    cout << "DPM2D::initLangevin:: current temperature: " << setprecision(12) << getTemperature() << endl;
  } else {
    cout << "DPM2D::initLangevin:: reading velocities" << endl;
  }
}

void DPM2D::langevinLoop() {
  this->sim_->integrate();
}

void DPM2D::initLangevin2(double Temp, double gamma, bool readState) {
  this->sim_ = new Langevin2(this, SimConfig(Temp, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(d_vel.size());
  this->sim_->d_rando.resize(d_vel.size());
  this->sim_->d_thermalVel.resize(d_vel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  resetLastPositions();
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  calcParticlePositions();
  d_particleInitPos = d_particlePos;
  if(readState == false) {
    //this->sim_->injectKineticEnergy();
    cout << "DPM2D::initLangevin2:: current temperature: " << setprecision(12) << getTemperature() << endl;
  } else {
    cout << "DPM2D::initLangevin2:: reading velocities" << endl;
  }
}

void DPM2D::langevin2Loop() {
  this->sim_->integrate();
}

void DPM2D::initActiveLangevin(double Temp, double Dr, double driving, double gamma, bool readState) {
  this->sim_ = new ActiveLangevin(this, SimConfig(Temp, Dr, driving));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(numParticles * nDim);
  this->sim_->d_rando.resize(numParticles * nDim);
  this->sim_->d_pActiveAngle.resize(numParticles);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  resetLastPositions();
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  calcParticlePositions();
  d_particleInitPos = d_particlePos;
  if(readState == false) {
    computeParticleAngleFromVel();
    //this->sim_->injectKineticEnergy();
  }
  cout << "DPM2D::initActiveLangevin:: current temperature: " << setprecision(12) << getTemperature() << endl;
}

void DPM2D::activeLangevinLoop() {
  this->sim_->integrate();
}

void DPM2D::initNVE(double Temp, bool readState) {
  this->sim_ = new NVE(this, SimConfig(Temp, 0, 0));
  this->sim_->noiseVar = sqrt(Temp);
  this->sim_->d_thermalVel.resize(d_particlePos.size());
  resetLastPositions();
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  calcParticlePositions();
  d_particleInitPos = d_particlePos;
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  cout << "DPM2D::initNVE:: current temperature: " << setprecision(12) << getTemperature() << endl;
}

void DPM2D::NVELoop() {
  this->sim_->integrate();
}

void DPM2D::initBrownian(double Temp, double gamma, bool readState) {
  this->sim_ = new Brownian(this, SimConfig(Temp, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  resetLastPositions();
  if(readState == false) {
    this->sim_->injectKineticEnergy();
  }
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  calcParticlePositions();
  d_particleInitPos = d_particlePos;
  cout << "DPM2D::initBrownian:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
}

void DPM2D::brownianLoop() {
  this->sim_->integrate();
}

void DPM2D::initActiveBrownian(double Dr, double driving, bool readState) {
  this->sim_ = new ActiveBrownian(this, SimConfig(0, Dr, driving));
  this->sim_->d_rand.resize(numParticles);
  resetLastPositions();
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
  }
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  calcParticlePositions();
  d_particleInitPos = d_particlePos;
  cout << "DPM2D::initActiveBrownian:: current temperature: " << setprecision(12) << getTemperature() << endl;
}

void DPM2D::activeBrownianLoop() {
  this->sim_->integrate();
}

void DPM2D::initActiveBrownianPlastic(double Dr, double driving, double gamma, bool readState) {
  this->sim_ = new ActiveBrownianPlastic(this, SimConfig(0, Dr, driving));
  this->sim_->lcoeff1 = exp(-gamma * dt);
  this->sim_->d_rand.resize(numParticles);
  d_l0Vel.resize(numVertices);
  thrust::fill(d_l0Vel.begin(), d_l0Vel.end(), double(0));
  resetLastPositions();
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  calcParticlePositions();
  d_particleInitPos = d_particlePos;
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    computeParticleAngleFromVel();
  }
  cout << "DPM2D::initActiveBrownianPlastic:: current temperature: " << setprecision(12) << getTemperature() << endl;
}

void DPM2D::activeBrownianPlasticLoop() {
  this->sim_->integrate();
}

//************************* rigid particle simulators ************************//
void DPM2D::initRigidLangevin(double Temp, double gamma, bool readState) {
  this->sim_ = new RigidLangevin(this, SimConfig(Temp, 0, 0));
  this->sim_->gamma = gamma;
  this->sim_->noiseVar = sqrt(2. * Temp * gamma);
  this->sim_->lcoeff1 = 0.25 * dt * sqrt(dt) * gamma * this->sim_->noiseVar;
  this->sim_->lcoeff2 = 0.5 * sqrt(dt) * this->sim_->noiseVar;
  this->sim_->lcoeff3 = (0.5 / sqrt(3)) * sqrt(dt) * dt * this->sim_->noiseVar;
  this->sim_->d_rand.resize(d_particleVel.size());
  this->sim_->d_rando.resize(d_particleVel.size());
  this->sim_->d_thermalVel.resize(d_particleVel.size());
  thrust::fill(this->sim_->d_thermalVel.begin(), this->sim_->d_thermalVel.end(), double(0));
  initRotationalVariables(getNumVertices(), getNumParticles());
  initDeltaVariables(getNumVertices(), getNumParticles());
  resetLastPositions();
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  calcParticlePositions();
  d_particleInitPos = d_particlePos;
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    cout << "DPM2D::initRigidLangevin:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
  } else {
    cout << "DPM2D::initRigidLangevin:: reading velocities" << endl;
  }
}

void DPM2D::rigidLangevinLoop() {
  this->sim_->integrate();
}

//************************* rigid particle simulators ************************//
void DPM2D::initRigidNVE(double Temp, bool readState) {
  this->sim_ = new RigidNVE(this, SimConfig(Temp, 0, 0));
  this->sim_->noiseVar = sqrt(Temp);
  this->sim_->d_thermalVel.resize(d_particlePos.size());
  initRotationalVariables(getNumVertices(), getNumParticles());
  initDeltaVariables(getNumVertices(), getNumParticles());
  resetLastPositions();
  d_particleInitPos.resize(numParticles * nDim);
  thrust::fill(d_particleInitPos.begin(), d_particleInitPos.end(), double(0));
  calcParticlePositions();
  d_particleInitPos = d_particlePos;
  if(readState == false) {
    this->sim_->injectKineticEnergy();
    cout << "DPM2D::initRigidNVE:: current temperature: " << setprecision(12) << getParticleTemperature() << endl;
  } else {
    cout << "DPM2D::initRigidNVE:: reading velocities" << endl;
  }
}

void DPM2D::rigidNVELoop() {
  this->sim_->integrate();
}
