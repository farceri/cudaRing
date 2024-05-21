//
// Author: Yuxuan Cheng
// Date:   10-09-2021
//
// DEFINITION OF INTEGRATION FUNCTIONS

#include "../include/Simulator.h"
#include "../include/defs.h"
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>

using namespace std;
// vertex updates
__global__ void kernelUpdateVertexPos(double* pos, const double* vel, const double timeStep);
__global__ void kernelUpdateVertexVel(double* vel, const double* force, const double timeStep);
//__global__ void kernelExtractThermalVertexVel(double* vel, const double* r1, const double* r2, const double amplitude);
__global__ void kernelUpdateBrownianVertexVel(double* vel, const double* force, double* thermalVel, const double mobility);
__global__ void kernelUpdateActiveVertexVel(double* vel, const double* force, double* pAngle, const double driving, const double mobility);
// rigid updates
__global__ void kernelUpdateParticlePos(double* pPos, const double* pVel, const double timeStep);
__global__ void kernelUpdateRigidPos(double* pPos, const double* pVel, double* pAngle, const double* pAngvel, const double timeStep);
__global__ void kernelUpdateRigidBrownianVel(double* pVel, const double* pForce, double* pAngvel, const double* pTorque, double* thermalVel, const double mobility);
__global__ void kernelUpdateRigidActiveVel(double* pVel, const double* pForce, double* pActiveAngle, double* pAngvel, const double* pTorque, const double driving, const double mobility);
__global__ void kernelUpdateActiveParticleVel(double* pVel, const double* pForce, double* pAngle, const double driving, const double mobility);
// momentum conservation
__global__ void kernelConserveVertexMomentum(double* vel);
__global__ void kernelConserveParticleMomentum(double* pVel);


//********************************** langevin ********************************//
void Langevin::integrate() {
  updateVelocity(0.5 * dpm_->dt);
  updatePosition(0.5 * dpm_->dt);
  updateThermalVel();
  updatePosition(0.5 * dpm_->dt);
  dpm_->checkNeighbors();
  dpm_->calcForceEnergy();
  updateVelocity(0.5 * dpm_->dt);
  //conserveMomentum();
}

void Langevin::injectKineticEnergy() {
  // generate random numbers between 0 and Tscale for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(0);
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles * dpm_->nDim, d_thermalVel.begin(), gaussNum(0.f,noiseVar));
  long s_nDim(dpm_->nDim);
  auto r = thrust::counting_iterator<long>(0);
  double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  const double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
  const long* pIdList = thrust::raw_pointer_cast(&(dpm_->d_particleIdList[0]));
  //const long* nVList = thrust::raw_pointer_cast(&(dpm_->d_numVertexInParticleList[0]));

  auto injectThermalVel = [=] __device__ (long vertexId) {
    long particleId = pIdList[vertexId];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      vel[vertexId * s_nDim + dim] = thermalVel[particleId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + dpm_->numVertices, injectThermalVel);
  //kernelConserveVertexMomentum<<<1, dpm_->dimBlock>>>(vel);
}

void Langevin::updatePosition(double timeStep) {
	double* pos = thrust::raw_pointer_cast(&dpm_->d_pos[0]);
	const double* vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  kernelUpdateVertexPos<<<dpm_->dimGrid, dpm_->dimBlock>>>(pos, vel, timeStep);
}

void Langevin::updateVelocity(double timeStep) {
	double* vel = thrust::raw_pointer_cast(&dpm_->d_vel[0]);
	const double* force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));
  kernelUpdateVertexVel<<<dpm_->dimGrid, dpm_->dimBlock>>>(vel, force, timeStep);
}

void Langevin::updateThermalVel() {
  // generate random numbers between 0 and 1 for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(0);
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles * dpm_->nDim, d_thermalVel.begin(), gaussNum(0.f,noiseVar));
  // update thermal velocity
  long s_nDim(dpm_->nDim);
  double s_lcoeff1(lcoeff1);
  double s_lcoeff2(lcoeff2);
  auto r = thrust::counting_iterator<long>(0);
  double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
  const long* pIdList = thrust::raw_pointer_cast(&(dpm_->d_particleIdList[0]));

  auto langevinUpdateThermalVel = [=] __device__ (long vertexId) {
    long particleId = pIdList[vertexId];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      vel[vertexId * s_nDim + dim] = s_lcoeff1 * vel[vertexId * s_nDim + dim] + s_lcoeff2 * thermalVel[particleId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + dpm_->numVertices, langevinUpdateThermalVel);
}

void Langevin::conserveMomentum() {
  double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  kernelConserveVertexMomentum<<<1, dpm_->dimBlock>>>(vel);
}

//********************************* langevin2 ********************************//
void Langevin2::integrate() {
  updateThermalVel();
  updateVelocity(0.5*dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->checkNeighbors();
  dpm_->calcForceEnergy();
  updateVelocity(0.5*dpm_->dt);
  //conserveMomentum();
}

void Langevin2::updateThermalVel() {
  // extract two noises and compute noise terms
  thrust::counting_iterator<long> index_sequence_begin1(0);
  thrust::transform(index_sequence_begin1, index_sequence_begin1 + dpm_->numParticles * dpm_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  thrust::counting_iterator<long> index_sequence_begin2(0);
  thrust::transform(index_sequence_begin2, index_sequence_begin2 + dpm_->numParticles * dpm_->nDim, d_rando.begin(), gaussNum(0.f,1.f));
  // update thermal velocity
  long s_nDim(dpm_->nDim);
  double s_lcoeff1(lcoeff1);
  double s_lcoeff2(lcoeff2);
  double s_lcoeff3(lcoeff3);
  auto r = thrust::counting_iterator<long>(0);
  double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  double *rando = thrust::raw_pointer_cast(&d_rando[0]);
  double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
  //const long* nVList = thrust::raw_pointer_cast(&(dpm_->d_numVertexInParticleList[0]));

  auto langevinUpdateThermalNoise = [=] __device__ (long particleId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      thermalVel[particleId * s_nDim + dim] = s_lcoeff1 * (0.5 * rand[particleId * s_nDim + dim] + rando[particleId * s_nDim + dim] / sqrt(3));// / sqrt(nVList[particleId]);
      rand[particleId * s_nDim + dim] *= s_lcoeff2;// / sqrt(nVList[particleId]);
      rando[particleId * s_nDim + dim] *= s_lcoeff3;// / sqrt(nVList[particleId]);
    }
  };

  thrust::for_each(r, r + dpm_->numParticles, langevinUpdateThermalNoise);
}

void Langevin2::updateVelocity(double timeStep) {
  long s_nDim(dpm_->nDim);
  double s_dt(timeStep);
  double s_gamma(gamma);
  auto r = thrust::counting_iterator<long>(0);
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
	double* vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
	const double* force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));
  const long* pIdList = thrust::raw_pointer_cast(&(dpm_->d_particleIdList[0]));

  auto langevinUpdateVertexVel = [=] __device__ (long vId) {
    long pId = pIdList[vId];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      vel[vId * s_nDim + dim] += s_dt * (force[vId * s_nDim + dim] - vel[vId * s_nDim + dim] * s_gamma);
      vel[vId * s_nDim + dim] -= 0.5 * s_dt * s_dt * (force[vId * s_nDim + dim] - vel[vId * s_nDim + dim] * s_gamma) / s_gamma;
      vel[vId * s_nDim + dim] += rand[pId * s_nDim + dim] - thermalVel[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + dpm_->numVertices, langevinUpdateVertexVel);
}

void Langevin2::updatePosition(double timeStep) {
  long s_nDim(dpm_->nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
  const double *rando = thrust::raw_pointer_cast(&d_rando[0]);
	double* pos = thrust::raw_pointer_cast(&(dpm_->d_pos[0]));
	const double* vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  const long* pIdList = thrust::raw_pointer_cast(&(dpm_->d_particleIdList[0]));

  auto langevinUpdateVertexPos = [=] __device__ (long vId) {
    long pId = pIdList[vId];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pos[vId * s_nDim + dim] += s_dt * vel[vId * s_nDim + dim] + rando[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + dpm_->numVertices, langevinUpdateVertexPos);
}


//***************************** active langevin2 *****************************//
void ActiveLangevin::integrate() {
  updateThermalVel();
  updateVelocity(0.5*dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->checkNeighbors();
  dpm_->calcForceEnergy();
  updateVelocity(0.5*dpm_->dt);
  //conserveMomentum();
}

void ActiveLangevin::updateThermalVel() {
  // extract two noises and compute noise terms
  thrust::counting_iterator<long> index_sequence_begin1(0);
  thrust::transform(index_sequence_begin1, index_sequence_begin1 + dpm_->numParticles * dpm_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  thrust::counting_iterator<long> index_sequence_begin2(0);
  thrust::transform(index_sequence_begin2, index_sequence_begin2 + dpm_->numParticles * dpm_->nDim, d_rando.begin(), gaussNum(0.f,1.f));
  // update thermal velocity
  long s_nDim(dpm_->nDim);
  double s_lcoeff1(lcoeff1);
  double s_lcoeff2(lcoeff2);
  double s_lcoeff3(lcoeff3);
  auto r = thrust::counting_iterator<long>(0);
  double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  double *rando = thrust::raw_pointer_cast(&d_rando[0]);
  double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);

  auto langevinUpdateThermalNoise = [=] __device__ (long particleId) {
    #pragma unroll (MAXDIM)
    for (long dim = 0; dim < s_nDim; dim++) {
      thermalVel[particleId * s_nDim + dim] = s_lcoeff1 * (0.5 * rand[particleId * s_nDim + dim] + rando[particleId * s_nDim + dim] / sqrt(3));
      rand[particleId * s_nDim + dim] *= s_lcoeff2;
      rando[particleId * s_nDim + dim] *= s_lcoeff3;
    }
  };

  thrust::for_each(r, r + dpm_->numParticles, langevinUpdateThermalNoise);

  // generate active forces
  double amplitude = sqrt(2. * config.Dr * dpm_->dt);
  thrust::counting_iterator<long> index_sequence_begin(0);
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles, d_pActiveAngle.begin(), gaussNum(0.f,1.f));
  double s_driving(config.driving);
  auto s = thrust::counting_iterator<long>(0);
  const double *pActiveAngle = thrust::raw_pointer_cast(&d_pActiveAngle[0]);
  double *pAngle = thrust::raw_pointer_cast(&(dpm_->d_particleAngle[0]));
	double* force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));
  const long* pIdList = thrust::raw_pointer_cast(&(dpm_->d_particleIdList[0]));

  auto addActiveParticleForceToVertex = [=] __device__ (long vId) {
    long pId = pIdList[vId];
    pAngle[pId] += amplitude * pActiveAngle[pId];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      force[vId * s_nDim + dim] += s_driving * ((1 - dim) * cos(pAngle[pId]) + dim * sin(pAngle[pId]));
    }
  };

  thrust::for_each(s, s + dpm_->numVertices, addActiveParticleForceToVertex);
}


//************************************ NVE ***********************************//
void NVE::integrate() {
  updateVelocity(0.5 * dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->checkNeighbors();
  dpm_->calcForceEnergy();
  updateVelocity(0.5 * dpm_->dt);
  //conserveMomentum();
}

//******************************** brownian **********************************//
void Brownian::integrate() {
  updateVelocity(dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->checkNeighbors();
  dpm_->calcForceEnergy();
  //conserveMomentum();
}

void Brownian::updateVelocity(double timeStep) {
  double mobility = 1/gamma;
  // generate random numbers between 0 and 1 for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(0);
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles * dpm_->nDim, d_thermalVel.begin(), gaussNum(0.f,noiseVar));
  // update vertex velocity
  double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  const double *force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));
  double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
  kernelUpdateBrownianVertexVel<<<dpm_->dimGrid, dpm_->dimBlock>>>(vel, force, thermalVel, mobility);
}

//**************************** active brownian *******************************//
void ActiveBrownian::integrate() {
  updateVelocity(dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->checkNeighbors();
  dpm_->calcForceEnergy();
  //conserveMomentum();
}

void ActiveBrownian::updateVelocity(double timeStep) {
  double amplitude = sqrt(2. * config.Dr);
  double mobility = 1/gamma;
  // generate random numbers between 0 and 1 for angle update
  thrust::counting_iterator<long> index_sequence_begin(0);
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles, d_rand.begin(), gaussNum(0.f,1.f));
  // update particle direction
  auto r = thrust::counting_iterator<long>(0);
  double *pAngle = thrust::raw_pointer_cast(&(dpm_->d_particleAngle[0]));
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);

  auto updateAngle = [=] __device__ (long particleId) {
    pAngle[particleId] += amplitude * rand[particleId];
  };

  thrust::for_each(r, r + dpm_->numParticles, updateAngle);
  // update vertex velocity with overdamped active dynamics
  double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  const double *force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));
  kernelUpdateActiveVertexVel<<<dpm_->dimGrid, dpm_->dimBlock>>>(vel, force, pAngle, config.driving, mobility);
}

//******************* active brownian with damping on l0 *********************//
void ActiveBrownianPlastic::integrate() {
  updateVelocity(dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->checkNeighbors();
  dpm_->calcForceEnergy();
  //conserveMomentum();
}

void ActiveBrownianPlastic::updatePosition(double timeStep) {
	double* pos = thrust::raw_pointer_cast(&dpm_->d_pos[0]);
	const double* vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  kernelUpdateVertexPos<<<dpm_->dimGrid, dpm_->dimBlock>>>(pos, vel, timeStep);
  // update rest length
  auto r = thrust::counting_iterator<long>(0);
  double s_dt(timeStep);
  double s_kl = 1.;
  double s_lcoeff1(lcoeff1);
  const double *length = thrust::raw_pointer_cast(&(dpm_->d_length[0]));
  double *l0 = thrust::raw_pointer_cast(&(dpm_->d_l0[0]));
  double *l0Vel = thrust::raw_pointer_cast(&(dpm_->d_l0Vel[0]));

  auto firstUpdateRestLength = [=] __device__ (long vertexId) {
    l0Vel[vertexId] += s_kl * (length[vertexId] - l0[vertexId]) * s_dt * 0.5;
    l0[vertexId] += l0Vel[vertexId] * s_dt * 0.5;
    l0Vel[vertexId] = s_lcoeff1 * l0Vel[vertexId];
  };

  thrust::for_each(r, r + dpm_->numVertices, firstUpdateRestLength);
}

void ActiveBrownianPlastic::updateVelocity(double timeStep) {
  double amplitude = sqrt(2. * config.Dr);
  double mobility = 1/gamma;
  // generate random numbers between 0 and 1 for angle update
  thrust::counting_iterator<long> index_sequence_begin(0);
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles, d_rand.begin(), gaussNum(0.f,1.f));
  // update particle direction
  auto r = thrust::counting_iterator<long>(0);
  double *pAngle = thrust::raw_pointer_cast(&(dpm_->d_particleAngle[0]));
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);

  auto updateAngle = [=] __device__ (long particleId) {
    pAngle[particleId] += amplitude * rand[particleId];
  };

  thrust::for_each(r, r + dpm_->numParticles, updateAngle);
  // update vertex velocity with overdamped active dynamics
  double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  const double *force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));
  kernelUpdateActiveVertexVel<<<dpm_->dimGrid, dpm_->dimBlock>>>(vel, force, pAngle, config.driving, mobility);
  // update rest length
  auto s = thrust::counting_iterator<long>(0);
  double s_dt(timeStep);
  double s_kl = 1.;
  const double *length = thrust::raw_pointer_cast(&(dpm_->d_length[0]));
  double *l0 = thrust::raw_pointer_cast(&(dpm_->d_l0[0]));
  double *l0Vel = thrust::raw_pointer_cast(&(dpm_->d_l0Vel[0]));

  auto secondUpdateRestLength = [=] __device__ (long vertexId) {
		l0[vertexId] += l0Vel[vertexId] * s_dt * 0.5;
		l0Vel[vertexId] += s_kl * (length[vertexId] - l0[vertexId]) * s_dt * 0.5;
  };

  thrust::for_each(r, r + dpm_->numVertices, secondUpdateRestLength);
}

//****************************** rigid langevin ******************************//
void RigidLangevin::integrate() {
  dpm_->resetParticleLastPositions();
  dpm_->resetParticleLastAngles();
  updateThermalVel();
  updateVelocity(0.5*dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->translateVertices();
  dpm_->rotateVertices();
  dpm_->checkNeighbors();
  dpm_->calcRigidForceEnergy();
  updateVelocity(0.5*dpm_->dt);
}

void RigidLangevin::injectKineticEnergy() {
  // generate random numbers between 0 and Tscale for thermal noise
  double amplitude = sqrt(config.Tinject);
  thrust::counting_iterator<long> index_sequence_begin(0);
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles * dpm_->nDim, dpm_->d_particleVel.begin(), gaussNum(0.f,amplitude));
  double *pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
  kernelConserveParticleMomentum<<<1, dpm_->dimBlock>>>(pVel);
}

void RigidLangevin::updateVelocity(double timeStep) {
  long s_nDim(dpm_->nDim);
  double s_dt(timeStep);
  double s_gamma(gamma);
  auto r = thrust::counting_iterator<long>(0);
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const double *thermalVel = thrust::raw_pointer_cast(&d_thermalVel[0]);
	double* pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
	double *pAngvel = thrust::raw_pointer_cast(&(dpm_->d_particleAngvel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));
	const double *pTorque = thrust::raw_pointer_cast(&(dpm_->d_particleTorque[0]));

  auto langevinUpdateRigidVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += s_dt * (pForce[pId * s_nDim + dim] - pVel[pId * s_nDim + dim] * s_gamma);
      pVel[pId * s_nDim + dim] -= 0.5 * s_dt * s_dt * (pForce[pId * s_nDim + dim] - pVel[pId * s_nDim + dim] * s_gamma) / s_gamma;
      pVel[pId * s_nDim + dim] += rand[pId * s_nDim + dim] - thermalVel[pId * s_nDim + dim];
    }
		pAngvel[pId] += s_dt * pTorque[pId];
  };

  thrust::for_each(r, r + dpm_->numParticles, langevinUpdateRigidVel);
  //kernelUpdateRigidBrownianVel<<<dpm_->partDimGrid, dpm_->dimBlock>>>(pVel, pForce, pAngvel, pTorque, thermalVel, mobility);
}

void RigidLangevin::updatePosition(double timeStep) {
  long s_nDim(dpm_->nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
  const double *rando = thrust::raw_pointer_cast(&d_rando[0]);
	double *pPos = thrust::raw_pointer_cast(&(dpm_->d_particlePos[0]));
	const double *pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
  double* pAngle = thrust::raw_pointer_cast(&(dpm_->d_particleAngle[0]));
	const double* pAngvel = thrust::raw_pointer_cast(&(dpm_->d_particleAngvel[0]));

  auto langevinUpdateRigidPos = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pPos[pId * s_nDim + dim] += s_dt * pVel[pId * s_nDim + dim] + rando[pId * s_nDim + dim];
    }
		pAngle[pId] += s_dt * pAngvel[pId];
  };

  thrust::for_each(r, r + dpm_->numParticles, langevinUpdateRigidPos);
  //kernelUpdateRigidPos<<<dpm_->partDimGrid, dpm_->dimBlock>>>(pPos, pVel, pAngle, pAngvel, timeStep);
}

void RigidLangevin::conserveMomentum() {
  double *pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
  kernelConserveParticleMomentum<<<1, dpm_->dimBlock>>>(pVel);
}


//************************************ Rigid NVE ***********************************//
void RigidNVE::integrate() {
  dpm_->resetParticleLastPositions();
  dpm_->resetParticleLastAngles();
  updateVelocity(0.5 * dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->translateVertices();
	dpm_->rotateVertices();
  dpm_->checkNeighbors();
  dpm_->calcRigidForceEnergy();
  updateVelocity(0.5 * dpm_->dt);
}

void RigidNVE::updatePosition(double timeStep) {
  long s_nDim(dpm_->nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
	double *pPos = thrust::raw_pointer_cast(&(dpm_->d_particlePos[0]));
	const double *pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
  double* pAngle = thrust::raw_pointer_cast(&(dpm_->d_particleAngle[0]));
	const double* pAngvel = thrust::raw_pointer_cast(&(dpm_->d_particleAngvel[0]));

  auto langevinUpdateRigidPos = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pPos[pId * s_nDim + dim] += s_dt * pVel[pId * s_nDim + dim];
    }
		pAngle[pId] += s_dt * pAngvel[pId];
  };

  thrust::for_each(r, r + dpm_->numParticles, langevinUpdateRigidPos);
}

void RigidNVE::updateVelocity(double timeStep) {
  long s_nDim(dpm_->nDim);
  double s_dt(timeStep);
  auto r = thrust::counting_iterator<long>(0);
	double* pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
	double *pAngvel = thrust::raw_pointer_cast(&(dpm_->d_particleAngvel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));
	const double *pTorque = thrust::raw_pointer_cast(&(dpm_->d_particleTorque[0]));

  auto langevinUpdateRigidVel = [=] __device__ (long pId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      pVel[pId * s_nDim + dim] += s_dt * pForce[pId * s_nDim + dim];
    }
		pAngvel[pId] += s_dt * pTorque[pId];
  };

  thrust::for_each(r, r + dpm_->numParticles, langevinUpdateRigidVel);
}

void RigidNVE::conserveMomentum() {
  double *pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
  kernelConserveParticleMomentum<<<1, dpm_->dimBlock>>>(pVel);
}