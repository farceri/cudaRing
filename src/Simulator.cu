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
// momentum conservation for rigid particles
__global__ void kernelConserveParticleMomentum(double* pVel);


//********************************** langevin ********************************//
void Langevin::integrate() {
  updateVelocity(0.5 * dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->checkNeighbors();
  dpm_->calcForceEnergy();
  updateThermalVel();
  updateVelocity(0.5 * dpm_->dt);
}

void Langevin::injectKineticEnergy() {
  double amplitude(sqrt(config.Tinject));
  // generate random numbers between 0 and Tscale for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles * dpm_->nDim, d_rand.begin(), gaussNum(0.f,amplitude));
  long s_nDim(dpm_->nDim);
  auto r = thrust::counting_iterator<long>(0);
  double* vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const long* pIdList = thrust::raw_pointer_cast(&(dpm_->d_particleIdList[0]));

  auto injectThermalVel = [=] __device__ (long vertexId) {
    auto particleId = pIdList[vertexId];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      vel[vertexId * s_nDim + dim] = rand[particleId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + dpm_->numVertices, injectThermalVel);
}

void Langevin::updateThermalVel() {
  // generate random numbers between 0 and 1 for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles * dpm_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  // update thermal velocity
  long s_nDim(dpm_->nDim);
  double s_gamma(gamma);
  double s_noise(noise);
  auto r = thrust::counting_iterator<long>(0);
  double *force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  const long* pIdList = thrust::raw_pointer_cast(&(dpm_->d_particleIdList[0]));

  auto langevinAddThermostatForces = [=] __device__ (long vertexId) {
    auto pId = pIdList[vertexId];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      force[vertexId * s_nDim + dim] += s_noise * rand[pId * s_nDim + dim] - s_gamma * vel[vertexId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + dpm_->numVertices, langevinAddThermostatForces);
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

void Langevin::conserveMomentum() {
  typedef thrust::device_vector<double>::iterator Iterator;
  strided_range<Iterator> vel_x(dpm_->d_vel.begin(), dpm_->d_vel.end(), 2);
  strided_range<Iterator> vel_y(dpm_->d_vel.begin() + 1, dpm_->d_vel.end(), 2);
  double meanVelx = thrust::reduce(vel_x.begin(), vel_x.end(), double(0), thrust::plus<double>()) / dpm_->numVertices;
  double meanVely = thrust::reduce(vel_y.begin(), vel_y.end(), double(0), thrust::plus<double>()) / dpm_->numVertices;

  long s_nDim(dpm_->nDim);
  auto r = thrust::counting_iterator<long>(0);
  double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  auto subtractDrift = [=] __device__ (long vertexId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      vel[vertexId * s_nDim + dim] -= ((1 - dim) * meanVelx + dim * meanVely);
    }
  };

  thrust::for_each(r, r + dpm_->numVertices, subtractDrift);
}

//****************************** BAOAB langevin ******************************//
void BAOAB::integrate() {
  updateVelocity(0.5 * dpm_->dt);
  updatePosition(0.5 * dpm_->dt);
  updateThermalVel();
  updatePosition(0.5 * dpm_->dt);
  dpm_->checkNeighbors();
  dpm_->calcForceEnergy();
  updateVelocity(0.5 * dpm_->dt);
}

void BAOAB::updateThermalVel() {
  // generate random numbers between 0 and 1 for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles * dpm_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  // update thermal velocity
  long s_nDim(dpm_->nDim);
  double s_gamma(gamma); // this is exp(-gamma dt)
  double s_noise(noise); // this is sqrt(1 - exp(-2 gamma dt)) sqrt(m kb T)
  auto r = thrust::counting_iterator<long>(0);
  double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const long* pIdList = thrust::raw_pointer_cast(&(dpm_->d_particleIdList[0]));

  auto BAOABUpdateThermalVel = [=] __device__ (long vertexId) {
    long pId = pIdList[vertexId];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      vel[vertexId * s_nDim + dim] = s_gamma * vel[vertexId * s_nDim + dim] + s_noise * rand[pId * s_nDim + dim];
    }
  };

  thrust::for_each(r, r + dpm_->numVertices, BAOABUpdateThermalVel);
}

//******************************** brownian **********************************//
void Brownian::integrate() {
  dpm_->checkNeighbors();
  dpm_->calcForceEnergy();
  updateThermalVel();
  updatePosition(dpm_->dt);
}

void Brownian::updateThermalVel() {
  // generate random numbers between 0 and 1 for thermal noise
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
  thrust::transform(index_sequence_begin, index_sequence_begin + dpm_->numParticles * dpm_->nDim, d_rand.begin(), gaussNum(0.f,1.f));
  // assign overdamped velocity
  long s_nDim(dpm_->nDim);
  double s_gamma(gamma);
  double s_noise(noise);
  auto r = thrust::counting_iterator<long>(0);
  double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  const double *rand = thrust::raw_pointer_cast(&d_rand[0]);
  const double *force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));
  const long *pIdList = thrust::raw_pointer_cast(&(dpm_->d_particleIdList[0]));

  auto updateBrownianVel = [=] __device__ (long vertexId) {
    long pId = pIdList[vertexId];
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
		  vel[vertexId * s_nDim + dim] = (force[vertexId * s_nDim + dim] + s_noise * rand[pId * s_nDim + dim]) / s_gamma;
    }
  };

  thrust::for_each(r, r + dpm_->numVertices, updateBrownianVel);
}

//**************************** driven brownian *******************************//
void DrivenBrownian::integrate() {
  dpm_->checkNeighbors();
  dpm_->calcForceEnergy();
  updateThermalVel();
  updatePosition(dpm_->dt);
}

void DrivenBrownian::updateThermalVel() {
  // assign overdamped velocity
  long s_nDim(dpm_->nDim);
  double s_gamma(gamma);
  auto r = thrust::counting_iterator<long>(0);
  double *vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
  const double *force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));

  auto updateDrivenBrownianVel = [=] __device__ (long vertexId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
		  vel[vertexId * s_nDim + dim] = force[vertexId * s_nDim + dim] / s_gamma;
    }
  };

  thrust::for_each(r, r + dpm_->numVertices, updateDrivenBrownianVel);
}

//************************************ NVE ***********************************//
void NVE::integrate() {
  updateVelocity(0.5 * dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->checkNeighbors();
  dpm_->calcForceEnergy();
  updateVelocity(0.5 * dpm_->dt);
}

//********************** nve with velocity rescaling ************************//
void NVERescale::integrate() {
  injectKineticEnergy();
  updateVelocity(0.5 * dpm_->dt);
  updatePosition(dpm_->dt);
  dpm_->checkNeighbors();
  dpm_->calcForceEnergy();
  updateVelocity(0.5 * dpm_->dt);
}

void NVERescale::injectKineticEnergy() {
  double scale = sqrt(config.Tinject / dpm_->getTemperature());
  long s_nDim(dpm_->nDim);
  auto r = thrust::counting_iterator<long>(0);
	double* vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));

  auto scaleVel = [=] __device__ (long vId) {
    #pragma unroll (MAXDIM)
		for (long dim = 0; dim < s_nDim; dim++) {
      vel[vId * s_nDim + dim] *= scale;
    }
  };

  thrust::for_each(r, r + dpm_->numVertices, scaleVel);
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
  thrust::counting_iterator<long> index_sequence_begin(lrand48());
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