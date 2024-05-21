//
// Author: Francesco Arceri
// Date:   10-26-2021
//
// FUNCTIONS FOR FIRE CLASS

#include "../include/FIRE.h"
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>

using namespace std;

//********************** constructor and deconstructor ***********************//
FIRE::FIRE(DPM2D * dpmPtr):dpm_(dpmPtr){
	// Note that mass is used only for particle-level FIRE
	d_mass.resize(dpm_->numParticles * dpm_->nDim);
	// Set variables to zero
	thrust::fill(d_mass.begin(), d_mass.end(), double(0));
}

FIRE::~FIRE() {
	d_mass.clear();
	d_velSquared.clear();
	d_forceSquared.clear();
};

// initilize the minimizer
void FIRE::initMinimizer(double a_start_, double f_dec_, double f_inc_, double f_a_, double fire_dt_, double fire_dt_max_, double a_, long minStep_, long numStep_, long numDOF_) {
	a_start = a_start_;
	f_dec = f_dec_;
	f_inc = f_inc_;
	f_a = f_a_;
	fire_dt = fire_dt_;
	fire_dt_max = fire_dt_max_;
	a = a_;
	minStep = minStep_;
	numStep = numStep_;
	d_velSquared.resize(numDOF_ * dpm_->nDim);
	d_forceSquared.resize(numDOF_ * dpm_->nDim);
}

//*************************** particle minimizer *****************************//
// update position and velocity in response to an applied force
void FIRE::updateParticlePositionAndVelocity() {
	long f_nDim(dpm_->nDim);
	double d_fire_dt(fire_dt);
	auto r = thrust::counting_iterator<long>(0);
	double* pPos = thrust::raw_pointer_cast(&(dpm_->d_particlePos[0]));
	double* pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));
	const double* mass = thrust::raw_pointer_cast(&d_mass[0]);

	auto perParticleUpdatePosAndVel = [=] __device__ (long particleId) {
		double totalForce(0);
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < f_nDim; dim++) {
			pVel[particleId * f_nDim + dim] += 0.5 * d_fire_dt * pForce[particleId * f_nDim + dim] / mass[particleId * f_nDim + dim];
			pPos[particleId * f_nDim + dim] += d_fire_dt * pVel[particleId * f_nDim + dim];
			totalForce += pForce[particleId * f_nDim + dim];
		}
		//If the total force on a particle is zero, then zero out the velocity as well
		if (totalForce == 0) {
			#pragma unroll (MAXDIM)
			for (long dim = 0; dim < f_nDim; dim++) {
				pVel[particleId * f_nDim + dim] = 0;
			}
		}
	};

	thrust::for_each(r, r + dpm_->numParticles, perParticleUpdatePosAndVel);
}

// update velocity in response to an applied force and return the maximum displacement in the previous step
void FIRE::updateParticleVelocity() {
	long f_nDim(dpm_->nDim);
	double d_fire_dt(fire_dt);
	auto r = thrust::counting_iterator<long>(0);
	double* pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));
	const double* mass = thrust::raw_pointer_cast(&d_mass[0]);

	auto perParticleUpdateVel = [=] __device__ (long particleId) {
		double totalForce(0);
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < f_nDim; dim++) {
			pVel[particleId * f_nDim + dim] += 0.5 * d_fire_dt * pForce[particleId * f_nDim + dim] / mass[particleId * f_nDim + dim];
			totalForce += pForce[particleId * f_nDim + dim];
		}
		//If the total force on a particle is zero, then zero out the velocity as well
		if (totalForce == 0) {
			for (long dim = 0; dim < f_nDim; dim++) {
				pVel[particleId * f_nDim + dim] = 0;
			}
		}
	};

	thrust::for_each(r, r + dpm_->numParticles, perParticleUpdateVel);
}

// bend the velocity towards the force
void FIRE::bendParticleVelocityTowardsForce() {
	double velNormSquared = 0, forceNormSquared = 0;
	// get the dot product between the velocity and the force
	double vDotF = double(thrust::inner_product(dpm_->d_particleVel.begin(), dpm_->d_particleVel.end(), dpm_->d_particleForce.begin(), double(0)));
	//cout << "FIRE::bendVelocityTowardsForceFIRE: vDotF = " << setprecision(precision) << vDotF << endl;
	if (vDotF < 0) {
		// if vDotF is negative, then we are going uphill, so let's stop and reset
		thrust::fill(dpm_->d_particleVel.begin(), dpm_->d_particleVel.end(), double(0));
		numStep = 0;
		fire_dt = std::max(fire_dt * f_dec, fire_dt_max / 10); // go to a shorter dt
		a = a_start; // start fresh with a more radical mixing between force and velocity
	} else if (numStep > minStep) {
		// if enough time has passed then let's start to increase the inertia
		fire_dt = std::min(fire_dt * f_inc, fire_dt_max);
		a *= f_a; // increase the inertia
	}
	// calculate the ratio of the norm squared of the velocity and the force
  thrust::transform(dpm_->d_particleVel.begin(), dpm_->d_particleVel.end(), d_velSquared.begin(), square());
  thrust::transform(dpm_->d_particleForce.begin(), dpm_->d_particleForce.end(), d_forceSquared.begin(), square());
	velNormSquared = thrust::reduce(d_velSquared.begin(), d_velSquared.end(), double(0), thrust::plus<double>());
	forceNormSquared = thrust::reduce(d_forceSquared.begin(), d_forceSquared.end(), double(0), thrust::plus<double>());
	// check FIRE convergence
	if (forceNormSquared == 0) {
		// if the forceNormSq is zero, then there is no force and we are done, so zero out the velocity
		cout << "FIRE::bendVelocityTowardsForceFIRE: forceNormSquared is zero" << endl;
		thrust::fill(dpm_->d_particleVel.begin(), dpm_->d_particleVel.end(), double(0));
	} else {
		double velForceNormRatio = sqrt(velNormSquared / forceNormSquared);
		double f_a(a);
		auto r = thrust::counting_iterator<long>(0);
		double* pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
		const double* pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));
		auto perDOFBendParticleVelocity = [=] __device__ (long i) {
			pVel[i] = (1 - f_a) * pVel[i] + f_a * pForce[i] * velForceNormRatio;
		};

		thrust::for_each(r, r + dpm_->d_particleVel.size(), perDOFBendParticleVelocity);
	}
}

// set the mass for each degree of freedom
void FIRE::setParticleMass() {
	d_mass.resize(dpm_->numParticles * dpm_->nDim);
	for (long particleId = 0; particleId < dpm_->numParticles; particleId++) {
		for (long dim = 0; dim < dpm_->nDim; dim++) {
			d_mass[particleId * dpm_->nDim + dim] = PI / (dpm_->d_particleRad[particleId] * dpm_->d_particleRad[particleId]);
		}
	}
}

// set FIRE time step
void FIRE::setFIRETimeStep(double fire_dt_) {
	fire_dt = fire_dt_;
	fire_dt_max = 10 * fire_dt_;
}

// Run the inner loop of the FIRE algorithm
void FIRE::minimizerParticleLoop() {
	// Move the system forward, based on the previous velocities and forces
	updateParticlePositionAndVelocity();
	// Calculate the new set of forces at the new step
	dpm_->checkParticleNeighbors();
	dpm_->calcParticleForceEnergy();
	// update the velocity based on the current forces
	updateParticleVelocity();
	// Bend the velocity towards the force
	bendParticleVelocityTowardsForce();
	// Increase the number of steps since the last restart
	numStep++;
}

//********************** rigid bumpy particle minimizer **********************//
// update position and velocity in response to an applied force
void FIRE::updateRigidPosAng(double timeStep) {
	long f_nDim(dpm_->nDim);
	double d_fire_dt(timeStep);
	auto r = thrust::counting_iterator<long>(0);
	double* pPos = thrust::raw_pointer_cast(&dpm_->d_particlePos[0]);
	const double* pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
	double* pAngle = thrust::raw_pointer_cast(&dpm_->d_particleAngle[0]);
	const double* pAngvel = thrust::raw_pointer_cast(&(dpm_->d_particleAngvel[0]));

	auto perParticleUpdatePosAng = [=] __device__ (long particleId) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < f_nDim; dim++) {
			pPos[particleId * f_nDim + dim] += d_fire_dt * pVel[particleId * f_nDim + dim];
		}
		pAngle[particleId] += d_fire_dt * pAngvel[particleId];
	};

	thrust::for_each(r, r + dpm_->numParticles, perParticleUpdatePosAng);
}

// update velocity in response to an applied force and return the maximum displacement in the previous step
void FIRE::updateRigidVelAngvel(double timeStep) {
	long f_nDim(dpm_->nDim);
	double d_fire_dt(timeStep);
	auto r = thrust::counting_iterator<long>(0);
	double* pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
	const double* pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));
	double* pAngvel = thrust::raw_pointer_cast(&(dpm_->d_particleAngvel[0]));
	const double* pTorque = thrust::raw_pointer_cast(&(dpm_->d_particleTorque[0]));

	auto perParticleUpdateVelAngvel = [=] __device__ (long particleId) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < f_nDim; dim++) {
			pVel[particleId * f_nDim + dim] += d_fire_dt * pForce[particleId * f_nDim + dim];
		}
		pAngvel[particleId] += d_fire_dt * pTorque[particleId];
	};

	thrust::for_each(r, r + dpm_->numParticles, perParticleUpdateVelAngvel);
}

// check and reset fire params
void FIRE::checkRigidFIREParams() {
	// get the dot product between the velocity and the force
	double vDotF = thrust::inner_product(dpm_->d_particleVel.begin(), dpm_->d_particleVel.end(), dpm_->d_particleForce.begin(), double(0));
	if (vDotF > 0) {
		// Increase the number of steps since the last restart
		numStep++;
		// if enough time has passed then let's start to increase the inertia
		if (numStep > minStep) {
			fire_dt = std::min(fire_dt * f_inc, fire_dt_max);
			a *= f_a;
		}
	} else {
		// start fresh with a more radical mixing between force and velocity
		numStep = 0;
		fire_dt = std::max(fire_dt * f_dec, fire_dt_max / 10);
		a = a_start;
		// take a step back
		long f_nDim(dpm_->nDim);
		double d_fire_dt(0.5 * fire_dt);
		auto r = thrust::counting_iterator<long>(0);
		double* pPos = thrust::raw_pointer_cast(&(dpm_->d_particlePos[0]));
		const double* pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
		double* pAngle = thrust::raw_pointer_cast(&(dpm_->d_particleAngle[0]));
		const double* pAngvel = thrust::raw_pointer_cast(&(dpm_->d_particleAngvel[0]));

		auto perParticleRevertPosAng = [=] __device__ (long particleId) {
			#pragma unroll (MAXDIM)
			for (long dim = 0; dim < f_nDim; dim++) {
				pPos[particleId * f_nDim + dim] -= d_fire_dt * pVel[particleId * f_nDim + dim];
			}
			pAngle[particleId] -= d_fire_dt * pAngvel[particleId];
		};

		thrust::for_each(r, r + dpm_->numParticles, perParticleRevertPosAng);
		thrust::fill(dpm_->d_particleVel.begin(), dpm_->d_particleVel.end(), double(0));
		thrust::fill(dpm_->d_particleAngvel.begin(), dpm_->d_particleAngvel.end(), double(0));
	}
}

void FIRE::bendRigidVelocityTowardsForce() {
	// calculate the ratio of the norm squared of the velocity and the force
  thrust::transform(dpm_->d_particleVel.begin(), dpm_->d_particleVel.end(), d_velSquared.begin(), square());
  thrust::transform(dpm_->d_particleForce.begin(), dpm_->d_particleForce.end(), d_forceSquared.begin(), square());
	double velNormSquared = thrust::reduce(d_velSquared.begin(), d_velSquared.end(), double(0), thrust::plus<double>());
	double forceNormSquared = thrust::reduce(d_forceSquared.begin(), d_forceSquared.end(), double(0), thrust::plus<double>());
	// check FIRE convergence
	if (forceNormSquared > 0) {
		double velForceNormRatio = sqrt(velNormSquared / forceNormSquared);
		double f_a(a);
		auto r = thrust::counting_iterator<long>(0);
		double* pVel = thrust::raw_pointer_cast(&(dpm_->d_particleVel[0]));
		const double* pForce = thrust::raw_pointer_cast(&(dpm_->d_particleForce[0]));

		auto perDOFBendParticleVelocity = [=] __device__ (long i) {
			pVel[i] = (1 - f_a) * pVel[i] + f_a * pForce[i] * velForceNormRatio;
		};

		thrust::for_each(r, r + dpm_->d_particleVel.size(), perDOFBendParticleVelocity);
	}
}

// Run the inner loop of the FIRE algorithm
void FIRE::minimizerRigidLoop() {
	dpm_->d_particleInitPos = dpm_->getParticlePositions();
	dpm_->d_particleInitAngle = dpm_->getParticleAngles();
	// check and reset fire params
	checkRigidFIREParams();
	// Update the velocity
	updateRigidVelAngvel(0.5 * fire_dt);
	// Bend the velocity towards the force
	bendRigidVelocityTowardsForce();
	// Move the system forward, based on the previous velocities and forces
	updateRigidPosAng(fire_dt);
	// Update vertex positions
	dpm_->translateVertices();
	dpm_->rotateVertices();
	dpm_->checkNeighbors();
	// Calculate the new set of forces at the new step
	dpm_->calcRigidForceEnergy();
	// Update the velocity
	updateRigidVelAngvel(0.5 * fire_dt);
}

//***************************** vertex minimizer *****************************//
// update position and velocity in response to an applied force
void FIRE::updateVertexPosition(double timeStep) {
	long f_nDim(dpm_->nDim);
	double d_fire_dt(timeStep);
	auto r = thrust::counting_iterator<long>(0);
	double* pos = thrust::raw_pointer_cast(&dpm_->d_pos[0]);
	const double* vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));

	auto perVertexUpdatePos = [=] __device__ (long vertexId) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < f_nDim; dim++) {
			pos[vertexId * f_nDim + dim] += d_fire_dt * vel[vertexId * f_nDim + dim];
		}
	};

	thrust::for_each(r, r + dpm_->numVertices, perVertexUpdatePos);
}

// update velocity in response to an applied force and return the maximum displacement in the previous step
void FIRE::updateVertexVelocity(double timeStep) {
	long f_nDim(dpm_->nDim);
	double d_fire_dt(timeStep);
	auto r = thrust::counting_iterator<long>(0);
	double* vel = thrust::raw_pointer_cast(&dpm_->d_vel[0]);
	const double* force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));

	auto perVertexUpdateVel = [=] __device__ (long vertexId) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < f_nDim; dim++) {
			vel[vertexId * f_nDim + dim] += d_fire_dt * force[vertexId * f_nDim + dim];
		}
	};

	thrust::for_each(r, r + dpm_->numVertices, perVertexUpdateVel);
}

// check and reset fire params
void FIRE::checkFIREParams() {
	// get the dot product between the velocity and the force
	double vDotF = thrust::inner_product(dpm_->d_vel.begin(), dpm_->d_vel.end(), dpm_->d_force.begin(), double(0));
	if (vDotF > 0) {
		// Increase the number of steps since the last restart
		numStep++;
		// if enough time has passed then let's start to increase the inertia
		if (numStep > minStep) {
			fire_dt = std::min(fire_dt * f_inc, fire_dt_max);
			a *= f_a;
		}
	} else {
		// start fresh with a more radical mixing between force and velocity
		numStep = 0;
		fire_dt = std::max(fire_dt * f_dec, fire_dt_max / 10);
		a = a_start;
		// take a step back
		long f_nDim(dpm_->nDim);
		double d_fire_dt(0.5 * fire_dt);
		auto r = thrust::counting_iterator<long>(0);
		double* pos = thrust::raw_pointer_cast(&(dpm_->d_pos[0]));
		const double* vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));

		auto perVertexRevertPos = [=] __device__ (long vertexId) {
			#pragma unroll (MAXDIM)
			for (long dim = 0; dim < f_nDim; dim++) {
				pos[vertexId * f_nDim + dim] -= d_fire_dt * vel[vertexId * f_nDim + dim];
			}
		};

		thrust::for_each(r, r + dpm_->numVertices, perVertexRevertPos);
		thrust::fill(dpm_->d_vel.begin(), dpm_->d_vel.end(), double(0));
	}
}

// bend the velocity towards the force
void FIRE::bendVertexVelocityTowardsForce() {
	// calculate the ratio of the norm squared of the velocity and the force
  thrust::transform(dpm_->d_vel.begin(), dpm_->d_vel.end(), d_velSquared.begin(), square());
  thrust::transform(dpm_->d_force.begin(), dpm_->d_force.end(), d_forceSquared.begin(), square());
	double velNormSquared = thrust::reduce(d_velSquared.begin(), d_velSquared.end(), double(0), thrust::plus<double>());
	double forceNormSquared = thrust::reduce(d_forceSquared.begin(), d_forceSquared.end(), double(0), thrust::plus<double>());
	// check FIRE convergence
	if (forceNormSquared > 0) {
		double velForceNormRatio = sqrt(velNormSquared / forceNormSquared);
		double f_a(a);
		auto r = thrust::counting_iterator<long>(0);
		double* vel = thrust::raw_pointer_cast(&(dpm_->d_vel[0]));
		const double* force = thrust::raw_pointer_cast(&(dpm_->d_force[0]));

		auto perDOFBendVertexVelocity = [=] __device__ (long i) {
			vel[i] = (1 - f_a) * vel[i] + f_a * force[i] * velForceNormRatio;
		};

		thrust::for_each(r, r + dpm_->d_vel.size(), perDOFBendVertexVelocity);
	}
}

// Run the inner loop of the FIRE algorithm
void FIRE::minimizerVertexLoop() {
	// check and reset fire params
	checkFIREParams();
	// Update the velocity
	updateVertexVelocity(0.5 * fire_dt);
	// Bend the velocity towards the force
	bendVertexVelocityTowardsForce();
	// Move the system forward, based on the previous velocities and forces
	updateVertexPosition(fire_dt);
	// Calculate the new set of forces at the new step
	dpm_->checkNeighbors();
	dpm_->calcForceEnergy();
	// Update the velocity
	updateVertexVelocity(0.5 * fire_dt);
}
