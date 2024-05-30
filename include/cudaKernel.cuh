//
// Author: Francesco Arceri
// Date:   10-03-2021
//
// KERNEL FUNCTIONS THAT ACT ON THE DEVICE(GPU)

#ifndef DPM2DKERNEL_CUH_
#define DPM2DKERNEL_CUH_

#include "defs.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

__constant__ simControlStruct d_simControl;

__constant__ double* d_boxSizePtr;

__constant__ long d_nDim;
__constant__ long d_numParticles;
__constant__ long d_numVertexPerParticle;
__constant__ long d_numVertices;

__constant__ long* d_numVertexInParticleListPtr;
__constant__ long* d_firstVertexInParticleIdPtr;
__constant__ long* d_particleIdListPtr;

// time step
__constant__ double d_dt;
// dimensionality factor
__constant__ double d_rho0;
// vertex/particle energy costs
__constant__ double d_ea; // area
__constant__ double d_el; // segment
__constant__ double d_eb; // bending
__constant__ double d_ec; // interaction
// attractive constants
__constant__ double d_l1; // depth
__constant__ double d_l2; // range
// Lennard-Jones constants
__constant__ double d_LJcutoff;
__constant__ double d_LJecut;
// FENE constants
__constant__ double d_stiff;
__constant__ double d_extSq;

// vertex neighborList
__constant__ long* d_neighborListPtr;
__constant__ long* d_maxNeighborListPtr;
__constant__ long d_neighborListSize;
__constant__ long d_maxNeighbors;

// vertex linked list
__constant__ long d_numCells;
__constant__ double d_cellSize;
__constant__ long* d_headerPtr;
__constant__ long* d_linkedListPtr;
__constant__ long* d_cellIndexListPtr;

// particle neighborList
__constant__ long* d_partNeighborListPtr;
__constant__ long* d_partMaxNeighborListPtr;
__constant__ long d_partNeighborListSize;
__constant__ long d_partMaxNeighbors;


inline __device__ double pbcDistance(const double x1, const double x2, const long dim) {
	auto delta = x1 - x2;
	auto size = d_boxSizePtr[dim];
	return delta - size * round(delta / size); //round for distance, floor for position
}

inline __device__ double calcNorm(const double* segment) {
  	auto normSq = 0.0;
  	for (long dim = 0; dim < d_nDim; dim++) {
    	normSq += segment[dim] * segment[dim];
  	}
  	return sqrt(normSq);
}

inline __device__ double calcNormSq(const double* segment) {
  	auto normSq = 0.0;
  	for (long dim = 0; dim < d_nDim; dim++) {
    	normSq += segment[dim] * segment[dim];
  	}
  	return normSq;
}

inline __device__ double calcDistance(const double* thisVec, const double* otherVec) {
  	auto delta = 0.0;
	auto distanceSq = 0.0;
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
    	delta = pbcDistance(thisVec[dim], otherVec[dim], dim);
    	distanceSq += delta * delta;
  	}
  	return sqrt(distanceSq);
}

inline __device__ double calcDeltaAndDistance(const double* thisVec, const double* otherVec, double* deltaVec) {
	auto delta = 0.0;
	auto distanceSq = 0.0;
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
    	delta = pbcDistance(thisVec[dim], otherVec[dim], dim);
		deltaVec[dim] = delta;
    	distanceSq += delta * delta;
  	}
  	return sqrt(distanceSq);
}

inline __device__ double calcFixedBoundaryDistance(const double* thisVec, const double* otherVec) {
  	auto delta = 0.0;
	auto distanceSq = 0.0;
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
    	delta = thisVec[dim] - otherVec[dim];
    	distanceSq += delta * delta;
  	}
  	return sqrt(distanceSq);
}

inline __device__ void getSegment(const double* thisVec, const double* otherVec, double* segment) {
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
    	segment[dim] = thisVec[dim] - otherVec[dim];
  	}
}

inline __device__ void getDelta(const double* thisVec, const double* otherVec, double* delta) {
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
    	delta[dim] = pbcDistance(thisVec[dim], otherVec[dim], dim);
  	}
}

// get index of next vertex of the same particle
inline __device__ long getNextId(const long vertexId, const long particleId) {
	if(vertexId == d_numVertices - 1) {
		return vertexId - d_numVertexInParticleListPtr[particleId] + 1;
	}
  	auto nextId = vertexId + 1;
  	if( d_particleIdListPtr[nextId] == particleId ) {
    	return nextId;
  	} else {
   		return nextId - d_numVertexInParticleListPtr[particleId];// return first vertex in the particle
  	}
}

// get index of previous vertex of the same particle
inline __device__ long getPreviousId(const long vertexId, const long particleId) {
	if(vertexId == 0) {
		return d_numVertexInParticleListPtr[particleId] - 1;
	}
  	auto previousId = vertexId - 1;
  	if( d_particleIdListPtr[previousId] == particleId ) {
    	return previousId;
  	} else {
   		return previousId + d_numVertexInParticleListPtr[particleId];// return last vertex in the particle
  	}
}

inline __device__ void getVertexPos(const long vId, const double* pos, double* vPos) {
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
		vPos[dim] = pos[vId * d_nDim + dim];
	}
}

inline __device__ void getVertexPBCPos(const long vId, const double* pos, double* vPos) {
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
		vPos[dim] = pos[vId * d_nDim + dim];
		vPos[dim] -= floor(vPos[dim] / d_boxSizePtr[dim]) * d_boxSizePtr[dim];
	}
}

inline __device__ void getParticlePos(const long pId, const double* pPos, double* tPos) {
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
		tPos[dim] = pPos[pId * d_nDim + dim];
	}
}

inline __device__ void getVertex(const long thisId, const double* pos, const double* rad, double* otherPos, double& otherRad) {
  	getVertexPos(thisId, pos, otherPos);
  	otherRad = rad[thisId];
}


inline __device__ void getRelativeVertexPos(const long vId, const double* pos, double* vPos, const double* partPos) {
	getVertexPos(vId, pos, vPos);
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
    	vPos[dim] = pbcDistance(vPos[dim], partPos[dim], dim);
  	}
}

inline __device__ bool extractOtherVertex(const long vertexId, const long otherId, const double* pos, const double* rad, double* otherPos, double& otherRad) {
	if ((d_particleIdListPtr[vertexId] != d_particleIdListPtr[otherId]) && (otherId != -1)) {
		getVertex(otherId, pos, rad, otherPos, otherRad);
    	return true;
  	}
  	return false;
}

inline __device__ bool extractOtherParticle(const long particleId, const long otherId, const double* pPos, const double* pRad, double* otherPos, double& otherRad) {
	if ((particleId != otherId) && (otherId != -1)) {
		getParticlePos(otherId, pPos, otherPos);
		otherRad = pRad[particleId];
    	return true;
  	}
  	return false;
}

inline __device__ bool extractOtherParticlePos(const long particleId, const long otherId, const double* pPos, double* otherPos) {
	if ((particleId != otherId) && (otherId != -1)) {
		getParticlePos(otherId, pPos, otherPos);
    	return true;
  	}
  	return false;
}

inline __device__ bool extractNeighbor(const long vertexId, const long nListId, const double* pos, const double* rad, double* otherPos, double& otherRad) {
	auto otherId = d_neighborListPtr[vertexId*d_neighborListSize + nListId];
  	if ((vertexId != otherId) && (otherId != -1)) {
		#pragma unroll (MAXDIM)
    	for (long dim = 0; dim < d_nDim; dim++) {
      		otherPos[dim] = pos[otherId * d_nDim + dim];
    	}
    	otherRad = rad[otherId];
    	return true;
  	}
  return false;
}

inline __device__ bool extractNeighborPos(const long vertexId, const long nListId, const double* pos, double* otherPos) {
	auto otherId = d_neighborListPtr[vertexId*d_neighborListSize + nListId];
  	if ((vertexId != otherId) && (otherId != -1)) {
		#pragma unroll (MAXDIM)
    	for (long dim = 0; dim < d_nDim; dim++) {
      		otherPos[dim] = pos[otherId * d_nDim + dim];
    	}
    	return true;
  	}
  return false;
}

inline __device__ bool extractParticleNeighbor(const long particleId, const long nListId, const double* pPos, const double* pRad, double* otherPos, double& otherRad) {
	auto otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
  	if ((particleId != otherId) && (otherId != -1)) {
		#pragma unroll (MAXDIM)
    	for (long dim = 0; dim < d_nDim; dim++) {
      		otherPos[dim] = pPos[otherId * d_nDim + dim];
    	}
    	otherRad = pRad[otherId];
    	return true;
  	}
  return false;
}

inline __device__ bool extractParticleNeighborPos(const long particleId, const long nListId, const double* pPos, double* otherPos) {
	auto otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
  	if ((particleId != otherId) && (otherId != -1)) {
		#pragma unroll (MAXDIM)
    	for (long dim = 0; dim < d_nDim; dim++) {
      		otherPos[dim] = pPos[otherId * d_nDim + dim];
    	}
    	return true;
  	}
  	return false;
}

//works only for 2D
inline __device__ void calcParticlePos(const long particleId, const double* pos, double* partPos) {
  	// compute particle center of mass
  	double currentPos[MAXDIM], delta[MAXDIM];
	auto nextId = -1;
	auto firstId = d_firstVertexInParticleIdPtr[particleId];
	getVertexPos(firstId, pos, currentPos);
	#pragma unroll (MAXDIM)
	for (long dim = 0; dim < d_nDim; dim++) {
		partPos[dim] = currentPos[dim];
	}
  	for (long currentId = firstId; currentId < firstId + d_numVertexInParticleListPtr[particleId]-1; currentId++) {
		nextId = getNextId(currentId, particleId);
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			delta[dim] = pbcDistance(pos[nextId * d_nDim + dim], currentPos[dim], dim);
			currentPos[dim] += delta[dim];
			partPos[dim] += currentPos[dim];
		}
	}
	#pragma unroll (MAXDIM)
	for (long dim = 0; dim < d_nDim; dim++) {
	  partPos[dim] /= d_numVertexInParticleListPtr[particleId];
	}
}

//works only for 2D
inline __device__ double getParticleArea(const long particleId, const double* pos) {
	auto tempArea = 0.0;
	auto nextId = -1;
	auto firstId = d_firstVertexInParticleIdPtr[particleId];
	double delta[MAXDIM], nextPos[MAXDIM], currentPos[MAXDIM];
	getVertexPos(firstId, pos, currentPos);
	// compute particle area via shoe-string method
	for (long currentId = firstId; currentId < firstId + d_numVertexInParticleListPtr[particleId]; currentId++) {
		nextId = getNextId(currentId, particleId);
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			delta[dim] = pbcDistance(pos[nextId * d_nDim + dim], currentPos[dim], dim);
			nextPos[dim] = currentPos[dim] + delta[dim];
		}
		tempArea += currentPos[0] * nextPos[1] - nextPos[0] * currentPos[1];
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			currentPos[dim] = nextPos[dim];
		}
	}
	return abs(tempArea) * 0.5;
}

inline __device__ double calcOverlap(const double* thisVec, const double* otherVec, const double radSum) {
  	return (1 - calcDistance(thisVec, otherVec) / radSum);
}

inline __device__ double calcFixedBoundaryOverlap(const double* thisVec, const double* otherVec, const double radSum) {
  	return (1 - calcFixedBoundaryDistance(thisVec, otherVec) / radSum);
}

inline __device__ void getNormalVector(const double* thisVec, double* normalVec) {
	normalVec[0] = thisVec[1];
	normalVec[1] = -thisVec[0];
}

inline __device__ double calcAngle(const double* nSegment, const double* pSegment) {
	auto midSine = nSegment[0] * pSegment[1] - nSegment[1] * pSegment[0];
	auto midCosine = nSegment[0] * pSegment[0] + nSegment[1] * pSegment[1];
	return atan2(midSine, midCosine);
}

inline __device__ double calcAreaForceEnergy(const double pA0, const double pA, const double* nPos, const double* pPos, double* vertexForce) {
	auto deltaA = (pA / pA0) - 1.; // area variation
	auto gradMultiple = d_ea * deltaA / pA0;
  	vertexForce[0] += 0.5 * gradMultiple * (pPos[1] - nPos[1]);
  	vertexForce[1] += 0.5 * gradMultiple * (nPos[0] - pPos[0]);
  	return (0.5 * d_ea * deltaA * deltaA);
}

inline __device__ double calcPerimeterForceEnergy(const double tL0, const double pL0, const double nLength, const double pLength, const double* vPos, const double* nPos, const double* pPos, double* vertexForce) {
  	//compute length variations
  	auto pDelta = (pLength / pL0) - 1.;
  	auto nDelta = (nLength / tL0) - 1.;
	// compute force
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
    	vertexForce[dim] += d_el * ( ( nDelta * (nPos[dim] - vPos[dim]) / (tL0 * nLength) ) - ( pDelta * (vPos[dim] - pPos[dim]) / (pL0 * pLength) ) );
  	}
  	return (0.5 * d_el * pDelta * pDelta);
}

inline __device__ double calcFENEPerimeterForceEnergy(const double tL0, const double pL0, const double nLength, const double pLength, const double* vPos, const double* nPos, const double* pPos, double* vertexForce) {
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
    	vertexForce[dim] += 2 * d_el * d_stiff * d_extSq * ((nLength / (d_extSq * tL0 * tL0 - nLength * nLength)) * (nPos[dim] - vPos[dim]) / nLength - (pLength / (d_extSq * pL0 * pL0 - pLength * pLength)) * (vPos[dim] - pPos[dim]) / pLength);
  	}
  	return d_el * d_stiff * d_extSq * log(1 - pLength * pLength / (d_extSq * pL0 * pL0));
}

inline __device__ double calcBendingForceEnergy(const double* preSegment, const double* nextSegment, const double thisAngleDelta, const double nextAngleDelta, const double preAngleDelta, double* vertexForce) {
	double preNormalSegment[MAXDIM], nextNormalSegment[MAXDIM];
	// get normal segments
	getNormalVector(preSegment, preNormalSegment);
	getNormalVector(nextSegment, nextNormalSegment);
	// compute angle variations
	auto preVar = (thisAngleDelta - preAngleDelta) / calcNormSq(preSegment);
	auto nextVar = (thisAngleDelta - nextAngleDelta) / calcNormSq(nextSegment);
	// compute force
	#pragma unroll (MAXDIM)
	for (long dim = 0; dim < d_nDim; dim++) {
		vertexForce[dim] += d_eb * (preVar * preNormalSegment[dim] + nextVar * nextNormalSegment[dim]);
	}
	return (0.5 * d_eb * thisAngleDelta * thisAngleDelta);
}

// this, next and previous are for vertices belonging to the same particle
__global__ void kernelCalcShapeForceEnergy(const double* a0, const double* area, const double* particlePos, const double* l0, const double* theta0, const double* theta, const double* pos, double* force, double* energy) {
	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  	long particleId = d_particleIdListPtr[vertexId];
  	if (vertexId < d_numVertices) {
		double vertexPos[MAXDIM], nextPos[MAXDIM], previousPos[MAXDIM], partPos[MAXDIM];
		//double secondNextPos[MAXDIM], secondPreviousPos[MAXDIM];
		double nextSegment[MAXDIM], previousSegment[MAXDIM];
		//double secondNextSegment[MAXDIM], secondPreviousSegment[MAXDIM];
		auto particleArea = area[particleId];
		auto shapeEnergy = 0.0;
		// get interacting vertices' indices
		auto nextId = getNextId(vertexId, particleId);
	  	auto previousId = getPreviousId(vertexId, particleId);
	  	//auto secondNextId = getNextId(nextId, particleId);
	  	//auto secondPreviousId = getPreviousId(previousId, particleId);
		//printf("vertexId: %ld, previousId: %ld, nextId: %ld \n", vertexId, previousId, nextId);
		// zero out the existing force and get particlePos
    	for (long dim = 0; dim < d_nDim; dim++) {
			force[vertexId * d_nDim + dim] = 0.0;
			partPos[dim] = particlePos[particleId * d_nDim + dim];
		}
		// get positions relative to particle center of mass
		getRelativeVertexPos(vertexId, pos, vertexPos, partPos);
	  	getRelativeVertexPos(nextId, pos, nextPos, partPos);
	  	getRelativeVertexPos(previousId, pos, previousPos, partPos);
		// area force
		shapeEnergy += calcAreaForceEnergy(a0[particleId], particleArea, nextPos, previousPos, &force[vertexId*d_nDim]) / d_numVertexInParticleListPtr[particleId];
		// perimeter force
		getSegment(nextPos, vertexPos, nextSegment);
		getSegment(vertexPos, previousPos, previousSegment);
	  	auto previousLength = calcNorm(previousSegment);
	  	auto nextLength = calcNorm(nextSegment);
	  	switch (d_simControl.monomerType) {
			case simControlStruct::monomerEnum::harmonic:
			shapeEnergy += calcPerimeterForceEnergy(l0[vertexId], l0[previousId], nextLength, previousLength, vertexPos, nextPos, previousPos, &force[vertexId*d_nDim]);
			break;
			case simControlStruct::monomerEnum::FENE:
			shapeEnergy += calcFENEPerimeterForceEnergy(l0[vertexId], l0[previousId], nextLength, previousLength, vertexPos, nextPos, previousPos, &force[vertexId*d_nDim]);
			break;
		}
		// bending force
		//getRelativeVertexPos(secondNextId, pos, secondNextPos, partPos);
	  	//getRelativeVertexPos(secondPreviousId, pos, secondPreviousPos, partPos);
		//getSegment(secondNextPos, nextPos, secondNextSegment);
		//getSegment(previousPos, secondPreviousPos, secondPreviousSegment);
		//theta[previousId] = calcAngle(previousSegment, secondPreviousSegment);
	  	auto previousAngleDelta = theta[previousId] - theta0[previousId];
		//theta[vertexId] = calcAngle(nextSegment, previousSegment);
	  	auto thisAngleDelta = theta[vertexId] - theta0[vertexId];
		//theta[nextId] = calcAngle(secondNextSegment, nextSegment);
	 	auto nextAngleDelta = theta[nextId] - theta0[nextId];
		shapeEnergy += calcBendingForceEnergy(previousSegment, nextSegment, thisAngleDelta, nextAngleDelta, previousAngleDelta, &force[vertexId*d_nDim]);
    	energy[vertexId] = shapeEnergy;
		//printf("\n shape: vertexId %ld \t fx: %.13e \t fy: %.13e \n", vertexId, force[vertexId*d_nDim], force[vertexId*d_nDim+1]);
	}
}

inline __device__ double calcContactInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
  	double delta[MAXDIM];
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	auto overlap = 1 - distance / radSum;
	if (overlap > 0) {
		auto gradMultiple = d_ec * overlap / radSum;
		#pragma unroll (MAXDIM)
	  	for (long dim = 0; dim < d_nDim; dim++) {
	    	currentForce[dim] += gradMultiple * delta[dim] / distance;
	  	}
	  	return (0.5 * d_ec * overlap * overlap) * 0.5;
	}
	return 0.0;
}

inline __device__ double calcContactInteraction2(const double* thisPos, const double* otherPos, const double radSum, double* currentForce, double* otherForce) {
  	double delta[MAXDIM];
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	auto overlap = 1 - distance / radSum;
	if (overlap > 0) {
		auto gradMultiple = d_ec * overlap / radSum;
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			atomicAdd(&currentForce[dim], gradMultiple * delta[dim] / distance);
			atomicAdd(&otherForce[dim], -gradMultiple * delta[dim] / distance);
		}
	  	return (0.5 * d_ec * overlap * overlap) * 0.5;
	}
	return 0.0;
}

inline __device__ double calcLJForceShift(const double radSum) {
	auto ratio6 = pow(d_LJcutoff, 6);
	return 24 * d_ec * (2 / ratio6 - 1) / (d_LJcutoff * radSum * ratio6);
}

inline __device__ double calcLJInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
  	double delta[MAXDIM];
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance <= (d_LJcutoff * radSum)) {
		auto forceShift = calcLJForceShift(radSum);
		auto gradMultiple = 24 * d_ec * (2 * ratio12 - ratio6) / distance - forceShift;
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    	currentForce[dim] += gradMultiple * delta[dim] / distance;
	  	}
		return 0.5 * (4 * d_ec * (ratio12 - ratio6) - d_LJecut + forceShift * (distance - d_LJcutoff * radSum));
	}
	return 0.0;
}

inline __device__ double calcLJInteraction2(const double* thisPos, const double* otherPos, const double radSum, double* currentForce, double* otherForce) {
	double delta[MAXDIM];
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance <= (d_LJcutoff * radSum)) {
		auto forceShift = calcLJForceShift(radSum);
		auto gradMultiple = 24 * d_ec * (2 * ratio12 - ratio6) / distance - forceShift;
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    	atomicAdd(&currentForce[dim], gradMultiple * delta[dim] / distance);
	    	atomicAdd(&otherForce[dim], -gradMultiple * delta[dim] / distance);
	  	}
		return 0.5 * (4 * d_ec * (ratio12 - ratio6) - d_LJecut + forceShift * (distance - d_LJcutoff * radSum));
	}
	return 0.0;
}

inline __device__ double calcAdhesiveInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
  	auto gradMultiple = 0.0;
	auto epot = 0.0;
	double delta[MAXDIM];
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	auto overlap = 1 - distance / radSum;
	if(distance < (1 + d_l2) * radSum) {
		if (distance < (1 + d_l1) * radSum) {
			gradMultiple = d_ec * overlap / radSum;
			epot = 0.5 * d_ec * (overlap * overlap - d_l1 * d_l2) * 0.5;
		} else {//if ((distance >= (1 + d_l1) * radSum) && (distance < (1 + d_l2) * radSum)) {
			gradMultiple = -(d_ec * d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) / radSum;
			epot = -(0.5 * (d_ec * d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) * (overlap + d_l2)) * 0.5;
		}
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    	currentForce[dim] += gradMultiple * delta[dim] / distance;
	  	}
		return epot;
	} else {
		return 0.0;
	}
}

inline __device__ double calcAdhesiveInteraction2(const double* thisPos, const double* otherPos, const double radSum, double* currentForce, double* otherForce) {
  	auto gradMultiple = 0.0;
	auto epot = 0.0;
	double delta[MAXDIM];
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	auto overlap = 1 - distance / radSum;
	if(distance < (1 + d_l2) * radSum) {
		if (distance < (1 + d_l1) * radSum) {
			gradMultiple = d_ec * overlap / radSum;
			epot = 0.5 * d_ec * (overlap * overlap - d_l1 * d_l2) * 0.5;
		} else {//if ((distance >= (1 + d_l1) * radSum) && (distance < (1 + d_l2) * radSum)) {
			gradMultiple = -(d_ec * d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) / radSum;
			epot = -(0.5 * (d_ec * d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) * (overlap + d_l2)) * 0.5;
		}
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    	atomicAdd(&currentForce[dim], gradMultiple * delta[dim] / distance);
	    	atomicAdd(&otherForce[dim], -gradMultiple * delta[dim] / distance);
	  	}
		return epot;
	} else {
		return 0.0;
	}
}

inline __device__ double calcWCAInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
	double delta[MAXDIM];
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance <= (WCAcut * radSum)) {
		auto gradMultiple = 24 * d_ec * (2 * ratio12 - ratio6) / distance;
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			currentForce[dim] += gradMultiple * delta[dim] / distance;
		}
		return 0.5 * d_ec * (4 * (ratio12 - ratio6) + 1);
	}
	return 0;
}

inline __device__ double calcWCAInteraction2(const double* thisPos, const double* otherPos, const double radSum, double* currentForce, double* otherForce) {
	double delta[MAXDIM];
	auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	auto ratio = radSum / distance;
	auto ratio6 = pow(ratio, 6);
	auto ratio12 = ratio6 * ratio6;
	if (distance <= (WCAcut * radSum)) {
		auto gradMultiple = 24 * d_ec * (2 * ratio12 - ratio6) / distance;
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			atomicAdd(&currentForce[dim], 0.5 * gradMultiple * delta[dim] / distance);
	    	atomicAdd(&otherForce[dim], -0.5 * gradMultiple * delta[dim] / distance);
		}
		return 0.5 * d_ec * (4 * (ratio12 - ratio6) + 1);
	} else {
		return 0.0;
	}
}

// this and other are for vertices belonging to neighbor particles
__global__ void kernelCalcVertexInteraction(const double* rad, const double* pos, double* force, double* energy) {
  	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (vertexId < d_numVertices) {
    	auto otherRad = 0.0;
		double thisPos[MAXDIM], otherPos[MAXDIM];
		// we don't zero out the force because we always call this function
		// after kernelCalcShapeForceEnergy where the force is zeroed out
		getVertexPos(vertexId, pos, thisPos);
    	auto thisRad = rad[vertexId];
    	// interaction between vertices of neighbor particles
    	for (long nListId = 0; nListId < d_maxNeighborListPtr[vertexId]; nListId++) {
      		if(extractNeighbor(vertexId, nListId, pos, rad, otherPos, otherRad)) {
				auto radSum = thisRad + otherRad;
				switch (d_simControl.potentialType) {
					case simControlStruct::potentialEnum::harmonic:
					energy[vertexId] += calcContactInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::lennardJones:
					energy[vertexId] += calcLJInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::adhesive:
					energy[vertexId] += calcAdhesiveInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::wca:
					energy[vertexId] += calcWCAInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim]);
					break;
				}
				//if(calcOverlap(thisPos, otherPos, radSum) > 0) printf("\n vertexId %ld \t neighbor: %ld \t overlap %lf \t %lf \t distance: %lf \t radSum: %lf \t thisRad: %lf \t otherRad: %lf \n", vertexId, d_neighborListPtr[vertexId*d_neighborListSize + nListId], calcOverlap(thisPos, otherPos, radSum), 1-calcDistance(thisPos, otherPos)/radSum, calcDistance(thisPos, otherPos), radSum, thisRad, otherRad);
				//printf("interaction: vertexId %ld \t neighborId %ld \t fx: %e \t fy: %e \t overlap: %e \n", vertexId, d_neighborListPtr[vertexId*d_neighborListSize + nListId], force[vertexId*d_nDim], force[vertexId*d_nDim+1], calcOverlap(thisPos, otherPos, radSum));
			}
		}
  	}
}

// this and other are for vertices belonging to neighbor particles
__global__ void kernelCalcAllToAllVertexInteraction(const double* rad, const double* pos, double* force, double* energy) {
  	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (vertexId < d_numVertices) {
    	auto otherRad = 0.0;
		double thisPos[MAXDIM], otherPos[MAXDIM];
		// we don't zero out the force because we always call this function
		// after kernelCalcShapeForceEnergy where the force is zeroed out
		getVertexPos(vertexId, pos, thisPos);
    	auto thisRad = rad[vertexId];
    	// interaction between vertices of neighbor particles
    	for (long otherId = 0; otherId < d_numVertices; otherId++) {
			if(extractOtherVertex(vertexId, otherId, pos, rad, otherPos, otherRad)) {
				auto radSum = thisRad + otherRad;
				switch (d_simControl.potentialType) {
					case simControlStruct::potentialEnum::harmonic:
					energy[vertexId] += calcContactInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::lennardJones:
					energy[vertexId] += calcLJInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::adhesive:
					energy[vertexId] += calcAdhesiveInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::wca:
					energy[vertexId] += calcWCAInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim]);
					break;
				}
				//if(calcOverlap(thisPos, otherPos, radSum) > 0) printf("\n vertexId %ld \t neighbor: %ld \t overlap %lf \t %lf \t distance: %lf \t radSum: %lf \t thisRad: %lf \t otherRad: %lf \n", vertexId, d_neighborListPtr[vertexId*d_neighborListSize + nListId], calcOverlap(thisPos, otherPos, radSum), 1-calcDistance(thisPos, otherPos)/radSum, calcDistance(thisPos, otherPos), radSum, thisRad, otherRad);
				//printf("interaction: vertexId %ld \t neighborId %ld \t fx: %e \t fy: %e \t overlap: %e \n", vertexId, d_neighborListPtr[vertexId*d_neighborListSize + nListId], force[vertexId*d_nDim], force[vertexId*d_nDim+1], calcOverlap(thisPos, otherPos, radSum));
			}
		}
  	}
}

// this and other are for vertices belonging to neighbor particles
__global__ void kernelCalcVertexInteraction2(const double* rad, const double* pos, double* force, double* energy) {
  	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (vertexId < d_numVertices) {
		thread_block block = this_thread_block();
    	auto otherRad = 0.0;
		auto interaction = 0.0;
		double thisPos[MAXDIM], otherPos[MAXDIM];
		// we don't zero out the force because we always call this function
		// after kernelCalcShapeForceEnergy where the force is zeroed out
		getVertexPos(vertexId, pos, thisPos);
    	auto thisRad = rad[vertexId];
    	// interaction between vertices of neighbor particles
    	for (long nListId = 0; nListId < d_maxNeighborListPtr[vertexId]; nListId++) {
      		if(extractNeighbor(vertexId, nListId, pos, rad, otherPos, otherRad)) {
				auto otherId = d_neighborListPtr[vertexId*d_neighborListSize + nListId];
				auto radSum = thisRad + otherRad;
				switch (d_simControl.potentialType) {
					case simControlStruct::potentialEnum::harmonic:
					interaction = calcContactInteraction2(thisPos, otherPos, radSum, &force[vertexId*d_nDim], &force[otherId*d_nDim]);
					atomicAdd(&energy[vertexId], 0.5 * interaction);
					atomicAdd(&energy[otherId], 0.5 * interaction);
					break;
					case simControlStruct::potentialEnum::lennardJones:
					interaction = calcLJInteraction2(thisPos, otherPos, radSum, &force[vertexId*d_nDim], &force[otherId*d_nDim]);
					atomicAdd(&energy[vertexId], 0.5 * interaction);
					atomicAdd(&energy[otherId], 0.5 * interaction);
					break;
					case simControlStruct::potentialEnum::adhesive:
					interaction = calcAdhesiveInteraction2(thisPos, otherPos, radSum, &force[vertexId*d_nDim], &force[otherId*d_nDim]);
					atomicAdd(&energy[vertexId], 0.5 * interaction);
					atomicAdd(&energy[otherId], 0.5 * interaction);
					break;
					case simControlStruct::potentialEnum::wca:
					interaction = calcWCAInteraction2(thisPos, otherPos, radSum, &force[vertexId*d_nDim], &force[otherId*d_nDim]);
					atomicAdd(&energy[vertexId], 0.5 * interaction);
					atomicAdd(&energy[otherId], 0.5 * interaction);
					break;
				}
				block.sync();
				//if(calcOverlap(thisPos, otherPos, radSum) > 0) printf("\n vertexId %ld \t neighbor: %ld \t overlap %lf \t %lf \t distance: %lf \t radSum: %lf \t thisRad: %lf \t otherRad: %lf \n", vertexId, d_neighborListPtr[vertexId*d_neighborListSize + nListId], calcOverlap(thisPos, otherPos, radSum), 1-calcDistance(thisPos, otherPos)/radSum, calcDistance(thisPos, otherPos), radSum, thisRad, otherRad);
				//printf("interaction: vertexId %ld \t neighborId %ld \t fx: %e \t fy: %e \t overlap: %e \n", vertexId, d_neighborListPtr[vertexId*d_neighborListSize + nListId], force[vertexId*d_nDim], force[vertexId*d_nDim+1], calcOverlap(thisPos, otherPos, radSum));
			}
		}
  	}
}

inline __device__ double calcGradMultipleAndEnergy(const double* thisPos, const double* otherPos, const double radSum, double &epot) {
	double overlap, ratio, ratio6, ratio12, forceShift;
	auto distance = calcDistance(thisPos, otherPos);
	switch (d_simControl.potentialType) {
		case simControlStruct::potentialEnum::harmonic:
		overlap = 1 - distance / radSum;
		if(overlap > 0) {
			epot = 0.5 * d_ec * overlap * overlap * 0.5;
			return d_ec * overlap / radSum;
		} else {
			return 0;
		}
		break;
		case simControlStruct::potentialEnum::lennardJones:
		ratio = radSum / distance;
		ratio12 = pow(ratio, 12);
		ratio6 = pow(ratio, 6);
		if (distance <= (d_LJcutoff * radSum)) {
			forceShift = calcLJForceShift(radSum);
			epot = 0.5 * (4 * d_ec * (ratio12 - ratio6) - d_LJecut + forceShift * (distance - d_LJcutoff * radSum));
			return 24 * d_ec * (2 * ratio12 - ratio6) / distance - forceShift;
		} else {
			return 0;
		}
		break;
		case simControlStruct::potentialEnum::adhesive:
		overlap = 1 - distance / radSum;
		if (distance < (1 + d_l1) * radSum) {
			epot = 0.5 * d_ec * (overlap * overlap - d_l1 * d_l2) * 0.5;
			return d_ec * overlap / radSum;
		} else if ((distance >= (1 + d_l1) * radSum) && (distance < (1 + d_l2) * radSum)) {
			epot = -(0.5 * (d_ec * d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) * (overlap + d_l2)) * 0.5;
			return -(d_ec * d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) / radSum;
		} else {
			return 0;
		}
		break;
		case simControlStruct::potentialEnum::wca:
		ratio = radSum / distance;
		ratio6 = pow(ratio, 6);
		ratio12 = ratio6 * ratio6;
		if (distance <= (WCAcut * radSum)) {
			epot = 0.5 * d_ec * (4 * (ratio12 - ratio6) + 1);
			return 4 * d_ec * (12 * ratio12 - 6 * ratio6) / distance;
		} else {
			return 0;
		}
		break;
		default:
		return 0;
		break;
	}
}

// clockwise projection
inline __device__ double getProjection(const double* thisPos, const double* otherPos, const double* previousPos, const double length) {
	return (pbcDistance(thisPos[0], previousPos[0], 0) * pbcDistance(otherPos[0], previousPos[0], 0) + pbcDistance(thisPos[1], previousPos[1], 1) * pbcDistance(otherPos[1], previousPos[1], 1)) / (length * length);

}

inline __device__ void getProjectionPos(const double* previousPos, const double* segment, double* projPos, const double proj) {
	auto reducedProj = max(0.0, min(1.0, proj));
	for (long dim = 0; dim < d_nDim; dim++) {
		projPos[dim] = previousPos[dim] + reducedProj * segment[dim];
	}
}

inline __device__ double calcCross(const double* thisPos, const double* otherPos, const double* previousPos) {
	return pbcDistance(previousPos[0], otherPos[0],0) * pbcDistance(otherPos[1], thisPos[1],1) - pbcDistance(otherPos[0], thisPos[0],0) * pbcDistance(previousPos[1], otherPos[1],1);
}

inline __device__ double calcVertexSegmentInteraction(const double* thisPos, const double* projPos, const double* otherPos, const double* previousPos, const double length, const double radSum, double* thisForce, double* otherForce, double* previousForce) {
	//double segment[MAXDIM];
	auto epot = 0.0;
	auto gradMultiple = calcGradMultipleAndEnergy(thisPos, projPos, radSum, epot);
	if (gradMultiple != 0) {
		auto cross = calcCross(thisPos, otherPos, previousPos);
		auto absCross = fabs(cross);
		auto sign = cross / absCross;
		// this vertex
	  	atomicAdd(&thisForce[0], gradMultiple * sign * pbcDistance(previousPos[1], otherPos[1], 1) / length);
	  	atomicAdd(&thisForce[1], gradMultiple * sign * pbcDistance(otherPos[0], previousPos[0], 0) / length);
		// other vertex
	  	atomicAdd(&otherForce[0], gradMultiple * (sign * pbcDistance(thisPos[1], previousPos[1], 1) + absCross * pbcDistance(previousPos[0], otherPos[0], 0) / (length * length)) / length);
	  	atomicAdd(&otherForce[1], gradMultiple * (sign * pbcDistance(previousPos[0], thisPos[0], 0) + absCross * pbcDistance(previousPos[1], otherPos[1], 1) / (length * length)) / length);
		// previous vertex
	  	atomicAdd(&previousForce[0], gradMultiple * (sign * pbcDistance(otherPos[1], thisPos[1], 1) - absCross * pbcDistance(previousPos[0], otherPos[0], 0) / (length * length)) / length);
	  	atomicAdd(&previousForce[1], gradMultiple * (sign * pbcDistance(thisPos[0], otherPos[0], 0) - absCross * pbcDistance(previousPos[1], otherPos[1], 1) / (length * length)) / length);
	  	return epot;
	}
	return 0.;
}

inline __device__ double calcVertexVertexInteraction(const double* thisPos, const double* previousPos, const double radSum, double* thisForce, double* otherForce) {
	double delta[MAXDIM];
	auto epot = 0.0;
	auto gradMultiple = calcGradMultipleAndEnergy(thisPos, previousPos, radSum, epot);
	if (gradMultiple != 0) {
		auto distance = calcDeltaAndDistance(thisPos, previousPos, delta);
		for (long dim = 0; dim < d_nDim; dim++) {
			atomicAdd(&thisForce[dim], gradMultiple * delta[dim] / distance);
			atomicAdd(&otherForce[dim], -gradMultiple * delta[dim] / distance);
		}
		return epot;
	}
	return 0.;
}

inline __device__ double checkAngle(double angle, double limit) {
	if(angle < 0) {
		angle += 2*PI;
	}
	return angle - limit;
}

// interaction force between vertices computed sequentially for each particle
__global__ void kernelCalcSmoothInteraction(const double* rad, const double* pos, double* force, double* pEnergy) {
	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (vertexId < d_numVertices) {
		//thread_block block = this_thread_block();
		auto otherRad = 0.0;
		auto interaction = 0.0;
		double thisPos[MAXDIM], otherPos[MAXDIM], previousPos[MAXDIM], secondPreviousPos[MAXDIM];
		double projPos[MAXDIM], segment[MAXDIM], previousSegment[MAXDIM], interSegment[MAXDIM];//, relSegment[MAXDIM], relPos[MAXDIM];
		getVertexPos(vertexId, pos, thisPos);
		auto thisRad = rad[vertexId];
		auto particleId = d_particleIdListPtr[vertexId];
		// loop through vertices in neighborList
		for (long nListId = 0; nListId < d_maxNeighborListPtr[vertexId]; nListId++) {
			if(extractNeighbor(vertexId, nListId, pos, rad, otherPos, otherRad)) {
				auto otherId = d_neighborListPtr[vertexId*d_neighborListSize + nListId];
				auto radSum = thisRad + otherRad;
				auto otherParticleId = d_particleIdListPtr[otherId];
				// compute projection of vertexId on segment between otherId and previousId
				auto previousId = getPreviousId(otherId, otherParticleId);
				getVertexPos(previousId, pos, previousPos);
				getDelta(otherPos, previousPos, segment);
				//getDelta(thisPos, previousPos, relSegment);
				//for (long dim = 0; dim < d_nDim; dim++) {
				//	relPos[dim] = previousPos[dim] + relSegment[dim];
				//}
				auto length = calcNorm(segment);
				auto projection = getProjection(thisPos, otherPos, previousPos, length);
				// check if the interaction is vertex-segment
				if(projection >= 0 && projection < 1) {
					getProjectionPos(previousPos, segment, projPos, projection);
					interaction = calcVertexSegmentInteraction(thisPos, projPos, otherPos, previousPos, length, radSum, &force[vertexId*d_nDim], &force[otherId*d_nDim], &force[previousId*d_nDim]);
					atomicAdd(&pEnergy[particleId], interaction);
					atomicAdd(&pEnergy[otherParticleId], interaction);
					//pEnergy[particleId] += interaction;
					//pEnergy[otherParticleId] += interaction;
					//__syncthreads();
				} else if(projection < 0) {
					auto secondPreviousId = getPreviousId(previousId, otherParticleId);
					getVertexPos(secondPreviousId, pos, secondPreviousPos);
					getDelta(secondPreviousPos, previousPos, previousSegment);
					length = calcNorm(previousSegment);
					//getDelta(thisPos, secondPreviousPos, relSegment);
					//for (long dim = 0; dim < d_nDim; dim++) {
					//	relPos[dim] = secondPreviousPos[dim] + relSegment[dim];
					//}
					auto previousProj = getProjection(thisPos, previousPos, secondPreviousPos, length);
					switch (d_simControl.concavityType) {
						case simControlStruct::concavityEnum::off:
						if(previousProj >= 1) {
							interaction = calcVertexVertexInteraction(thisPos, previousPos, radSum, &force[vertexId*d_nDim], &force[previousId*d_nDim]);
							atomicAdd(&pEnergy[particleId], interaction);
							atomicAdd(&pEnergy[otherParticleId], interaction);
							//pEnergy[particleId] += interaction;
							//pEnergy[otherParticleId] += interaction;
							//__syncthreads();
						}
						break;
						case simControlStruct::concavityEnum::on:
						// check if the vertex-vertex interaction is concave or convex
						getDelta(thisPos, previousPos, interSegment);
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
								interaction = calcVertexVertexInteraction(previousPos, thisPos, radSum, &force[vertexId*d_nDim], &force[previousId*d_nDim]);
								atomicAdd(&pEnergy[particleId], interaction);
								atomicAdd(&pEnergy[otherParticleId], interaction);
								//pEnergy[particleId] += interaction;
								//pEnergy[otherParticleId] += interaction;
								//__syncthreads();
							} else if(previousProj > 1) {
								interaction = calcVertexVertexInteraction(thisPos, previousPos, radSum, &force[vertexId*d_nDim], &force[previousId*d_nDim]);
								atomicAdd(&pEnergy[particleId], interaction);
								atomicAdd(&pEnergy[otherParticleId], interaction);
								//pEnergy[particleId] += interaction;
								//pEnergy[otherParticleId] += interaction;
								//__syncthreads();
							}
						}
						break;
					}
				}
			}
		}
	//block.sync();
  	}
}

inline __device__ bool isVertexInCell(const double* vertexPos, const long vertexId, const long cellId) {
	auto cIdx = static_cast<long>(vertexPos[0] / d_cellSize);
	auto cIdy = static_cast<long>(vertexPos[1] / d_cellSize);
	auto cId = cIdx * d_numCells + cIdy;
	printf("cId %ld cIdx %ld cIdy %ld \n", cId, cIdx, cIdy);
	if(cId == cellId) return true;
	else return false;
}

inline __device__ long getNeighborCellId(const long cellIdx, const long cellIdy, const long dx, const long dy) {
	// check boundary conditions
	auto cIdx = cellIdx + dx;
	if(cIdx >= d_numCells) {
		cIdx -= d_numCells;
	} else if(cIdx < 0) {
		cIdx += d_numCells;
	}
	auto cIdy = cellIdy + dy;
	if(cIdy >= d_numCells) {
		cIdy -= d_numCells;
	} else if(cIdy < 0) {
		cIdy += d_numCells;
	}
	return cIdx * d_numCells + cIdy;
}

__global__ void myKernel() {
	long cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
	long cellIdy = blockIdx.y * blockDim.y + threadIdx.y;
	printf("cellIdx cellIdy: %ld %ld\n", cellIdx, cellIdy);
}

// interaction force between vertices computed sequentially for each particle
__global__ void kernelCalcCellListSmoothInteraction(const double* rad, const double* pos, double* force, double* pEnergy) {
	long cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
	long cellIdy = blockIdx.y * blockDim.y + threadIdx.y;
	printf("cellIdx cellIdy: %ld %ld\n", cellIdx, cellIdy);
  	if (cellIdx < d_numCells && cellIdy < d_numCells) {
		auto cellId = cellIdx * d_numCells + cellIdy;
		//printf("cellId: %ld\n", cellId);
		auto interaction = 0.0;
		double thisPos[MAXDIM], otherPos[MAXDIM], previousPos[MAXDIM], secondPreviousPos[MAXDIM];
		double projPos[MAXDIM], segment[MAXDIM], previousSegment[MAXDIM], interSegment[MAXDIM], relSegment[MAXDIM];// relPos[MAXDIM]
		// loop through vertices in cellId
		for (long vertexId = d_headerPtr[cellId]; vertexId != -1L; vertexId = d_linkedListPtr[vertexId]) {
			getVertexPos(vertexId, pos, thisPos);
			auto thisRad = rad[vertexId];
			auto particleId = d_particleIdListPtr[vertexId];
			for (long dx = -1; dx <= 1; dx++) {
				for (long dy = -1; dy <= 1; dy++) {
					//auto cellIdx = static_cast<long>(thisPos[0] / d_cellSize);
    				//auto cellIdy = static_cast<long>(thisPos[1] / d_cellSize);
					auto otherCellId = getNeighborCellId(cellIdx, cellIdy, dx, dy);
					for (long otherId = d_headerPtr[otherCellId]; otherId != -1L; otherId = d_linkedListPtr[otherId]) {
						auto otherParticleId = d_particleIdListPtr[otherId];
						if ((vertexId != otherId) && (particleId != otherParticleId)) {
							getVertexPos(otherId, pos, otherPos);
							auto otherRad = rad[otherId];
							auto radSum = thisRad + otherRad;
							// compute projection of vertexId on segment between otherId and previousId
							auto previousId = getPreviousId(otherId, otherParticleId);
							getVertexPos(previousId, pos, previousPos);
							getDelta(otherPos, previousPos, segment);
							getDelta(thisPos, previousPos, relSegment);
							for (long dim = 0; dim < d_nDim; dim++) {
								thisPos[dim] = previousPos[dim] + relSegment[dim];
							}
							auto length = calcNorm(segment);
							auto projection = getProjection(thisPos, otherPos, previousPos, length);
							// check if the interaction is vertex-segment
							if(projection > 0 && projection <= 1) {
								getProjectionPos(previousPos, segment, projPos, projection);
								interaction = calcVertexSegmentInteraction(thisPos, projPos, otherPos, previousPos, length, radSum, &force[vertexId*d_nDim], &force[otherId*d_nDim], &force[previousId*d_nDim]);
								printf("CellId: %ld \t vertexId: %ld \t otherCellId: %ld \t otherVertexId: %ld interaction %lf \n", cellId, vertexId, otherCellId, otherId, interaction);
								atomicAdd(&pEnergy[particleId], interaction);
								atomicAdd(&pEnergy[otherParticleId], interaction);
								//block.sync();
							} else if(projection <= 0) {
								auto secondPreviousId = getPreviousId(previousId, otherParticleId);
								getVertexPos(secondPreviousId, pos, secondPreviousPos);
								getDelta(secondPreviousPos, previousPos, previousSegment);
								length = calcNorm(previousSegment);
								auto previousProj = getProjection(thisPos, previousPos, secondPreviousPos, length);
								switch (d_simControl.concavityType) {
									case simControlStruct::concavityEnum::off:
									if(previousProj > 1) {
										interaction = calcVertexVertexInteraction(thisPos, previousPos, radSum, &force[vertexId*d_nDim], &force[previousId*d_nDim]);
										//printf("CellId: %ld \t vertexId: %ld \t otherCellId: %ld \t otherVertexId: %ld interaction %lf \n", cellId, vertexId, otherCellId, otherId, interaction);
										atomicAdd(&pEnergy[particleId], interaction);
										atomicAdd(&pEnergy[otherParticleId], interaction);
									}
									break;
									case simControlStruct::concavityEnum::on:
									// check if the vertex-vertex interaction is concave or convex
									getDelta(thisPos, previousPos, interSegment);
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
									if((projection <= 0 && isCapInteraction) || (projection > 0 && isInverseInteraction)) {
										if(isInverseInteraction) {
											interaction = calcVertexVertexInteraction(previousPos, thisPos, radSum, &force[vertexId*d_nDim], &force[previousId*d_nDim]);
											//printf("CellId: %ld \t vertexId: %ld \t otherCellId: %ld \t otherVertexId: %ld interaction %lf \n", cellId, vertexId, otherCellId, otherId, interaction);
											atomicAdd(&pEnergy[particleId], interaction);
											atomicAdd(&pEnergy[otherParticleId], interaction);
										} else if(previousProj > 1) {
											interaction = calcVertexVertexInteraction(thisPos, previousPos, radSum, &force[vertexId*d_nDim], &force[previousId*d_nDim]);
											//printf("CellId: %ld \t vertexId: %ld \t otherCellId: %ld \t otherVertexId: %ld interaction %lf \n", cellId, vertexId, otherCellId, otherId, interaction);
											atomicAdd(&pEnergy[particleId], interaction);
											atomicAdd(&pEnergy[otherParticleId], interaction);
										}
									}
									break;
								}
							}
						}
					}
				}
			}
		}
	__syncthreads();
  	}
}

inline __device__ double calcSingleVertexInteraction(const double* thisPos, const double* otherPos, const double radSum, double* thisForce) {
  	double delta[MAXDIM];
	auto epot = 0.0;
	auto gradMultiple = calcGradMultipleAndEnergy(thisPos, otherPos, radSum, epot);
	if (gradMultiple != 0) {
		auto distance = calcDeltaAndDistance(thisPos, otherPos, delta);
		for (long dim = 0; dim < d_nDim; dim++) {
			thisForce[dim] += gradMultiple * delta[dim] / distance;
		}
		return epot;
	}
	return 0.;
}

inline __device__ double calcSegmentInteraction(const double* thisPos, const double* projPos, const double* otherPos, const double* previousPos, const double radSum, double* thisForce) {
	double segment[MAXDIM];
	auto epot = 0.0;
	// compute segment and the overlap between its center and this vertex
	auto cross = calcCross(thisPos, otherPos, previousPos);
	auto absCross = fabs(cross);
	auto gradMultiple = calcGradMultipleAndEnergy(projPos, thisPos, radSum, epot);
	getDelta(otherPos, previousPos, segment);
	auto length = calcNorm(segment);
	if (gradMultiple != 0) {
		auto sign = cross / absCross;
		// this vertex
	  	thisForce[0] += gradMultiple * sign * pbcDistance(previousPos[1], otherPos[1], 1) / length;
	  	thisForce[1] += gradMultiple * sign * pbcDistance(otherPos[0], previousPos[0], 0) / length;
		return epot;
	}
	return 0.;
}

inline __device__ double calcSegment1Interaction(const double* thisPos, const double* projPos, const double* otherPos, const double* previousPos, const double radSum, double* otherForce) {
	double segment[MAXDIM];
	auto epot = 0.0;
	// compute segment and the overlap between its center and this vertex
	auto cross = calcCross(thisPos, otherPos, previousPos);
	auto absCross = fabs(cross);
	auto gradMultiple = calcGradMultipleAndEnergy(projPos, thisPos, radSum, epot);
	getDelta(otherPos, previousPos, segment);
	auto length = calcNorm(segment);
	if (gradMultiple != 0) {
		auto sign = cross / absCross;
		// other vertex
	  	otherForce[0] += gradMultiple * (sign * pbcDistance(thisPos[1], previousPos[1], 1) + absCross * pbcDistance(previousPos[0], otherPos[0], 0) / (length * length)) / length;
	  	otherForce[1] += gradMultiple * (sign * pbcDistance(previousPos[0], thisPos[0], 0) + absCross * pbcDistance(previousPos[1], otherPos[1], 1) / (length * length)) / length;
		return epot;
	}
	return 0.;
}


inline __device__ double calcSegment2Interaction(const double* thisPos, const double* projPos, const double* otherPos, const double* previousPos, const double radSum, double* previousForce) {
	double segment[MAXDIM];
	auto epot = 0.0;
	// compute segment and the overlap between its center and this vertex
	auto cross = calcCross(thisPos, otherPos, previousPos);
	auto absCross = fabs(cross);
	auto gradMultiple = calcGradMultipleAndEnergy(projPos, thisPos, radSum, epot);
	getDelta(otherPos, previousPos, segment);
	auto length = calcNorm(segment);
	if (gradMultiple != 0) {
		auto sign = cross / absCross;
		// previous vertex
	  	previousForce[0] += gradMultiple * (sign * pbcDistance(otherPos[1], thisPos[1], 1) - absCross * pbcDistance(previousPos[0], otherPos[0], 0) / (length * length)) / length;
	  	previousForce[1] += gradMultiple * (sign * pbcDistance(thisPos[0], otherPos[0], 0) - absCross * pbcDistance(previousPos[1], otherPos[1], 1) / (length * length)) / length;
	  	return epot;
	}
	return 0.;
}

inline __device__ double getProjection2(const double* thisPos, const double* otherPos, const double* previousPos, const double* segment) {
	auto length = calcNorm(segment);
	return (pbcDistance(previousPos[0], otherPos[0], 0) * pbcDistance(previousPos[0], thisPos[0], 0) + pbcDistance(previousPos[1], otherPos[1], 1) * pbcDistance(previousPos[1], thisPos[1], 1)) / (length * length);
}

// interaction force between vertices fully computed in parallel (work in progress)
__global__ void kernelCalcVertexSmoothInteraction(const double* rad, const double* pos, double* force, double* energy) {
	long vertexId = blockIdx.x * blockDim.x + threadIdx.x; // vertexId is thisId
  	if (vertexId < d_numVertices) {
		auto interaction = 0.0;
		auto otherRad = 0.0;
		double thisPos[MAXDIM], otherPos[MAXDIM], previousPos[MAXDIM], nextPos[MAXDIM], thisPreviousPos[MAXDIM], thisNextPos[MAXDIM];
		double segment[MAXDIM], projPos[MAXDIM], nextProjPos[MAXDIM], thisProjPos[MAXDIM], thisNextProjPos[MAXDIM];
		// we don't zero out the force because we always call this function
		// after kernelCalcShapeForceEnergy where the force is already zeroed out
		getVertexPos(vertexId, pos, thisPos);
		auto thisRad = rad[vertexId];
		auto particleId = d_particleIdListPtr[vertexId];
		// interaction between vertices of neighbor particles
		for (long nListId = 0; nListId < d_maxNeighborListPtr[vertexId]; nListId++) {
      		if(extractNeighbor(vertexId, nListId, pos, rad, otherPos, otherRad)) {
				auto otherId = d_neighborListPtr[vertexId*d_neighborListSize + nListId];
				auto radSum = thisRad + otherRad;
				auto otherParticleId = d_particleIdListPtr[otherId];
				// compute projection of vertexId on segment between previousId and otherId
				auto previousId = getPreviousId(otherId, otherParticleId);
				getVertexPos(previousId, pos, previousPos);
				getDelta(otherPos, previousPos, segment); // this should be previousPos - otherPos
				auto proj = getProjection2(thisPos, otherPos, previousPos, segment);
				getProjectionPos(previousPos, segment, projPos, proj);
				auto projDistance = calcDistance(projPos, thisPos);
				// compute projection of vertexId on segment between otherId and nextId
				auto nextId = getNextId(otherId, otherParticleId);
				getVertexPos(nextId, pos, nextPos);
				getDelta(nextPos, otherPos, segment);
				auto nextProj = getProjection2(thisPos, nextPos, otherPos, segment);
				getProjectionPos(otherPos, segment, nextProjPos, nextProj);
				auto nextProjDistance = calcDistance(projPos, thisPos);
				// compute projection of otherId on segment between thisPreviousId and vertexId
				auto thisPreviousId = getPreviousId(vertexId, particleId);
				getVertexPos(thisPreviousId, pos, thisPreviousPos);
				getDelta(thisPos, thisPreviousPos, segment);
				auto thisProj = getProjection2(otherPos, thisPos, thisPreviousPos, segment);
				getProjectionPos(thisPreviousPos, segment, thisProjPos, thisProj);
				auto thisProjDistance = calcDistance(otherPos, thisProjPos);
				// compute projection of otherId on segment between vertexId and thisNextId
				auto thisNextId = getNextId(vertexId, particleId);
				getVertexPos(thisNextId, pos, thisNextPos);
				getDelta(thisNextPos, thisPos, segment);
				auto thisNextProj = getProjection2(otherPos, thisNextPos, thisPos, segment);
				getProjectionPos(thisPos, segment, thisNextProjPos, thisNextProj);
				auto thisNextProjDistance = calcDistance(otherPos, thisNextProjPos);
				// find the shortest distance between thisId and otherId
				if(projDistance < nextProjDistance && projDistance < thisProjDistance && projDistance < thisNextProjDistance) {
					if(proj > 0 && proj <= 1) {
						interaction = calcSegment1Interaction(thisPos, projPos, otherPos, previousPos, radSum, &force[vertexId*d_nDim]);
					} else if(proj > 1) {
						interaction = calcSingleVertexInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim]);
					} else {
						interaction = calcSingleVertexInteraction(thisPos, previousPos, radSum, &force[vertexId*d_nDim]);
					}
				}
				if(nextProjDistance < projDistance && nextProjDistance < thisProjDistance && nextProjDistance < thisNextProjDistance) {
					if(nextProj > 0 && nextProj <= 1) {
						interaction  = calcSegmentInteraction(thisPos, nextProjPos, nextPos, otherPos, radSum, &force[vertexId*d_nDim]);
					} else if(nextProj > 1) {
						interaction = calcSingleVertexInteraction(thisPos, nextPos, radSum, &force[vertexId*d_nDim]);
					} else {
						interaction = calcSingleVertexInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim]);
					}
				}
				if(thisProjDistance < projDistance && thisProjDistance < nextProjDistance && thisProjDistance < thisNextProjDistance) {
					if(thisProj > 0 && thisProj <= 1) {
						interaction = calcSegment1Interaction(otherPos, thisProjPos, thisPos, thisPreviousPos, radSum, &force[vertexId*d_nDim]);
					} else if(thisProj > 1) {
						interaction = calcSingleVertexInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim]);
					}
				}
				if(thisNextProjDistance < projDistance && thisNextProjDistance < nextProjDistance && thisNextProjDistance < thisProjDistance) {
					if(thisNextProj > 0 && thisNextProj <= 1) {
						interaction = calcSegment2Interaction(otherPos, thisNextProjPos, thisNextPos, thisPos, radSum, &force[vertexId*d_nDim]);
					} else if(thisNextProj < 0) {
						interaction = calcSingleVertexInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim]);
					}
				}
				energy[vertexId] += interaction;
			}
		}
  	}
}

// this function checks what type of interactions are on a vertex
// if a vertex has line-vertex interaction and vertex-vertex interaction
// the force function will only consider vertex-line to avoid tangential forces (frictionless dpm)
__global__ void kernelCheckVertexLineInteraction(const double* pos, long* vertexLineFlag) {
	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (vertexId < d_numVertices) {
		auto neighborId = -1;
		double thisPos[MAXDIM], otherPos[MAXDIM], previousPos[MAXDIM], segment[MAXDIM];
		// we don't zero out the force because we always call this function
		// after kernelCalcShapeForceEnergy where the force is already zeroed out
		getVertexPos(vertexId, pos, thisPos);
		// interaction between vertices of neighbor particles
		for (long nListId = 0; nListId < d_maxNeighborListPtr[vertexId]; nListId++) {
      		if(extractNeighborPos(vertexId, nListId, pos, otherPos)) {
				auto otherId = d_neighborListPtr[vertexId*d_neighborListSize + nListId];
				auto otherParticleId = d_particleIdListPtr[otherId];
				// get previous vertex to vertex on neighbor particle
				auto previousId = getPreviousId(otherId, otherParticleId);
				getVertexPos(previousId, pos, previousPos);
				// compute projection of vertexId on segment between otherId and previousId
				getDelta(otherPos, previousPos, segment);
				auto length = calcNorm(segment);
				auto projection = getProjection(thisPos, otherPos, previousPos, length);
				// check if the interaction is vertex-segment
				if(projection > 0 && projection <= 1) {
					vertexLineFlag[vertexId*d_neighborListSize + nListId] = 1;
				}
				for (long mListId = 0; mListId < d_maxNeighborListPtr[otherId]; mListId++) {
					neighborId = d_neighborListPtr[otherId*d_neighborListSize + mListId];
					if(neighborId == vertexId) {
						vertexLineFlag[otherId*d_neighborListSize + mListId] = 1;
					}
				}
				for (long mListId = 0; mListId < d_maxNeighborListPtr[previousId]; mListId++) {
					neighborId = d_neighborListPtr[previousId*d_neighborListSize + mListId];
					if(neighborId == vertexId) {
						vertexLineFlag[previousId*d_neighborListSize + mListId] = 1;
					}
				}
			}
		}
  	}
}

// particle-particle contact interaction
__global__ void kernelCalcParticleInteraction(const double* pRad, const double* pPos, double* pForce, double* pEnergy) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
    	auto otherRad = 0.0;
		double thisPos[MAXDIM], otherPos[MAXDIM];
		// zero out the force and get particle positions
		for (long dim = 0; dim < d_nDim; dim++) {
			pForce[particleId * d_nDim + dim] = 0.0;
			thisPos[dim] = pPos[particleId * d_nDim + dim];
		}
    	auto thisRad = pRad[particleId];
    	pEnergy[particleId] = 0.0;
    	// interaction between vertices of neighbor particles
    	for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
      		if (extractParticleNeighbor(particleId, nListId, pPos, pRad, otherPos, otherRad)) {
				auto radSum = thisRad + otherRad;
				switch (d_simControl.potentialType) {
					case simControlStruct::potentialEnum::harmonic:
					pEnergy[particleId] += calcContactInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::lennardJones:
					pEnergy[particleId] += calcLJInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::adhesive:
					pEnergy[particleId] += calcAdhesiveInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::wca:
					pEnergy[particleId] += calcWCAInteraction(thisPos, otherPos, radSum, &pForce[particleId*d_nDim]);
					break;
				}
			//if(particleId == 116 && d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId] == 109) printf("particleId %ld \t neighbor: %ld \t overlap %e \n", particleId, d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId], calcOverlap(thisPos, otherPos, radSum));
			}
    	}
  	}
}

__global__ void kernelCalcVertexForceTorque(const double* rad, const double* pos, const double* pPos, double* force, double* torque, double* energy) {
  	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (vertexId < d_numVertices) {
		auto particleId = d_particleIdListPtr[vertexId];
    	auto otherRad = 0.0;
		double thisPos[MAXDIM], otherPos[MAXDIM], partPos[MAXDIM];
		for (long dim = 0; dim < d_nDim; dim++) {
			force[vertexId * d_nDim + dim] = 0;
		}
		torque[vertexId] = 0.0;
		energy[vertexId] = 0.0;
		getVertexPos(vertexId, pos, thisPos);
    	auto thisRad = rad[vertexId];
    	// interaction between vertices of neighbor particles
    	for (long nListId = 0; nListId < d_maxNeighborListPtr[vertexId]; nListId++) {
      		if(extractNeighbor(vertexId, nListId, pos, rad, otherPos, otherRad)) {
				//if(vertexId == 0) printf("vertexId %ld \t neighbor: %ld \t force %lf \t %lf \n", vertexId, d_neighborListPtr[vertexId*d_neighborListSize + nListId], force[vertexId * d_nDim], force[vertexId * d_nDim + 1]);
				auto radSum = thisRad + otherRad;
				switch (d_simControl.potentialType) {
					case simControlStruct::potentialEnum::harmonic:
					energy[vertexId] += calcContactInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::lennardJones:
					energy[vertexId] += calcLJInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::adhesive:
					energy[vertexId] += calcAdhesiveInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim]);
					break;
					case simControlStruct::potentialEnum::wca:
					energy[vertexId] += calcWCAInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim]);
					break;
				}
			}
		}
		getParticlePos(particleId, pPos, partPos);
		getRelativeVertexPos(vertexId, pos, thisPos, partPos);
		torque[vertexId] = (thisPos[0] * force[vertexId * d_nDim + 1] - thisPos[1] * force[vertexId * d_nDim]);
  	}
}

__global__ void kernelCalcVertexSmoothForceTorque(const double* rad, const double* pos, const double* pPos, double* force, double* torque, double* pEnergy) {
  	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (vertexId < d_numVertices) {
		//thread_block block = this_thread_block();
		auto particleId = d_particleIdListPtr[vertexId];
    	auto otherRad = 0.0;
		auto interaction = 0.0;
		double thisPos[MAXDIM], otherPos[MAXDIM], partPos[MAXDIM], previousPos[MAXDIM], segment[MAXDIM], relSegment[MAXDIM];
		double secondPreviousPos[MAXDIM], previousSegment[MAXDIM], projPos[MAXDIM];
		for (long dim = 0; dim < d_nDim; dim++) {
			force[vertexId * d_nDim + dim] = 0;
		}
		torque[vertexId] = 0.0;
		getVertexPos(vertexId, pos, thisPos);
    	auto thisRad = rad[vertexId];
    	// interaction between vertices of neighbor particles
    	for (long nListId = 0; nListId < d_maxNeighborListPtr[vertexId]; nListId++) {
      		if(extractNeighbor(vertexId, nListId, pos, rad, otherPos, otherRad)) {
				auto otherId = d_neighborListPtr[vertexId*d_neighborListSize + nListId];
				auto radSum = thisRad + otherRad;
				auto otherParticleId = d_particleIdListPtr[otherId];
				// compute projection of vertexId on segment between otherId and previousId
				auto previousId = getPreviousId(otherId, otherParticleId);
				getVertexPos(previousId, pos, previousPos);
				getDelta(otherPos, previousPos, segment);
				getDelta(thisPos, previousPos, relSegment);
				for (long dim = 0; dim < d_nDim; dim++) {
					thisPos[dim] = previousPos[dim] + relSegment[dim];
				}
				auto length = calcNorm(segment);
				auto projection = getProjection(thisPos, otherPos, previousPos, length);
				// check if the interaction is vertex-segment
				if(projection > 0 && projection <= 1) {
					getProjectionPos(previousPos, segment, projPos, projection);
					interaction = calcVertexSegmentInteraction(thisPos, projPos, otherPos, previousPos, length, radSum, &force[vertexId*d_nDim], &force[otherId*d_nDim], &force[previousId*d_nDim]);
					atomicAdd(&pEnergy[particleId], interaction);
					atomicAdd(&pEnergy[otherParticleId], interaction);
					//if(interaction!=0) printf("particleId: %ld \t vertexId: %ld \t otherId: %ld \t previousId: %ld \t interaction %lf \n", particleId, vertexId, otherId, previousId, pEnergy[particleId]);
					//block.sync();
				} else if(projection <= 0) {
					auto secondPreviousId = getPreviousId(previousId, otherParticleId);
					getVertexPos(secondPreviousId, pos, secondPreviousPos);
					getDelta(secondPreviousPos, previousPos, previousSegment);
					length = calcNorm(previousSegment);
					auto previousProj = getProjection(thisPos, previousPos, secondPreviousPos, length);
					if(previousProj > 1) {
						interaction = calcVertexVertexInteraction(thisPos, previousPos, radSum, &force[vertexId*d_nDim], &force[previousId*d_nDim]);
						atomicAdd(&pEnergy[particleId], interaction);
						atomicAdd(&pEnergy[otherParticleId], interaction);
						//if(interaction != 0) printf("particleId: %ld \t vertexId: %ld \t otherId: %ld \t interaction %lf \n", particleId, vertexId, otherId, pEnergy[particleId]);
						//block.sync();
					}
				}
			}
		}
		getParticlePos(particleId, pPos, partPos);
		getRelativeVertexPos(vertexId, pos, thisPos, partPos);
		torque[vertexId] = (thisPos[0] * force[vertexId * d_nDim + 1] - thisPos[1] * force[vertexId * d_nDim]);
	//__syncthreads();
	}
}

__global__ void kernelCalcParticleRigidForceEnergy(const double* force, const double* torque, const double* energy, double* pForce, double* pTorque, double* pEnergy) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if(particleId < d_numParticles) {
		for (long dim = 0; dim < d_nDim; dim++) {
			pForce[particleId * d_nDim + dim] = 0.0;
		}
		pTorque[particleId] = 0.0;
		pEnergy[particleId] = 0.0;
		auto firstVertex = d_firstVertexInParticleIdPtr[particleId];
		auto lastVertex = firstVertex + d_numVertexInParticleListPtr[particleId];
		for (long vertexId = firstVertex; vertexId < lastVertex; vertexId++) {
			for (long dim = 0; dim < d_nDim; dim++) {
				pForce[particleId * d_nDim + dim] += force[vertexId * d_nDim + dim];
			}
			pTorque[particleId] += torque[vertexId];
			pEnergy[particleId] += energy[vertexId];
		}
	}
}

__global__ void kernelCalcParticleSmoothRigidForceEnergy(const double* force, const double* torque, double* pForce, double* pTorque) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if(particleId < d_numParticles) {
		for (long dim = 0; dim < d_nDim; dim++) {
			pForce[particleId * d_nDim + dim] = 0.0;
		}
		pTorque[particleId] = 0.0;
		auto firstVertex = d_firstVertexInParticleIdPtr[particleId];
		auto lastVertex = firstVertex + d_numVertexInParticleListPtr[particleId];
		for (long vertexId = firstVertex; vertexId < lastVertex; vertexId++) {
			for (long dim = 0; dim < d_nDim; dim++) {
				pForce[particleId * d_nDim + dim] += force[vertexId * d_nDim + dim];
			}
			pTorque[particleId] += torque[vertexId];
		}
	}
}

__global__ void kernelCalcStressTensor(const double* perPStress, double* stress) {
  long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleId < d_numParticles) {
		//diagonal terms
		stress[0] += perPStress[0];
		stress[3] += perPStress[3];
		// cross terms
		stress[1] += perPStress[1];
		stress[2] += perPStress[2];
	}
}

// this function is written for contact potential only
__global__ void kernelCalcPerParticleStressTensor(const double* rad, const double* pos, const double* pPos, double* perPStress) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		auto firstId = d_firstVertexInParticleIdPtr[particleId];
    	double thisRad, otherRad, radSum;
		auto distance = 0.0;
		auto overlap = 0.0;
		auto gradMultiple = 0.0;
		double relativePos[MAXDIM], thisPos[MAXDIM], otherPos[MAXDIM], delta[MAXDIM], forces[MAXDIM];
		// zero out perParticleStress
		for (long dim2 = 0; dim2 < (d_nDim * d_nDim); dim2++) {
			perPStress[particleId * (d_nDim * d_nDim) + dim2] = 0;
		}
		// iterate over vertices in particle
		for (long vertexId = firstId; vertexId < firstId + d_numVertexInParticleListPtr[particleId]; vertexId++) {
			getVertexPos(vertexId, pos, thisPos);
    		thisRad = rad[vertexId];
			getRelativeVertexPos(vertexId, pos, relativePos, &pPos[particleId*d_nDim]);
    		// stress between vertices of neighbor particles
    		for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
      			if(extractNeighbor(vertexId, nListId, pos, rad, otherPos, otherRad)) {
					radSum = thisRad + otherRad;
					overlap = calcOverlap(thisPos, otherPos, radSum);
					if (overlap > 0) {
						gradMultiple = d_ec * overlap / radSum;
						distance = calcDistance(thisPos, otherPos);
						for (long dim = 0; dim < d_nDim; dim++) {
							delta[dim] = pbcDistance(otherPos[dim], thisPos[dim], dim);
							forces[dim] = gradMultiple * delta[dim] / (distance * radSum);
							relativePos[dim] += delta[dim] * 0.5; // distance from center of mass to contact location
						}
						//diagonal terms
						perPStress[particleId * (d_nDim * d_nDim)] += relativePos[0] * forces[0];
						perPStress[particleId * (d_nDim * d_nDim) + 3] += relativePos[1] * forces[1];
						// cross terms
						perPStress[particleId * (d_nDim * d_nDim) + 1] += relativePos[0] * forces[1];
						perPStress[particleId * (d_nDim * d_nDim) + 2] += relativePos[1] * forces[0];
					}
				}
			}
		}
	}
}

//works only for 2D
__global__ void kernelCalcParticleShape(const double* pos, double* length, double* theta, double* area, double* perimeter) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		auto segmentLength = 0.0;
		auto tempArea = 0.0;
		perimeter[particleId] = 0.0;
		auto nextId = -1;
		auto previousId = -1;
		auto firstId = d_firstVertexInParticleIdPtr[particleId];
		double delta[MAXDIM], nextPos[MAXDIM], currentPos[MAXDIM], nextSegment[MAXDIM], previousPos[MAXDIM], previousSegment[MAXDIM];
		getVertexPos(firstId, pos, currentPos);
		// compute particle area via shoe-string method
		for (long currentId = firstId; currentId < firstId + d_numVertexInParticleListPtr[particleId]; currentId++) {
			nextId = getNextId(currentId, particleId);
			previousId = getPreviousId(currentId, particleId); // added
			segmentLength = 0.0;
			for (long dim = 0; dim < d_nDim; dim++) {
				delta[dim] = pbcDistance(pos[nextId * d_nDim + dim], currentPos[dim], dim);
				nextPos[dim] = currentPos[dim] + delta[dim];
				segmentLength += delta[dim] * delta[dim];
				delta[dim] = pbcDistance(pos[previousId * d_nDim + dim], currentPos[dim], dim); // added
				previousPos[dim] = currentPos[dim] + delta[dim]; // added + because previous is already before current (on a line delta is already negative)
			}
			getSegment(nextPos, currentPos, nextSegment);
            getSegment(currentPos, previousPos, previousSegment);
            theta[currentId] = calcAngle(nextSegment, previousSegment);
			length[currentId] = sqrt(segmentLength);
			perimeter[particleId] += length[currentId];
			tempArea += currentPos[0] * nextPos[1] - nextPos[0] * currentPos[1];
			for (long dim = 0; dim < d_nDim; dim++) {
				currentPos[dim] = nextPos[dim];
			}
		}
		area[particleId] = abs(tempArea) * 0.5;
	}
}

__global__ void kernelCalcParticlePositions(const double* pos, double* particlePos) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		calcParticlePos(particleId, pos, &particlePos[particleId*d_nDim]);
	}
}

__global__ void kernelCalcVertexArea(const double* rad, double* vertexArea) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		auto vertexRad = rad[d_firstVertexInParticleIdPtr[particleId]];
		vertexArea[particleId] = vertexRad * vertexRad * (0.5 * d_numVertexInParticleListPtr[particleId] - 1);
	}
}

__global__ void kernelScaleVertexPositions(const double* particlePos, double* pos, double scale) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double distance[MAXDIM];
		auto firstVertex = d_firstVertexInParticleIdPtr[particleId];
		auto lastVertex = firstVertex + d_numVertexInParticleListPtr[particleId];
		for (long vertexId = firstVertex; vertexId < lastVertex; vertexId++) {
			for (long dim = 0; dim < d_nDim; dim++) {
				//distance[dim] = pbcDistance(pos[vertexId * d_nDim + dim], particlePos[particleId * d_nDim + dim], d_boxSizePtr[dim]);
				distance[dim] = pos[vertexId * d_nDim + dim] - particlePos[particleId * d_nDim + dim];
      			pos[vertexId * d_nDim + dim] += (scale - 1) * distance[dim];
			}
		}
	}
}

__global__ void kernelCheckPBC(double* pos) {
	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
	if (vertexId < d_numVertices) {
		for (long dim = 0; dim < d_nDim; dim++) {
			pos[vertexId * d_nDim + dim] -= round(pos[vertexId * d_nDim + dim] / d_boxSizePtr[dim]) * d_boxSizePtr[dim];
		}
	}
}

__global__ void kernelCheckParticlePBC(double* pPos) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		for (long dim = 0; dim < d_nDim; dim++) {
			pPos[particleId * d_nDim + dim] -= round(pPos[particleId * d_nDim + dim] / d_boxSizePtr[dim]) * d_boxSizePtr[dim];
		}
	}
}

__global__ void kernelTranslateVertices(const double* pPos, const double* pLastPos, double* pos) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double delta[MAXDIM];
		getDelta(&pPos[particleId * d_nDim], &pLastPos[particleId * d_nDim], delta);
		auto firstVertex = d_firstVertexInParticleIdPtr[particleId];
		auto lastVertex = firstVertex + d_numVertexInParticleListPtr[particleId];
		for (long vertexId = firstVertex; vertexId < lastVertex; vertexId++) {
			for (long dim = 0; dim < d_nDim; dim++) {
				pos[vertexId * d_nDim + dim] += delta[dim];
				// pbc check
				//pos[vertexId * d_nDim + dim] -= round(pos[vertexId * d_nDim + dim] / d_boxSizePtr[dim]) * d_boxSizePtr[dim];
			}
		}
	}
}

__global__ void kernelRotateVertices(const double* pPos, const double* pAngle, const double* pLastAngle, double* pos) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double delta[MAXDIM], newPos[MAXDIM], deltaAngle;
		deltaAngle = pAngle[particleId] - pLastAngle[particleId];
		auto firstVertex = d_firstVertexInParticleIdPtr[particleId];
		auto lastVertex = firstVertex + d_numVertexInParticleListPtr[particleId];
		for (long vertexId = firstVertex; vertexId < lastVertex; vertexId++) {
			getDelta(&pos[vertexId * d_nDim], &pPos[particleId * d_nDim], delta);
			newPos[0] = delta[0] * cos(deltaAngle) - delta[1] * sin(deltaAngle);
			newPos[1] = delta[0] * sin(deltaAngle) + delta[1] * cos(deltaAngle);
			for (long dim = 0; dim < d_nDim; dim++) {
				pos[vertexId * d_nDim + dim] += (newPos[dim] - delta[dim]);
				// pbc check
				//pos[vertexId * d_nDim + dim] -= round(pos[vertexId * d_nDim + dim] / d_boxSizePtr[dim]) * d_boxSizePtr[dim];
			}
		}
	}
}

__global__ void kernelTranslateAndRotateVertices(const double* pPos, const double* pLastPos, const double* pAngle, const double* pLastAngle, double* pos) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double newPos[MAXDIM], relPos[MAXDIM], deltaAngle;
		deltaAngle = pAngle[particleId] - pLastAngle[particleId];
		auto firstVertex = d_firstVertexInParticleIdPtr[particleId];
		auto lastVertex = firstVertex + d_numVertexInParticleListPtr[particleId];
		for (long vertexId = firstVertex; vertexId < lastVertex; vertexId++) {
			getDelta(&pos[vertexId * d_nDim], &pLastPos[particleId * d_nDim], relPos);
			newPos[0] = relPos[0] * cos(deltaAngle) - relPos[1] * sin(deltaAngle);
			newPos[1] = relPos[0] * sin(deltaAngle) + relPos[1] * cos(deltaAngle);
			for (long dim = 0; dim < d_nDim; dim++) {
				pos[vertexId * d_nDim + dim] = pPos[particleId * d_nDim + dim] + newPos[dim];
				//pos[vertexId * d_nDim + dim] -= round(pos[vertexId * d_nDim + dim] / d_boxSizePtr[dim]) * d_boxSizePtr[dim];
			}
		}
	}
}

//************************** neighbors and contacts **************************//
__global__ void kernelCalcNeighborList(const double* pos, const double* rad, const double cutDistance) {
  	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
	if (vertexId < d_numVertices) {
		auto otherParticleId = -1;
		auto addedNeighbor = 0;
		double otherRad, radSum;
		double thisPos[MAXDIM], otherPos[MAXDIM];
		getVertexPos(vertexId, pos, thisPos);
		auto thisRad = rad[vertexId];
		auto particleId = d_particleIdListPtr[vertexId];

		for (long otherId = 0; otherId < d_numVertices; otherId++) {
			otherParticleId = d_particleIdListPtr[otherId];
			if(otherParticleId != particleId) {
				if(extractOtherVertex(vertexId, otherId, pos, rad, otherPos, otherRad)) {
					bool isNeighbor = false;
					radSum = thisRad + otherRad;
					isNeighbor = (-calcOverlap(thisPos, otherPos, radSum) < cutDistance);
					//isNeighbor = (calcDistance(thisPos, otherPos) < (cutDistance * radSum));
					if (addedNeighbor < d_neighborListSize) {
						d_neighborListPtr[vertexId * d_neighborListSize + addedNeighbor] = otherId*isNeighbor -1*(!isNeighbor);
					}
					addedNeighbor += isNeighbor;
				}
			}
		}
		d_maxNeighborListPtr[vertexId] = addedNeighbor;
	}
}

__global__ void kernelCalcParticleNeighbors(const double* pos, const double* rad, const long neighborLimit, long* neighborList, long* numNeighbors) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		auto addedNeighbor = 0;
		auto newNeighborId = -1;
		double thisRad, otherRad;
		double thisPos[MAXDIM], otherPos[MAXDIM];
		auto firstVertex = d_firstVertexInParticleIdPtr[particleId];
		auto lastVertex = firstVertex + d_numVertexInParticleListPtr[particleId];
		for(long vertexId = firstVertex; vertexId < lastVertex; vertexId++) {
			getVertex(vertexId, pos, rad, thisPos, thisRad);
			// compute vertex contacts and fill out particle contact list
			for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
				if (extractNeighbor(vertexId, nListId, pos, rad, otherPos, otherRad)) {
					newNeighborId = d_particleIdListPtr[d_neighborListPtr[vertexId * d_neighborListSize + nListId]];
					bool isNewNeighbor = true;
					for (long neighId = 0; neighId < neighborLimit; neighId++) {
						if(newNeighborId == neighborList[particleId * neighborLimit + neighId]) {
							isNewNeighbor = false;
						}
					}
					if(isNewNeighbor) {
						neighborList[particleId * neighborLimit + addedNeighbor] = newNeighborId;
						addedNeighbor++;
					}
				}
			}
		}
		numNeighbors[particleId] = addedNeighbor;
	}
}

__global__ void kernelCalcParticleNeighborList(const double* pPos, const double* pRad, const double cutDistance) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		auto addedNeighbor = 0;
		double otherRad, radSum;
		double thisPos[MAXDIM], otherPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		auto thisRad = pRad[particleId];

		for (long otherId = 0; otherId < d_numParticles; otherId++) {
			if(extractOtherParticle(particleId, otherId, pPos, pRad, otherPos, otherRad)) {
				bool isNeighbor = false;
				radSum = thisRad + otherRad;
				//isNeighbor = (-calcOverlap(thisPos, otherPos, radSum) < cutDistance);
				isNeighbor = (calcDistance(thisPos, otherPos) < (cutDistance * radSum));
				if (addedNeighbor < d_partNeighborListSize) {
					d_partNeighborListPtr[particleId * d_partNeighborListSize + addedNeighbor] = otherId*isNeighbor -1*(!isNeighbor);
					//if(isNeighbor == true && particleId == 116) printf("particleId %ld \t otherId: %ld \t isNeighbor: %i \n", particleId, otherId, isNeighbor);
				}
				addedNeighbor += isNeighbor;
			}
		}
		d_partMaxNeighborListPtr[particleId] = addedNeighbor;
  	}
}

__global__ void kernelCalcContacts(const double* pos, const double* rad, const double gapSize, const long contactLimit, long* contactList, long* numContacts) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		auto addedContact = 0;
		auto newContactId = -1;
		double thisRad, otherRad, radSum;
		double thisPos[MAXDIM], otherPos[MAXDIM];
		for(long vertexId = d_firstVertexInParticleIdPtr[particleId]; vertexId < d_firstVertexInParticleIdPtr[particleId] + d_numVertexInParticleListPtr[particleId]; vertexId++) {
			getVertex(vertexId, pos, rad, thisPos, thisRad);
			// compute vertex contacts and fill out particle contact list
			for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
				if (extractNeighbor(vertexId, nListId, pos, rad, otherPos, otherRad)) {
					radSum = thisRad + otherRad;
					if (calcOverlap(thisPos, otherPos, radSum) > (-gapSize)) {
						if (addedContact < contactLimit) {
							newContactId = d_particleIdListPtr[d_neighborListPtr[vertexId * d_neighborListSize + nListId]];
							bool isNewContact = true;
							for (long contactId = 0; contactId < contactLimit; contactId++) {
								if(newContactId == contactList[particleId * contactLimit + contactId]) {
									isNewContact = false;
								}
							}
							if(isNewContact) {
								contactList[particleId * contactLimit + addedContact] = newContactId;
								addedContact++;
							}
						}
					}
				}
			}
		}
		numContacts[particleId] = addedContact;
	}
}

__global__ void kernelCalcContactVectorList(const double* pPos, const long* contactList, const long contactListSize, const long maxContacts, double* contactVectorList) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double thisPos[MAXDIM], otherPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
		for (long cListId = 0; cListId < maxContacts; cListId++) {
			long otherId = contactList[particleId * contactListSize + cListId];
			if ((particleId != otherId) && (otherId != -1)) {
				extractOtherParticlePos(particleId, otherId, pPos, otherPos);
				//Calculate the contactVector and put it into contactVectorList, which is a maxContacts*nDim by numParticle array
				getDelta(thisPos, otherPos, &contactVectorList[particleId*(maxContacts*d_nDim) + cListId*d_nDim]);
			}
		}
	}
}

__global__ void kernelCalcNeighborForces(const double* pos, const double *rad, double *neighforce) {
	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (vertexId < d_numVertices) {
    	double otherRad, radSum, overlap, gradMultiple, distance;
		double thisPos[MAXDIM], otherPos[MAXDIM];
		// we don't zero out the force because we always call this function
		// after kernelCalcShapeForceEnergy where the force is zeroed out
		getVertexPos(vertexId, pos, thisPos);
    	auto thisRad = rad[vertexId];
    	// interaction between vertices of neighbor particles
    	for (long nListId = 0; nListId < d_maxNeighborListPtr[vertexId]; nListId++) {
      		if (extractNeighbor(vertexId, nListId, pos, rad, otherPos, otherRad)) {
        		radSum = thisRad + otherRad;
				overlap = calcOverlap(thisPos, otherPos, radSum), gradMultiple, distance;
				gradMultiple = (overlap > 0) * d_ec * overlap / radSum;
				distance = calcDistance(thisPos, otherPos);
			  	for (long dim = 0; dim < d_nDim; dim++) {
			    	neighforce[vertexId*d_neighborListSize + nListId*d_nDim+dim] += gradMultiple * pbcDistance(thisPos[dim], otherPos[dim], dim) / distance;
				}
			}
    	}
  	}
}

//******************************** observables *******************************//
__global__ void kernelCalcVertexDistanceSq(const double* pos, const double* initialPos, double* delta) {
	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
	if (vertexId < d_numVertices) {
		double distance[MAXDIM];
		calcDeltaAndDistance(&pos[vertexId*d_nDim], &initialPos[vertexId*d_nDim], distance);
		for (long dim = 0; dim < d_nDim; dim++) {
			delta[vertexId * d_nDim + dim] = distance[dim]*distance[dim];
		}
	}
}

__global__ void kernelCalcVertexDisplacement(const double* pos, const double* lastPos, double* disp) {
	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
	if (vertexId < d_numVertices) {
		disp[vertexId] = calcDistance(&pos[vertexId*d_nDim], &lastPos[vertexId*d_nDim]);
	}
}

__global__ void kernelCheckVertexDisplacement(const double* pos, const double* lastPos, int* flag, double cutoff) {
	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
	if (vertexId < d_numVertices) {
		auto displacement = calcDistance(&pos[vertexId*d_nDim], &lastPos[vertexId*d_nDim]);
		if(3 * displacement > cutoff) {
			flag[vertexId] = 1;
		}
	}
}

__global__ void kernelCalcParticleDistanceSq(const double* pPos, const double* pInitialPos, double* pDelta) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double delta[MAXDIM];
		calcDeltaAndDistance(&pPos[particleId*d_nDim], &pInitialPos[particleId*d_nDim], delta);
		for (long dim = 0; dim < d_nDim; dim++) {
			pDelta[particleId * d_nDim + dim] = delta[dim]*delta[dim];
		}
	}
}

__global__ void kernelCalcParticleDisplacement(const double* pPos, const double* pPreviousPos, double* pDisp) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		pDisp[particleId] = calcDistance(&pPos[particleId*d_nDim], &pPreviousPos[particleId*d_nDim]);
	}
}

// use isotropic formula to compute scattering function as reported in https://www.nature.com/articles/srep36702
__global__ void kernelCalcVertexScatteringFunction(const double* pos, const double* initialPos, double* vSF, const double waveNum) {
	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
	if (vertexId < d_numVertices) {
		auto distance = calcDistance(&pos[vertexId*d_nDim], &initialPos[vertexId*d_nDim]);
		vSF[vertexId] = sin(waveNum * distance) / (waveNum * distance);
	}
}

__global__ void kernelCalcParticleScatteringFunction(const double* pPos, const double* pInitialPos, double* pSF, const double waveNum) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		auto distance = calcDistance(&pPos[particleId*d_nDim], &pInitialPos[particleId*d_nDim]);
		pSF[particleId] = sin(waveNum * distance) / (waveNum * distance);
	}
}

__global__ void kernelCalcHexaticOrderParameter(const double* pPos, double* psi6) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if(particleId < d_numParticles) {
		double thisPos[MAXDIM], otherPos[MAXDIM], delta[MAXDIM];
		auto angle = 0.0;
		// get particle position
		for (long dim = 0; dim < d_nDim; dim++) {
			thisPos[dim] = pPos[particleId * d_nDim + dim];
		}
    	// extract neighbor particles
    	for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
      		if (extractParticleNeighborPos(particleId, nListId, pPos, otherPos)) {
				getDelta(thisPos, otherPos, delta);
				angle = atan2(delta[1], delta[0]);
				psi6[particleId] += sin(6 * angle) / (6 * angle);
			}
		}
		psi6[particleId] /= d_partMaxNeighborListPtr[particleId];
	}
}

//******************************** integrators *******************************//
__global__ void kernelExtractThermalVertexVel(double* vel, const double* r1, const double* r2, const double amplitude) {
	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (vertexId < d_numVertices) {
		double rNum[MAXDIM];
		rNum[0] = sqrt(-2.0 * log(r1[vertexId])) * cos(2.0 * PI * r2[vertexId]);
		rNum[1] = sqrt(-2.0 * log(r1[vertexId])) * sin(2.0 * PI * r2[vertexId]);
		for (long dim = 0; dim < d_nDim; dim++) {
			vel[vertexId * d_nDim + dim] = amplitude * rNum[dim];
		}
  	}
}

__global__ void kernelExtractThermalParticleVel(double* pVel, const double* r1, const double* r2, const double amplitude) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		double rNum[MAXDIM];
		rNum[0] = sqrt(-2.0 * log(r1[particleId])) * cos(2.0 * PI * r2[particleId]);
		rNum[1] = sqrt(-2.0 * log(r1[particleId])) * sin(2.0 * PI * r2[particleId]);
		for (long dim = 0; dim < d_nDim; dim++) {
			pVel[particleId * d_nDim + dim] = amplitude * rNum[dim];
		}
  	}
}

__global__ void kernelUpdateVertexPos(double* pos, const double* vel, const double timeStep) {
  	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (vertexId < d_numVertices) {
    	for (long dim = 0; dim < d_nDim; dim++) {
			pos[vertexId * d_nDim + dim] += timeStep * vel[vertexId * d_nDim + dim];
		}
  	}
}

__global__ void kernelUpdateParticlePos(double* pPos, const double* pVel, const double timeStep) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
    	for (long dim = 0; dim < d_nDim; dim++) {
			pPos[particleId * d_nDim + dim] += timeStep * pVel[particleId * d_nDim + dim];
		}
  	}
}

__global__ void kernelUpdateVertexVel(double* vel, const double* force, const double timeStep) {
  	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (vertexId < d_numVertices) {
    	for (long dim = 0; dim < d_nDim; dim++) {
			vel[vertexId * d_nDim + dim] += timeStep * force[vertexId * d_nDim + dim];
		}
  	}
}

__global__ void kernelUpdateRigidPos(double* pPos, const double* pVel, double* pAngle, const double* pAngvel, const double timeStep) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
    	for (long dim = 0; dim < d_nDim; dim++) {
			pPos[particleId * d_nDim + dim] += timeStep * pVel[particleId * d_nDim + dim];
		}
		pAngle[particleId] += timeStep * pAngvel[particleId];
  	}
}

__global__ void kernelUpdateBrownianVertexVel(double* vel, const double* force, double* thermalVel, const double mobility) {
	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (vertexId < d_numVertices) {
		auto particleId = d_particleIdListPtr[vertexId];
		for (long dim = 0; dim < d_nDim; dim++) {
			vel[vertexId * d_nDim + dim] = mobility * force[vertexId * d_nDim + dim] + thermalVel[particleId * d_nDim + dim];
		}
  	}
}

__global__ void kernelUpdateActiveVertexVel(double* vel, const double* force, double* pAngle, const double driving, const double mobility) {
	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (vertexId < d_numVertices) {
		auto particleId = d_particleIdListPtr[vertexId];
		auto angle = pAngle[particleId];
		for (long dim = 0; dim < d_nDim; dim++) {
			vel[vertexId * d_nDim + dim] = mobility * (force[vertexId * d_nDim + dim] + driving * ((1 - dim) * cos(angle) + dim * sin(angle)));
		}
  	}
}

__global__ void kernelUpdateActiveParticleVel(double* pVel, const double* pForce, double* pAngle, const double driving, const double mobility) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		auto angle = pAngle[particleId];
		for (long dim = 0; dim < d_nDim; dim++) {
			pVel[particleId * d_nDim + dim] = mobility * (pForce[particleId * d_nDim + dim] + driving * ((1 - dim) * cos(angle) + dim * sin(angle)));
		}
  	}
}

__global__ void kernelUpdateRigidBrownianVel(double* pVel, const double* pForce, double* pAngvel, const double* pTorque, double* thermalVel, const double mobility) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		for (long dim = 0; dim < d_nDim; dim++) {
			pVel[particleId * d_nDim + dim] = mobility * pForce[particleId * d_nDim + dim] + thermalVel[particleId * d_nDim + dim];
		}
		pAngvel[particleId] = mobility * pTorque[particleId];
  	}
}

__global__ void kernelUpdateRigidActiveVel(double* pVel, const double* pForce, double* pActiveAngle, double* pAngvel, const double* pTorque, const double driving, const double mobility) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
		auto activeAngle = pActiveAngle[particleId];
		for (long dim = 0; dim < d_nDim; dim++) {
			pVel[particleId * d_nDim + dim] = mobility * pForce[particleId * d_nDim + dim] + driving * ((1 - dim) * cos(activeAngle) + dim * sin(activeAngle));
		}
		pAngvel[particleId] = mobility * pTorque[particleId];
  	}
}

__global__ void kernelConserveVertexMomentum(double* vel) {
  	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  	__shared__ double COMP[MAXDIM];
  	if (threadIdx.x == 0) {
    	for (long dim = 0; dim < d_nDim; dim++) {
			COMP[dim] = 0.0;
		}
  	}
  	__syncthreads();

  	if (vertexId < d_numVertices) {
    	for (long dim = 0; dim < d_nDim; dim++) {
			atomicAdd(&COMP[dim], vel[vertexId * d_nDim + dim]);
    	}
  	}
  	__syncthreads();

  	if (threadIdx.x == 0) {
    	for (long dim = 0; dim < d_nDim; dim++) {
			COMP[dim] /= d_numVertices;
		}
  	}
  	__syncthreads();

  	if (vertexId < d_numVertices) {
    	for (long dim = 0; dim < d_nDim; dim++) {
			vel[vertexId * d_nDim + dim] -= COMP[dim];
		}
  	}
}

__global__ void kernelConserveParticleMomentum(double* pVel) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	__shared__ double COMP[MAXDIM];
  	if (threadIdx.x == 0) {
    	for (long dim = 0; dim < d_nDim; dim++) {
			COMP[dim] = 0.0;
		}
  	}
  	__syncthreads();

  	if (particleId < d_numParticles) {
    	for (long dim = 0; dim < d_nDim; dim++) {
			atomicAdd(&COMP[dim], pVel[particleId * d_nDim + dim]);
    	}
  	}
  	__syncthreads();

  	if (threadIdx.x == 0) {
    	for (long dim = 0; dim < d_nDim; dim++) {
			COMP[dim] /= d_numParticles;
		}
  	}
  	__syncthreads();

  	if (particleId < d_numParticles) {
    	for (long dim = 0; dim < d_nDim; dim++) {
			pVel[particleId * d_nDim + dim] -= COMP[dim];
		}
  	}
}

__global__ void kernelConserveSubSetMomentum(double* pVel, const long firstId) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	__shared__ double COMP[MAXDIM];
  	if (threadIdx.x == 0) {
    	for (long dim = 0; dim < d_nDim; dim++) {
			COMP[dim] = 0.0;
		}
  	}
  	__syncthreads();

  	if (particleId < d_numParticles && particleId > firstId) {
    	for (long dim = 0; dim < d_nDim; dim++) {
			atomicAdd(&COMP[dim], pVel[particleId * d_nDim + dim]);
    	}
  	}
  	__syncthreads();

  	if (threadIdx.x == 0) {
    	for (long dim = 0; dim < d_nDim; dim++) {
			COMP[dim] /= (d_numParticles - firstId);
		}
  	}
  	__syncthreads();

  	if (particleId < d_numParticles && particleId > firstId) {
    	for (long dim = 0; dim < d_nDim; dim++) {
			pVel[particleId * d_nDim + dim] -= COMP[dim];
		}
  	}
}

__global__ void kernelSumParticleVelocity(const double* pVel, double* velSum) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		for (long dim = 0; dim < d_nDim; dim++) {
			atomicAdd(&velSum[dim], pVel[particleId * d_nDim + dim]);
		}
	}
}

__global__ void kernelSubtractParticleDrift(double* pVel, const double* velSum) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		for (long dim = 0; dim < d_nDim; dim++) {
			atomicAdd(&pVel[particleId * d_nDim + dim], -velSum[dim]/d_numParticles);
		}
	}
}


#endif /* DPM2DKERNEL_CUH_ */
