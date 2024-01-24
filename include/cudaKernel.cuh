//
// Author: Francesco Arceri
// Date:   10-03-2021
//
// KERNEL FUNCTIONS THAT ACT ON THE DEVICE(GPU)

#ifndef DPM2DKERNEL_CUH_
#define DPM2DKERNEL_CUH_

#include "defs.h"
#include <stdio.h>

__constant__ simControlStruct d_simControl;

__constant__ long d_dimBlock;
__constant__ long d_dimGrid;
__constant__ long d_partDimGrid;

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

// vertex neighborList
__constant__ long* d_neighborListPtr;
__constant__ long* d_maxNeighborListPtr;
__constant__ long d_neighborListSize;
// make the neighborLoops only go up to neighborMax
__constant__ long d_maxNeighbors;

// particle neighborList
__constant__ long* d_partNeighborListPtr;
__constant__ long* d_partMaxNeighborListPtr;
__constant__ long d_partNeighborListSize;
// make the neighborLoops only go up to neighborMax
__constant__ long d_partMaxNeighbors;


inline __device__ double pbcDistance(const double x1, const double x2, const long dim) {
	double delta = x1 - x2, size = d_boxSizePtr[dim];
	return delta - size * round(delta / size); //round for distance, floor for position
}

inline __device__ double calcNorm(const double* segment) {
  double normSq = 0.;
  for (long dim = 0; dim < d_nDim; dim++) {
    normSq += segment[dim] * segment[dim];
  }
  return sqrt(normSq);
}

inline __device__ double calcNormSq(const double* segment) {
  double normSq = 0.;
  for (long dim = 0; dim < d_nDim; dim++) {
    normSq += segment[dim] * segment[dim];
  }
  return normSq;
}

inline __device__ double calcDistance(const double* thisVec, const double* otherVec) {
  double delta, distanceSq = 0.;
	#pragma unroll (MAXDIM)
  for (long dim = 0; dim < d_nDim; dim++) {
    delta = pbcDistance(thisVec[dim], otherVec[dim], dim);
    distanceSq += delta * delta;
  }
  return sqrt(distanceSq);
}

inline __device__ double calcDeltaAndDistance(const double* thisVec, const double* otherVec, double* deltaVec) {
	double delta, distanceSq = 0.;
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
    	delta = pbcDistance(thisVec[dim], otherVec[dim], dim);
		deltaVec[dim] = delta;
    	distanceSq += delta * delta;
  	}
  	return sqrt(distanceSq);
}

inline __device__ double calcFixedBoundaryDistance(const double* thisVec, const double* otherVec) {
  double delta, distanceSq = 0.;
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
  long nextId = vertexId + 1;
  long whichParticle = d_particleIdListPtr[nextId];
  if( whichParticle == particleId ) {
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
  long previousId = vertexId - 1;
  long whichParticle = d_particleIdListPtr[previousId];
  if( whichParticle == particleId ) {
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
	long particleId = d_particleIdListPtr[vertexId];
	long otherParticleId = d_particleIdListPtr[otherId];
	if ((particleId != otherParticleId) && (otherId != -1)) {
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
	long otherId = d_neighborListPtr[vertexId*d_neighborListSize + nListId];
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

inline __device__ bool extractParticleNeighbor(const long particleId, const long nListId, const double* pPos, const double* pRad, double* otherPos, double& otherRad) {
	long otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
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
	long otherId = d_partNeighborListPtr[particleId*d_partNeighborListSize + nListId];
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
	long nextId, currentId, firstId = d_firstVertexInParticleIdPtr[particleId];
	getVertexPos(firstId, pos, currentPos);
	#pragma unroll (MAXDIM)
	for (long dim = 0; dim < d_nDim; dim++) {
		partPos[dim] = currentPos[dim];
	}
  for (currentId = firstId; currentId < firstId + d_numVertexInParticleListPtr[particleId]-1; currentId++) {
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
	double tempArea = 0;
	long nextId, currentId, firstId = d_firstVertexInParticleIdPtr[particleId];
	double delta[MAXDIM], nextPos[MAXDIM], currentPos[MAXDIM];
	getVertexPos(firstId, pos, currentPos);
	// compute particle area via shoe-string method
	for (currentId = firstId; currentId < firstId + d_numVertexInParticleListPtr[particleId]; currentId++) {
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
  double midSine, midCosine;
  midSine = nSegment[0] * pSegment[1] - nSegment[1] * pSegment[0];
  midCosine = nSegment[0] * pSegment[0] + nSegment[1] * pSegment[1];
  return atan2(midSine, midCosine);
}

inline __device__ double calcAreaForceEnergy(const double pA0, const double pA, const double* nPos, const double* pPos, double* vertexForce) {
	double deltaA = (pA / pA0) - 1.; // area variation
	double gradMultiple = d_ea * deltaA / pA0;
  vertexForce[0] += 0.5 * gradMultiple * (pPos[1] - nPos[1]);
  vertexForce[1] += 0.5 * gradMultiple * (nPos[0] - pPos[0]);
  return (0.5 * d_ea * deltaA * deltaA);
}

inline __device__ double calcPerimeterForceEnergy(const double tL0, const double pL0, const double nLength, const double pLength, const double* vPos, const double* nPos, const double* pPos, double* vertexForce) {
  double pDelta, nDelta;
  //compute length variations
  pDelta = (pLength / pL0) - 1.;
  nDelta = (nLength / tL0) - 1.;
	// compute force
	#pragma unroll (MAXDIM)
  for (long dim = 0; dim < d_nDim; dim++) {
    vertexForce[dim] += d_el * ( ( nDelta * (nPos[dim] - vPos[dim]) / (tL0 * nLength) ) - ( pDelta * (vPos[dim] - pPos[dim]) / (pL0 * pLength) ) );
  }
  return (0.5 * d_el * pDelta * pDelta);
}

inline __device__ double calcBendingForceEnergy(const double* preSegment, const double* nextSegment, const double thisAngleDelta, const double nextAngleDelta, const double preAngleDelta, double* vertexForce) {
	double preNormalSegment[MAXDIM], nextNormalSegment[MAXDIM];
	double preVar, nextVar;
	// get normal segments
  getNormalVector(preSegment, preNormalSegment);
  getNormalVector(nextSegment, nextNormalSegment);
  // compute angle variations
  preVar = (thisAngleDelta - preAngleDelta) / calcNormSq(preSegment);
  nextVar = (thisAngleDelta - nextAngleDelta) / calcNormSq(nextSegment);
  // compute force
	#pragma unroll (MAXDIM)
  for (long dim = 0; dim < d_nDim; dim++) {
    vertexForce[dim] += d_eb * (preVar * preNormalSegment[dim] + nextVar * nextNormalSegment[dim]);
  }
  return (0.5 * d_eb * thisAngleDelta * thisAngleDelta);
}

// this, next and previous are for vertices belonging to the same particle
__global__ void kernelCalcShapeForceEnergy(const double* a0, const double* area, const double* particlePos, const double* l0, const double* theta0, const double* pos, double* force, double* energy) {
  long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  long particleId = d_particleIdListPtr[vertexId];
  if (vertexId < d_numVertices) {
		double vertexPos[MAXDIM], nextPos[MAXDIM], previousPos[MAXDIM], partPos[MAXDIM];
		double secondNextPos[MAXDIM], secondPreviousPos[MAXDIM];
		double nextSegment[MAXDIM], previousSegment[MAXDIM];
		double secondNextSegment[MAXDIM], secondPreviousSegment[MAXDIM];
		double previousLength, nextLength, particleArea = area[particleId], shapeEnergy = 0.;
		double thisAngleDelta, nextAngleDelta, previousAngleDelta;
		// get interacting vertices' indices
		long nextId = getNextId(vertexId, particleId);
	  long previousId = getPreviousId(vertexId, particleId);
	  long secondNextId = getNextId(nextId, particleId);
	  long secondPreviousId = getPreviousId(previousId, particleId);
		//printf("vertexId: %ld, previousId: %ld, nextId: %ld \n", vertexId, previousId, nextId);
		// zero out the existing force and get particlePos
    for (long dim = 0; dim < d_nDim; dim++) {
			force[vertexId * d_nDim + dim] = 0.;
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
	  previousLength = calcNorm(previousSegment);
	  nextLength = calcNorm(nextSegment);
		shapeEnergy += calcPerimeterForceEnergy(l0[vertexId], l0[previousId], nextLength, previousLength, vertexPos, nextPos, previousPos, &force[vertexId*d_nDim]);
		// bending force
		getRelativeVertexPos(secondNextId, pos, secondNextPos, partPos);
	  getRelativeVertexPos(secondPreviousId, pos, secondPreviousPos, partPos);
		getSegment(secondNextPos, nextPos, secondNextSegment);
		getSegment(previousPos, secondPreviousPos, secondPreviousSegment);
	  previousAngleDelta = calcAngle(previousSegment, secondPreviousSegment) - theta0[previousId];
	  thisAngleDelta = calcAngle(nextSegment, previousSegment) - theta0[vertexId];
	  nextAngleDelta = calcAngle(secondNextSegment, nextSegment) - theta0[nextId];
		shapeEnergy += calcBendingForceEnergy(previousSegment, nextSegment, thisAngleDelta, nextAngleDelta, previousAngleDelta, &force[vertexId*d_nDim]);
    energy[vertexId] = shapeEnergy;
		//printf("\n shape: vertexId %ld \t fx: %.13e \t fy: %.13e \n", vertexId, force[vertexId*d_nDim], force[vertexId*d_nDim+1]);
	}
}

inline __device__ double calcContactInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
  double overlap, gradMultiple = 0, distance, delta[MAXDIM];
	distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	overlap = 1 - distance / radSum;
	if (overlap > 0) {
		gradMultiple = d_ec * overlap / radSum;
		#pragma unroll (MAXDIM)
	  for (long dim = 0; dim < d_nDim; dim++) {
	    currentForce[dim] += gradMultiple * delta[dim] / distance;
	  }
	  return (0.5 * d_ec * overlap * overlap) * 0.5;
	}
	return 0.;
}

inline __device__ double calcLJForceShift(const double radSum, const double radSum6) {
	double distance, distance6;
	distance = d_LJcutoff * radSum;
	distance6 = pow(distance, 6);
	return 24 * d_ec * radSum6 * (1 / distance6 - 2*radSum6 / (distance6 * distance6)) / distance;
}

inline __device__ double calcLJInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
  double distance, distance6, radSum6, forceShift, gradMultiple = 0, epot = 0.;
	double delta[MAXDIM];
	distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	distance6 = pow(distance, 6);
	radSum6 = pow(radSum, 6);
	if (distance <= (d_LJcutoff * radSum)) {
		forceShift = calcLJForceShift(radSum, radSum6);
		gradMultiple = -24 * d_ec * radSum6 * (1 / distance6 - 2*radSum6 / (distance6 * distance6)) / distance + forceShift;
		epot = 0.5 * d_ec * (4 * (radSum6 * radSum6 / (distance6 * distance6) - radSum6 / distance6) - d_LJecut);
		epot -= 0.5 * forceShift * (distance - d_LJcutoff * radSum);
	} else {
		epot = 0.;
	}
	if (gradMultiple != 0) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    currentForce[dim] += gradMultiple * delta[dim] / distance;
	  }
	}
	return epot;
}

inline __device__ double calcAdhesiveInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
  double overlap, distance, gradMultiple = 0, epot = 0.;
	double delta[MAXDIM];
	distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	overlap = 1 - distance / radSum;
	if (distance < (1 + d_l1) * radSum) {
		gradMultiple = d_ec * overlap / radSum;
		epot = 0.5 * d_ec * (overlap * overlap - d_l1 * d_l2) * 0.5;
	} else if ((distance >= (1 + d_l1) * radSum) && (distance < (1 + d_l2) * radSum)) {
		gradMultiple = -(d_ec * d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) / radSum;
		epot = -(0.5 * (d_ec * d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) * (overlap + d_l2)) * 0.5;
	} else {
		epot = 0.;
	}
	if (gradMultiple != 0) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
	    currentForce[dim] += gradMultiple * delta[dim] / distance;
	  }
	}
	return epot;
}

inline __device__ double calcWCAInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce) {
  double distance, distance6, radSum6, gradMultiple = 0, epot = 0.;
  double delta[MAXDIM];
  distance = calcDeltaAndDistance(thisPos, otherPos, delta);
  distance6 = pow(distance, 6);
  radSum6 = pow(radSum, 6);
  if (distance <= (WCAcut * radSum)) {
	gradMultiple = -24 * d_ec * radSum6 * (1 / distance6 - 2*radSum6 / (distance6 * distance6)) / distance;
	epot = 0.5 * d_ec * (4 * (radSum6 * radSum6 / (distance6 * distance6) - radSum6 / distance6) + 1);
	} else {
		epot = 0.;
	}
	if (gradMultiple != 0) {
		#pragma unroll (MAXDIM)
		for (long dim = 0; dim < d_nDim; dim++) {
			currentForce[dim] += gradMultiple * delta[dim] / distance;
		}
	}
	return epot;
}

// this and other are for vertices belonging to neighbor particles
__global__ void kernelCalcVertexInteraction(const double* rad, const double* pos, double* force, double* energy) {
  	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (vertexId < d_numVertices) {
    	double thisRad, otherRad, radSum;
		double thisPos[MAXDIM], otherPos[MAXDIM];
		// we don't zero out the force because we always call this function
		// after kernelCalcShapeForceEnergy where the force is zeroed out
		getVertexPos(vertexId, pos, thisPos);
    	thisRad = rad[vertexId];
    	// interaction between vertices of neighbor particles
    	for (long nListId = 0; nListId < d_maxNeighborListPtr[vertexId]; nListId++) {
      		if(extractNeighbor(vertexId, nListId, pos, rad, otherPos, otherRad)) {
				radSum = thisRad + otherRad;
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

inline __device__ double calcGradMultipleAndEnergy(const double* thisPos, const double* otherPos, const double radSum, double epot) {
	double distance, overlap, distance6, radSum6;
	distance = calcDistance(thisPos, otherPos);
	epot = 0;
	switch (d_simControl.potentialType) {
		case simControlStruct::potentialEnum::harmonic:
		overlap = 1 - distance / radSum;
		if(overlap > 0) {
			epot = 0.5 * d_ec * overlap * overlap;
			return d_ec * overlap / radSum;
		} else {
			return 0;
		}
		break;
		case simControlStruct::potentialEnum::lennardJones:
		double forceShift;
		distance6 = pow(distance, 6);
		radSum6 = pow(radSum, 6);
		if (distance <= (d_LJcutoff * radSum)) {
			forceShift = calcLJForceShift(radSum, radSum6);
			epot = 0.5 * d_ec * (4 * (radSum6 * radSum6 / (distance6 * distance6) - radSum6 / distance6) - d_LJecut) - forceShift * (distance - d_LJcutoff * radSum);
			return -24 * d_ec * radSum6 * (1 / distance6 - 2*radSum6 / (distance6 * distance6)) / distance + forceShift;
		} else {
			return 0;
		}
		break;
		case simControlStruct::potentialEnum::adhesive:
		overlap = 1 - distance / radSum;
		if (distance < (1 + d_l1) * radSum) {
			epot = 0.5 * d_ec * (overlap * overlap - d_l1 * d_l2);
			return d_ec * overlap / radSum;
		} else if ((distance >= (1 + d_l1) * radSum) && (distance < (1 + d_l2) * radSum)) {
			epot = -(0.5 * (d_ec * d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) * (overlap + d_l2));
			return -(d_ec * d_l1 / (d_l2 - d_l1)) * (overlap + d_l2) / radSum;
		} else {
			return 0;
		}
		break;
		case simControlStruct::potentialEnum::wca:
		distance6 = pow(distance, 6);
		radSum6 = pow(radSum, 6);
		if (distance <= (WCAcut * radSum)) {
			epot = 0.5 * d_ec * (4 * (radSum6 * radSum6 / (distance6 * distance6) - radSum6 / distance6) + 1);
			return -24 * d_ec * radSum6 * (1 / distance6 - 2*radSum6 / (distance6 * distance6)) / distance;
		} else {
			return 0;
		}
		break;
	}
}

// clockwise projection
inline __device__ double getProjection(const double* thisPos, const double* otherPos, const double* previousPos, const double length) {
	double proj;
	proj = pbcDistance(thisPos[0], otherPos[0], 0) * pbcDistance(previousPos[0], otherPos[0], 0) + pbcDistance(thisPos[1], otherPos[1], 1) * pbcDistance(previousPos[1], otherPos[1], 1);
	return proj / (length * length);
}

inline __device__ void getProjectionPos(const double* previousPos, const double* segment, double* projPos, const double proj) {
	double reducedProj = max(0.0, min(1.0, proj));
	for (long dim = 0; dim < d_nDim; dim++) {
		projPos[dim] = previousPos[dim] + reducedProj * segment[dim];
	}
}

inline __device__ double calcCross(const double* thisPos, const double* otherPos, const double* previousPos) {
	return pbcDistance(previousPos[0], otherPos[0],0) * pbcDistance(otherPos[1], thisPos[1],1) - pbcDistance(otherPos[0], thisPos[0],0) * pbcDistance(previousPos[1], otherPos[1],1);
}

inline __device__ double calcVertexSegmentInteraction(const double* thisPos, const double* projPos, const double* otherPos, const double* previousPos, const double radSum, const double length, double* thisForce, double* otherForce, double* previousForce) {
  double gradMultiple, cross, absCross, sign, epot = 0;
	// compute segment and the overlap between its center and this vertex
	cross = calcCross(thisPos, otherPos, previousPos);
	absCross = fabs(cross);
	gradMultiple = calcGradMultipleAndEnergy(thisPos, projPos, radSum, epot);
	if (gradMultiple > 0) {
		sign = cross / absCross;
		// this vertex
	  thisForce[0] += gradMultiple * sign * pbcDistance(previousPos[1], otherPos[1], 1) / length;
	  thisForce[1] += gradMultiple * sign * pbcDistance(otherPos[0], previousPos[0], 0) / length;
		// other vertex
	  otherForce[0] += gradMultiple * (sign * pbcDistance(thisPos[1], previousPos[1], 1) + absCross * pbcDistance(previousPos[0], otherPos[0], 0) / (length * length)) / length;
	  otherForce[1] += gradMultiple * (sign * pbcDistance(previousPos[0], thisPos[0], 0) + absCross * pbcDistance(previousPos[1], otherPos[1], 1) / (length * length)) / length;
		// previous vertex
	  previousForce[0] += gradMultiple * (sign * pbcDistance(otherPos[1], thisPos[1], 1) - absCross * pbcDistance(previousPos[0], otherPos[0], 0) / (length * length)) / length;
	  previousForce[1] += gradMultiple * (sign * pbcDistance(thisPos[0], otherPos[0], 0) - absCross * pbcDistance(previousPos[1], otherPos[1], 1) / (length * length)) / length;
	  return epot * 0.5;
	}
	return 0.;
}

inline __device__ double calcVertexVertexInteraction(const double* thisPos, const double* otherPos, const double radSum, double* currentForce, double sign) {
  double gradMultiple, distance, epot = 0, delta[MAXDIM];
	gradMultiple = calcGradMultipleAndEnergy(thisPos, otherPos, radSum, epot);
	if (gradMultiple > 0) {
		distance = calcDeltaAndDistance(thisPos, otherPos, delta);
	  for (long dim = 0; dim < d_nDim; dim++) {
	    currentForce[dim] += gradMultiple * delta[dim] / distance;
	  }
	  return epot * 0.5;
	}
	return 0.;
}

inline __device__ double checkAngle(double angle, double limit) {
	if(angle < 0) {
		angle += 2*PI;
	}
	return angle - limit;
}

// this and other are for vertices belonging to neighbor particles
__global__ void kernelCalcSmoothVertexInteraction(const double* rad, const double* pos, double* force, double* pEnergy) {
	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
	long particleId = d_particleIdListPtr[vertexId];
  	if (vertexId < d_numVertices) {
		long otherParticleId, otherId, previousId, secondPreviousId;
		double thisRad, otherRad, radSum, segmentLength, projection, interaction = 0., sign = 1;
		double thisPos[MAXDIM], otherPos[MAXDIM], previousPos[MAXDIM], secondPreviousPos[MAXDIM];
		double projPos[MAXDIM], segment[MAXDIM], previousSegment[MAXDIM], interSegment[MAXDIM];
		double endEndAngle, endCapAngle, inverseEndEndAngle;
		bool isCapConvexInteraction, isCapConcaveInteraction, isCapInteraction, isInverseInteraction, isConcaveInteraction;
		// we don't zero out the force because we always call this function
		// after kernelCalcShapeForceEnergy where the force is already zeroed out
		getVertexPos(vertexId, pos, thisPos);
		thisRad = rad[vertexId];
		// interaction between vertices of neighbor particles
		for (long nListId = 0; nListId < d_maxNeighborListPtr[vertexId]; nListId++) {
      		if(extractNeighbor(vertexId, nListId, pos, rad, otherPos, otherRad)) {
				otherId = d_neighborListPtr[vertexId*d_neighborListSize + nListId];
				otherParticleId = d_particleIdListPtr[otherId];
				radSum = thisRad + otherRad;
				// get previous vertex to vertex on neighbor particle
				previousId = getPreviousId(otherId, otherParticleId);
				getVertexPos(previousId, pos, previousPos);
				// compute projection of vertexId on segment between otherId and previousId
				getDelta(otherPos, previousPos, segment);
			  	segmentLength = calcNorm(segment);
				projection = getProjection(thisPos, otherPos, previousPos, segmentLength);
				getProjectionPos(previousPos, segment, projPos, projection);
				// check if the interaction is vertex-segment
				if(projection < 1 && projection > 0) { // vertex-segment interaction
					interaction += calcVertexSegmentInteraction(thisPos, projPos, otherPos, previousPos, radSum, segmentLength, &force[vertexId*d_nDim], &force[otherId*d_nDim], &force[previousId*d_nDim]);
				} else {
					switch (d_simControl.concavityType) {
						case simControlStruct::concavityEnum::off:
						if(projection <= 0) { // vertex-vertex interaction
							sign = 1;
							interaction += calcVertexVertexInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim], sign);
						} else if(projection > 0) { // vertex-vertex interaction
							sign = -1;
							interaction += calcVertexVertexInteraction(thisPos, previousPos, radSum, &force[vertexId*d_nDim], sign);
						} else {
							interaction = 0.;
						}
						break;
						case simControlStruct::concavityEnum::on:
						// check if the vertex-vertex interaction is concave or convex
						secondPreviousId = getPreviousId(previousId, otherParticleId);
						getVertexPos(secondPreviousId, pos, secondPreviousPos);
						getDelta(secondPreviousPos, previousPos, previousSegment);
						getDelta(thisPos, previousPos, interSegment);
						endEndAngle = atan2(interSegment[0]*segment[1] - interSegment[1]*segment[0], interSegment[0]*segment[0] + interSegment[1]*segment[1]);
						checkAngle(endEndAngle, PI/2);
						endCapAngle = atan2(previousSegment[0]*segment[1] - previousSegment[1]*segment[0], previousSegment[0]*segment[0] + previousSegment[0]*segment[0]);
						checkAngle(endCapAngle, PI);
						isCapConvexInteraction = (endEndAngle >= 0 && endEndAngle <= endCapAngle);
						isCapConcaveInteraction = (endCapAngle < 0 && endEndAngle > PI - fabs(endCapAngle) && endEndAngle < PI);
						isCapInteraction = (isCapConvexInteraction || isCapConcaveInteraction);
						// check if the interaction is inverse
						inverseEndEndAngle = (endEndAngle - 2*PI * (endEndAngle > PI));
						isConcaveInteraction = (endCapAngle < 0 && inverseEndEndAngle < 0 && inverseEndEndAngle >= endCapAngle);
						endEndAngle = PI - endEndAngle + fabs(endCapAngle);
						endEndAngle -= 2*PI * (endEndAngle > 2*PI);
						endEndAngle += 2*PI * (endEndAngle < 0);
						isInverseInteraction = (isConcaveInteraction || (endCapAngle > 0 && (endEndAngle < endCapAngle)));
						if(projection <= 0 && isCapInteraction) { // vertex-vertex interaction
							sign = 1;
							interaction += calcVertexVertexInteraction(thisPos, otherPos, radSum, &force[vertexId*d_nDim], sign);
						} else if(projection > 0 && isInverseInteraction) { // vertex-vertex interaction
							sign = -1;
							interaction += calcVertexVertexInteraction(thisPos, previousPos, radSum, &force[vertexId*d_nDim], sign);
						} else {
							interaction = 0.;
						}
						break;
					}
				}
			}
		}
		pEnergy[particleId] += interaction;
  	}
}

// particle-particle contact interaction
__global__ void kernelCalcParticleInteraction(const double* pRad, const double* pPos, double* pForce, double* pEnergy) {
  	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  	if (particleId < d_numParticles) {
    	double thisRad, otherRad, radSum;
		double thisPos[MAXDIM], otherPos[MAXDIM];
		// zero out the force and get particle positions
		for (long dim = 0; dim < d_nDim; dim++) {
			pForce[particleId * d_nDim + dim] = 0;
			thisPos[dim] = pPos[particleId * d_nDim + dim];
		}
    	thisRad = pRad[particleId];
    	pEnergy[particleId] = 0;
    	// interaction between vertices of neighbor particles
    	for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
      		if (extractParticleNeighbor(particleId, nListId, pPos, pRad, otherPos, otherRad)) {
				radSum = thisRad + otherRad;
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
		long particleId = d_particleIdListPtr[vertexId];
    	double thisRad, otherRad, radSum;
		double thisPos[MAXDIM], otherPos[MAXDIM], partPos[MAXDIM];
		for (long dim = 0; dim < d_nDim; dim++) {
			force[vertexId * d_nDim + dim] = 0;
		}
		torque[vertexId] = 0;
		energy[vertexId] = 0;
		getVertexPos(vertexId, pos, thisPos);
    	thisRad = rad[vertexId];
    	// interaction between vertices of neighbor particles
    	for (long nListId = 0; nListId < d_maxNeighborListPtr[vertexId]; nListId++) {
      		if(extractNeighbor(vertexId, nListId, pos, rad, otherPos, otherRad)) {
				//if(vertexId == 0) printf("vertexId %ld \t neighbor: %ld \t force %lf \t %lf \n", vertexId, d_neighborListPtr[vertexId*d_neighborListSize + nListId], force[vertexId * d_nDim], force[vertexId * d_nDim + 1]);
				radSum = thisRad + otherRad;
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

__global__ void kernelCalcParticleRigidForceEnergy(const double* force, const double* torque, const double* energy, double* pForce, double* pTorque, double* pEnergy) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if(particleId < d_numParticles) {
		for (long dim = 0; dim < d_nDim; dim++) {
			pForce[particleId * d_nDim + dim] = 0;
		}
		pTorque[particleId] = 0;
		pEnergy[particleId] = 0;
		long firstVertex = d_firstVertexInParticleIdPtr[particleId];
		long lastVertex = firstVertex + d_numVertexInParticleListPtr[particleId];
		for (long vertexId = firstVertex; vertexId < lastVertex; vertexId++) {
			for (long dim = 0; dim < d_nDim; dim++) {
				pForce[particleId * d_nDim + dim] += force[vertexId * d_nDim + dim];
			}
			pTorque[particleId] += torque[vertexId];
			pEnergy[particleId] += energy[vertexId];
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

__global__ void kernelCalcPerParticleStressTensor(const double* rad, const double* pos, const double* pPos, double* perPStress) {
  long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleId < d_numParticles) {
		long vertexId, firstId = d_firstVertexInParticleIdPtr[particleId];
    double thisRad, otherRad, radSum;
		double overlap, gradMultiple, distance, scalingFactor = d_rho0 / (d_boxSizePtr[0] * d_boxSizePtr[1]);
		double relativePos[MAXDIM], thisPos[MAXDIM], otherPos[MAXDIM], deltas[MAXDIM], forces[MAXDIM];
		// zero out perParticleStress
		for (long dim2 = 0; dim2 < (d_nDim * d_nDim); dim2++) {
			perPStress[particleId * (d_nDim * d_nDim) + dim2] = 0;
		}
		// iterate over vertices in particle
		for (vertexId = firstId; vertexId < firstId + d_numVertexInParticleListPtr[particleId]; vertexId++) {
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
							deltas[dim] = pbcDistance(otherPos[dim], thisPos[dim], dim);
							forces[dim] = gradMultiple * deltas[dim] / (distance * radSum);
							relativePos[dim] += deltas[dim] * 0.5; // distance from center of mass to contact location
						}
						//diagonal terms
						perPStress[particleId * (d_nDim * d_nDim)] += relativePos[0] * forces[0] * scalingFactor;
						perPStress[particleId * (d_nDim * d_nDim) + 3] += relativePos[1] * forces[1] * scalingFactor;
						// cross terms
						perPStress[particleId * (d_nDim * d_nDim) + 1] += relativePos[0] * forces[1] * scalingFactor;
						perPStress[particleId * (d_nDim * d_nDim) + 2] += relativePos[1] * forces[0] * scalingFactor;
					}
				}
			}
		}
	}
}

//works only for 2D
__global__ void kernelCalcParticlesShape(const double* pos, double* length, double* area, double* perimeter) {
  long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleId < d_numParticles) {
		double tempPerimeter, tempArea = 0;
		perimeter[particleId] = 0;
		long nextId, currentId, firstId = d_firstVertexInParticleIdPtr[particleId];
		double delta[MAXDIM], nextPos[MAXDIM], currentPos[MAXDIM];
		getVertexPos(firstId, pos, currentPos);
		// compute particle area via shoe-string method
		for (currentId = firstId; currentId < firstId + d_numVertexInParticleListPtr[particleId]; currentId++) {
			nextId = getNextId(currentId, particleId);
			tempPerimeter = 0;
			for (long dim = 0; dim < d_nDim; dim++) {
				delta[dim] = pbcDistance(pos[nextId * d_nDim + dim], currentPos[dim], dim);
				nextPos[dim] = currentPos[dim] + delta[dim];
				tempPerimeter += delta[dim] * delta[dim];
			}
			length[currentId] = sqrt(tempPerimeter);
			perimeter[particleId] += length[currentId];
			tempArea += currentPos[0] * nextPos[1] - nextPos[0] * currentPos[1];
			for (long dim = 0; dim < d_nDim; dim++) {
				currentPos[dim] = nextPos[dim];
			}
		}
		area[particleId] = abs(tempArea) * 0.5;
	}
}

__global__ void kernelCalcParticlesPositions(const double* pos, double* particlePos) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		calcParticlePos(particleId, pos, &particlePos[particleId*d_nDim]);
	}
}

__global__ void kernelCalcVertexArea(const double* rad, double* vertexArea) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double vertexRad = rad[d_firstVertexInParticleIdPtr[particleId]];
		vertexArea[particleId] = vertexRad * vertexRad * (0.5 * d_numVertexInParticleListPtr[particleId] - 1);
	}
}

__global__ void kernelScaleVertexPositions(const double* particlePos, double* pos, double scale) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double distance[MAXDIM];
		long firstVertex = d_firstVertexInParticleIdPtr[particleId];
		long lastVertex = firstVertex + d_numVertexInParticleListPtr[particleId];
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
		long firstVertex = d_firstVertexInParticleIdPtr[particleId];
		long lastVertex = firstVertex + d_numVertexInParticleListPtr[particleId];
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
		long firstVertex = d_firstVertexInParticleIdPtr[particleId];
		long lastVertex = firstVertex + d_numVertexInParticleListPtr[particleId];
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
		long firstVertex = d_firstVertexInParticleIdPtr[particleId];
		long lastVertex = firstVertex + d_numVertexInParticleListPtr[particleId];
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
    long addedNeighbor = 0;
    double thisRad, otherRad, radSum;
    double thisPos[MAXDIM], otherPos[MAXDIM];
    getVertexPos(vertexId, pos, thisPos);
    thisRad = rad[vertexId];

    for (long otherId = 0; otherId < d_numVertices; otherId++) {
      if(extractOtherVertex(vertexId, otherId, pos, rad, otherPos, otherRad)) {
        bool isNeighbor = false;
        radSum = thisRad + otherRad;
        isNeighbor = (-calcOverlap(thisPos, otherPos, radSum) < cutDistance);
        if (addedNeighbor < d_neighborListSize) {
					d_neighborListPtr[vertexId * d_neighborListSize + addedNeighbor] = otherId*isNeighbor -1*(!isNeighbor);
				}
				addedNeighbor += isNeighbor;
      }
    }
    d_maxNeighborListPtr[vertexId] = addedNeighbor;
  }
}

__global__ void kernelCalcParticleNeighbors(const double* pos, const double* rad, const long neighborLimit, long* neighborList, long* numNeighbors) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		long addedNeighbor = 0, newNeighborId;
		double thisRad, otherRad;
		double thisPos[MAXDIM], otherPos[MAXDIM];
		for(long vertexId = d_firstVertexInParticleIdPtr[particleId]; vertexId < d_firstVertexInParticleIdPtr[particleId] + d_numVertexInParticleListPtr[particleId]; vertexId++) {
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
    long addedNeighbor = 0;
    double thisRad, otherRad, radSum;
    double thisPos[MAXDIM], otherPos[MAXDIM];
		getParticlePos(particleId, pPos, thisPos);
    thisRad = pRad[particleId];

    for (long otherId = 0; otherId < d_numParticles; otherId++) {
      if(extractOtherParticle(particleId, otherId, pPos, pRad, otherPos, otherRad)) {
        bool isNeighbor = false;
        radSum = thisRad + otherRad;
        isNeighbor = (-calcOverlap(thisPos, otherPos, radSum) < cutDistance);
				//isNeighbor = (calcDistance(thisPos, otherPos) < cutDistance);
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
		long addedContact = 0, newContactId;
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
    double thisRad, otherRad, radSum, overlap, gradMultiple, distance;
		double thisPos[MAXDIM], otherPos[MAXDIM];
		// we don't zero out the force because we always call this function
		// after kernelCalcShapeForceEnergy where the force is zeroed out
		getVertexPos(vertexId, pos, thisPos);
    thisRad = rad[vertexId];
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
		double distance = 0;
		distance = calcDistance(&pos[vertexId*d_nDim], &initialPos[vertexId*d_nDim]);
		vSF[vertexId] = sin(waveNum * distance) / (waveNum * distance);
	}
}

__global__ void kernelCalcParticleScatteringFunction(const double* pPos, const double* pInitialPos, double* pSF, const double waveNum) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleId < d_numParticles) {
		double distance = 0;
		distance = calcDistance(&pPos[particleId*d_nDim], &pInitialPos[particleId*d_nDim]);
		pSF[particleId] = sin(waveNum * distance) / (waveNum * distance);
	}
}

__global__ void kernelCalcHexaticOrderParameter(const double* pPos, double* psi6) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
	if(particleId < d_numParticles) {
		double thisPos[MAXDIM], otherPos[MAXDIM], delta[MAXDIM], theta;
		// get particle position
		for (long dim = 0; dim < d_nDim; dim++) {
			thisPos[dim] = pPos[particleId * d_nDim + dim];
		}
    // extract neighbor particles
    for (long nListId = 0; nListId < d_partMaxNeighborListPtr[particleId]; nListId++) {
      if (extractParticleNeighborPos(particleId, nListId, pPos, otherPos)) {
				getDelta(thisPos, otherPos, delta);
				theta = atan2(delta[1], delta[0]);
				psi6[particleId] += sin(6 * theta) / (6 * theta);
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
		long particleId = d_particleIdListPtr[vertexId];
		for (long dim = 0; dim < d_nDim; dim++) {
			vel[vertexId * d_nDim + dim] = mobility * force[vertexId * d_nDim + dim] + thermalVel[particleId * d_nDim + dim];
		}
  }
}

__global__ void kernelUpdateActiveVertexVel(double* vel, const double* force, double* pAngle, const double driving, const double mobility) {
	long vertexId = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertexId < d_numVertices) {
		long particleId = d_particleIdListPtr[vertexId];
		double angle = pAngle[particleId];
		for (long dim = 0; dim < d_nDim; dim++) {
			vel[vertexId * d_nDim + dim] = mobility * (force[vertexId * d_nDim + dim] + driving * ((1 - dim) * cos(angle) + dim * sin(angle)));
		}
  }
}

__global__ void kernelUpdateActiveParticleVel(double* pVel, const double* pForce, double* pAngle, const double driving, const double mobility) {
	long particleId = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleId < d_numParticles) {
		double angle = pAngle[particleId];
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
		double activeAngle = pActiveAngle[particleId];
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
