//
// Author: Francesco Arceri
// Date:   05-23-2022
//
// DEFINITION OF DPM3D FUNCTIONS

#include "../include/DPM3D.h"
#include "../include/Simulator.h"
#include "../include/FIRE.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

DPM3D::DPM3D(long nParticles, long dim, long nVertexPerParticle): DPM2D(nParticles, dim, nVertexPerParticle) {
  d_v0.resize(nVertexPerParticle);
  thrust::fill(d_v0.begin(), d_v0.end(), double(0));
}

DPM3D::~DPM3D() {
	// clear all vectors and pointers
	d_v0.clear();
}

void DPM3D::setPolyRandomSoftParticles(double phi0, double polyDispersity) {
  thrust::host_vector<double> boxSize(nDim);
  double r1, r2, randNum, mean, sigma, scale, boxLength = 1.;
  mean = 0.;
  sigma = sqrt(log(polyDispersity*polyDispersity + 1.));
  // generate polydisperse particle size
  for (long particleId = 0; particleId < numParticles; particleId++) {
    r1 = drand48();
    r2 = drand48();
    randNum = sqrt(-2. * log(r1)) * cos(2. * PI * r2);
    d_particleRad[particleId] = exp(mean + randNum * sigma);
    d_v0[particleId] = (4 * PI / 3) * d_particleRad[particleId] * d_particleRad[particleId] * d_particleRad[particleId];
  }
  scale = cbrt(getParticlePhi() / phi0);
  for (long dim = 0; dim < nDim; dim++) {
    boxSize[dim] = boxLength;
  }
  setBoxSize(boxSize);
  // extract random positions
  double volumeSum = 0;
  for (long particleId = 0; particleId < numParticles; particleId++) {
    d_particleRad[particleId] /= scale;
    d_v0[particleId] = (4 * PI / 3) * d_particleRad[particleId] * d_particleRad[particleId] * d_particleRad[particleId];
    for(long dim = 0; dim < nDim; dim++) {
      d_particlePos[particleId * nDim + dim] = d_boxSize[dim] * drand48();
    }
    volumeSum += d_v0[particleId];
  }
  // need to set this otherwise forces are zeros
  setLengthScaleToOne();
  //setSphericalLengthScale();
  cout << "DPM3D::setPolyRandomSoftParticles: particle packing fraction: " << getParticlePhi() << endl;
}

double DPM3D::getParticlePhi() {
  return double(thrust::reduce(d_v0.begin(), d_v0.end(), double(0), thrust::plus<double>())) / (d_boxSize[0] * d_boxSize[1] * d_boxSize[2]);
}

double DPM3D::getMeanParticleSize3D() {
  return cbrt(thrust::reduce(d_v0.begin(), d_v0.end(), double(0), thrust::plus<double>()) * 3 / (4 * PI * numParticles));
}
