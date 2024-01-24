//
// Author: Francesco Arceri
// Date:   05-23-2022
//
// DEFINITION OF 3D DPM OBJECT

#ifndef DPM3D_H
#define DPM3D_H

#include "DPM2D.h"

class DPM3D : public DPM2D {
public:
	thrust::device_vector<double> d_v0;
	DPM2D * dpm_;

	DPM3D() = default;
  DPM3D(long nParticles, long dim, long nVertexPerParticle);
	~DPM3D();

	void setPolyRandomSoftParticles(double phi0, double polyDispersity);
	double getParticlePhi();
	double getMeanParticleSize3D();

};

#endif
