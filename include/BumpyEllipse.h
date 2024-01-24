//
// Author: Yuxuan Cheng
// Date:   02-22-2022
//
// DEFINITION OF BUMPY PARTICLE OBJECT

#ifndef BUMPYELLIPSES_H
#define BUMPYELLIPSES_H

#include "DPM2D.h"

class BumpyEllipse : public DPM2D {
public:
	double ratio_a_b = 1.8;

	BumpyEllipse() = default;
  BumpyEllipse(long nParticles, long dim, long nVertexPerParticle);
  //~BumpyEllipse();

	virtual void setRatio(double calA0);
  void initVerticesOnParticles();
  void initEllipse(double ratio);

};

#endif
