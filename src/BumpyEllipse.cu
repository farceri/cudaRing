//
// Author: Yuxuan Cheng
// Date:   02-22-2022
//
// DEFINITION OF BUMPY PARTICLE OBJECT

#include "../include/BumpyEllipse.h"

double ComputeArcOverAngle(double r1, double r2, double angle, double angleSeg);
double GetLengthOfEllipse(double deltaAngle, double a, double b);
double GetAngleForArcLengthRecursively(double currentArcPos, double goalArcPos, double angle, double angleSeg, double a, double b);

BumpyEllipse::BumpyEllipse(long nParticles, long dim, long nVertexPerParticle):DPM2D(nParticles, dim, nVertexPerParticle){};

void BumpyEllipse::setRatio(double calA0) {
    // initialize vertices
    initEllipse(ratio_a_b);

    calcParticlesShape();
    // double currentCalA = 0;
    double currentCalA = d_perimeter[0] * d_perimeter[0] / (4 * PI * d_area[0]);
    while (!(currentCalA > calA0 * 0.999 && currentCalA < calA0 * 1.001))
    {
        if (currentCalA > calA0)
            ratio_a_b *= 0.999;
        else
            ratio_a_b *= 1.001;
        initEllipse(ratio_a_b);
        calcParticlesShape();
        currentCalA = d_perimeter[0] * d_perimeter[0] / (4 * PI * d_area[0]);
    }
}

void BumpyEllipse::initVerticesOnParticles() {
    setRatio(calA0);
    initEllipse(ratio_a_b);
}

void BumpyEllipse::initEllipse(double ratio)
{
    long firstVertexId = 0;
    for (long particleId = 0; particleId < numParticles; particleId++) {
        long numVertexInParticle = d_numVertexInParticleList[particleId];
        double randomAngle = 2* PI * (double)rand() / (RAND_MAX + 1.0);

        double a = sqrt(ratio * d_a0[particleId] / PI);
        double b = a / ratio;

        // Distance in radians between angles measured on the ellipse
        double deltaAngle = 0.001;
        double circumference = GetLengthOfEllipse(deltaAngle, a, b);

        double arcLength = circumference/ numVertexInParticle;

        double angle = 0;

        long vertexId = 0;
        // Loop until we get all the points out of the ellipse
        for (int numPoints = 0; numPoints < numVertexInParticle; numPoints++)
        {
            vertexId = firstVertexId + numPoints;
            angle = GetAngleForArcLengthRecursively(0, arcLength, angle, deltaAngle, a, b);

            double xTemp = a * cos(angle);
            double yTemp = b * sin(angle);

            double x = xTemp * cos(randomAngle) - yTemp * sin(randomAngle);
            double y = xTemp * sin(randomAngle) + yTemp * cos(randomAngle);

            d_pos[vertexId * nDim] = x + d_particlePos[particleId * nDim] + 1e-02 * d_l0[vertexId] * drand48();
            d_pos[vertexId * nDim + 1] = y + d_particlePos[particleId * nDim + 1] + 1e-02 * d_l0[vertexId] * drand48();
        }
        firstVertexId += numVertexInParticle;
    }
}
// https://stackoverflow.com/questions/6972331/how-can-i-generate-a-set-of-points-evenly-distributed-along-the-perimeter-of-an

double GetLengthOfEllipse(double deltaAngle, double a, double b)
{
	// Distance in radians between angles
	double numIntegrals = round(PI * 2.0 / deltaAngle);
	double length = 0;

	// integrate over the elipse to get the circumference
	for (int i = 0; i < numIntegrals; i++)
	{
		length += ComputeArcOverAngle(a, b, i * deltaAngle, deltaAngle);
	}

	return length;
}

double GetAngleForArcLengthRecursively(double currentArcPos, double goalArcPos, double angle, double angleSeg, double a, double b)
{
	double ARC_ACCURACY = 0.1;
	// Calculate arc length at new angle
	double nextSegLength = ComputeArcOverAngle(a, b, angle + angleSeg, angleSeg);

	// If we've overshot, reduce the delta angle and try again
	if (currentArcPos + nextSegLength > goalArcPos) {
		return GetAngleForArcLengthRecursively(currentArcPos, goalArcPos, angle, angleSeg / 2, a, b);

		// We're below the our goal value but not in range (
	} else if (currentArcPos + nextSegLength < goalArcPos - ((goalArcPos - currentArcPos) * ARC_ACCURACY)) {
		return GetAngleForArcLengthRecursively(currentArcPos + nextSegLength, goalArcPos, angle + angleSeg, angleSeg, a, b);

		// current arc length is in range (within error), so return the angle
	} else
		return angle;
}

double ComputeArcOverAngle(double r1, double r2, double angle, double angleSeg)
{
	double distance = 0.0;

	double dpt_sin = pow(r1 * sin(angle), 2.0);
	double dpt_cos = pow(r2 * cos(angle), 2.0);
	distance = sqrt(dpt_sin + dpt_cos);

	// Scale the value of distance
	return distance * angleSeg;
}
