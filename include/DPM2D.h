//
// Author: Francesco Arceri
// Date:   10-01-2021
//
// HEADER FILE FOR DPM2D CLASS

#ifndef DPM2D_H
#define DPM2D_H

#include "defs.h"
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace std;
using std::vector;
using std::string;

struct simControlStruct {
  enum class potentialEnum {harmonic, lennardJones, adhesive, wca} potentialType;
  enum class interactionEnum {vertexVertex, smooth} interactionType;
  enum class concavityEnum {on, off} concavityType;
};

// pointer-to-member function call macro
#define CALL_MEMBER_FN(object, ptrToMember) ((object).*(ptrToMember))

class DPM2D;
class FIRE;
class SimInterface;
//typedef void (DPM2D::*dpm2d)(void);

class DPM2D
{
public:

  // constructor and deconstructor
  DPM2D(long nParticles, long dim, long nVertexPerParticle);
  ~DPM2D();

  // Simulator
  FIRE * fire_;
  SimInterface * sim_;

  simControlStruct simControl;

  // variables for CUDA runtime details
  long dimGrid, dimBlock, partDimGrid;

  // dpm packing constants
  long nDim;
  long numParticles;
  long numVertexPerParticle;
  long numVertices;
  // the size distribution is defined by the number of vertices in the particles
  thrust::device_vector<long> d_numVertexInParticleList;
  thrust::device_vector<long> d_firstVertexInParticleId;
  // store the index of which particle each vertex belongs to
  thrust::device_vector<long> d_particleIdList;

  thrust::device_vector<double> d_boxSize;

  // time step
  double dt;
  // dimensional factor
  double rho0;
  double vertexRad;
  // vertex/particle energy consts
  double calA0;
  double ea; // area
  double el; // segment
  double eb; // bending
  double ec; // interaction
  // attraction constants
  double l1, l2;
  // Lennard-Jones constants
  double LJcutoff, LJecut;
  double cutoff, cutDistance;

  // vertex shape variables
  thrust::device_vector<double> d_rad;
  thrust::device_vector<double> d_l0;
  thrust::device_vector<double> d_length;
  thrust::device_vector<double> d_perimeter;
  thrust::device_vector<double> d_l0Vel;
  thrust::device_vector<double> d_a0;
  thrust::device_vector<double> d_area;
  thrust::device_vector<double> d_theta0;

  // dynamical variables
  thrust::device_vector<double> d_pos;
  thrust::device_vector<double> d_vel;
  thrust::device_vector<double> d_force;
  thrust::device_vector<double> d_energy;
  thrust::device_vector<double> d_lastPos;
  thrust::device_vector<double> d_disp;
  thrust::device_vector<double> d_initialPos;
  thrust::device_vector<double> d_delta;

  // particle variables
  thrust::device_vector<double> d_particlePos;
  thrust::device_vector<double> d_particleRad;
  thrust::device_vector<double> d_particleAngle;
  thrust::device_vector<double> d_particleLastPos;
  thrust::device_vector<double> d_particleDisp;
  thrust::device_vector<double> d_particleVel;
  thrust::device_vector<double> d_particleForce;
  thrust::device_vector<double> d_particleEnergy;
  thrust::device_vector<double> d_particleInitPos;
	thrust::device_vector<double> d_particleDelta;

  // rigid variables
  thrust::device_vector<double> d_torque;
  thrust::device_vector<double> d_particleAngvel;
  thrust::device_vector<double> d_particleTorque;
  thrust::device_vector<double> d_particleInitAngle;
  thrust::device_vector<double> d_particleLastAngle;
	thrust::device_vector<double> d_particleDeltaAngle;

  // stress
  thrust::device_vector<double> d_stress;
  thrust::device_vector<double> d_perParticleStress;

  //contact list
  thrust::device_vector<long> d_numContacts;
  thrust::device_vector<long> d_contactList;
  thrust::device_vector<double> d_contactVectorList;
  long maxContacts;
  long contactLimit;
  // neighbor list
  thrust::device_vector<long> d_neighborList;
  thrust::device_vector<long> d_maxNeighborList;
  long maxNeighbors;
	long neighborListSize;
	// particle neighbor list
  thrust::device_vector<long> d_numPartNeighbors;
  thrust::device_vector<long> d_partNeighborList;
  thrust::device_vector<long> d_partMaxNeighborList;
  long partMaxNeighbors;
	long partNeighborListSize;
  long neighborLimit;

  void initShapeVariables(long numVertices_, long numParticles_);

  void initDynamicalVariables(long numVertices_);

  void initParticleVariables(long numParticles_);

  void initDeltaVariables(long numVertices_, long numParticles_);

  void initRotationalVariables(long numVertices_, long numParticles_);

  void initContacts(long numParticles_);

  void initNeighbors(long numVertices_);

  void initParticleNeighbors(long numParticles_);

  void initParticleIdList();

  //setters and getters
  void syncSimControlToDevice();
  void syncSimControlFromDevice();

  void setPotentialType(simControlStruct::potentialEnum potentialType_);
	simControlStruct::potentialEnum getPotentialType();

  void setInteractionType(simControlStruct::interactionEnum interactionType_);
	simControlStruct::interactionEnum getInteractionType();

  void setConcavityType(simControlStruct::concavityEnum concavityType_);
	simControlStruct::concavityEnum getConcavityType();

  void setDimBlock(long dimBlock_);
  long getDimBlock();

  void setNDim(long nDim_);
  long getNDim();

  void setNumParticles(long numParticles_);
	long getNumParticles();

	void setNumVertices(long numVertices_);
	long getNumVertices();

  void setNumVertexPerParticle(long numVertexPerParticle_);
	long getNumVertexPerParticle();

  void setNumVertexInParticleList(thrust::host_vector<long> &numVertexInParticleList_);

  thrust::host_vector<long> getNumVertexInParticleList();

  void setLengthScale();

  void setLengthScaleToOne();

  void setBoxSize(thrust::host_vector<double> &boxSize_);
  thrust::host_vector<double> getBoxSize();

  // shape variables
  void setVertexRadii(thrust::host_vector<double> &rad_);
  thrust::host_vector<double> getVertexRadii();

  double getMaxRadius();

  void setRestAreas(thrust::host_vector<double> &a0_);
  thrust::host_vector<double> getRestAreas();

  void setRestLengths(thrust::host_vector<double> &l0_);
  thrust::host_vector<double> getRestLengths();

  void setRestAngles(thrust::host_vector<double> &theta0_);
  thrust::host_vector<double> getRestAngles();

  thrust::host_vector<double> getSegmentLengths();

  void setAreas(thrust::host_vector<double> &area_);
  thrust::host_vector<double> getAreas();

  thrust::host_vector<double> getPerimeters();

  double getMeanParticleSize();

  double getMeanParticleSigma();

  double getMinParticleSigma();

  double getMeanVertexRadius();

  double getMinVertexRadius();

  // particle variables
  void calcParticlesShape();

  void calcParticlesPositions();

  void setDefaultParticleRadii();

  void setParticleRadii(thrust::host_vector<double> &particleRad_);
  thrust::host_vector<double> getParticleRadii();

  void setParticlePositions(thrust::host_vector<double> &particlePos_);
  thrust::host_vector<double> getParticlePositions();

  void resetParticleLastPositions();

  void resetParticleLastAngles();

  void setParticleVelocities(thrust::host_vector<double> &particleVel_);
  thrust::host_vector<double> getParticleVelocities();

  void setParticleForces(thrust::host_vector<double> &particleForce_);
  thrust::host_vector<double> getParticleForces();

  thrust::host_vector<double> getParticleEnergies();

  void setParticleAngles(thrust::host_vector<double> &particleAngle_);
  thrust::host_vector<double> getParticleAngles();

  void setParticleAngularVelocities(thrust::host_vector<double> &particleAngvel_);
  thrust::host_vector<double> getParticleAngularVelocities();

  void setParticleTorques(thrust::host_vector<double> &particleTorque_);
  thrust::host_vector<double> getParticleTorques();

  // dynamical variables
  void setVertexPositions(thrust::host_vector<double> &pos_);
  thrust::host_vector<double> getVertexPositions();

  void resetLastPositions();

  void setVertexVelocities(thrust::host_vector<double> &vel_);
	thrust::host_vector<double> getVertexVelocities();

  void setVertexForces(thrust::host_vector<double> &force_);
	thrust::host_vector<double> getVertexForces();

  void setVertexTorques(thrust::host_vector<double> &torque_);
  thrust::host_vector<double> getVertexTorques();

  thrust::host_vector<double> getPerParticleStressTensor();

  thrust::host_vector<double> getStressTensor();

  double getPressure();

  double getTotalForceMagnitude();

  double getMaxUnbalancedForce();

  thrust::host_vector<long> getMaxNeighborList();

  thrust::host_vector<long> getNeighbors();

  thrust::host_vector<long> getContacts();

  void printNeighbors();

  void printContacts();

  double getPotentialEnergy();

  double getSmoothPotentialEnergy();

  double getKineticEnergy();

  double getTemperature();

  double getTotalEnergy();

  double getPhi();

  double getPreferredPhi();

  double getRigidPhi();

  double getParticlePhi();

  double get3DParticlePhi();

  double getVertexMSD();

  double getParticleMSD();

  double getMaxDisplacement();

  void setDisplacementCutoff(double cutoff_, double cutDistance_);

  void checkMaxDisplacement();

  double getParticleMaxDisplacement();

  double getDeformableWaveNumber();

  double getRigidWaveNumber();

  double getSoftWaveNumber();

  double getVertexISF();

  double getParticleISF(double waveNumber_);

  double getAreaFluctuation();

  // initialization functions
  void setMonoSizeDistribution();

  //void setBiSizeDistribution();

  void setPolySizeDistribution(double calA0, double polyDispersity);

  void setSinusoidalRestAngles(double thetaA, double thetaK);

  void setRandomParticles(double phi0, double extraRad);

  void setScaledRandomParticles(double phi0, double extraRad);

  void initVerticesOnParticles();

  void scaleBoxSize(double scale);

  void scaleVertexPositions(double scale);

  void scalePacking(double scale);

  void scaleParticlePacking();

  void scaleVertices(double scale);

  void scaleParticles(double scale);

  void pressureScaleParticles(double pscale);

  void scaleSoftParticles(double scale);

  void scaleParticleVelocity(double scale);

  void translateVertices();

  void rotateVertices();

  void translateAndRotateVertices();

  void computeParticleAngleFromVel();

  // force and energy
  void setEnergyCosts(double ea_, double el_, double eb_, double ec_);

  void setAttractionConstants(double l1_, double l2_);

  void setLJcutoff(double LJcutoff_);

  double setTimeScale(double dt_);

  double setTimeStep(double dt_);

  void calcForceEnergy();

  void calcRigidForceEnergy();

  void calcVertexForceTorque();

  void calcStressTensor();

  void calcPerParticleStressTensor();

  void calcNeighborForces();

  // contacts and neighbors
  void calcParticleNeighbors();

  void calcContacts(double gapSize);

  thrust::host_vector<long> getContactVectors(double gapSize);

  void calcNeighborList(double cutDistance);

  void syncNeighborsToDevice();

  void calcParticleNeighborList(double cutDistance);

  void syncParticleNeighborsToDevice();

  void calcParticleForceEnergy();

  void makeExternalParticleForce(double externalForce);

  void addExternalParticleForce();

  thrust::host_vector<double> getExternalParticleForce();

  double getParticleTotalForceMagnitude();

  double getParticleMaxUnbalancedForce();

  double getRigidMaxUnbalancedForce();

  double getParticleEnergy();

  double getParticleKineticEnergy();

  double getParticleTemperature();

  double getParticleDrift();

  thrust::host_vector<long> getParticleNeighbors();

  // minimizers
  void initFIRE(std::vector<double> &FIREparams, long minStep, long maxStep, long numDOF);

  void setParticleMassFIRE();

  void setTimeStepFIRE(double timeStep);

  void particleFIRELoop();

  void vertexFIRELoop();

  void initRigidFIRE(std::vector<double> &FIREparams, long minStep, long numStep, long numDOF, double cutDist);

  void rigidFIRELoop();

  // integrators
  void initLangevin(double Temp, double gamma, bool readState);

  void langevinLoop();

  void initLangevin2(double Temp, double gamma, bool readState);

  void langevin2Loop();

  void initActiveLangevin(double Temp, double Dr, double driving, double gamma, bool readState);

  void activeLangevinLoop();

  void initNVE(double Tin, bool readState);

  void NVELoop();

  void initBrownian(double Temp, double gamma, bool readState);

  void brownianLoop();

  void initActiveBrownian(double Dr, double driving, bool readState);

  void activeBrownianLoop();

  void initActiveBrownianPlastic(double Dr, double driving, double gamma, bool readState);

  void activeBrownianPlasticLoop();

  // integrators for rigid particles
  void initRigidLangevin(double Temp, double gamma, bool readState);

  void rigidLangevinLoop();

  void initRigidNVE(double Temp, bool readState);

  void rigidNVELoop();

};

#endif /* DPM2D_H */
