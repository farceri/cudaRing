//
// Author: Yuxuan Cheng
// Date:   10-09-2021
//
// HEADER FILE FOR INTEGRATOR CLASS
// We define different integrator as child classes of SimulatorInterface where
// all the essential integration functions are defined

#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "DPM2D.h"
#include "defs.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

class DPM2D;

class SimConfig // initializer
{
public:
  double Tinject;
  double Dr;
  double driving;
  SimConfig() = default;
  SimConfig(double Tin, double Dr, double driving):Tinject(Tin), Dr(Dr), driving(driving) {}
};

class SimInterface // integration functions
{
public:
  DPM2D * dpm_;
  SimConfig config;
  double lcoeff1;
  double lcoeff2;
  double lcoeff3;
  double noiseVar;
  double gamma = 1; // this is just a choice
  long firstIndex = 10;
  double mass = 1;
  thrust::device_vector<double> d_rand;
  thrust::device_vector<double> d_rando;
  thrust::device_vector<double> d_pActiveAngle; // for decoupled rotation and activity angles
  thrust::device_vector<double> d_thermalVel; // for brownian noise of soft particles

  SimInterface() = default;
  SimInterface(DPM2D * dpmPtr, SimConfig config):dpm_(dpmPtr),config(config){}
  ~SimInterface();

  virtual void injectKineticEnergy() = 0;
  virtual void updatePosition(double timeStep) = 0;
  virtual void updateVelocity(double timeStep) = 0;
  virtual void updateThermalVel() = 0;
  virtual void conserveMomentum() = 0;
  virtual void integrate() = 0;
};

//****************** integrators for deformable particles ********************//
// Langevin integrator child of SimulatorInterface
class Langevin: public SimInterface
{
public:
  Langevin() = default;
  Langevin(DPM2D * dpmPtr, SimConfig config) : SimInterface:: SimInterface(dpmPtr, config){;}

  virtual void injectKineticEnergy();
  virtual void updatePosition(double timeStep);
  virtual void updateVelocity(double timeStep);
  virtual void updateThermalVel();
  virtual void conserveMomentum();
  virtual void integrate();
};

// Langevin2 integrator child of Langevin
class Langevin2: public Langevin
{
public:
  Langevin2() = default;
  Langevin2(DPM2D * dpmPtr, SimConfig config) : Langevin:: Langevin(dpmPtr, config){;}

  virtual void updatePosition(double timeStep);
  virtual void updateVelocity(double timeStep);
  virtual void updateThermalVel();
  virtual void integrate();
};

// Active Langevin integrator child of Langevin2
class ActiveLangevin: public Langevin2
{
public:
  ActiveLangevin() = default;
  ActiveLangevin(DPM2D * dpmPtr, SimConfig config) : Langevin2:: Langevin2(dpmPtr, config){;}

  virtual void updateThermalVel();
  virtual void integrate();
};

// NVE integrator child of Langevin
class NVE: public Langevin
{
public:
  NVE() = default;
  NVE(DPM2D * dpmPtr, SimConfig config) : Langevin:: Langevin(dpmPtr, config){;}
  
  virtual void injectKineticEnergy();
  virtual void integrate();
};

// Brownian integrator child of NVE
class Brownian: public NVE
{
public:
  Brownian() = default;
  Brownian(DPM2D * dpmPtr, SimConfig config) : NVE:: NVE(dpmPtr, config){;}

  virtual void updateVelocity(double timeStep);
  virtual void integrate();
};

// Active Brownian integrator child of NVE
class ActiveBrownian: public NVE
{
public:
  ActiveBrownian() = default;
  ActiveBrownian(DPM2D * dpmPtr, SimConfig config) : NVE:: NVE(dpmPtr, config){;}

  virtual void updateVelocity(double timeStep);
  virtual void integrate();
};

// Active Brownian integrator with damping on l0 child of NVE
class ActiveBrownianPlastic: public NVE
{
public:
  ActiveBrownianPlastic() = default;
  ActiveBrownianPlastic(DPM2D * dpmPtr, SimConfig config) : NVE:: NVE(dpmPtr, config){;}

  virtual void updatePosition(double timeStep);
  virtual void updateVelocity(double timeStep);
  virtual void integrate();
};

//********************* integrators for rigid particles **********************//
// Rigid Langevin integrator child of Langevin2
class RigidLangevin: public Langevin2
{
public:
  RigidLangevin() = default;
  RigidLangevin(DPM2D * dpmPtr, SimConfig config) : Langevin2:: Langevin2(dpmPtr, config){;}

  virtual void injectKineticEnergy();
  virtual void updatePosition(double timeStep);
  virtual void updateVelocity(double timeStep);
  virtual void conserveMomentum();
  virtual void integrate();
};

//****************** integrators for deformable particles ********************//
// Langevin integrator child of SimulatorInterface
class RigidNVE: public RigidLangevin
{
public:
  RigidNVE() = default;
  RigidNVE(DPM2D * dpmPtr, SimConfig config) : RigidLangevin:: RigidLangevin(dpmPtr, config){;}

  virtual void updatePosition(double timeStep);
  virtual void updateVelocity(double timeStep);
  virtual void conserveMomentum();
  virtual void integrate();
};

#endif // SIMULATOR_H //
