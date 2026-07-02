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
  SimConfig() = default;
  SimConfig(double Tin):Tinject(Tin){}
};

class SimInterface // integration functions
{
public:
  DPM2D * dpm_;
  SimConfig config;
  double noise = 0; // this is just a choice
  double gamma = 1; // this is just a choice
  long firstIndex = 10;
  thrust::device_vector<double> d_rand;
  thrust::device_vector<double> d_rando;
  thrust::device_vector<double> d_thermalVel; // for brownian noise of soft particles

  SimInterface() = default;
  SimInterface(DPM2D * dpmPtr, SimConfig config):dpm_(dpmPtr),config(config){}
  virtual ~SimInterface() = default;

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

// BAOAB integrator child of Langevin
class BAOAB: public Langevin
{
public:
  BAOAB() = default;
  BAOAB(DPM2D * dpmPtr, SimConfig config) : Langevin:: Langevin(dpmPtr, config){;}

  virtual void updateThermalVel();
  virtual void integrate();
};

// Brownian integrator child of Langevin
class Brownian: public Langevin
{
public:
  Brownian() = default;
  Brownian(DPM2D * dpmPtr, SimConfig config) : Langevin:: Langevin(dpmPtr, config){;}

  virtual void updateThermalVel();
  virtual void integrate();
};

// Driven Brownian integrator child of Langevin
class DrivenBrownian: public Langevin
{
public:
  DrivenBrownian() = default;
  DrivenBrownian(DPM2D * dpmPtr, SimConfig config) : Langevin:: Langevin(dpmPtr, config){;}

  virtual void updateThermalVel();
  virtual void integrate();
};

// NVE integrator child of Langevin
class NVE: public Langevin
{
public:
  NVE() = default;
  NVE(DPM2D * dpmPtr, SimConfig config) : Langevin:: Langevin(dpmPtr, config){;}
  
  virtual void integrate();
};

// NVE integrator with velocity rescale child of Langevin
class NVERescale: public Langevin
{
public:
  NVERescale() = default;
  NVERescale(DPM2D * dpmPtr, SimConfig config) : Langevin:: Langevin(dpmPtr, config){;}

  virtual void injectKineticEnergy();
  virtual void integrate();
};

//********************* integrators for rigid particles **********************//
// Rigid Langevin integrator child of Langevin
class RigidLangevin: public Langevin
{
public:
  RigidLangevin() = default;
  RigidLangevin(DPM2D * dpmPtr, SimConfig config) : Langevin:: Langevin(dpmPtr, config){;}

  virtual void injectKineticEnergy();
  virtual void updatePosition(double timeStep);
  virtual void updateVelocity(double timeStep);
  virtual void conserveMomentum();
  virtual void integrate();
};

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
