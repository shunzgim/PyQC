#ifndef FULL_A_SIM_HPP
#define FULL_A_SIM_HPP

#include <iostream>
#include <complex>
#include <omp.h>
#include <string>
#include <map>


using StateType = std::complex<float>;

class fullASim
{
  public:
  
  fullASim();
  ~fullASim();
  void flush(int);
  int getLog2(int);
  void applyOneGate(StateType * buf, int, int);
  void applyControlOneGate(StateType * buf, int, int, int);
  void applyConstantModExp(int, int, int);
  StateType getOneAmplitudeFromBinstring(std::string);
  float getExpectation(int *, int);
  int getMeasureResultHandle(int);
  void grad_helper_init(StateType * buf, int *, int *, int);
  float grad_helper(StateType * buf, int *, int *, int);
  void show_state();

  StateType  * buffer;
  StateType  * tmp_buffer_1;
  StateType  * tmp_buffer_2;
  std::map<int, StateType *> state;
  
  bool state_init;
  size_t local_size;
  int qubit_nums;
};
#endif