#include <iostream>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include "../include/full_amplitude_sim.hpp"


fullASim::fullASim(){
  this->state_init = false;
}


fullASim::~fullASim(){ 
  delete [] this->buffer;
  delete [] this->tmp_buffer_1;
  delete [] this->tmp_buffer_2;
}


void fullASim::flush(int n)
{
  if(!this->state_init){
    this->qubit_nums = n;
    this->local_size = 1UL<<n;
    this->buffer = new StateType[this->local_size];
    this->tmp_buffer_1 =  new StateType[this->local_size];
    this->tmp_buffer_2 =  new StateType[this->local_size];
    this->state[0] = this->buffer;
    this->state[1] = this->tmp_buffer_1;
    this->state[2] = this->tmp_buffer_2;
    this->state_init = true;
  }else if(this->local_size!=1UL<<n){  
    this->qubit_nums = n;
    this->local_size = 1UL<<n;
    delete [] this->buffer;
    delete [] this->tmp_buffer_1;
    delete [] this->tmp_buffer_2;
    this->buffer = NULL;
    this->tmp_buffer_1 = NULL;
    this->tmp_buffer_2 = NULL;
    this->buffer = new StateType[this->local_size];
    this->tmp_buffer_1 =  new StateType[this->local_size];
    this->tmp_buffer_2 =  new StateType[this->local_size];
  }
#pragma omp parallel for
  for(std::size_t i=0; i<this->local_size; i++){
    this->buffer[i] = {0.0, 0.0};
  }
  this->buffer[0] = {1.0, 0.0};
}

void fullASim::applyOneGate(StateType * buf, int target, int tag=0)
{ // default state in buffer
  target = this->qubit_nums - 1 - target;
  std::size_t t = 1UL<<target;
  if(target < 23)  
  {
#pragma omp parallel for
    for(std::size_t i = 0; i < this->local_size; i+=2*t)
    {
      for(std::size_t j=i; j<i+t; j++)
      {
        StateType a = this->state[tag][j];
        StateType b = this->state[tag][j+t];
        this->state[tag][j] =  buf[0]*a + buf[1]*b;
        this->state[tag][j+t] = buf[2]*a + buf[3]*b;
      }
    }
  }
  else  
  {
    for(std::size_t i = 0; i < this->local_size; i+=2*t)
    {
#pragma omp parallel for
      for(std::size_t j=i; j<i+t; j++)
      {
        StateType a = this->state[tag][j];
        StateType b = this->state[tag][j+t];
        this->state[tag][j] =  buf[0]*a + buf[1]*b;
        this->state[tag][j+t] = buf[2]*a + buf[3]*b;
      }
    }
  }
  //for(std::size_t i = 0; i < this->local_size; i++)
  //  std::cout<<this->state[i]<<" ";
  //std::cout<<std::endl;
}


void fullASim::applyControlOneGate(StateType * buf,  int target, int control, int tag=0)
{
  target = this->qubit_nums - 1 - target;
  control = this->qubit_nums - 1 - control;
  std::size_t t = 1UL<<target;
  std::size_t c = 1UL<<control;
  if(target<control)  // T > C 
  {
#pragma omp parallel for
    for(std::size_t ci=c; ci<this->local_size; ci+=2*c)
    {
      for(std::size_t i=ci; i<ci+c; i+=2*t)
      {
        for(std::size_t j=i; j<i+t; j++)
        {
          StateType a = this->state[tag][j];
          StateType b = this->state[tag][j+t];
          this->state[tag][j] =  buf[0]*a + buf[1]*b;
          this->state[tag][j+t] = buf[2]*a + buf[3]*b;
        }
      }
    }
  }
  else  // C > T
  {
#pragma omp parallel for
    for(std::size_t ci=c; ci<this->local_size; ci+=2*t)
    {
      for(std::size_t i=ci; i<ci+t; i+=2*c)
      {
        for(std::size_t j=i; j<i+c; j++)
        {
          StateType a = this->state[tag][j];
          StateType b = this->state[tag][j+t];
          this->state[tag][j] =  buf[0]*a + buf[1]*b;
          this->state[tag][j+t] = buf[2]*a + buf[3]*b;
        }
      }
    }
  }
  //for(std::size_t i = 0; i < this->local_size; i++)
  //  std::cout<<this->state[i]<<" ";
  //std::cout<<std::endl;
}


StateType fullASim::getOneAmplitudeFromBinstring(std::string binstr)
{
    std::size_t id=0;
    int size = binstr.size();
    for(int i=0; i<size; i++){
      int t = binstr[i] - '0';
      id += t * 1<<(size-1-i);
    }
    if(id<0 || id>this->local_size-1)
      throw "error!";
    //std::cout<<binstr<<" "<<id<<std::endl;
    return this->state[0][id]; // state in buffer
}


float fullASim::getExpectation(int * target, int size)
{
  //std::cout<<size<<std::endl;
  //for(int i=0; i<size; i++)
  //  std::cout<<i<<" ";
  //std::cout<<std::endl;
  float expectation=0;
#pragma omp parallel for reduction(+:expectation)
  for(std::size_t i=0; i<this->local_size; i++){
    float delta = std::norm(this->state[0][i]);
    int coefficient=1;
    for(int j=0; j<size; j++){
      int bit = this->qubit_nums - 1 - target[j];
      if((1<<bit)&i)
        coefficient *= -1;
    }
    //std::cout<<"coefficient: "<<coefficient<<" delta:"<<delta<<std::endl;
    expectation += coefficient*delta;
  }
  //std::cout<<expectation<<std::endl;
  return expectation;
}


void fullASim::grad_helper_init(StateType * buf_list, int * target_list, int * size_list, int size){
  /*
  int s, t=0;
  std::cout<<"size："<<size<<std::endl;
  for(int i=0; i<size; i++){
    s = size_list[i];
    std::cout<<"s: "<<s<<std::endl;
    for(int j=0; j<s; j++){
      StateType tmp[4] = {buf_list[4*t],buf_list[4*t+1],buf_list[4*t+2],buf_list[4*t+3]};
      std::cout<<tmp[0]<<' '<<tmp[1]<<' '<<tmp[2]<<' '<<tmp[3]<<' '<<target_list[t]<<std::endl;
      t += 1;
    }
  }
  */
  int s, t=0;
  #pragma omp parallel for
    for(std::size_t i=0; i<this->local_size; i++){
      this->tmp_buffer_1[i] = {0.0, 0.0};
    }
  for(int i=0; i<size; i++){
    s = size_list[i];
    memcpy(this->tmp_buffer_2,this->buffer,sizeof(StateType)*this->local_size);
    //std::cout<<"---------------------------"<<i<<"---------------------------"<<std::endl;
    //for(std::size_t i=0; i<this->local_size; i++){
    //  std::cout<<this->buffer[i]<<"    "<<this->tmp_buffer_2[i]<<std::endl;
    //}
    //std::cout<<"---------------------------"<<i<<"---------------------------"<<std::endl;
    for(int j=0; j<s; j++){
      StateType tmp[4] = {buf_list[4*t],buf_list[4*t+1],buf_list[4*t+2],buf_list[4*t+3]};
      //std::cout<<target_list[t]<<" ";
      this->applyOneGate(tmp, target_list[t], 2); //作用在tmp_buffer_2上
      t += 1;
    }
    //std::cout<<std::endl;
    //for(std::size_t i=0; i<this->local_size; i++){
    //  std::cout<<this->buffer[i]<<"    "<<this->tmp_buffer_2[i]<<std::endl;
    //}

    #pragma omp parallel for
    for(std::size_t i=0; i<this->local_size; i++){
      this->tmp_buffer_1[i] += this->tmp_buffer_2[i];
    }
    //std::cout<<"---------------------------"<<i<<"---------------------------"<<std::endl;
    //for(std::size_t i=0; i<this->local_size; i++){
    //  std::cout<<this->buffer[i]<<"    "<<this->tmp_buffer_2[i]<<"    "<<this->tmp_buffer_1[i]<<std::endl;
    //}
  }
  //std::cout<<"---------------------------grad_helper_init_end---------------------------"<<std::endl;
  //for(std::size_t i=0; i<this->local_size; i++){
  //  std::cout<<this->buffer[i]<<"    "<<this->tmp_buffer_1[i]<<std::endl;
  //}  
}


float fullASim::grad_helper(StateType * buf_list, int * target_list, int * size_list, int size){
  /*
  int s, t=0;
  std::cout<<"size："<<size<<std::endl;
  for(int i=0; i<size; i++){
    s = size_list[i];
    std::cout<<"s: "<<s<<std::endl;
    for(int j=0; j<s; j++){
      StateType tmp[4] = {buf_list[4*t],buf_list[4*t+1],buf_list[4*t+2],buf_list[4*t+3]};
      std::cout<<tmp[0]<<' '<<tmp[1]<<' '<<tmp[2]<<' '<<tmp[3]<<' '<<target_list[t]<<std::endl;
      t += 1;
    }
  }
  return 3.14;
  */
 //std::cout<<"---------------------------grag_helper_start---------------------------"<<std::endl;
  //for(std::size_t i=0; i<this->local_size; i++){
  //  std::cout<<this->buffer[i]<<"    "<<this->tmp_buffer_1[i]<<std::endl;
  //}

  float res=0;
  int s, t=0;
  for(int i=0; i<size; i++){
    s = size_list[i];
    memcpy(this->tmp_buffer_2,this->buffer,sizeof(StateType)*this->local_size);
    for(int j=0; j<s; j++){
      StateType tmp[4] = {buf_list[4*t],buf_list[4*t+1],buf_list[4*t+2],buf_list[4*t+3]};
      this->applyOneGate(tmp, target_list[t], 2); //作用在tmp_buffer_2上
      t += 1;
    }
    #pragma omp parallel for reduction(+:res)
    for(std::size_t i=0; i<this->local_size; i++){
      StateType tmp = std::conj(this->tmp_buffer_1[i])*this->tmp_buffer_2[i];
      res += tmp.imag();
    }
  } 
  //std::cout<<"---------------------------grag_helper_end---------------------------"<<std::endl;
  //for(std::size_t i=0; i<this->local_size; i++){
  //  std::cout<<this->buffer[i]<<"    "<<this->tmp_buffer_1[i]<<std::endl;
  //}

  return -2*res;
}


void fullASim::show_state(){
  std::cout<<"buffer____buffer1_____buffer2"<<std::endl;
  for(std::size_t i=0; i<this->local_size; i++){
    std::cout<<this->state[0][i]<<"   "<<this->state[1][i]<<"   "<<this->state[2][i]<<std::endl;
  }
}

