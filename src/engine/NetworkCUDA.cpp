#include "NeuroGen/cuda/NetworkCUDA.cuh"
#include "NeuroGen/BrainModuleArchitecture.h"
#include <iostream>
#include <vector>
#include <string>
#include <memory>

// Forward declaration of the BrainModuleArchitecture class
class BrainModuleArchitecture;

// CUDA error checking macro
#define CUDA_CHECK_RETURN(call, retval)                                        \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__              \
                << " - " << cudaGetErrorString(error) << std::endl;            \
      return retval;                                                           \
    }                                                                          \
  } while (0)

void NetworkCUDA::setBrainArchitecture(
    std::shared_ptr<BrainModuleArchitecture> architecture) {
  std::lock_guard<std::mutex> lock(cuda_mutex_);

  brain_architecture_ = architecture;

  if (brain_architecture_ && is_initialized_) {
    // Update module count and assignments
    auto module_names = brain_architecture_->getModuleNames();
    num_modules_ = module_names.size();

    // Update learning state with module information
    if (d_learning_state_) {
      // Create module assignment mapping
      std::vector<int> module_assignments(num_neurons_, 0);

      // Simple assignment strategy - divide neurons among modules
      if (num_modules_ > 0) {
        size_t neurons_per_module = num_neurons_ / num_modules_;
        for (size_t i = 0; i < num_neurons_; ++i) {
          module_assignments[i] = static_cast<int>(i / neurons_per_module);
          if (module_assignments[i] >= static_cast<int>(num_modules_)) {
            module_assignments[i] = static_cast<int>(num_modules_ - 1);
          }
        }
      }

      // Copy module assignments to GPU
      GPULearningState h_learning_state;
      CUDA_CHECK_RETURN(cudaMemcpy(&h_learning_state, d_learning_state_,
                                 sizeof(GPULearningState),
                                 cudaMemcpyDeviceToHost),
                        void());

      CUDA_CHECK_RETURN(cudaMemcpy(h_learning_state.module_assignments,
                                 module_assignments.data(),
                                 num_neurons_ * sizeof(int),
                                 cudaMemcpyHostToDevice),
                        void());

      h_learning_state.num_modules = static_cast<int>(num_modules_);

      CUDA_CHECK_RETURN(cudaMemcpy(d_learning_state_, &h_learning_state,
                                 sizeof(GPULearningState),
                                 cudaMemcpyHostToDevice),
                        void());
    }

    // Update inter-module connections
    if (d_inter_module_state_) {
      auto connections = brain_architecture_->getConnections();

      // Update connection count
      GPUInterModuleState h_inter_module_state;
      CUDA_CHECK_RETURN(cudaMemcpy(&h_inter_module_state, d_inter_module_state_,
                                 sizeof(GPUInterModuleState),
                                 cudaMemcpyDeviceToHost),
                        void());

      h_inter_module_state.num_connections = static_cast<int>(
          std::min(connections.size(), size_t(100))); // Max 100 connections

      CUDA_CHECK_RETURN(cudaMemcpy(d_inter_module_state_, &h_inter_module_state,
                                 sizeof(GPUInterModuleState),
                                 cudaMemcpyHostToDevice),
                        void());
    }

    std::cout << "🔗 Brain architecture integrated with " << num_modules_
              << " modules" << std::endl;
  }
}