#include "NeuroGen/cuda/KernelLaunchWrappers.cuh"

// Include all the necessary kernel headers
#include "NeuroGen/cuda/IonChannelInitialization.cuh"
#include "NeuroGen/cuda/NeuronUpdateKernel.cuh"
#include "NeuroGen/cuda/CalciumDiffusionKernel.cuh"
#include "NeuroGen/cuda/EnhancedSTDPKernel.cuh"
#include "NeuroGen/cuda/EligibilityAndRewardKernels.cuh"
#include "NeuroGen/cuda/HomeostaticMechanismsKernel.cuh"

#include <iostream>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// Helper macro for checking CUDA errors
#define CUDA_CHECK_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA ERROR at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDA_CHECK_LAST_ERROR() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA KERNEL ERROR at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace KernelLaunchWrappers {

// (Other wrapper implementations remain the same)
void initialize_ion_channels(GPUNeuronState* neurons, int num_neurons) {
    const int num_blocks = (num_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    std::cout << "[CUDA] Launching ionChannelInitializationKernel with " << num_blocks << " blocks..." << std::endl;
    ionChannelInitializationKernel<<<num_blocks, THREADS_PER_BLOCK>>>(neurons, num_neurons);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    std::cout << "[CUDA] ionChannelInitializationKernel completed successfully" << std::endl;
}

void update_neuron_states(GPUNeuronState* neurons, float current_time, float dt, int num_neurons) {
    // Validate parameters before kernel launch
    if (neurons == nullptr) {
        std::cerr << "[CUDA ERROR] neurons pointer is NULL!" << std::endl;
        return;
    }
    if (num_neurons <= 0) {
        std::cerr << "[CUDA ERROR] invalid num_neurons: " << num_neurons << std::endl;
        return;
    }
    
    const int num_blocks = (num_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    std::cout << "[CUDA] Launching neuronUpdateKernel with " << num_blocks << " blocks..." << std::endl;
    std::cout << "[CUDA]   neurons=" << neurons << ", current_time=" << current_time 
              << ", dt=" << dt << ", num_neurons=" << num_neurons << std::endl;
    
    neuronUpdateKernel<<<num_blocks, THREADS_PER_BLOCK>>>(neurons, current_time, dt, num_neurons);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    std::cout << "[CUDA] neuronUpdateKernel completed successfully" << std::endl;
}

void update_calcium_dynamics(GPUNeuronState* neurons, float current_time, float dt, int num_neurons) {
    const int num_blocks = (num_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    std::cout << "[CUDA] Launching calciumDiffusionKernel with " << num_blocks << " blocks..." << std::endl;
    calciumDiffusionKernel<<<num_blocks, THREADS_PER_BLOCK>>>(neurons, current_time, dt, num_neurons);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    std::cout << "[CUDA] calciumDiffusionKernel completed successfully" << std::endl;
}

void run_stdp_and_eligibility(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float current_time,
    float dt,
    int num_synapses)
{
    const int num_blocks = (num_synapses + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    std::cout << "[CUDA] Launching enhancedSTDPKernel with " << num_blocks << " blocks..." << std::endl;
    enhancedSTDPKernel<<<num_blocks, THREADS_PER_BLOCK>>>(synapses, neurons, current_time, dt, num_synapses);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    std::cout << "[CUDA] enhancedSTDPKernel completed successfully" << std::endl;
}

void apply_reward_and_adaptation(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    float reward,
    float current_time,
    float dt,
    int num_synapses)
{
    const int num_blocks = (num_synapses + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    std::cout << "[CUDA] Launching applyRewardKernel with " << num_blocks << " blocks..." << std::endl;
    applyRewardKernel<<<num_blocks, THREADS_PER_BLOCK>>>(synapses, reward, dt, num_synapses);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    std::cout << "[CUDA] applyRewardKernel completed successfully" << std::endl;
    
    std::cout << "[CUDA] Launching adaptNeuromodulationKernel with " << num_blocks << " blocks..." << std::endl;
    adaptNeuromodulationKernel<<<num_blocks, THREADS_PER_BLOCK>>>(synapses, neurons, reward, num_synapses, current_time);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    std::cout << "[CUDA] adaptNeuromodulationKernel completed successfully" << std::endl;
}


// --- FIX: Wrapper function now accepts and passes current_time. ---
void run_homeostatic_mechanisms(
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    float current_time,
    int num_neurons,
    int num_synapses)
{
    const int neuron_blocks = (num_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    std::cout << "[CUDA] Launching synapticScalingKernel with " << neuron_blocks << " blocks..." << std::endl;
    // The kernel call now has the correct number of arguments.
    synapticScalingKernel<<<neuron_blocks, THREADS_PER_BLOCK>>>(neurons, synapses, num_neurons, num_synapses, current_time);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    std::cout << "[CUDA] synapticScalingKernel completed successfully" << std::endl;
    
    std::cout << "[CUDA] Launching intrinsicPlasticityKernel with " << neuron_blocks << " blocks..." << std::endl;
    intrinsicPlasticityKernel<<<neuron_blocks, THREADS_PER_BLOCK>>>(neurons, num_neurons);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    std::cout << "[CUDA] intrinsicPlasticityKernel completed successfully" << std::endl;
}

} // namespace KernelLaunchWrappers