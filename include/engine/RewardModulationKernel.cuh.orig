#ifndef REWARD_MODULATION_KERNEL_CUH
#define REWARD_MODULATION_KERNEL_CUH

#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// Constants for reward modulation
#define BASELINE_DOPAMINE 0.4f
#define REWARD_LEARNING_RATE 0.01f
#define PREDICTION_ERROR_THRESHOLD 0.1f

/**
 * Basic reward prediction error calculation kernel
 */
__global__ void rewardPredictionErrorKernel(GPUNeuronState* neurons,
                                           float external_reward,
                                           float* predicted_reward,
                                           float* prediction_error,
                                           float* dopamine_level,
                                           float current_time,
                                           float dt,
                                           int num_neurons) {
    // Simple implementation for compatibility
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *predicted_reward = 0.0f;
        *prediction_error = external_reward - *predicted_reward;
        *dopamine_level = BASELINE_DOPAMINE + *prediction_error * 0.5f;
        
        // Clamp dopamine level
        if (*dopamine_level > 1.0f) *dopamine_level = 1.0f;
        if (*dopamine_level < 0.0f) *dopamine_level = 0.0f;
    }
}

/**
 * Apply reward modulation to synaptic plasticity
 */
__global__ void rewardModulationKernel(GPUSynapse* synapses,
                                      GPUNeuronState* neurons,
                                      float external_reward,
                                      float dopamine_level,
                                      float prediction_error,
                                      float current_time,
                                      float dt,
                                      int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    if (synapse.active == 0) return;
    
    // Modulate plasticity based on dopamine level
    float reward_factor = 1.0f + (dopamine_level - BASELINE_DOPAMINE) * 2.0f;
    
    // Apply modulation to eligibility trace
    synapse.eligibility_trace *= reward_factor;
    synapse.plasticity_modulation = reward_factor;
}

/**
 * Update dopamine sensitivity adaptation
 */
__global__ void dopamineSensitivityAdaptationKernel(GPUSynapse* synapses,
                                                   GPUNeuronState* neurons,
                                                   float average_reward,
                                                   float current_time,
                                                   float dt,
                                                   int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    if (synapse.active == 0) return;
    
    // Adapt dopamine sensitivity based on reward history
    float adaptation_rate = 0.001f * dt;
    synapse.dopamine_sensitivity += adaptation_rate * (average_reward - BASELINE_DOPAMINE);
    
    // Clamp sensitivity
    if (synapse.dopamine_sensitivity > 2.0f) synapse.dopamine_sensitivity = 2.0f;
    if (synapse.dopamine_sensitivity < 0.1f) synapse.dopamine_sensitivity = 0.1f;
}

/**
 * Update reward trace
 */
__global__ void rewardTraceUpdateKernel(float* reward_trace,
                                       float external_reward,
                                       float dt) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float decay_factor = expf(-dt / 1000.0f); // 1 second time constant
        *reward_trace = *reward_trace * decay_factor + external_reward * (1.0f - decay_factor);
    }
}

#endif // REWARD_MODULATION_KERNEL_CUH