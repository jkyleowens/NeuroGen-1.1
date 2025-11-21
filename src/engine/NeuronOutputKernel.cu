#include <engine/NeuronOutputKernel.cuh>
#include <cuda_runtime.h>

__global__ void accumulateNeuronOutputsKernel(
    const GPUNeuronState* neurons,
    float* group_sums,
    int* group_counts,
    int num_neurons,
    int group_size,
    int num_outputs)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) {
        return;
    }

    int safe_group_size = (group_size > 0) ? group_size : 1;
    int group_idx = neuron_idx / safe_group_size;
    if (group_idx >= num_outputs) {
        group_idx = num_outputs - 1;
    }

    float value = neurons[neuron_idx].average_firing_rate;
    atomicAdd(&group_sums[group_idx], value);
    atomicAdd(&group_counts[group_idx], 1);
}

__global__ void finalizeNeuronOutputsKernel(
    float* group_sums,
    const int* group_counts,
    int num_outputs)
{
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= num_outputs) {
        return;
    }

    int count = group_counts[group_idx];
    if (count > 0) {
        group_sums[group_idx] /= static_cast<float>(count);
    } else {
        group_sums[group_idx] = 0.0f;
    }
}

