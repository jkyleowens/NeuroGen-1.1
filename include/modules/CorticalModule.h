#pragma once
#include <vector>
#include <string>
#include <memory>
#include "engine/NetworkCUDA.cuh"

// Represents a distinct functional region of the brain
class CorticalModule {
public:
    struct Config {
        std::string module_name;
        int num_neurons;
        bool enable_plasticity;
        float learning_rate;
    };

    CorticalModule(const Config& config, int gpu_device_id);
    ~CorticalModule();

    // 1. Input Processing: Receives signals from other modules or sensors
    // Converts high-level data into current injections for input neurons
    void receiveInput(const std::vector<float>& input_vector);

    // 2. The Core Step: Advances the CUDA simulation
    void update(float dt_ms, float reward_signal);

    // 3. Output Generation: Reads firing rates from output neurons
    // Returns a vector representation of this module's current thought/state
    std::vector<float> getOutputState() const;

    // 4. Plasticity Control: Modulate learning based on global state
    void setPlasticity(bool enabled);

private:
    Config config_;
    std::unique_ptr<NetworkCUDA> neural_engine_;
    
    // Buffer for inputs to avoid re-allocation
    std::vector<float> input_buffer_;
    
    // Helper to map linear inputs to specific neuron indices
    void mapInputToNeurons(const std::vector<float>& input);
};