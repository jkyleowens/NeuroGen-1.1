#include "modules/CorticalModule.h"
#include <iostream>

CorticalModule::CorticalModule(const Config& config, int gpu_device_id) 
    : config_(config) {
    
    CUDAConfig cuda_config;
    cuda_config.device_id = gpu_device_id;
    cuda_config.enable_learning_state_gpu = config.enable_plasticity;
    
    neural_engine_ = std::make_unique<NetworkCUDA>(cuda_config);
    
    NetworkConfig net_config;
    net_config.num_neurons = config.num_neurons;
    // 20% inhibitory, 80% excitatory standard balance
    net_config.percent_inhibitory = 0.2f; 
    
    auto result = neural_engine_->initialize(net_config);
    if (!result.first) {
        std::cerr << "Failed to init module " << config_.module_name 
                  << ": " << result.second << std::endl;
    }
}

CorticalModule::~CorticalModule() {
    // NetworkCUDA destructor handles GPU cleanup
}

void CorticalModule::receiveInput(const std::vector<float>& input_vector) {
    // In the new design, we don't just "processInput", we store it
    // The update() loop handles the actual kernel execution
    // This method just updates the d_inputs_ buffer in the engine
    neural_engine_->copyInputsToGPU(input_vector); 
    // Note: You might need to expose a 'copyInputsToGPU' method in NetworkCUDA
    // or use the existing processInput logic modified to not step time immediately.
}

void CorticalModule::update(float dt_ms, float reward_signal) {
    // Step the physics of the neurons
    neural_engine_->update(dt_ms, reward_signal, 0.0f);
}

std::vector<float> CorticalModule::getOutputState() const {
    // Retrieve firing rates or voltages from the output layer
    return neural_engine_->getNeuronOutputs(); 
    // You will map specific "output neurons" in NetworkCUDA to return this vector
}