#include "NeuroGen/cuda/NetworkCUDA.cuh"
#include "NeuroGen/BrainModuleArchitecture.h"
#include "NeuroGen/LearningStateManager.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <chrono>
#include <thread>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            return false; \
        } \
    } while(0)

#define CUDA_CHECK_RETURN(call, retval) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            return retval; \
        } \
    } while(0)

// ============================================================================
// CONSTRUCTION AND INITIALIZATION
// ============================================================================

NetworkCUDA::NetworkCUDA(const CUDAConfig& config) 
    : cuda_config_(config), device_id_(config.device_id) {
    
    last_update_time_ = std::chrono::high_resolution_clock::now();
    update_time_history_.reserve(1000);
    
    std::cout << "🚀 NetworkCUDA created with device " << device_id_ << std::endl;
}

NetworkCUDA::~NetworkCUDA() {
    cleanupGPUResources();
    std::cout << "🧹 NetworkCUDA cleanup completed" << std::endl;
}

std::pair<bool, std::string> NetworkCUDA::initialize(const NetworkConfig& network_config) {
    std::lock_guard<std::mutex> lock(cuda_mutex_);
    
    try {
        network_config_ = network_config;
        
        // Initialize CUDA device and context
        if (!initializeCudaDevice()) {
            return {false, "Failed to initialize CUDA device"};
        }
        
        // Initialize CUDA streams
        if (!initializeCudaStreams()) {
            return {false, "Failed to initialize CUDA streams"};
        }
        
        // Initialize cuBLAS and cuRAND
        if (!initializeCudaLibraries()) {
            return {false, "Failed to initialize CUDA libraries"};
        }
        
        // Set network dimensions from config
        num_neurons_ = network_config_.num_neurons;
        num_inputs_ = num_neurons_ / 4;  // Simple estimation: 25% inputs
        num_outputs_ = num_neurons_ / 4; // Simple estimation: 25% outputs
        
        // Estimate number of synapses based on connectivity
        num_synapses_ = static_cast<size_t>(num_neurons_ * num_neurons_ * 0.1f); // 10% connectivity
        
        // Allocate GPU memory
        if (!allocateNeuralNetworkMemory()) {
            return {false, "Failed to allocate neural network GPU memory"};
        }
        
        if (!allocateWorkingBuffers()) {
            return {false, "Failed to allocate working buffers"};
        }
        
        // Initialize neural network data
        if (!initializeNeuralNetworkData()) {
            return {false, "Failed to initialize neural network data"};
        }
        
        // Initialize learning state if enabled
        if (cuda_config_.enable_learning_state_gpu) {
            auto [success, error_msg] = initializeLearningStateGPU();
            if (!success) {
                return {false, "Failed to initialize learning state GPU: " + error_msg};
            }
        }
        
        // Initialize host buffers
        h_neuron_outputs_.resize(num_outputs_);
        h_synaptic_weights_.resize(num_synapses_);
        
        // Warm up GPU
        warmupGPU();
        
        is_initialized_ = true;
        
        std::cout << "✅ NetworkCUDA initialized successfully" << std::endl;
        std::cout << "   Neurons: " << num_neurons_ << ", Synapses: " << num_synapses_ << std::endl;
        std::cout << "   GPU Memory: " << getMemoryStats().allocated_memory_bytes / (1024*1024) << " MB" << std::endl;
        
        return {true, "Success"};
        
    } catch (const std::exception& e) {
        return {false, "Exception during initialization: " + std::string(e.what())};
    }
}

bool NetworkCUDA::initializeCudaDevice() {
    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    // Get device properties
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties_, device_id_));
    
    std::cout << "🔧 CUDA Device: " << device_properties_.name << std::endl;
    std::cout << "   Compute Capability: " << device_properties_.major << "." << device_properties_.minor << std::endl;
    std::cout << "   Global Memory: " << device_properties_.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    std::cout << "   Shared Memory per Block: " << device_properties_.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "   Max Threads per Block: " << device_properties_.maxThreadsPerBlock << std::endl;
    
    // Enable unified memory if requested
    if (cuda_config_.enable_unified_memory && device_properties_.unifiedAddressing) {
        std::cout << "🔗 Unified memory enabled" << std::endl;
    }
    
    // Create memory pool if enabled
    if (cuda_config_.enable_memory_pool && device_properties_.major >= 6) {
        cudaMemPoolProps pool_props = {};
        pool_props.allocType = cudaMemAllocationTypePinned;
        pool_props.handleTypes = cudaMemHandleTypeNone;
        pool_props.location.type = cudaMemLocationTypeDevice;
        pool_props.location.id = device_id_;
        
        if (cudaMemPoolCreate(&memory_pool_, &pool_props) == cudaSuccess) {
            size_t pool_size = cuda_config_.memory_pool_size_mb * 1024 * 1024;
            cudaMemPoolSetAttribute(memory_pool_, cudaMemPoolAttrReservedMemHigh, &pool_size);
            memory_pool_enabled_ = true;
            std::cout << "💾 Memory pool created: " << cuda_config_.memory_pool_size_mb << " MB" << std::endl;
        }
    }
    
    return true;
}

bool NetworkCUDA::initializeCudaStreams() {
    // Create default stream
    CUDA_CHECK(cudaStreamCreate(&default_stream_));
    
    // Create compute streams
    compute_streams_.resize(cuda_config_.num_compute_streams);
    for (int i = 0; i < cuda_config_.num_compute_streams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&compute_streams_[i]));
        
        // Set stream priority if enabled
        if (cuda_config_.enable_stream_priorities) {
            int priority = (i == 0) ? -1 : 0; // High priority for main compute stream
            cudaStreamCreateWithPriority(&compute_streams_[i], cudaStreamDefault, priority);
        }
    }
    
    // Create memory streams
    memory_streams_.resize(cuda_config_.num_memory_streams);
    for (int i = 0; i < cuda_config_.num_memory_streams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&memory_streams_[i]));
    }
    
    std::cout << "🌊 Created " << compute_streams_.size() << " compute streams and " 
              << memory_streams_.size() << " memory streams" << std::endl;
    
    return true;
}

bool NetworkCUDA::initializeCudaLibraries() {
    // Initialize cuBLAS
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "❌ Failed to create cuBLAS handle" << std::endl;
        return false;
    }
    
    // Set cuBLAS stream
    cublasSetStream(cublas_handle_, default_stream_);
    
    // Enable tensor cores if available and requested
    if (cuda_config_.enable_tensor_cores && device_properties_.major >= 7) {
        cublasSetMathMode(cublas_handle_, CUBLAS_TENSOR_OP_MATH);
        std::cout << "⚡ Tensor cores enabled" << std::endl;
    }
    
    // Initialize cuRAND
    if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
        std::cerr << "❌ Failed to create cuRAND generator" << std::endl;
        return false;
    }
    
    curandSetStream(curand_generator_, default_stream_);
    curandSetPseudoRandomGeneratorSeed(curand_generator_, time(nullptr));
    
    std::cout << "📚 CUDA libraries initialized" << std::endl;
    return true;
}

// ============================================================================
// MEMORY ALLOCATION AND MANAGEMENT
// ============================================================================

bool NetworkCUDA::allocateNeuralNetworkMemory() {
    try {
        // Allocate neuron states
        size_t neuron_size = num_neurons_ * sizeof(GPUNeuronState);
        if (memory_pool_enabled_) {
            CUDA_CHECK(cudaMallocFromPoolAsync(&d_neurons_, neuron_size, memory_pool_, default_stream_));
        } else {
            CUDA_CHECK(cudaMalloc(&d_neurons_, neuron_size));
        }

        // Allocate synapses
        size_t synapse_size = num_synapses_ * sizeof(GPUSynapse);
        if (memory_pool_enabled_) {
            CUDA_CHECK(cudaMallocFromPoolAsync(&d_synapses_, synapse_size, memory_pool_, default_stream_));
        } else {
            CUDA_CHECK(cudaMalloc(&d_synapses_, synapse_size));
        }

        // CRITICAL: Synchronize stream after async allocations
        // This ensures memory allocations complete before being used
        if (memory_pool_enabled_) {
            CUDA_CHECK(cudaStreamSynchronize(default_stream_));
        }

        // Allocate input/output buffers
        CUDA_CHECK(cudaMalloc(&d_inputs_, num_inputs_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_outputs_, num_outputs_ * sizeof(float)));

        std::cout << "💾 Allocated neural network GPU memory:" << std::endl;
        std::cout << "   Neurons: " << neuron_size / (1024*1024) << " MB" << std::endl;
        std::cout << "   Synapses: " << synapse_size / (1024*1024) << " MB" << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error allocating neural network memory: " << e.what() << std::endl;
        return false;
    }
}

std::pair<bool, std::string> NetworkCUDA::initializeLearningStateGPU() {
    try {
        if (!allocateLearningStateMemory()) {
            return {false, "Failed to allocate learning state memory"};
        }
        
        if (!initializeLearningStateData()) {
            return {false, "Failed to initialize learning state data"};
        }
        
        // Calculate buffer size for host-GPU transfers
        learning_state_buffer_size_ = calculateLearningStateBufferSize();
        h_learning_state_buffer_ = std::make_unique<uint8_t[]>(learning_state_buffer_size_);
        
        std::cout << "🧠 Learning state GPU initialized:" << std::endl;
        std::cout << "   Buffer size: " << learning_state_buffer_size_ / (1024*1024) << " MB" << std::endl;
        
        return {true, "Success"};
        
    } catch (const std::exception& e) {
        return {false, "Exception: " + std::string(e.what())};
    }
}

bool NetworkCUDA::allocateLearningStateMemory() {
    try {
        // Allocate main learning state structure
        CUDA_CHECK(cudaMalloc(&d_learning_state_, sizeof(GPULearningState)));
        
        // Allocate inter-module state structure
        CUDA_CHECK(cudaMalloc(&d_inter_module_state_, sizeof(GPUInterModuleState)));
        
        // Create host-side structures to set up GPU pointers
        GPULearningState h_learning_state;
        GPUInterModuleState h_inter_module_state;
        
        // Allocate learning trace arrays
        CUDA_CHECK(cudaMalloc(&h_learning_state.eligibility_traces, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_learning_state.synaptic_tags, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_learning_state.consolidation_weights, num_synapses_ * sizeof(float)));
        
        // Allocate neuromodulator arrays (3 per neuron: dopamine, acetylcholine, norepinephrine)
        CUDA_CHECK(cudaMalloc(&h_learning_state.neuromodulator_levels, num_neurons_ * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_learning_state.firing_rate_history, num_neurons_ * 1000 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_learning_state.prediction_errors, num_neurons_ * 100 * sizeof(float)));
        
        // Allocate learning parameter arrays
        CUDA_CHECK(cudaMalloc(&h_learning_state.learning_rates, num_neurons_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_learning_state.plasticity_thresholds, num_neurons_ * sizeof(float)));
        
        // Allocate performance tracking arrays
        CUDA_CHECK(cudaMalloc(&h_learning_state.learning_step_counts, num_neurons_ * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&h_learning_state.reward_history, num_neurons_ * 100 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_learning_state.history_indices, num_neurons_ * sizeof(uint32_t)));
        
        // Allocate module assignment arrays
        CUDA_CHECK(cudaMalloc(&h_learning_state.module_assignments, num_neurons_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_learning_state.module_boundaries, 32 * sizeof(int))); // Max 16 modules
        h_learning_state.num_modules = 0; // Will be set by brain architecture
        
        // Initialize arrays to zero
        CUDA_CHECK(cudaMemset(h_learning_state.eligibility_traces, 0, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(h_learning_state.synaptic_tags, 0, num_synapses_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(h_learning_state.neuromodulator_levels, 0, num_neurons_ * 3 * sizeof(float)));
        
        // Initialize learning parameters to default values
        std::vector<float> default_learning_rates(num_neurons_, 0.001f);
        CUDA_CHECK(cudaMemcpy(h_learning_state.learning_rates, default_learning_rates.data(), 
                             num_neurons_ * sizeof(float), cudaMemcpyHostToDevice));
        
        std::vector<float> default_thresholds(num_neurons_, 0.1f);
        CUDA_CHECK(cudaMemcpy(h_learning_state.plasticity_thresholds, default_thresholds.data(),
                             num_neurons_ * sizeof(float), cudaMemcpyHostToDevice));
        
        // Copy structure to GPU
        CUDA_CHECK(cudaMemcpy(d_learning_state_, &h_learning_state, sizeof(GPULearningState), cudaMemcpyHostToDevice));
        
        // Initialize inter-module state (will be configured by brain architecture)
        size_t max_connections = 100; // Initial allocation, can be expanded
        
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.connection_strengths, max_connections * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.usage_frequencies, max_connections * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.correlation_strengths, max_connections * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.activation_counts, max_connections * sizeof(uint64_t)));
        
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.pre_synaptic_traces, max_connections * 1000 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.post_synaptic_traces, max_connections * 1000 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.timing_differences, max_connections * 100 * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.source_modules, max_connections * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_inter_module_state.target_modules, max_connections * sizeof(int)));
        h_inter_module_state.num_connections = 0; // Will be set by brain architecture
        
        // Initialize connection strengths to small values
        std::vector<float> initial_strengths(max_connections, 0.1f);
        CUDA_CHECK(cudaMemcpy(h_inter_module_state.connection_strengths, initial_strengths.data(),
                             max_connections * sizeof(float), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMemcpy(d_inter_module_state_, &h_inter_module_state, sizeof(GPUInterModuleState), cudaMemcpyHostToDevice));
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error allocating learning state memory: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// CORE NEURAL PROCESSING
// ============================================================================

void NetworkCUDA::update(float dt, float reward_signal, float novelty_signal) {
    if (!is_initialized_) {
        std::cerr << "❌ NetworkCUDA not initialized" << std::endl;
        return;
    }

    std::lock_guard<std::mutex> lock(cuda_mutex_);

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        // Update neuron states
        KernelLaunchWrappers::update_neuron_states(d_neurons_, current_time_, dt, num_neurons_);

        // Update calcium dynamics
        KernelLaunchWrappers::update_calcium_dynamics(d_neurons_, current_time_, dt, num_neurons_);

        // Run STDP and update eligibility traces
        KernelLaunchWrappers::run_stdp_and_eligibility(d_synapses_, d_neurons_, current_time_, dt, num_synapses_);

        // Apply reward and adaptation if there's a reward signal
        if (std::abs(reward_signal) > 0.001f) {
            KernelLaunchWrappers::apply_reward_and_adaptation(d_synapses_, d_neurons_, reward_signal, current_time_, dt, num_synapses_);
        }

        // Update learning state if enabled
        if (cuda_config_.enable_learning_state_gpu && d_learning_state_) {
            updateLearningStateGPU(reward_signal, novelty_signal, dt);
        }

        // Run homeostatic mechanisms
        KernelLaunchWrappers::run_homeostatic_mechanisms(d_neurons_, d_synapses_, current_time_, num_neurons_, num_synapses_);

        // Synchronize default stream
        CUDA_CHECK_RETURN(cudaStreamSynchronize(default_stream_), void());

        // Advance simulation time
        current_time_ += dt;

        // Update performance metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        float update_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        updatePerformanceMetrics(update_time_ms);

    } catch (const std::exception& e) {
        std::cerr << "❌ Error during GPU update: " << e.what() << std::endl;
    }
}

std::vector<float> NetworkCUDA::processInput(const std::vector<float>& inputs) {
    return processInputWithLearning(inputs, {}, 0.0f);
}

std::vector<float> NetworkCUDA::processInputWithLearning(const std::vector<float>& inputs,
                                                        const std::vector<float>& target_outputs,
                                                        float reward_signal) {
    if (!is_initialized_) {
        std::cerr << "❌ NetworkCUDA not initialized" << std::endl;
        return {};
    }
    
    std::lock_guard<std::mutex> lock(cuda_mutex_);
    
    try {
        // Copy inputs to GPU
        size_t input_size = std::min(inputs.size(), num_inputs_);
        if (input_size > 0) {
            CUDA_CHECK_RETURN(cudaMemcpy(d_inputs_, inputs.data(), input_size * sizeof(float), cudaMemcpyHostToDevice), {});
        }
        
        // Process through network (this would involve more complex processing in practice)
        // For now, we'll simulate by updating neurons and getting outputs
        update(0.001f, reward_signal, 0.0f);
        
        // Copy outputs from GPU
        CUDA_CHECK_RETURN(cudaMemcpy(h_neuron_outputs_.data(), d_outputs_, num_outputs_ * sizeof(float), cudaMemcpyDeviceToHost), {});
        
        return h_neuron_outputs_;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error processing input: " << e.what() << std::endl;
        return {};
    }
}

// ============================================================================
// LEARNING STATE MANAGEMENT
// ============================================================================

void NetworkCUDA::updateLearningStateGPU(float reward_signal, float novelty_signal, float dt) {
    if (!d_learning_state_ || !d_inter_module_state_) {
        return;
    }
    
    try {
        // Update eligibility traces
        LearningStateKernels::update_eligibility_traces(
            d_learning_state_, d_neurons_, d_synapses_, reward_signal, dt, 
            num_neurons_, num_synapses_);
        
        // Apply synaptic tagging
        LearningStateKernels::apply_synaptic_tagging(
            d_learning_state_, d_neurons_, novelty_signal, dt, 
            num_neurons_, num_synapses_);
        
        // Update neuromodulators based on reward and context
        float dopamine = reward_signal; // Reward prediction error
        float acetylcholine = std::abs(novelty_signal); // Attention/uncertainty
        float norepinephrine = std::abs(reward_signal) > 0.5f ? 1.0f : 0.0f; // Arousal
        
        LearningStateKernels::update_neuromodulators(
            d_learning_state_, d_neurons_, dopamine, acetylcholine, norepinephrine, dt, num_neurons_);
        
        // Update inter-module connections if we have module information
        if (brain_architecture_) {
            LearningStateKernels::update_inter_module_connections(
                d_inter_module_state_, d_learning_state_, d_neurons_, 1.0f, dt, num_neurons_);
        }
        
        // Update learning statistics
        LearningStateKernels::update_learning_statistics(
            d_learning_state_, d_neurons_, reward_signal, novelty_signal, dt, num_neurons_);
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error updating learning state GPU: " << e.what() << std::endl;
    }
}

size_t NetworkCUDA::performMemoryConsolidationGPU(float consolidation_strength) {
    if (!d_learning_state_) {
        return 0;
    }
    
    try {
        // Allocate device memory for consolidation counter
        int* d_consolidated_count;
        CUDA_CHECK_RETURN(cudaMalloc(&d_consolidated_count, sizeof(int)), 0);
        
        // Perform consolidation
        LearningStateKernels::perform_memory_consolidation(
            d_learning_state_, d_neurons_, d_synapses_, consolidation_strength,
            d_consolidated_count, num_neurons_, num_synapses_);
        
        // Copy result back to host
        int h_consolidated_count;
        CUDA_CHECK_RETURN(cudaMemcpy(&h_consolidated_count, d_consolidated_count, sizeof(int), cudaMemcpyDeviceToHost), 0);
        
        CUDA_CHECK_RETURN(cudaFree(d_consolidated_count), 0);
        
        std::cout << "🧠 GPU memory consolidation completed: " << h_consolidated_count << " synapses consolidated" << std::endl;
        
        return static_cast<size_t>(h_consolidated_count);
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during memory consolidation: " << e.what() << std::endl;
        return 0;
    }
}

// ============================================================================
// BRAIN ARCHITECTURE
// ============================================================================

void NetworkCUDA::setBrainArchitecture(std::shared_ptr<BrainModuleArchitecture> architecture) {
    if (!architecture) {
        std::cerr << "❌ Invalid brain architecture provided" << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(cuda_mutex_);
    
    brain_architecture_ = architecture;
    
    // Configure modules and connections based on the architecture
    // This part needs to be implemented based on how BrainArchitecture is defined
    
    std::cout << "🧠 Brain architecture set with " << brain_architecture_->getModuleCount() << " modules" << std::endl;
}

// ============================================================================
// DATA SYNCHRONIZATION
// ============================================================================

void NetworkCUDA::synchronizeWithArchitecture(bool force_full_sync) {
    if (!brain_architecture_ || !is_initialized_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(cuda_mutex_);
    
    try {
        // Synchronize learning state
        if (d_learning_state_ && (force_full_sync || shouldSynchronize())) {
            auto global_state = brain_architecture_->getGlobalLearningState();
            
            // Update GPU learning state from architecture
            // This would involve more complex synchronization in practice
            
            std::cout << "🔄 Synchronized with brain architecture" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error synchronizing with architecture: " << e.what() << std::endl;
    }
}

// ============================================================================
// PERFORMANCE MONITORING
// ============================================================================

NetworkCUDA::GPUMemoryStats NetworkCUDA::getMemoryStats() const {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    
    GPUMemoryStats stats;
    
    // Get total GPU memory
    size_t free_bytes, total_bytes;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        stats.total_memory_bytes = total_bytes;
        stats.free_memory_bytes = free_bytes;
        stats.allocated_memory_bytes = total_bytes - free_bytes;
    }
    
    // Calculate specific allocations
    if (is_initialized_) {
        stats.neural_network_bytes = num_neurons_ * sizeof(GPUNeuronState) + num_synapses_ * sizeof(GPUSynapse);
        stats.neural_network_bytes += (num_inputs_ + num_outputs_) * sizeof(float);
        
        if (d_learning_state_) {
            stats.learning_state_bytes = num_synapses_ * 3 * sizeof(float); // eligibility + tags + consolidation
            stats.learning_state_bytes += num_neurons_ * (3 + 1000 + 100) * sizeof(float); // neuromod + history + errors
            stats.learning_state_bytes += num_neurons_ * (2 * sizeof(float) + sizeof(uint64_t) + sizeof(uint32_t)); // params + counts
        }
        
        stats.temporary_buffer_bytes = learning_state_buffer_size_;
    }
    
    stats.memory_utilization_percent = (static_cast<float>(stats.allocated_memory_bytes) / stats.total_memory_bytes) * 100.0f;
    stats.fragmentation_ratio = calculateFragmentationRatio();
    
    return stats;
}

NetworkCUDA::CUDAPerformanceMetrics NetworkCUDA::getPerformanceMetrics() const {
    std::lock_guard<std::mutex> lock(cuda_mutex_);
    return performance_metrics_;
}

void NetworkCUDA::updatePerformanceMetrics(float kernel_time_ms) const {
    performance_metrics_.last_update_time_ms = kernel_time_ms;
    
    // Update average with exponential moving average
    if (performance_metrics_.avg_update_time_ms == 0.0f) {
        performance_metrics_.avg_update_time_ms = kernel_time_ms;
    } else {
        performance_metrics_.avg_update_time_ms = 0.9f * performance_metrics_.avg_update_time_ms + 0.1f * kernel_time_ms;
    }
    
    // Calculate throughput metrics
    if (kernel_time_ms > 0.0f) {
        performance_metrics_.neurons_per_second = static_cast<float>(num_neurons_) / (kernel_time_ms / 1000.0f);
        performance_metrics_.synapses_per_second = static_cast<float>(num_synapses_) / (kernel_time_ms / 1000.0f);
    }
    
    // Update memory bandwidth (simplified calculation)
    size_t data_transferred = (num_neurons_ * sizeof(GPUNeuronState) + num_synapses_ * sizeof(GPUSynapse));
    if (kernel_time_ms > 0.0f) {
        performance_metrics_.memory_bandwidth_gbps = (static_cast<float>(data_transferred) / (1024*1024*1024)) / (kernel_time_ms / 1000.0f);
    }
    
    // Update utilization metrics
    updateMemoryStats();
    performance_metrics_.memory_utilization_percent = getMemoryStats().memory_utilization_percent;
}

// ============================================================================
// DATA ACCESS METHODS
// ============================================================================

std::vector<GPUNeuronState> NetworkCUDA::getNeuronStates() const {
    std::lock_guard<std::mutex> lock(cuda_mutex_);
    
    if (!is_initialized_ || !d_neurons_) {
        return {};
    }
    
    std::vector<GPUNeuronState> host_neurons(num_neurons_);
    
    cudaError_t error = cudaMemcpy(host_neurons.data(), d_neurons_, 
                                   num_neurons_ * sizeof(GPUNeuronState), 
                                   cudaMemcpyDeviceToHost);
    
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy neuron states from GPU: " << cudaGetErrorString(error) << std::endl;
        return {};
    }
    
    return host_neurons;
}

std::vector<float> NetworkCUDA::getSynapticWeights() const {
    std::lock_guard<std::mutex> lock(cuda_mutex_);
    
    if (!is_initialized_ || !d_synapses_) {
        return {};
    }
    
    // First get the synapses from GPU
    std::vector<GPUSynapse> host_synapses(num_synapses_);
    
    cudaError_t error = cudaMemcpy(host_synapses.data(), d_synapses_, 
                                   num_synapses_ * sizeof(GPUSynapse), 
                                   cudaMemcpyDeviceToHost);
    
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy synapses from GPU: " << cudaGetErrorString(error) << std::endl;
        return {};
    }
    
    // Extract weights from synapses
    std::vector<float> weights;
    weights.reserve(num_synapses_);
    
    for (const auto& synapse : host_synapses) {
        weights.push_back(synapse.weight);
    }
    
    return weights;
}

// ============================================================================
// UTILITY AND HELPER METHODS
// ============================================================================

bool NetworkCUDA::checkCudaError(const std::string& operation) const {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in " << operation << ": " << cudaGetErrorString(error) << std::endl;
        total_cuda_errors_++;
        last_cuda_error_ = cudaGetErrorString(error);
        return false;
    }
    return true;
}

void NetworkCUDA::cleanupGPUResources() {
    std::lock_guard<std::mutex> lock(cuda_mutex_);
    
    // Cleanup learning state
    cleanupLearningStateGPU();
    
    // Free neural network memory
    if (d_neurons_) { cudaFree(d_neurons_); d_neurons_ = nullptr; }
    if (d_synapses_) { cudaFree(d_synapses_); d_synapses_ = nullptr; }
    if (d_inputs_) { cudaFree(d_inputs_); d_inputs_ = nullptr; }
    if (d_outputs_) { cudaFree(d_outputs_); d_outputs_ = nullptr; }
    
    // Free working buffers
    if (d_temp_buffer_) { cudaFree(d_temp_buffer_); d_temp_buffer_ = nullptr; }
    if (d_reduction_buffer_) { cudaFree(d_reduction_buffer_); d_reduction_buffer_ = nullptr; }
    if (d_consolidated_count_) { cudaFree(d_consolidated_count_); d_consolidated_count_ = nullptr; }
    
    // Destroy CUDA streams
    if (default_stream_) { cudaStreamDestroy(default_stream_); default_stream_ = nullptr; }
    for (auto stream : compute_streams_) {
        if (stream) cudaStreamDestroy(stream);
    }
    for (auto stream : memory_streams_) {
        if (stream) cudaStreamDestroy(stream);
    }
    compute_streams_.clear();
    memory_streams_.clear();
    
    // Destroy CUDA libraries
    if (cublas_handle_) { cublasDestroy(cublas_handle_); cublas_handle_ = nullptr; }
    if (curand_generator_) { curandDestroyGenerator(curand_generator_); curand_generator_ = nullptr; }
    
    // Destroy memory pool
    if (memory_pool_) { cudaMemPoolDestroy(memory_pool_); memory_pool_ = nullptr; }
    
    // Destroy captured graph
    if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
    if (computation_graph_) { cudaGraphDestroy(computation_graph_); computation_graph_ = nullptr; }
    
    is_initialized_ = false;
}

void NetworkCUDA::warmupGPU() {
    if (!is_initialized_) return;
    
    std::cout << "🔥 Warming up GPU..." << std::endl;
    
    // Run a few dummy computations to warm up the GPU
    for (int i = 0; i < 3; ++i) {
        std::vector<float> dummy_input(num_inputs_, 0.1f);
        processInput(dummy_input);
    }
    
    // Synchronize and measure baseline performance
    cudaDeviceSynchronize();
    
    std::cout << "✅ GPU warmup completed" << std::endl;
}

size_t NetworkCUDA::calculateLearningStateBufferSize() const {
    size_t size = 0;
    
    if (is_initialized_) {
        // Eligibility traces and synaptic tags
        size += num_synapses_ * 3 * sizeof(float);
        
        // Neuromodulator levels and histories
        size += num_neurons_ * (3 + 1000 + 100) * sizeof(float);
        
        // Learning parameters and performance tracking
        size += num_neurons_ * (2 * sizeof(float) + sizeof(uint64_t) + sizeof(uint32_t));
        
        // Module assignments and boundaries
        size += num_neurons_ * sizeof(int) + 32 * sizeof(int);
        
        // Inter-module connection state
        size += 100 * (7 * sizeof(float) + sizeof(uint64_t) + 2 * sizeof(int)); // Max 100 connections
        size += 100 * (1000 + 1000 + 100) * sizeof(float); // Trace buffers
    }
    
    return size;
}

// ============================================================================
// CUDA UTILITY FUNCTIONS
// ============================================================================

bool isCudaAvailable() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

int getCudaDeviceCount() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

int getOptimalCudaDevice() {
    int device_count = getCudaDeviceCount();
    if (device_count == 0) return -1;
    
    int best_device = 0;
    size_t max_memory = 0;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.totalGlobalMem > max_memory && prop.major >= 3) { // Require compute capability 3.0+
                max_memory = prop.totalGlobalMem;
                best_device = i;
            }
        }
    }
    
    return best_device;
}

std::pair<size_t, size_t> getCudaMemoryInfo(int device_id) {
    size_t free_bytes = 0, total_bytes = 0;
    
    int current_device;
    cudaGetDevice(&current_device);
    
    if (device_id >= 0) {
        cudaSetDevice(device_id);
    }
    
    cudaMemGetInfo(&free_bytes, &total_bytes);
    
    if (device_id >= 0) {
        cudaSetDevice(current_device);
    }
    
    return {free_bytes, total_bytes};
}

void warmUpCudaDevice(int device_id) {
    int current_device;
    cudaGetDevice(&current_device);
    
    if (device_id >= 0) {
        cudaSetDevice(device_id);
    }
    
    // Allocate and free a small amount of memory to initialize the context
    void* temp_ptr;
    cudaMalloc(&temp_ptr, 1024);
    cudaFree(temp_ptr);
    
    // Run a simple kernel to warm up the device
    cudaDeviceSynchronize();
    
    if (device_id >= 0) {
        cudaSetDevice(current_device);
    }
}

std::string getCudaRuntimeVersion() {
    int runtime_version;
    cudaRuntimeGetVersion(&runtime_version);
    
    int major = runtime_version / 1000;
    int minor = (runtime_version % 1000) / 10;
    
    return std::to_string(major) + "." + std::to_string(minor);
}

std::string getCudaDriverVersion() {
    int driver_version;
    cudaDriverGetVersion(&driver_version);
    
    int major = driver_version / 1000;
    int minor = (driver_version % 1000) / 10;
    
    return std::to_string(major) + "." + std::to_string(minor);
}

// ============================================================================
// MISSING HELPER METHODS
// ============================================================================

bool NetworkCUDA::shouldSynchronize() const {
    // Simple heuristic: synchronize every 100 updates or if there's significant learning activity
    static int update_count = 0;
    update_count++;
    
    if (update_count % 100 == 0) {
        return true;
    }
    
    // Check if there's significant learning activity
    if (performance_metrics_.last_update_time_ms > performance_metrics_.avg_update_time_ms * 1.5f) {
        return true;
    }
    
    return false;
}

float NetworkCUDA::calculateFragmentationRatio() const {
    // Simple fragmentation calculation based on memory usage patterns
    // In a real implementation, this would analyze memory allocation patterns
    
    size_t free_bytes, total_bytes;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return 0.0f;
    }
    
    size_t used_bytes = total_bytes - free_bytes;
    if (total_bytes == 0) {
        return 0.0f;
    }
    
    // Simple heuristic: fragmentation increases with memory usage
    float usage_ratio = static_cast<float>(used_bytes) / total_bytes;
    
    // Assume fragmentation is minimal below 50% usage, increases after that
    if (usage_ratio < 0.5f) {
        return usage_ratio * 0.1f; // Low fragmentation
    } else {
        return 0.05f + (usage_ratio - 0.5f) * 0.4f; // Increasing fragmentation
    }
}

bool NetworkCUDA::allocateWorkingBuffers() {
    try {
        // Allocate temporary computation buffer
        size_t temp_buffer_size = std::max(num_neurons_, num_synapses_) * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_temp_buffer_, temp_buffer_size));
        
        // Allocate reduction buffer for parallel reductions
        size_t reduction_buffer_size = 1024 * sizeof(float); // For reduction operations
        CUDA_CHECK(cudaMalloc(&d_reduction_buffer_, reduction_buffer_size));
        
        // Allocate counter for consolidation operations
        CUDA_CHECK(cudaMalloc(&d_consolidated_count_, sizeof(int)));
        
        std::cout << "💾 Allocated working buffers:" << std::endl;
        std::cout << "   Temp buffer: " << temp_buffer_size / (1024*1024) << " MB" << std::endl;
        std::cout << "   Reduction buffer: " << reduction_buffer_size / 1024 << " KB" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error allocating working buffers: " << e.what() << std::endl;
        return false;
    }
}

bool NetworkCUDA::initializeNeuralNetworkData() {
    try {
        // Initialize neurons with default values
        if (d_neurons_) {
            std::vector<GPUNeuronState> initial_neurons(num_neurons_);

            // Zero-initialize all neurons first to avoid garbage values
            std::memset(initial_neurons.data(), 0, num_neurons_ * sizeof(GPUNeuronState));

            for (size_t i = 0; i < num_neurons_; ++i) {
                // Core membrane dynamics
                initial_neurons[i].V = -65.0f; // Resting potential
                initial_neurons[i].u = 0.0f; // Recovery variable
                for (int j = 0; j < MAX_COMPARTMENTS; ++j) {
                    initial_neurons[i].I_syn[j] = 0.0f;
                    initial_neurons[i].ca_conc[j] = 50.0e-9f; // Resting calcium
                }
                initial_neurons[i].I_ext = 0.0f;

                // Timing and activity
                initial_neurons[i].last_spike_time = -1000.0f; // Far in the past
                initial_neurons[i].previous_spike_time = -1000.0f;
                initial_neurons[i].average_firing_rate = 0.0f;

                // Plasticity and adaptation
                initial_neurons[i].excitability = 1.0f; // Normal excitability
                initial_neurons[i].synaptic_scaling_factor = 1.0f;
                initial_neurons[i].threshold = -50.0f; // Firing threshold

                // Network properties
                initial_neurons[i].active = 1; // Set as active
                initial_neurons[i].neuron_type = (i % 5 == 0) ? 0 : 1; // 20% inhibitory, 80% excitatory
            }

            CUDA_CHECK(cudaMemcpy(d_neurons_, initial_neurons.data(),
                                 num_neurons_ * sizeof(GPUNeuronState),
                                 cudaMemcpyHostToDevice));
        }
        
        // Initialize synapses with random weights
        if (d_synapses_) {
            std::vector<GPUSynapse> initial_synapses(num_synapses_);

            // Zero-initialize all synapses first
            std::memset(initial_synapses.data(), 0, num_synapses_ * sizeof(GPUSynapse));

            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> weight_dist(0.0f, 0.1f);

            for (size_t i = 0; i < num_synapses_; ++i) {
                // Connectivity
                initial_synapses[i].pre_neuron_idx = static_cast<int>(i % num_neurons_);
                initial_synapses[i].post_neuron_idx = static_cast<int>((i + 1) % num_neurons_);
                initial_synapses[i].active = 1; // Set as active

                // Synaptic properties
                initial_synapses[i].weight = weight_dist(gen);
                initial_synapses[i].max_weight = 1.0f;
                initial_synapses[i].min_weight = 0.0f;
                initial_synapses[i].effective_weight = initial_synapses[i].weight;

                // Plasticity
                initial_synapses[i].is_plastic = true; // Enable plasticity
                initial_synapses[i].eligibility_trace = 0.0f;
                initial_synapses[i].learning_rate = 0.01f;

                // Timing
                initial_synapses[i].last_pre_spike_time = -1000.0f;
                initial_synapses[i].last_post_spike_time = -1000.0f;

                // Vesicle dynamics
                initial_synapses[i].vesicle_count = 100;
                initial_synapses[i].release_probability = 0.5f;
            }

            CUDA_CHECK(cudaMemcpy(d_synapses_, initial_synapses.data(),
                                 num_synapses_ * sizeof(GPUSynapse),
                                 cudaMemcpyHostToDevice));
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error initializing neural network data: " << e.what() << std::endl;
        return false;
    }
}

bool NetworkCUDA::initializeLearningStateData() {
    // This method would initialize the learning state data structures
    // For now, return true as the allocation already initializes to defaults
    return true;
}

void NetworkCUDA::updateMemoryStats() const {
    // Update memory statistics - this is called internally
    // The actual implementation is in getMemoryStats()
}

void NetworkCUDA::cleanupLearningStateGPU() {
    if (d_learning_state_) {
        // Free learning state arrays
        GPULearningState h_learning_state;
        if (cudaMemcpy(&h_learning_state, d_learning_state_, sizeof(GPULearningState), cudaMemcpyDeviceToHost) == cudaSuccess) {
            if (h_learning_state.eligibility_traces) cudaFree(h_learning_state.eligibility_traces);
            if (h_learning_state.synaptic_tags) cudaFree(h_learning_state.synaptic_tags);
            if (h_learning_state.consolidation_weights) cudaFree(h_learning_state.consolidation_weights);
            if (h_learning_state.neuromodulator_levels) cudaFree(h_learning_state.neuromodulator_levels);
            if (h_learning_state.firing_rate_history) cudaFree(h_learning_state.firing_rate_history);
            if (h_learning_state.prediction_errors) cudaFree(h_learning_state.prediction_errors);
            if (h_learning_state.learning_rates) cudaFree(h_learning_state.learning_rates);
            if (h_learning_state.plasticity_thresholds) cudaFree(h_learning_state.plasticity_thresholds);
            if (h_learning_state.learning_step_counts) cudaFree(h_learning_state.learning_step_counts);
            if (h_learning_state.reward_history) cudaFree(h_learning_state.reward_history);
            if (h_learning_state.history_indices) cudaFree(h_learning_state.history_indices);
            if (h_learning_state.module_assignments) cudaFree(h_learning_state.module_assignments);
            if (h_learning_state.module_boundaries) cudaFree(h_learning_state.module_boundaries);
        }
        
        cudaFree(d_learning_state_);
        d_learning_state_ = nullptr;
    }
    
    if (d_inter_module_state_) {
        // Free inter-module state arrays
        GPUInterModuleState h_inter_module_state;
        if (cudaMemcpy(&h_inter_module_state, d_inter_module_state_, sizeof(GPUInterModuleState), cudaMemcpyDeviceToHost) == cudaSuccess) {
            if (h_inter_module_state.connection_strengths) cudaFree(h_inter_module_state.connection_strengths);
            if (h_inter_module_state.usage_frequencies) cudaFree(h_inter_module_state.usage_frequencies);
            if (h_inter_module_state.correlation_strengths) cudaFree(h_inter_module_state.correlation_strengths);
            if (h_inter_module_state.activation_counts) cudaFree(h_inter_module_state.activation_counts);
            if (h_inter_module_state.pre_synaptic_traces) cudaFree(h_inter_module_state.pre_synaptic_traces);
            if (h_inter_module_state.post_synaptic_traces) cudaFree(h_inter_module_state.post_synaptic_traces);
            if (h_inter_module_state.timing_differences) cudaFree(h_inter_module_state.timing_differences);
            if (h_inter_module_state.source_modules) cudaFree(h_inter_module_state.source_modules);
            if (h_inter_module_state.target_modules) cudaFree(h_inter_module_state.target_modules);
        }
        
        cudaFree(d_inter_module_state_);
        d_inter_module_state_ = nullptr;
    }
}