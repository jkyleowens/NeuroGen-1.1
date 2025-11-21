#include "persistence/CheckpointReader.h"
#include "persistence/CheckpointWriter.h"
#include <cassert>
#include <filesystem>
#include <iostream>

int main() {
    namespace fs = std::filesystem;

    persistence::BrainSnapshot snapshot;
    snapshot.training_step = 123;
    snapshot.cognitive_cycles = 10;
    snapshot.tokens_processed = 2048;
    snapshot.average_reward = 0.42f;
    snapshot.time_since_consolidation_ms = 250.0f;

    persistence::ModuleSnapshot module;
    module.module_index = 0;
    module.config.module_name = "TestModule";
    module.config.num_neurons = 2;
    module.config.enable_plasticity = true;
    module.config.learning_rate = 0.01f;
    module.config.fanout_per_neuron = 4;
    module.config.num_inputs = 2;
    module.config.num_outputs = 2;
    module.config.dopamine_sensitivity = 0.5f;
    module.config.serotonin_sensitivity = 0.3f;
    module.config.inhibition_level = 0.1f;
    module.config.attention_threshold = 0.2f;
    module.config.excitability_bias = 1.0f;
    module.dopamine_level = 0.25f;
    module.serotonin_level = 0.15f;
    module.working_memory = {0.1f, 0.2f, 0.3f};

    module.neurons.resize(2);
    module.neurons[0].V = -65.0f;
    module.neurons[1].V = -64.5f;

    module.synapses.resize(1);
    module.synapses[0].weight = 0.75f;
    module.synapses[0].pre_neuron_idx = 0;
    module.synapses[0].post_neuron_idx = 1;

    snapshot.modules.push_back(module);

    persistence::ConnectionSnapshot connection;
    connection.name = "TestConnection";
    connection.source_module = "TestModule";
    connection.target_module = "TestModule";
    connection.is_excitatory = true;
    connection.plasticity_enabled = true;
    connection.current_strength = 1.25f;
    connection.gating_threshold = 0.1f;
    connection.plasticity_rate = 0.02f;
    connection.attention_modulation = 0.9f;
    connection.average_activity = 0.4f;
    connection.total_transmitted = 1.0f;
    connection.activation_count = 5;
    connection.pre_synaptic_trace = 0.3f;
    connection.post_synaptic_trace = 0.2f;
    snapshot.connections.push_back(connection);

    snapshot.optimizer_state.learning_state_blob = {0x01, 0x02, 0x03};
    snapshot.rng_state.seeds = {42, 99};

    fs::path temp_path = fs::temp_directory_path() / "neurogen_checkpoint_test.ngchk";

    persistence::CheckpointWriter writer(temp_path.string());
    bool write_ok = writer.write(snapshot);
    assert(write_ok && "Failed to write checkpoint");

    persistence::CheckpointReader reader(temp_path.string());
    auto loaded_snapshot = reader.read();
    assert(loaded_snapshot.has_value() && "Failed to read checkpoint");

    const auto& restored = *loaded_snapshot;
    assert(restored.modules.size() == snapshot.modules.size());
    assert(restored.modules[0].neurons.size() == snapshot.modules[0].neurons.size());
    assert(restored.modules[0].synapses.size() == snapshot.modules[0].synapses.size());
    assert(restored.connections.size() == snapshot.connections.size());
    assert(restored.optimizer_state.learning_state_blob == snapshot.optimizer_state.learning_state_blob);
    assert(restored.rng_state.seeds == snapshot.rng_state.seeds);

    fs::remove(temp_path);

    std::cout << "Checkpoint round-trip test passed." << std::endl;
    return 0;
}

