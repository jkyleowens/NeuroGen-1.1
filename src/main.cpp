#include <iostream>
#include <memory>
#include <string>
#include "modules/BrainOrchestrator.h"
#include "interfaces/TokenEmbedding.h"
#include "interfaces/OutputDecoder.h"
#include "interfaces/TrainingLoop.h"

void printBanner() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     NeuroGen Modular Brain Architecture v1.1                  ║
║     Biologically-Inspired Neural Language Model               ║
║                                                               ║
║     🧠 6 Specialized Brain Modules                            ║
║     🔗 Dynamic Inter-Module Connections                       ║
║     💭 Cognitive Cycle Processing                             ║
║     🎓 Continual Learning System                              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
)" << std::endl;
}

void printModuleInfo() {
    std::cout << "\n📋 Brain Modules:\n";
    std::cout << "  1. Thalamus (2,048 neurons) - Sensory gating and attention\n";
    std::cout << "  2. Wernicke's Area (16,384 neurons) - Language comprehension\n";
    std::cout << "  3. Broca's Area (16,384 neurons) - Language production\n";
    std::cout << "  4. Hippocampus (8,192 neurons) - Episodic memory & consolidation\n";
    std::cout << "  5. PFC (32,768 neurons) - Executive control & working memory\n";
    std::cout << "  6. Basal Ganglia (4,096 neurons) - Action selection & RL\n";
    std::cout << "\n  Total: 79,872 neurons across 6 specialized modules\n";
}

int main(int argc, char** argv) {
    printBanner();
    
    // Parse command line arguments
    std::string mode = "train";  // Default mode
    std::string checkpoint_to_load;
    if (argc > 1) {
        mode = argv[1];
    }
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--load=", 0) == 0) {
            checkpoint_to_load = arg.substr(7);
        }
    }
    
    try {
        // ====================================================================
        // 1. Initialize Brain Orchestrator
        // ====================================================================
        std::cout << "\n🚀 Initializing Neural System...\n" << std::endl;
        
        BrainOrchestrator::Config brain_config;
        brain_config.gpu_device_id = 0;
        brain_config.time_step_ms = 1.0f;
        brain_config.enable_parallel_execution = true;
        brain_config.enable_consolidation = true;
        brain_config.consolidation_interval_ms = 10000.0f;  // Every 10 seconds
        
        auto brain = std::make_shared<BrainOrchestrator>(brain_config);
        
        // Initialize all modules
        brain->initializeModules();
        printModuleInfo();
        
        // Create the connectome
        brain->createConnectome();

        if (!checkpoint_to_load.empty()) {
            std::cout << "\n📂 Loading checkpoint: " << checkpoint_to_load << std::endl;
            if (!brain->loadCheckpoint(checkpoint_to_load)) {
                std::cerr << "⚠️  Continuing without checkpoint due to load failure." << std::endl;
            }
        }
        
        // ====================================================================
        // 2. Initialize Token Embedding Interface
        // ====================================================================
        std::cout << "\n📝 Initializing Token Embedding System..." << std::endl;
        
        TokenEmbedding::Config embedding_config;
        embedding_config.vocab_size = 10000;      // 10k vocabulary
        embedding_config.embedding_dim = 512;     // 512D embeddings
        embedding_config.use_random_init = true;
        embedding_config.normalization = 1.0f;
        embedding_config.vocab_file = "";
        
        auto embeddings = std::make_shared<TokenEmbedding>(embedding_config);
        embeddings->initialize();
        
        // ====================================================================
        // 3. Initialize Output Decoder
        // ====================================================================
        std::cout << "\n🔊 Initializing Output Decoder..." << std::endl;
        
        OutputDecoder::Config decoder_config;
        decoder_config.vocab_size = embedding_config.vocab_size;
        decoder_config.output_dim = 16384;  // Broca's output dimension
        decoder_config.temperature = 1.0f;
        decoder_config.top_k = 50;
        decoder_config.top_p = 0.9f;
        decoder_config.beam_width = 5;
        decoder_config.strategy = OutputDecoder::SamplingStrategy::TOP_P;
        
        auto decoder = std::make_shared<OutputDecoder>(decoder_config, embeddings);
        
        // ====================================================================
        // 4. Run based on mode
        // ====================================================================
        
        if (mode == "train") {
            // Training mode
            std::cout << "\n🎓 Entering Training Mode...\n" << std::endl;
            
            TrainingLoop::Config train_config;
            train_config.max_epochs = 10;
            train_config.batch_size = 8;
            train_config.learning_rate = 0.001f;
            train_config.sequence_length = 32;
            train_config.enable_validation = false;
            train_config.validation_interval = 100;
            train_config.reward_discount = 0.99f;
            train_config.train_data_path = "";  // Will use demo data
            train_config.val_data_path = "";
            train_config.checkpoint_dir = "./checkpoints";
            
            TrainingLoop trainer(train_config, brain, embeddings, decoder);
            trainer.train();
            
            // Test generation after training
            std::cout << "\n\n🎯 Testing Generation...\n" << std::endl;
            std::string prompt = "The neural network";
            std::string generated = trainer.generate(prompt, 20);
            std::cout << "\n  Generated: " << generated << "\n" << std::endl;
            
        } else if (mode == "generate") {
            // Generation mode
            std::cout << "\n💭 Entering Generation Mode...\n" << std::endl;
            
            TrainingLoop::Config train_config;
            train_config.max_epochs = 0;
            train_config.batch_size = 1;
            train_config.learning_rate = 0.0f;
            train_config.sequence_length = 32;
            train_config.enable_validation = false;
            train_config.validation_interval = 0;
            train_config.reward_discount = 0.99f;
            train_config.train_data_path = "";
            train_config.val_data_path = "";
            train_config.checkpoint_dir = "";
            
            TrainingLoop generator(train_config, brain, embeddings, decoder);
            
            // Interactive generation
            std::string prompt;
            std::cout << "\nEnter prompt (or 'quit' to exit): ";
            std::getline(std::cin, prompt);
            
            while (prompt != "quit") {
                std::string generated = generator.generate(prompt, 30);
                std::cout << "\n  Generated: " << generated << "\n" << std::endl;
                
                std::cout << "\nEnter prompt (or 'quit' to exit): ";
                std::getline(std::cin, prompt);
            }
            
        } else if (mode == "demo") {
            // Demo mode - show cognitive cycle
            std::cout << "\n🎬 Entering Demo Mode - Cognitive Cycle Visualization...\n" << std::endl;
            
            // Process a single token and show phase transitions
            auto test_embedding = embeddings->encode("hello");
            
            for (int cycle = 0; cycle < 5; ++cycle) {
                std::cout << "\n--- Cognitive Cycle " << (cycle + 1) << " ---" << std::endl;
                
                for (int step = 0; step < 50; ++step) {  // ~500ms cycle
                    auto output = brain->cognitiveStep(test_embedding);
                    
                    auto stats = brain->getStats();
                    
                    // Print phase transitions
                    static auto last_phase = BrainOrchestrator::CognitivePhase::SENSATION;
                    if (stats.current_phase != last_phase) {
                        std::string phase_name;
                        switch (stats.current_phase) {
                            case BrainOrchestrator::CognitivePhase::SENSATION:
                                phase_name = "SENSATION"; break;
                            case BrainOrchestrator::CognitivePhase::PERCEPTION:
                                phase_name = "PERCEPTION"; break;
                            case BrainOrchestrator::CognitivePhase::INTEGRATION:
                                phase_name = "INTEGRATION"; break;
                            case BrainOrchestrator::CognitivePhase::SELECTION:
                                phase_name = "SELECTION"; break;
                            case BrainOrchestrator::CognitivePhase::ACTION:
                                phase_name = "ACTION"; break;
                        }
                        std::cout << "  → Phase: " << phase_name << " (t=" 
                                 << stats.total_time_ms << "ms)" << std::endl;
                        last_phase = stats.current_phase;
                    }
                    
                    // If we got output, decode it
                    if (!output.empty()) {
                        std::string token = decoder->decodeToString(output);
                        std::cout << "  ✓ Output Token: " << token << std::endl;
                    }
                }
            }
            
            // Print final statistics
            auto final_stats = brain->getStats();
            std::cout << "\n📊 Final Statistics:" << std::endl;
            std::cout << "  Total time: " << final_stats.total_time_ms << " ms" << std::endl;
            std::cout << "  Cognitive cycles: " << final_stats.cognitive_cycles << std::endl;
            std::cout << "  Tokens processed: " << final_stats.tokens_processed << std::endl;
            std::cout << "  Average reward: " << final_stats.average_reward << std::endl;
            
            std::cout << "\n  Module Activities:" << std::endl;
            for (const auto& [name, activity] : final_stats.module_activities) {
                std::cout << "    " << name << ": " << activity << std::endl;
            }
            
        } else {
            std::cerr << "Unknown mode: " << mode << std::endl;
            std::cerr << "Usage: " << argv[0] << " [train|generate|demo] [--load=/path/to/checkpoint]" << std::endl;
            return 1;
        }
        
        std::cout << "\n✨ NeuroGen Modular Brain - Session Complete ✨\n" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Fatal Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

