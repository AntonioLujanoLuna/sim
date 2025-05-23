#!/usr/bin/env python3
"""
Performance testing script for the Emergent Intelligence Ecosystem.

This script tests all the performance improvements and validates that the
identified issues have been resolved.
"""

import time
import sys
import traceback
from pathlib import Path

# Add the emergent_ecosystem package to the path
sys.path.insert(0, str(Path(__file__).parent))

from emergent_ecosystem.config import Config, DEMO_CONFIG, RESEARCH_CONFIG, PERFORMANCE_CONFIG
from emergent_ecosystem.core.simulation import EmergentIntelligenceSimulation
from emergent_ecosystem.core.error_handling import error_handler, validate_configuration
from emergent_ecosystem.core.memory_manager import memory_manager


def test_configuration_fix():
    """Test that the configuration error has been fixed"""
    print("🔧 Testing Configuration Fix...")
    
    try:
        # Test basic config creation
        config = Config()
        assert hasattr(config, 'species_configs'), "species_configs property missing"
        
        species_configs = config.species_configs
        assert isinstance(species_configs, dict), "species_configs should be a dict"
        assert 'predator' in species_configs, "predator config missing"
        assert 'herbivore' in species_configs, "herbivore config missing"
        
        print("✅ Configuration fix verified - species_configs property working correctly")
        
        # Test configuration validation
        assert validate_configuration(config), "Configuration validation failed"
        print("✅ Configuration validation working")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_performance_optimizations():
    """Test performance optimizations"""
    print("\n⚡ Testing Performance Optimizations...")
    
    try:
        # Create a simulation with performance config
        config = PERFORMANCE_CONFIG
        simulation = EmergentIntelligenceSimulation(config)
        
        # Test spatial indexing
        assert hasattr(simulation, 'performance_optimizer'), "Performance optimizer missing"
        assert hasattr(simulation.performance_optimizer, 'spatial_grid'), "Spatial grid missing"
        
        print("✅ Spatial indexing system initialized")
        
        # Run a few simulation steps and measure performance
        start_time = time.time()
        
        for i in range(10):
            simulation.update()
            
            # Check that spatial index is being updated
            grid_cells = len(simulation.performance_optimizer.spatial_grid.grid)
            individuals_in_grid = len(simulation.performance_optimizer.spatial_grid.individual_positions)
            
            if i == 5:  # Check halfway through
                assert individuals_in_grid > 0, "Spatial index not being populated"
                print(f"✅ Spatial index working: {individuals_in_grid} individuals in {grid_cells} cells")
        
        elapsed_time = time.time() - start_time
        avg_time_per_step = elapsed_time / 10
        
        print(f"✅ Performance test completed: {avg_time_per_step:.3f}s per step")
        
        # Get performance metrics
        metrics = simulation.get_performance_summary()
        assert 'optimizer_metrics' in metrics, "Performance metrics missing"
        
        print("✅ Performance monitoring working")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test improved error handling"""
    print("\n🛡️ Testing Error Handling...")
    
    try:
        # Test error handler initialization
        assert error_handler is not None, "Error handler not initialized"
        
        # Test safe community detection with empty graph
        import networkx as nx
        from emergent_ecosystem.core.error_handling import safe_community_detection
        
        empty_graph = nx.Graph()
        communities = safe_community_detection(empty_graph)
        assert communities == [], "Safe community detection failed for empty graph"
        
        print("✅ Safe community detection working")
        
        # Test error logging
        error_handler.reset_error_tracking()
        initial_error_count = len(error_handler.error_history)
        
        # Simulate an error
        try:
            raise ValueError("Test error")
        except ValueError as e:
            error_handler._log_error('test', 'test_operation', e)
        
        assert len(error_handler.error_history) > initial_error_count, "Error logging not working"
        
        print("✅ Error logging working")
        
        # Test error summary
        summary = error_handler.get_error_summary()
        assert 'total_errors' in summary, "Error summary missing fields"
        assert 'error_counts' in summary, "Error counts missing"
        
        print("✅ Error reporting working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False


def test_memory_management():
    """Test memory management improvements"""
    print("\n🧠 Testing Memory Management...")
    
    try:
        # Test memory manager initialization
        assert memory_manager is not None, "Memory manager not initialized"
        
        # Test memory usage monitoring
        usage = memory_manager.get_memory_usage()
        assert 'rss_mb' in usage, "Memory usage monitoring not working"
        assert 'percent' in usage, "Memory percentage missing"
        
        print(f"✅ Memory monitoring working: {usage['rss_mb']:.1f}MB ({usage['percent']:.1f}%)")
        
        # Test memory limits
        original_limit = memory_manager.limits['individual_memory']
        memory_manager.set_memory_limits(individual_memory=150)
        assert memory_manager.limits['individual_memory'] == 150, "Memory limit setting failed"
        memory_manager.limits['individual_memory'] = original_limit  # Restore
        
        print("✅ Memory limit configuration working")
        
        # Test memory cleanup with a simulation
        config = DEMO_CONFIG
        simulation = EmergentIntelligenceSimulation(config)
        
        # Run simulation to generate some data
        for _ in range(5):
            simulation.update()
        
        # Test memory cleanup
        cleanup_result = memory_manager.optimize_memory(simulation, force=True)
        assert 'items_cleaned' in cleanup_result, "Memory cleanup not working"
        
        print(f"✅ Memory cleanup working: {cleanup_result['items_cleaned']} items cleaned")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        traceback.print_exc()
        return False


def test_dependency_requirements():
    """Test that all required dependencies are available"""
    print("\n📦 Testing Dependencies...")
    
    required_packages = [
        'numpy', 'matplotlib', 'networkx', 'scipy', 'sklearn',
        'pandas', 'seaborn', 'tqdm', 'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies available")
        return True


def test_simulation_stability():
    """Test that the simulation runs stably without crashes"""
    print("\n🔄 Testing Simulation Stability...")
    
    try:
        config = DEMO_CONFIG
        simulation = EmergentIntelligenceSimulation(config)
        
        print(f"Starting simulation with {len(simulation.individuals)} individuals...")
        
        # Run for more steps to test stability
        steps_to_run = 50
        successful_steps = 0
        
        for step in range(steps_to_run):
            try:
                simulation.update()
                successful_steps += 1
                
                # Check population health
                alive_individuals = [ind for ind in simulation.individuals if ind.is_alive()]
                if len(alive_individuals) == 0:
                    print("⚠️ All individuals died - this might be expected behavior")
                    break
                
                # Progress indicator
                if step % 10 == 0:
                    print(f"  Step {step}: {len(alive_individuals)} individuals alive")
                    
            except Exception as e:
                print(f"❌ Simulation failed at step {step}: {e}")
                break
        
        success_rate = successful_steps / steps_to_run
        print(f"✅ Simulation stability: {successful_steps}/{steps_to_run} steps successful ({success_rate:.1%})")
        
        # Get final performance summary
        performance_summary = simulation.get_performance_summary()
        print(f"✅ Final performance summary available: {len(performance_summary)} metrics")
        
        return success_rate > 0.8  # 80% success rate threshold
        
    except Exception as e:
        print(f"❌ Simulation stability test failed: {e}")
        traceback.print_exc()
        return False


def test_o_n_squared_fix():
    """Test that O(n²) complexity has been reduced"""
    print("\n📈 Testing O(n²) Complexity Fix...")
    
    try:
        # Test with different population sizes
        small_config = Config()
        small_config.simulation.initial_population = 20
        small_config.simulation.max_population = 30
        
        large_config = Config()
        large_config.simulation.initial_population = 60
        large_config.simulation.max_population = 80
        
        # Test small population
        small_sim = EmergentIntelligenceSimulation(small_config)
        start_time = time.time()
        for _ in range(5):
            small_sim.update()
        small_time = time.time() - start_time
        
        # Test large population
        large_sim = EmergentIntelligenceSimulation(large_config)
        start_time = time.time()
        for _ in range(5):
            large_sim.update()
        large_time = time.time() - start_time
        
        # Calculate time ratio
        population_ratio = large_config.simulation.initial_population / small_config.simulation.initial_population
        time_ratio = large_time / small_time
        
        print(f"Population ratio: {population_ratio:.1f}x")
        print(f"Time ratio: {time_ratio:.1f}x")
        
        # If it were O(n²), time ratio should be close to population_ratio²
        # With optimizations, it should be much better
        expected_o_n_squared_ratio = population_ratio ** 2
        
        if time_ratio < expected_o_n_squared_ratio * 0.7:  # 30% improvement threshold
            print(f"✅ O(n²) optimization working: {time_ratio:.1f}x vs expected {expected_o_n_squared_ratio:.1f}x")
            return True
        else:
            print(f"⚠️ O(n²) optimization may need more work: {time_ratio:.1f}x vs expected {expected_o_n_squared_ratio:.1f}x")
            return False
            
    except Exception as e:
        print(f"❌ O(n²) complexity test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all performance tests"""
    print("🚀 Emergent Intelligence Ecosystem - Performance Test Suite")
    print("=" * 60)
    
    tests = [
        ("Configuration Fix", test_configuration_fix),
        ("Performance Optimizations", test_performance_optimizations),
        ("Error Handling", test_error_handling),
        ("Memory Management", test_memory_management),
        ("Dependencies", test_dependency_requirements),
        ("O(n²) Complexity Fix", test_o_n_squared_fix),
        ("Simulation Stability", test_simulation_stability),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n🎉 All tests passed! The performance improvements are working correctly.")
        return 0
    else:
        print(f"\n⚠️ {total-passed} test(s) failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 