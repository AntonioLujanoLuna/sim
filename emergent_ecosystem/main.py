"""
Main entry point for the Emergent Intelligence Ecosystem simulation.

This script sets up and runs the complete simulation with all systems integrated.
"""

import argparse
import os
import sys
import time
from typing import Optional

from .config import Config, DEMO_CONFIG, RESEARCH_CONFIG, PERFORMANCE_CONFIG
from .core.simulation import EmergentIntelligenceSimulation
from .visualization.main_display import create_visualization

# Add this at the top of main.py to handle import paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Emergent Intelligence Ecosystem Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration Profiles:
  demo        - Interactive demo with visualization (default)
  research    - Research configuration with detailed tracking
  performance - High-performance configuration for large populations

Examples:
  python -m emergent_ecosystem.main --profile demo
  python -m emergent_ecosystem.main --profile research --steps 10000
  python -m emergent_ecosystem.main --no-viz --profile performance
        """
    )
    
    parser.add_argument(
        '--profile', 
        choices=['demo', 'research', 'performance'],
        default='demo',
        help='Configuration profile to use'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=5000,
        help='Number of simulation steps to run'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Run without visualization (headless mode)'
    )
    
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='Interval for saving statistics (0 to disable)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory for output files'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducible results'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def get_config(profile: str) -> Config:
    """Get configuration based on profile"""
    configs = {
        'demo': DEMO_CONFIG,
        'research': RESEARCH_CONFIG,
        'performance': PERFORMANCE_CONFIG
    }
    return configs.get(profile, DEMO_CONFIG)


def setup_simulation(config: Config, seed: Optional[int] = None) -> EmergentIntelligenceSimulation:
    """Setup and initialize the simulation"""
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    
    simulation = EmergentIntelligenceSimulation(config)
    return simulation


def run_headless_simulation(simulation: EmergentIntelligenceSimulation, 
                           steps: int, save_interval: int = 100, 
                           output_dir: str = 'output', verbose: bool = False):
    """Run simulation without visualization"""
    import os
    import json
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running headless simulation for {steps} steps...")
    print(f"Output directory: {output_dir}")
    
    start_time = time.time()
    
    for step in range(steps):
        simulation.update()
        
        # Progress reporting
        if verbose and step % 100 == 0:
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"Step {step}/{steps} ({rate:.1f} steps/sec)")
        
        # Save statistics
        if save_interval > 0 and step % save_interval == 0:
            stats = simulation.statistics_tracker.get_comprehensive_report()
            
            # Save to JSON
            stats_file = os.path.join(output_dir, f'stats_step_{step:06d}.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            if verbose:
                print(f"Saved statistics to {stats_file}")
    
    # Final statistics
    final_stats = simulation.statistics_tracker.get_comprehensive_report()
    final_file = os.path.join(output_dir, 'final_statistics.json')
    with open(final_file, 'w') as f:
        json.dump(final_stats, f, indent=2, default=str)
    
    # Emergence events
    emergence_events = simulation.emergence_detector.get_all_events()
    emergence_file = os.path.join(output_dir, 'emergence_events.json')
    with open(emergence_file, 'w') as f:
        json.dump(emergence_events, f, indent=2, default=str)
    
    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.2f} seconds")
    print(f"Final population: {len(simulation.individuals)}")
    print(f"Total emergence events: {len(emergence_events)}")
    print(f"Results saved to: {output_dir}")


def run_interactive_simulation(simulation: EmergentIntelligenceSimulation, 
                              steps: int, config: Config):
    """Run simulation with interactive visualization"""
    try:
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
    except ImportError:
        print("Error: matplotlib not available for visualization")
        print("Install with: pip install matplotlib")
        sys.exit(1)
    
    print(f"Starting interactive simulation with visualization...")
    print(f"Configuration: {config.__class__.__name__}")
    print(f"Population limit: {config.max_population}")
    print(f"Environment size: {config.width}x{config.height}")
    
    # Create visualization
    viz = create_visualization(simulation, config)
    
    # Start animation
    animation = viz.start_animation(frames=steps, interval=50)
    
    print("Visualization started. Close the window to exit.")
    
    try:
        viz.show()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Falling back to headless mode...")
        run_headless_simulation(simulation, steps, verbose=True)


def print_simulation_info(config: Config, args):
    """Print simulation configuration information"""
    print("=" * 60)
    print("EMERGENT INTELLIGENCE ECOSYSTEM SIMULATION")
    print("=" * 60)
    print(f"Profile: {args.profile}")
    print(f"Steps: {args.steps}")
    print(f"Visualization: {'Disabled' if args.no_viz else 'Enabled'}")
    print(f"Random seed: {args.seed if args.seed else 'Random'}")
    print()
    print("Configuration:")
    print(f"  Environment: {config.width}x{config.height}")
    print(f"  Max population: {config.max_population}")
    print(f"  Initial population: {config.initial_population}")
    print(f"  Species: {len(config.species_configs)}")
    print()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Get configuration
    config = get_config(args.profile)
    
    # Print simulation info
    print_simulation_info(config, args)
    
    # Setup simulation
    try:
        simulation = setup_simulation(config, args.seed)
    except Exception as e:
        print(f"Error setting up simulation: {e}")
        sys.exit(1)
    
    # Run simulation
    try:
        if args.no_viz:
            run_headless_simulation(
                simulation, 
                args.steps, 
                args.save_interval, 
                args.output_dir,
                args.verbose
            )
        else:
            run_interactive_simulation(simulation, args.steps, config)
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
