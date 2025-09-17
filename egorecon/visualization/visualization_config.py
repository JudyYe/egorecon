#!/usr/bin/env python3
"""Configuration for the visualization system"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VisualizationConfig:
    """Configuration for visualization system"""
    
    # Basic settings
    enabled: bool = False
    exp_name: str = "default_experiment"
    save_dir: str = "runs/default"
    
    # Data paths
    mano_models_dir: str = "data/mano_models"
    hand_articulations_path: str = "data/hand_articulations.pkl"
    generation_data_path: str = "data/generation.pkl"
    
    # Performance settings
    visualization_frequency: int = 100
    enhanced_scene_frames: int = 500
    
    # Visualization options
    save_training_snapshots: bool = True
    save_evaluation_results: bool = True
    save_final_trajectory: bool = True
    save_enhanced_scene: bool = True
    
    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments"""
        return cls(
            enabled=getattr(args, 'use_rerun', False),
            exp_name=getattr(args, 'exp_name', 'default_experiment'),
            save_dir=getattr(args, 'save_dir', 'runs/default'),
            mano_models_dir=getattr(args, 'mano_models_dir', 'data/mano_models'),
            hand_articulations_path=getattr(args, 'hand_articulations_path', 'data/hand_articulations.pkl'),
            generation_data_path=getattr(args, 'generation_data_path', 'data/generation.pkl'),
            visualization_frequency=getattr(args, 'visualization_frequency', 100),
            enhanced_scene_frames=getattr(args, 'enhanced_scene_frames', 500)
        )
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.enabled:
            return True
        
        # Check required paths
        required_paths = [
            self.mano_models_dir,
            self.hand_articulations_path,
            self.generation_data_path
        ]
        
        for path in required_paths:
            if not Path(path).exists():
                print(f"⚠️  Required path not found: {path}")
                return False
        
        return True
    
    def get_summary(self) -> str:
        """Get configuration summary"""
        if not self.enabled:
            return "Visualization disabled"
        
        summary = f"🎯 Visualization Configuration:\n"
        summary += f"  ✓ Experiment: {self.exp_name}\n"
        summary += f"  ✓ Save directory: {self.save_dir}\n"
        summary += f"  ✓ MANO models: {self.mano_models_dir}\n"
        summary += f"  ✓ Hand articulations: {self.hand_articulations_path}\n"
        summary += f"  ✓ Generation data: {self.generation_data_path}\n"
        summary += f"  ✓ Frequency: every {self.visualization_frequency} steps\n"
        summary += f"  ✓ Enhanced scene frames: {self.enhanced_scene_frames}\n"
        
        return summary
