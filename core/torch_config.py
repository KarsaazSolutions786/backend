import torch
import os
from utils.logger import logger

def configure_torch_memory():
    """Configure PyTorch memory settings for optimal performance in limited environments."""
    try:
        # Set memory allocation strategy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.5)  # Use only 50% of available GPU memory
        else:
            # CPU-specific optimizations
            torch.set_num_threads(1)  # Limit CPU threads
            torch.set_num_interop_threads(1)
            
            # Set memory management parameters
            if hasattr(torch, 'set_memory_fraction'):
                torch.set_memory_fraction(0.5)  # Use only 50% of available memory
            
            # Disable gradient computation if not needed
            torch.set_grad_enabled(False)
            
        logger.info("PyTorch memory configuration applied successfully")
        
    except Exception as e:
        logger.warning(f"Failed to configure PyTorch memory settings: {e}")
        logger.warning("Continuing with default memory settings") 