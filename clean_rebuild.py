"""
Script to clean up Python cache and rebuild models from scratch
"""

import os
import shutil
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CleanRebuild")

def clean_pycache():
    """Remove all __pycache__ directories"""
    
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                logger.info(f"Removing {pycache_path}")
                shutil.rmtree(pycache_path)
    
    logger.info("Python cache cleaned")

def remove_models():
    """Remove all model files"""
    
    model_dir = "models"
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith(".pth"):
                file_path = os.path.join(model_dir, file)
                logger.info(f"Removing {file_path}")
                os.remove(file_path)
    
    logger.info("Model files removed")

if __name__ == "__main__":
    logger.info("Starting cleanup...")
    clean_pycache()
    remove_models()
    logger.info("Cleanup completed")
    
    # Now run initialize_models.py
    logger.info("Rebuilding models...")
    os.system(f"{sys.executable} initialize_models.py")
    logger.info("Models rebuilt")
