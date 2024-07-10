import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import logger, main
from simCLR_IT.module_simCLR_IT import SimCLRModuleIT

CONFIG = {
    "x_train_path": "data/x_train_40k.npy",
    "y_train_path": "data/y_train_40k.npy",
    "model_save_path": "python/simCLR+InceptionTime/simCLR+IT.pth",
    "batch_size": 1024,
    "num_workers": 8,
    "max_epochs": 150,
    "learning_rate": 0.02,
    "val_split": 0.1,
    "model_type": "it",
}

if __name__ == "__main__":
    main(CONFIG, SimCLRModuleIT)
    logger.info("Training completed.")
