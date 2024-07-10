from python.shared import logger, main
from python.simCLR_resnet.module_simCLR_RN import SimCLRModuleRN

CONFIG = {
    "x_train_path": "data/x_train_40k.npy",
    "y_train_path": "data/y_train_40k.npy",
    "model_save_path": "python/simCLR+resnet/simCLR+RN.pth",
    "batch_size": 1024,
    "num_workers": 8,
    "max_epochs": 200,
    "learning_rate": 0.02,
    "val_split": 0.1,
    "model_type": "rn",
}

if __name__ == "__main__":
    try:
        main(CONFIG, SimCLRModuleRN)
        logger.info("Training completed.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
