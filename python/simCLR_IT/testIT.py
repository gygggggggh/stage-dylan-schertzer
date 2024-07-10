from module_simCLR_IT import SimCLRModuleIT

from ..shared_test import main, setup_logging

# Constants
LOG_FILE = "testIT.log"
MODEL_PATHS = "checkpointsIT"
MODEL_PATH = "checkpointsIT/simclr-it-epoch=145.ckpt"

TRAIN_DATA_PATH = {"x": "data/x_train_40k.npy", "y": "data/y_train_40k.npy"}
TEST_DATA_PATH = {"x": "data/x_test.npy", "y": "data/y_test.npy"}

CONFIG = {
    "num_seeds": 20,
    "n_values": [5, 10, 50, 100],
    "batch_size": 1024,
    "num_workers": 8,
}

if __name__ == "__main__":
    setup_logging(LOG_FILE)
    main(
        SimCLRModuleIT,
        MODEL_PATHS,
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
        CONFIG,
        evaluate_all_checkpoints=False,
        model_path=MODEL_PATH,
    )
    print("Done.")
