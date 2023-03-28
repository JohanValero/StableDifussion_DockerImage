import os
import sys
import time
import torch
import logging

from util.DeepLearning_Manager import DeepLearningManager
from google.cloud import pubsub_v1

# Block execution with error, if there not a avalible GPU.
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU available.")

if not os.path.exists(f'./steps'):
    os.mkdir(f'./steps')

if not os.path.exists(f'./logs'):
    os.mkdir(f'./logs')

# ------------------------------------------------------------------------------------------
#   init_logger
#   Configurate the log of the application.
#   
#   AUTOR               YYYY-MM-DD  COMMENT
#   =================== =========== ========================================================
#   JOHAN VALERO        2023-01-12  Creation.
# ------------------------------------------------------------------------------------------
def init_logger():
    print("LOGGING LEVEL: ", os.getenv('LOGGING_LEVEL', 'CRITICAL'))
    vLogFileName   = 'logs/log_file.txt'
    vFileHandler   = logging.FileHandler(filename = vLogFileName)
    vStdoutHandler = logging.StreamHandler(sys.stdout)
    vHandlers      = [vStdoutHandler, vFileHandler]
    
    logging.basicConfig(
        encoding = 'utf-8',
        format   = '[%(levelname)s:%(asctime)s:%(threadName)s] %(message)s',
        level    = os.getenv('LOGGING_LEVEL', 'INFO'),
        handlers = vHandlers,
        force    = True
    )

init_logger()

# Set the PubSub subscriber config.
gSUBSCRIPTION_PATH : str = pubsub_v1.SubscriberClient.subscription_path(
    os.getenv('GOOGLE_CLOUD_PROJECT_ID'),
    'GOOGLE_PUBSUB_SUBSCRIPTION',
)

def main():
    vForceCPU : bool = os.getenv("FORCE_CPU", False) == "1"
    gDevices : list[ DeepLearningManager ] = [
        DeepLearningManager(i, gSUBSCRIPTION_PATH, vForceCPU) for i in range(torch.cuda.device_count())
    ]

    # Start the threads with the DL managers.
    for vDevice in gDevices:
        vDevice.start()

    try:
        while True:
            for vDevice in gDevices:
                vDevice.join(timeout = 1)
            time.sleep(2)
    except KeyboardInterrupt as ex:
        print("KeyboardInterrupt")
    except:
        raise

if __name__ == "__main__":
    main()