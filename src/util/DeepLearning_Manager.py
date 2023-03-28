import os
import time
import json
import torch
import logging
import threading
import util.ModelLoader as ModelLoader

from google.api_core import retry
from google.cloud import pubsub_v1

class DeepLearningManager( threading.Thread ):
    def __init__(self, iCudaIndex : int, iSubscriptionPath : str, iForceCPU : bool = False) -> None:
        self.cuda_index    : int = iCudaIndex
        self.device_name   : str = 'cuda:' + str(self.cuda_index)
        self.gpu_name      : str = torch.cuda.get_device_name(self.cuda_index)
        self.configuration : ModelLoader.StableConfiguration = ModelLoader.StableConfiguration(
            None, None, None, None, None, None, None, None, None, None, None, None, None,
            self.device_name, None, None, None, None, None, None, None, self.cuda_index,
            "./resources/model_" + str(self.cuda_index) + ".ckpt", None,
            None, None, None, None, None, None, True
        )
        self.subscription_path : str = iSubscriptionPath
        self.subscriber        : pubsub_v1.SubscriberClient = pubsub_v1.SubscriberClient()
        threading.Thread.__init__(
            self,
            name = self.device_name,
            daemon = True
        )
        if iForceCPU:
            logging.warning("CPU forced for testing.")
            self.configuration.half_precision = False
            self.configuration.device_name = "cpu"
        if not os.path.exists(f'./steps/cuda_{self.cuda_index}'):
            os.mkdir(f'./steps/cuda_{self.cuda_index}')
    
    def set_pull_response(self, iPullResponse : any) -> None:
        if self.pull_response is not None:
            raise RuntimeError("Object not completely free.")
        self.pull_response = iPullResponse
        self.processing = True

    def get_message(self) -> tuple[ str, pubsub_v1.types.PubsubMessage ]:
        vResponse : pubsub_v1.PullResponse = self.subscriber.pull(
            request = {
                "subscription": self.subscription_path,
                "return_immediately": True,
                "max_messages": 1
            },
            retry = retry.Retry(deadline = 100)
        )
        if len(vResponse.received_messages) == 0:
            return None, None
        return vResponse.received_messages[0].ack_id, vResponse.received_messages[0].message
    
    def ack(self, iAck_Id : str) -> None:
        self.subscriber.acknowledge(request = {
            "subscription": self.subscription_path,
            "ack_ids": [iAck_Id]
        })
    
    def nack(self, iAck_Id : str) -> None:
        raise NotImplementedError("Pending of the development.")
    
    def run(self) -> None:
        vAck_Id   : str = None
        vMessage  : pubsub_v1.types.PubsubMessage = None
        vJsonData : dict = None

        with self.subscriber:
            logging.info("Thread opened.")
            while True:
                vAck_Id, vMessage = self.get_message()
                if vAck_Id is None:
                    time.sleep(1)
                    continue
                
                try:
                    torch.cuda.empty_cache()
                    vJsonData = json.loads(vMessage.data.decode("utf-8"))
                    logging.info(str(vJsonData))

                    self.configuration.prompt       = vJsonData["p"]
                    self.configuration.model_privacy= vJsonData["mp"]
                    self.configuration.image_width  = vJsonData["w"]
                    self.configuration.image_height = vJsonData["h"]
                    self.configuration.cfg          = vJsonData["cfg"]
                    self.configuration.steps        = vJsonData["st"]
                    self.configuration.batch_size   = vJsonData["i#"]
                    self.configuration.img_channels = None
                    self.configuration.seed         = vJsonData["s"]
                    self.configuration.time_stamp   = vJsonData["t"]
                    self.configuration.sampler      = vJsonData["sa"]
                    self.configuration.negative_prompt = vJsonData["np"]
                    
                    vModelId = vJsonData["mi"]
                    if self.configuration.model_id is None or vModelId != self.configuration.model_id:
                        logging.info(f"Loading model {vModelId}.")
                        self.configuration.model_id = vModelId
                        self.configuration.user_id  = None
                        self.configuration.model_architecture = vJsonData["ma"]
                        if self.configuration.model_architecture == 1:
                            self.configuration.file_configuration = "./resources/v1-inference.yaml"
                        else:
                            self.configuration.file_configuration = "./resources/v2.inference-v.yaml"
                        self.configuration.extract_ema   = False
                        self.configuration.pipeline_type = None
                        self.configuration.prediction_type = "epsilon"
                        
                        #self.configuration.device_name = "cpu" # Only for test time in my trash pc.
                        ModelLoader.load_configuration(self.configuration)
                            # Convert noise image to latent space.
                        ModelLoader.load_unet(self.configuration)
                        # In the future load HyperNetworks, NetWork additional to the U-Net.
                            # Must in the future verify the VAE default or other VAE.
                            # The VAE decoder, convert Latent space in the image.
                        ModelLoader.load_vae(self.configuration)
                        # In the future load new customized embedings of the user.
                        ModelLoader.load_embeding(self.configuration)
                        ModelLoader.load_scheduler(self.configuration)
                    else:
                        if self.configuration.sampler != vJsonData["sa"]:
                            self.configuration.sampler = vJsonData["sa"]
                            ModelLoader.load_scheduler(self.configuration)
                    
                    ModelLoader.prompt2img(self.configuration, save_int = False)
                    logging.info(f"Image {self.configuration.time_stamp} generated.")
                    
                    self.ack(vAck_Id)
                except Exception as ex:
                    logging.fatal(ex)
                    raise

    def getMemoryReserved(self) -> float:
        return round(torch.cuda.memory_reserved (self.cuda_index)/1024**3, 2)

    def getMemoryAllocate(self) -> float:
        return round(torch.cuda.memory_allocated(self.cuda_index)/1024**3, 2)
    
    def __str__(self) -> str:
        vData : dict = {
            "cuda_index": self.cuda_index,
            "device_name": self.device_name,
            "gpu_name": self.gpu_name,
            "model_type": self.model_type,
            "model_privacy": self.model_privacy,
            "model_id": self.model_id,
            "memory_reserved": self.getMemoryReserved(),
            "memoer_allocated": self.getMemoryAllocate()
        }
        return str(vData)