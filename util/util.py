from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple

import logging
import os
import time
from pathlib import Path
import io
import matplotlib.pyplot as plt

import tensorflow as tf

def setup_logger(path):
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    lf = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    fh_es = logging.FileHandler(filename=path, mode="a", encoding="utf-8")
    fh_es.setFormatter(lf)
    logger.addHandler(fh_es)

    console = logging.StreamHandler()
    console.setFormatter(lf)
    logger.addHandler(console)

    return logger


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


class Timer:
    def __init__(self, timestamp=time.time(), names=None):
        if names is None:
            names = ["default"]
        if "default" not in names:
            names.append("default")
        self.timestamp = timestamp
        self.names = names
        self.stack = {k: [timestamp] for k in names}

    def tick(self, name="default"):
        self.stack[name].append(time.time())
        return self.stack[name][-1] - self.stack[name][-2]


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_root: str = field(
        default=None,
        metadata={"help": "root path of the dataset"}
    )
    dataset: str = field(
        default=None,
        metadata={"help": "which dataset to use, e.g., ipinyou or criteo"}
    )
    advertiser: str = field(
        default=None,
        metadata={"help": "which advertiser to use"}
    )
    price_scale: float = field(
        default=1.0,
        metadata={"help": "bid price and market price will be scaled as 1/price_scale"}
    )
    bid_scale: float = field(
        default=1.0,
        metadata={"help": "bid price will be scaled as 1/bid_scale to simulate censored data"}
    )


@dataclass
class ModelArguments:
    """

    """
    embedding_dimension: int = field(
        default=64,
        metadata={"help": "maximal embedding dimension of each field"}
    )
    hidden_dimension: int = field(
        default=128,
        metadata={"help": "dimension of hidden state"}
    )
    K: int = field(
        default=5,
        metadata={"help": "number of modes for the distribution"}
    )
    family: str = field(
        default="gaussian",
        metadata={"help": "distribution family of mixture model"}
    )
    drop_out: float = field(
        default=0.2,
        metadata={"help": "drop out rate of mixing coefficient"}
    )
    first_price: bool = field(
        default=False,
        metadata={"help": "whether to use first price auctions? False to second price auctions"}
    )



@dataclass
class TrainingArguments:
    """

    """
    CUDA_VISIBLE_DEVICES: str = field(
        default=None,
        metadata={"help": "The GPUs to use"}
    )
    seed: int = field(
        default=43,
        metadata={"help": "random seed"}
    )
    output_dir: str = field(
        default="./result/temp/",
        metadata={"help": "path to save the model"}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "dose overwrite the output directory"}
    )
    batch_size: int = field(
        default=1024,
        metadata={"help": "batch_size for train and test"}
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"help": "learning rate"}
    )
    epoch: int = field(
        default=10,
        metadata={"help": "training epoch"}
    )
    debug: bool = field(
        default=False,
        metadata={"help": "whether to print debbug information"}
    )
    nomalize_loss: bool = field(
        default=False,
        metadata={"help": "whether to normalize loss"}
    )

