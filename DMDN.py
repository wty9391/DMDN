import os
import sys
import codecs
import json
import pickle
import datetime
sys.path.append(os.path.abspath(os.path.join(".", "util")))
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from transformers import HfArgumentParser

import traceback
import contextlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.backend.set_floatx('float32')
EPSILON = tf.keras.backend.epsilon()

import util
import dataset_processor
from dataset_processor import Processor
from truthful_bidder import Truthful_bidder

# Some helper code to demonstrate the kinds of errors you might encounter.
@contextlib.contextmanager
def assert_raises(error_class):
  try:
    yield
  except error_class as e:
    print('Caught expected exception \n  {}:'.format(error_class))
    traceback.print_exc(limit=2)
  except Exception as e:
    raise e
  else:
    raise Exception('Expected {} to be raised but no error was raised!'.format(
        error_class))

def main():
    parser = HfArgumentParser((util.ModelArguments, util.DataTrainingArguments, util.TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger = util.setup_logger(os.path.join(training_args.output_dir, 'log.log'))
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("data parameters %s", data_args)
    logger.info("model parameters %s", model_args)

    np.random.seed(training_args.seed)
    tf.random.set_seed(training_args.seed)

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and not training_args.overwrite_output_dir
    ):
        logger.warning(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = training_args.CUDA_VISIBLE_DEVICES
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    import myModels

    timer = util.Timer()
    processor = Processor(dataset=data_args.dataset,
                          advertiser=data_args.advertiser,
                          root=data_args.dataset_root,
                          logger=logger)

    if training_args.overwrite_output_dir or not Processor.is_cache_available(training_args.output_dir):
        logger.info("dataset cache is not available, start to cache in {}".format(training_args.output_dir))
        processor.process(output_dir=training_args.output_dir)
    else:
        logger.info("dataset cache is available, reuse cache in {}".format(training_args.output_dir))
        processor.load_from_cache(training_args.output_dir)

    timer.tick()
    train_size = len(processor.train.index)
    test_size = len(processor.test.index)
    features = pd.concat([processor.train, processor.test], ignore_index=True)

    y_col = None  # column name of click
    z_col = None  # column name of pay price
    b_col = None  # column name of bid price
    numerical_cols = None

    if processor.dataset == "ipinyou":
        y_col = dataset_processor.ipinyou_col["click"]
        z_col = dataset_processor.ipinyou_col["pay_price"]
        b_col = dataset_processor.ipinyou_col["bid_price"]
        numerical_cols = dataset_processor.ipinyou_col["feature_numerical"]
    elif processor.dataset == "criteo":
        y_col = dataset_processor.criteo_col["click"]
        z_col = dataset_processor.criteo_col["pay_price"]
        b_col = dataset_processor.criteo_col["bid_price"]
        numerical_cols = dataset_processor.criteo_col["feature_numerical"]

        pass
    else:
        pass
    del processor

    data_args.max_price = features[z_col].max()
    logger.info("max_price {}".format(data_args.max_price))
    y = features.pop(y_col)
    z = features.pop(z_col) / data_args.price_scale
    b = None
    if b_col is None:
        b_col = "bidprice"
    else:
        b = features.pop(b_col) / data_args.price_scale / data_args.bid_scale

    ohe = OneHotEncoder(categories='auto', handle_unknown='ignore', sparse=True)
    ohe.fit(features)
    features_ohe = ohe.transform(features)

    truthful_bidder_path = os.path.join(training_args.output_dir, "truthful_bidder.pickle")
    if training_args.overwrite_output_dir or not os.path.exists(truthful_bidder_path):
        logger.info("truthful bidder is not available, start to cache in {}".format(truthful_bidder_path))
        bidder = Truthful_bidder(verbose=0)
        bidder.fit(features_ohe[:train_size], y[:train_size], z[:train_size])
        pickle.dump(bidder, open(truthful_bidder_path, 'wb'))
    else:
        logger.info("Truthful_bidder is available, reuse cache in {}".format(truthful_bidder_path))
        bidder = pickle.load(open(truthful_bidder_path, 'rb'))

    auc = bidder.evaluate(features_ohe[train_size:], y[train_size:])
    logger.info("Truthful_bidder been trained/loaded, use time {:.2f}s, CTR auc:{:.2f}".format(timer.tick(), auc))

    b = bidder.bid(features_ohe) / data_args.price_scale  / data_args.bid_scale
    pd.DataFrame(columns=["bid_price"], data=b[:train_size]).to_csv(os.path.join(training_args.output_dir, "train_b.csv"), mode='w', index=False, sep='\t', header=True)
    pd.DataFrame(columns=["bid_price"], data=b[train_size:]).to_csv(os.path.join(training_args.output_dir, "test_b.csv"), mode='w', index=False, sep='\t', header=True)
    # TODO: important! define winning rule
    win = b > z
    del features_ohe

    if model_args.first_price:
        # if we simulate first price auctions
        # the observed z is equal to b, for the training dataset
        z[:train_size] = b[:train_size]

    # categorical
    categorical_vocabulary = {}
    categorical_vocabulary_path = os.path.join(training_args.output_dir, "categorical_vocabulary.json")
    all_columns = list(features.columns)
    for col in all_columns:
        if col in numerical_cols:
            continue
        else:
            c = pd.Categorical(features[col])
            features[col] = c
            features[col] = features[col].cat.codes
            categorical_vocabulary[col] = c.categories
            # categorical_vocabulary[col] = features[col].unique()

    with codecs.open(categorical_vocabulary_path, 'wb+', 'utf-8') as f:
        f.write(json.dumps({k:v.tolist() for k,v in categorical_vocabulary.items()}))

    logger.info("categorical features have been processed, use time {:.2f}s".format(timer.tick()))
    logger.info("categorical features' schema has beed saved in {}".format(categorical_vocabulary_path))

    train_x = features[:train_size]
    train_y = y[:train_size]
    train_z = z[:train_size]
    train_b = b[:train_size]
    train_win = win[:train_size]
    train_inputs = dict(train_x)
    train_inputs[z_col] = train_z
    train_inputs[b_col] = train_b
    train_outputs = {"pay_price": train_z, "win": train_win, "bid_price": train_b}

    test_x = features[train_size:]
    test_y = y[train_size:]
    test_z = z[train_size:]
    test_b = b[train_size:]
    test_win = win[train_size:]
    test_inputs = dict(test_x)
    test_inputs[z_col] = test_z
    test_inputs[b_col] = test_b
    test_outputs = {"pay_price": test_z, "win": test_win, "bid_price": test_b}

    test_pdf_dict = dict(test_z.value_counts())
    test_pdf = [0] * (int(data_args.max_price) + 1) # include zero
    for i in range(len(test_pdf)):
        test_pdf[i] = test_pdf_dict.get(i/data_args.price_scale, 0) / test_size
    assert np.abs(sum(test_pdf)-1) < 1e-4, "sum(test_pdf) must be 1, value: {:.5f}".format(sum(test_pdf))

    logger.info("dataset\t size\t winning rate\t")
    logger.info("train\t {:d}\t {:.3f}\t".format(train_size, train_win.sum() / train_size))
    logger.info("test\t {:d}\t nan\t".format(test_size))

    logger.info("train AMP of win\t train AMP of lose\t train AMP\t test AMP")
    logger.info("train\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t".format(
        train_z[train_win].mean(),
        train_z[~train_win].mean(),
        train_z.mean(),
        test_z.mean()))

    price_model = "first_price" if model_args.first_price else "second_price"
    log_dir = os.path.join(".", "runs", price_model, data_args.advertiser, model_args.family, str(model_args.K), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    tb_writer = tf.summary.create_file_writer(log_dir)

    model = myModels.Deep_Mixture_Density_Model(training_args,
                                                model_args,
                                                data_args,
                                                feature_col=all_columns,
                                                numerical_col=numerical_cols,
                                                pay_price_col=z_col,
                                                bid_price_col=b_col,
                                                categorical_vocabulary=categorical_vocabulary,
                                                true_pdf = test_pdf,
                                                logger=logger,
                                                tb_writer=tb_writer)

    keras.utils.plot_model(model.uncompiled_model,
                           os.path.join(training_args.output_dir, "model_architecture.png"),
                           show_shapes=True)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(training_args.learning_rate,
                                                                 decay_steps=1,
                                                                 decay_rate=0.999,
                                                                 staircase=False)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-5),
                  run_eagerly=None)
    with tb_writer.as_default():
        tf.summary.scalar('model/K', model_args.K, 0)
        tf.summary.scalar('model/auc', auc, 0)
        tf.summary.scalar('model/max_price', data_args.max_price, 0)
        tf.summary.scalar('train/size', train_size, 0)
        tf.summary.scalar('train/size_win', train_win.sum(), 0)
        tf.summary.scalar('train/winning_rate', train_win.sum() / train_size, 0)
        tf.summary.scalar('train/AMP', train_z.mean(), 0)
        tf.summary.scalar('train/AMP_lose', train_z[~train_win].mean(), 0)
        tf.summary.scalar('train/AMP_win', train_z[train_win].mean(), 0)

        tf.summary.scalar('test/size', test_size, 0)
        tf.summary.scalar('test/AMP', test_z.mean(), 0)

    model.fit(
        train_inputs,
        train_outputs,
        epochs=training_args.epoch,
        batch_size=training_args.batch_size,
        validation_data=(test_inputs, test_outputs),
        verbose=1 if training_args.debug else 2,
        callbacks=[tensorboard_callback])

    return


if __name__ == "__main__":
    rs = main()

