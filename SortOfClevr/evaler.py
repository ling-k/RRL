from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

from util import log

from input_ops import create_input_ops, check_data_id
from vqa_util import NUM_COLOR

import os
import time
import numpy as np
import tensorflow as tf
from vqa_util import *

class EvalManager(object):
    def __init__(self):
        # collection of batches (not flattened)
        self._ids = []
        self._predictions = []
        self._groundtruths = []

    def add_batch(self, id, prediction, groundtruth):

        # for now, store them all (as a list of minibatch chunks)
        self._ids.append(id)
        self._predictions.append(prediction)
        self._groundtruths.append(groundtruth)

    def report(self):
        # report L2 loss
        log.info("Computing scores...")
        correct_prediction_nr = 0
        count_nr = 0
        correct_prediction_r = 0
        count_r = 0

        for id, pred, gt in zip(self._ids, self._predictions, self._groundtruths):
            for i in range(pred.shape[0]):
                # relational
                if np.argmax(gt[i, :]) < NUM_COLOR:
                    count_r += 1
                    if np.argmax(pred[i, :]) == np.argmax(gt[i, :]):
                        correct_prediction_r += 1
                # non-relational
                else:
                    count_nr += 1
                    if np.argmax(pred[i, :]) == np.argmax(gt[i, :]):
                        correct_prediction_nr += 1

        avg_nr = float(correct_prediction_nr)/count_nr
        log.infov("Average accuracy of non-relational questions: {}%".format(avg_nr*100))
        avg_r = float(correct_prediction_r)/count_r
        log.infov("Average accuracy of relational questions: {}%".format(avg_r*100))
        avg = float(correct_prediction_r+correct_prediction_nr)/(count_r+count_nr)
        log.infov("Average accuracy: {}%".format(avg*100))


class Evaler(object):

    @staticmethod
    def get_model_class(model_name):
        if model_name == 'baseline':
            from model_baseline import Model
        elif model_name == 'rn':
            from model_rn import Model
        elif model_name == 'ilp':
            from model_ilp import Model
        else:
            raise ValueError(model_name)
        return Model

    def __init__(self,
                 config,
                 dataset):
        self.config = config
        self.train_dir = config.train_dir
        log.info("self.train_dir = %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        self.dataset = dataset

        check_data_id(dataset, config.data_id)
        _, self.batch = create_input_ops(dataset, self.batch_size,
                                         data_id=config.data_id,
                                         is_training=False,
                                         shuffle=False)

        # --- create model ---
        Model = self.get_model_class(config.model)
        log.infov("Using Model class : %s", Model)
        self.model = Model(config)

        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.step_op = tf.no_op(name='step_no_op')

        tf.set_random_seed(1234)

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = tf.Session(config=session_config)

        # --- checkpoint and monitoring ---
        self.saver = tf.train.Saver(max_to_keep=100)

        self.checkpoint_path = config.checkpoint_path
        if self.checkpoint_path is None and self.train_dir:
            self.checkpoint_path = tf.train.latest_checkpoint(self.train_dir)
        if self.checkpoint_path is None:
            log.warn("No checkpoint is given. Just random initialization :-)")
            self.session.run(tf.global_variables_initializer())
        else:
            log.info("Checkpoint path : %s", self.checkpoint_path)

    def eval_run(self):
        # load checkpoint
        if self.checkpoint_path:
            self.saver.restore(self.session, self.checkpoint_path)
            log.info("Loaded from checkpoint!")

        log.infov("Start 1-epoch Inference and Evaluation")

        log.info("# of examples = %d", len(self.dataset))
        length_dataset = len(self.dataset)

        max_steps = int(length_dataset / self.batch_size) + 1
        log.info("max_steps = %d", max_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.session,
                                               coord=coord, start=True)

        evaler = EvalManager()
        try:
            for s in xrange(max_steps):
                step, loss, step_time, batch_chunk, prediction_pred, prediction_gt = \
                    self.run_single_step(self.batch)
                self.log_step_message(s, loss, step_time)
                evaler.add_batch(batch_chunk['id'], prediction_pred, prediction_gt)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        try:
            coord.join(threads, stop_grace_period_secs=3)
        except RuntimeError as e:
            log.warn(str(e))

        evaler.report()
        log.infov("Evaluation complete.")

    def run_single_step(self, batch, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        [xxxx,xo,x0,step, accuracy, all_preds, all_targets, _] = self.session.run(
            [self.model.mdl.xxxx,  self.model.XO,self.model.X0,self.global_step, self.model.accuracy, self.model.all_preds, self.model.a, self.step_op],
            feed_dict=self.model.get_feed_dict(batch_chunk)
        )

        _end_time = time.time()
        # xo = 
        obj = xo['obj']
        rect = xo['rectangle']
        cl0=np.round(10*xo['is_color_0'])/10
        cl1=np.round(10*xo['is_color_1'])/10
        cl2=np.round(10*xo['is_color_2'])/10
        cl3=np.round(10*xo['is_color_3'])/10
        cl4=np.round(10*xo['is_color_4'])/10
        cl5=np.round(10*xo['is_color_5'])/10
        
        
        # al5=np.round(10*xo['is_l2_5'])/10
        


        oks=np.argmax( batch_chunk['q'][np.argwhere(np.argmax( all_targets,-1 )==np.argmax( all_preds,-1 )).flatten(),6:],-1 )
        return step, accuracy, (_end_time - _start_time), batch_chunk, all_preds, all_targets

    def log_step_message(self, step, accuracy, step_time, is_train=False):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "batch total-accuracy (test): {test_accuracy:.2f}% " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         test_accuracy=accuracy*100,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time,
                         )
               )


def check_data_path(path):
    if os.path.isfile(os.path.join(path, 'data.hy')) \
           and os.path.isfile(os.path.join(path, 'id.txt')):
        return True
    else:
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16*1)
    parser.add_argument('--model', type=str, default='ilp', choices=['rn', 'baseline'])
    # parser.add_argument('--checkpoint_path', type=str,default='./train_dir/ilp-default-Sort-of-CLEVR_default_lr_0.0025-20190619-195552/model-32000')
    # parser.add_argument('--checkpoint_path', type=str,default='./train_dir/ilp-default-Sort-of-CLEVR_default_lr_0.0025-20190619-115754/model-42000')
    parser.add_argument('--checkpoint_path', type=str,default='./train_dir/ilp-default-Sort-of-CLEVR_default_lr_0.002-20190807-173045/model-80000')
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--dataset_path', type=str, default='Sort-of-CLEVR_default')
    parser.add_argument('--data_id', nargs='*', default=None)
    config = parser.parse_args()

    path = os.path.join('./datasets', config.dataset_path)

    if check_data_path(path):
        import sort_of_clevr as dataset
    else:
        raise ValueError(path)

    config.data_info = dataset.get_data_info()
    config.conv_info = dataset.get_conv_info()
    dataset_train, dataset_test = dataset.create_default_splits(path)

    evaler = Evaler(config, dataset_test)

    # qs1=[]
    # qs2=[]
    # ans=[]
    # for id in dataset_test._ids:
    #     dt = dataset_train.get_data(id)
    #     qs1.append( np.argmax(dt[1][:6]) )
    #     qs2.append( np.argmax(dt[1][6:]) )
    #     ans.append( np.argmax(dt[2]) )


    log.warning("dataset: %s", config.dataset_path)
    evaler.eval_run()

if __name__ == '__main__':
    main()
