""" Test the activation statistics hook. """

import unittest
import os
import tempfile
import pickle
import numpy as np
import chainer
import chainer.functions as F  # testing purpose
import chainer.links as L
from chainer import testing
from chainer import Function, FunctionNode, gradient_check, report, training, Variable
from chainer import datasets, initializers, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.datasets import mnist  # for trainer test

from chainerlp import utils
from chainerlp.hooks.act_stat_hook import ActStatFuncHook

utils.set_random_seed(0)


class TestActStatHook(unittest.TestCase):
    def test_forward(self):
        """ ActStatFuncHook should work properly for the forward pass
            of a model. Properly means the input data should be correctly
            collected.
        """
        hook = ActStatFuncHook()
        with hook:
            data = np.random.random((3, 3)).astype(np.float32) - 0.5
            x = chainer.Variable(data)
            _ = F.relu(x)

        self.assertEqual(len(hook.call_history), 1)  # one history recorded
        idx, label, in_stats, out_stats = list(hook.call_history.values())[0]

        self.assertEqual(idx, 1)
        self.assertEqual(label, "ReLU")
        self.assertEqual(len(in_stats), 1)  # single input
        # check stats
        self.assertTrue(np.allclose(in_stats[0][0], data.min()))
        self.assertTrue(np.allclose(in_stats[0][1], data.max()))
        self.assertTrue(np.allclose(in_stats[0][2], data.mean()))
        self.assertTrue(np.allclose(in_stats[0][3], data.std()))
        # output data should be None
        self.assertIsNone(out_stats)

    def test_backward(self):
        """ ActStatFuncHook should work properly for the backward pass
            of a model as well.
        """
        hook = ActStatFuncHook()
        with hook:
            shape = (3, 3)
            data = np.random.random(shape).astype(np.float32) - 0.5
            x = chainer.Variable(data)
            y = F.relu(x)
            y_grad = np.ones(shape, dtype=np.float32)
            y.grad = y_grad  # assign grad
            # NOTE: y.grad will be None after being used.
            y.backward()

        self.assertEqual(len(hook.call_history), 2)  # two functions
        idx, label, in_stats, out_stats = list(hook.call_history.values())[0]

        self.assertEqual(idx, 1)
        self.assertEqual(label, "ReLU")
        self.assertEqual(len(in_stats), 1)  # single input
        # check stats
        self.assertTrue(np.allclose(in_stats[0][0], data.min()))
        self.assertTrue(np.allclose(in_stats[0][1], data.max()))
        self.assertTrue(np.allclose(in_stats[0][2], data.mean()))
        self.assertTrue(np.allclose(in_stats[0][3], data.std()))
        # output data should be None
        self.assertEqual(len(out_stats), 1)
        self.assertTrue(np.allclose(1.0, out_stats[0][0]))
        self.assertTrue(np.allclose(1.0, out_stats[0][1]))
        self.assertTrue(np.allclose(1.0, out_stats[0][2]))
        self.assertTrue(np.allclose(0.0, out_stats[0][3]))

    def test_snapshot(self):
        """ Manually call the snapshot function. """

        with tempfile.TemporaryDirectory() as snapshot_dir:
            # Use a temporary directory to store snapshots

            hook = ActStatFuncHook(snapshot_dir=snapshot_dir)
            n_iter = 3
            call_histories = []
            with hook:
                shape = (3, 3)

                for _ in range(n_iter):
                    data = np.random.random(shape).astype(np.float32) - 0.5

                    x = chainer.Variable(data)
                    y = F.relu(x)
                    y_grad = np.ones(shape, dtype=np.float32)
                    y.grad = y_grad  # assign grad
                    # NOTE: y.grad will be None after being used.
                    y.backward()

                    # cache call histroy
                    call_histories.append(hook.call_history)
                    hook.snapshot()

            # called snapshot for n_iter times
            self.assertEqual(len(os.listdir(snapshot_dir)), n_iter)
            self.assertEqual(len(hook.call_history), 0)  # all cleared

            # check all the snapshot files, compare them with call_histories.
            for i, snapshot_name in enumerate(sorted(os.listdir(snapshot_dir))):
                self.assertIn("_{}".format(i + 1), snapshot_name)

                fp = os.path.join(snapshot_dir, snapshot_name)
                with open(fp, "rb") as f:
                    call_history = pickle.load(f)

                # TODO: just verify keys at this stage
                self.assertListEqual(
                    list(call_history.keys()), list(call_histories[i].keys())
                )

    def test_trainer(self):
        """ Test a normal training procedure. """
        train, _ = mnist.get_mnist()
        batchsize = 128

        train_iter = iterators.SerialIterator(train, batchsize)

        class MLP(Chain):
            def __init__(self, n_mid_units=100, n_out=10):
                super(MLP, self).__init__()
                with self.init_scope():
                    self.l1 = L.Linear(None, n_mid_units)
                    self.l2 = L.Linear(None, n_mid_units)
                    self.l3 = L.Linear(None, n_out)

            def forward(self, x):
                h1 = F.relu(self.l1(x))
                h2 = F.relu(self.l2(h1))
                return self.l3(h2)

        gpu_id = -1
        n_iter = 10

        model = MLP()
        model = L.Classifier(model)

        optimizer = optimizers.MomentumSGD()
        optimizer.setup(model)
        updater = training.updaters.StandardUpdater(
            train_iter, optimizer, device=gpu_id
        )

        with tempfile.TemporaryDirectory() as out_dir:
            trainer = training.Trainer(updater, (n_iter, "iteration"), out=out_dir)

            hook = ActStatFuncHook(trainer=trainer, snapshot_dir=out_dir)
            with hook:
                trainer.run()
                hook.snapshot()  # final snapshot

            self.assertEqual(len(os.listdir(out_dir)), n_iter)
            for snapshot_name in sorted(
                os.listdir(out_dir),
                key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]),
            ):
                fp = os.path.join(out_dir, snapshot_name)
                with open(fp, "rb") as f:
                    call_history = pickle.load(f)

                # all stats are filled
                # TODO: for now we only test this property
                for key, val in call_history.items():
                    if "Grad" not in val[1] and "Accuracy" not in val[1]:  # label
                        self.assertIsNotNone(val[2])
                        self.assertIsNotNone(val[3])


testing.run_module(__name__, __file__)
