""" Utility functions for Jupyter notebooks """
import os
import json
import subprocess
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import chainer
import chainer.links as L
from chainerlp.links import models
from chainerlp.training.load_only_updator import LoadOnlyUpdator


def get_train_dir(
    dtype=None, dataset=None, model=None, rootdir=None, seed=None, **kwargs
):
    """ Get the training directory. """
    train_dir = "{rootdir}/{dataset}/{model}_{dtype}".format(
        dataset=dataset, model=model, dtype=dtype, rootdir=rootdir
    )
    if seed is not None:
        train_dir += "_{}".format(seed)
    for key, val in kwargs.items():
        if val is None:
            continue
        if isinstance(val, float) and np.isnan(val):
            continue
        train_dir += "_{}_{}".format(key, val)

    return train_dir


######################################
# Cluster related                    #
######################################
USER = os.environ.get("USER", None)
SERVER = os.environ.get("SERVER", None)
CLI = os.environ.get("CLUSTER_CLI", None)
DEFAULT_SPEC_FILE = "{}spec.yml".format(CLI)
PROJDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
CLI_RUN_ARGS = [
    # '--auto-persist',
]


def download_from_cluster(
    dtype,
    dataset,
    model,
    job_id,
    rootdir,
    files=None,
    ignore_job_done=False,
    seed=None,
    persist=True,
    verbose=False,
    **kwargs
):
    """ Download result from the cluster to local directory """
    if isinstance(job_id, int):
        job_id = str(job_id)

    info = get_job_info(job_id)
    if not is_job_done(info) and not ignore_job_done:
        return None
    if persist:
        execute_job_command(job_id, "persist-results")
    if files is None:
        files = ["log"]

    train_dir = get_train_dir(dtype, dataset, model, rootdir, seed=seed, **kwargs)
    args = [*files, "--dest", train_dir]
    if verbose:
        print("Downloading to directory: {} ...".format(train_dir))

    return execute_job_command(job_id, "get-results", args=args)


def run_on_cluster(args, cli_args=None):
    """ Launch a run task on cluster """
    if cli_args is None:
        cli_args = []
    args = [
        CLI,
        "--user",
        USER,
        "--server",
        SERVER,
        "run",
        *cli_args,
        "--",
        *args,
    ]
    return subprocess.run(args=args, capture_output=True, cwd=PROJDIR)


def train_on_cluster(
    train_script,
    train_args,
    cli_args=None,
    label="train_on_cluster",
    spec_file=DEFAULT_SPEC_FILE,
    verbose=False,
):
    """ Launch training task on the cluster without MPI environment """
    if cli_args is None:
        cli_args = []
    cli_args.extend(
        [
            "-A",
            "nvidia.k8s.pfn.io/cuda-version=10.0",
            "--auto-persist",
            "--spec",
            spec_file,
            "--label",
            label,
        ]
    )
    args = [
        "python3",
        train_script,
        *train_args,
    ]
    return run_on_cluster(args, cli_args=cli_args)


def train_cifar_on_cluster(
    arch,
    dataset,
    dtype,
    n_epoch,
    schedule,
    manual_seed=0,
    learnrate=0.1,
    weight_decay=1e-4,
    lr_decay=0.1,
    warmup_lr_ratio=0.1,
    n_warmup_epoch=None,
    snapshot_freq=10,
    use_fixup=False,
    custom_label=None,
):
    """ Train model on CIFAR-10 on the cluster environment.
        Single GPU is required.
    """
    train_script = "examples/cifar/train_cifar.py"
    spec_file = os.path.join(PROJDIR, "examples", "cifar", DEFAULT_SPEC_FILE)
    label = "{arch}-{dataset}-{dtype}-{seed}".format(
        arch=arch, dataset=dataset, dtype=dtype, seed=manual_seed
    )
    if custom_label is not None:
        label = custom_label + "-" + label  # add a prefix

    schedule_ = [str(s) for s in schedule]
    warmup_lr = learnrate * warmup_lr_ratio

    train_args = [
        "--dataset",
        dataset,
        "--arch",
        arch,
        "--dtype",
        dtype,
        "-b",
        "128",
        "-e",
        str(n_epoch),
        "-s",
        *schedule_,
        "--learnrate",
        str(learnrate),
        "--manual-seed",
        str(manual_seed),
        "--device",
        "0",
        "--weight-decay",
        str(weight_decay),
        "--lr-decay",
        str(lr_decay),
        "--out",
        "/home/user/results",
        "--snapshot-freq",
        str(snapshot_freq),
    ]

    if use_fixup:
        train_args.append("--use-fixup")
    if n_warmup_epoch is not None:
        train_args.extend(
            ["--warmup-lr", str(warmup_lr), "--warmup-epoch", str(n_warmup_epoch),]
        )

    cli_args = []

    # HACK: need to launch on larger GPUs
    if dtype == "float32" and arch == "resnet1202":
        cli_args.extend(
            ["-A", "nvidia.k8s.pfn.io/gpu_model=Tesla-P100-PCIE-16GB",]
        )

    return train_on_cluster(train_script, train_args, label=label, spec_file=spec_file)


def train_imagenet_on_cluster(
    arch,
    dataset="imagenet",
    dtype="float32",
    manual_seed=0,
    first_bn_mixed16=None,
    dataset_dir="/home/user/data/imagenet",
    mpi="4x4",
    snapshot_freq=1,
    verbose=False,
):
    """ Launch ImageNet training task on cluster """
    train_script = "examples/imagenet/train_imagenet_multi.py"
    spec_file = os.path.join(PROJDIR, "examples", "imagenet", DEFAULT_SPEC_FILE)
    label = "{arch}-{dataset}-{dtype}".format(arch=arch, dataset=dataset, dtype=dtype)

    train_args = [
        "--dataset-dir",
        dataset_dir,
        "-a",
        arch,
        "--dtype",
        dtype,
        "--manual-seed",
        str(manual_seed),
        "--out",
        "/home/user/results",
        "--snapshot-freq",
        str(snapshot_freq),
    ]
    if first_bn_mixed16:
        train_args.append("--first-bn-mixed16")

    cli_args = ["--mpi", mpi]

    return train_on_cluster(
        train_script,
        train_args,
        label=label,
        cli_args=cli_args,
        spec_file=spec_file,
        verbose=verbose,
    )


def execute_job_command(job_id, command, args=None):
    """ Collect job information by job_id """
    if isinstance(job_id, int):
        job_id = str(job_id)
    if args is None:
        args = []
    args = [
        CLI,
        "--user",
        USER,
        "--server",
        SERVER,
        "job",
        job_id,
        command,
        *args,
    ]
    return subprocess.run(args=args, capture_output=True)


def get_job_info(job_id):
    """ Collect job information """
    while True:
        p = execute_job_command(job_id, "info")
        # NOTE: this task may fail
        if p.returncode == 0:
            break
        else:
            print("[WARN] Retrying get job info of {} ...".format(job_id))

    return p.stdout.decode("utf-8")


def get_job_attr(job_info, attr_name):
    """ Get the content from a job attribute """
    lines = [s.strip() for s in job_info.split("\n")]

    # Find the line that starts with the attribute
    cmd_line = next((l for l in lines if l.startswith(attr_name)), None)
    if cmd_line is None:
        return None

    cmd = cmd_line.split(" ")[1:]  # first entry is 'cmmand:'
    return cmd


def get_job_id(p):
    """ Collect job_id from a job launching process p """
    s = p.stderr.decode("utf-8")
    try:
        job_id = s.split("\n")[-3].split(" ")[-1]  # TODO: improve
        return int(job_id)  # may raise value error
    except ValueError:
        return None


def is_job_failed(job_info):
    return "failed" in job_info


def is_job_done(job_info):
    return "completed successfully" in job_info


def is_job_running(job_info):
    return "running" in job_info


def get_job_status(job_info):
    if is_job_failed(job_info):
        return "FAILED"
    if is_job_done(job_info):
        return "DONE"
    if is_job_running(job_info):
        return "RUNNING"

    return "UNKNOWN"


def relaunch_failed_job(job_id, spec_file):
    """ Relaunch the selected failed job.
        spec_file should be provided or we cannot know which spec you're using
    """
    job_info = get_job_info(job_id)
    if not is_job_failed(job_info):
        return None

    args = get_job_attr(job_info, "command")
    label = get_job_attr(job_info, "label")
    attrs = get_job_attr(job_info, "attributes")

    cli_args = [*CLI_RUN_ARGS, "--spec", spec_file]
    if label is not None:
        cli_args.extend(["--label", *label])
    if attrs is not None:
        cli_args.extend(["-A", *attrs])

    return run_on_cluster(args, cli_args=cli_args)


def relaunch_failed_jobs(tasks, spec_file, verbose=False):
    """ Relaunch jobs that are failed from the given list """
    job_cnts = 0  # number of newly launched jobs

    for i, task in enumerate(tasks):
        job_id = str(task[-1])  # the last entry

        # Try to launch until succeed
        while True:
            p = relaunch_failed_job(job_id, spec_file)
            if p is None:  # NOTE: when the job is not failed
                break

            if verbose:
                print("==> Re-launching failed task: {} ...".format(task))
            new_id = get_job_id(p)
            if new_id is not None:
                break

        # If a new process is launched
        if p is not None:
            tasks[i][-1] = new_id
            job_cnts += 1

    return job_cnts


def check_task_status(tasks, verbose=False):
    """ Go through all the tasks and see how their running status are """
    print(time.ctime())  # Print a timestamp (NECESSARY)

    stats = {}

    for task in tasks:
        job_id = str(task[-1])  # needs to be str
        info = get_job_info(job_id)
        status = get_job_status(info)
        if verbose:
            print("Status of task {}:\t{}".format(task, status))

        # update the statistics
        if status not in stats:
            stats[status] = 0
        stats[status] += 1

    return stats


def cleanup_jobs(jobs, verbose=False):
    """ Cleanup the jobs remaining on the cluster """
    for job_id in jobs:
        if verbose:
            print("==> Cleaning-up job {} ...".format(job_id))
        job_info = get_job_info(job_id)
        status = get_job_status(job_info)
        if status == "RUNNING":  # kill if the job is running
            if verbose:
                print("==> Killing ...")
            execute_job_command(job_id, "kill")

        # unpersist results
        execute_job_command(job_id, "unpersist-results")


######################################
# File IO                            #
######################################


def load_train_log(
    dtype=None,
    dataset=None,
    model=None,
    rootdir=None,
    seed=None,
    train_dir=None,
    **kwargs
):
    """ Load the log file by Pandas """
    if train_dir is None:
        assert dtype in ["float32", "float16", "mixed16"]
        train_dir = get_train_dir(
            dtype=dtype,
            dataset=dataset,
            model=model,
            rootdir=rootdir,
            seed=seed,
            **kwargs
        )
    if not os.path.isdir(train_dir):
        print("[WARN] train_dir to load data cannot be found: {}".format(train_dir))
        return None

    fp = "{}/log".format(train_dir)
    with open(fp, "r") as f:
        df = pd.DataFrame(json.loads(f.read()))
    return df


def plot_train_log(
    dataset=None,
    model=None,
    seed=None,
    rootdir=None,
    savedir=None,
    dtypes=None,
    x_axis="epoch",
    fig=None,
    ax1=None,
    ax2=None,
    label_suffix=None,
    **kwargs
):
    """ Plot the training log of a single model. """
    assert isinstance(rootdir, str)

    if dtypes is None:
        dtypes = ["float32", "float16"]
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 4))

    ax1.set_ylabel("Validation accuracy (%)")
    ax1.set_xlabel(x_axis)
    ax2.set_ylabel("Train loss")
    ax2.set_xlabel(x_axis)

    for dtype in dtypes:
        train_log = load_train_log(dtype, dataset, model, rootdir, seed=seed, **kwargs)

        if train_log is None:
            continue
        if label_suffix is None:
            label_suffix = " ({})".format(seed) if seed is not None else ""

        dtype_in_label = dtype[0] + dtype[-2:]
        val_acc = train_log[~train_log["validation/main/accuracy"].isnull()]
        ax1.plot(
            val_acc[x_axis],
            val_acc["validation/main/accuracy"] * 100,
            label="{} {}{}".format(model, dtype_in_label, label_suffix),
        )
        ax2.plot(
            train_log[x_axis],
            train_log["main/loss"],
            label="{} {}{}".format(model, dtype_in_label, label_suffix),
        )

    ax1.legend()
    ax2.legend()
    fig.suptitle("{} on {}".format(model, dataset))
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        fig.savefig(os.path.join(savedir, "{}_{}_train_log.pdf".format(model, dataset)))


def plot_train_logs(
    dataset, models, rootdir=None, savedir=None, x_axis="epoch", dtypes=None, seeds=None
):
    """ Plot training logs of multiple models
    
        There will be two subplots, on the left is the validation accuracy, 
        on the right is training loss.
    """
    if seeds is None:
        seeds = [None]
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 4))

    for model in models:
        for seed in seeds:
            plot_train_log(
                dataset,
                model,
                seed=seed,
                rootdir=rootdir,
                savedir=None,
                x_axis=x_axis,
                dtypes=dtypes,
                fig=fig,
                ax1=ax1,
                ax2=ax2,
            )

    ax1.legend()
    ax2.legend()
    fig.suptitle("Results on {}".format(dataset))
    if savedir is not None:
        fig.savefig(
            os.path.join(
                savedir, "{}_{}_train_log.pdf".format("_".join(models), dataset)
            )
        )


def load_snapshot_and_create_model(
    arch,
    snapshot_name=None,
    snapshot_dir=None,
    dataset="imagenet",
    dtype="float32",
    n_class=1000,
    rootdir=None,
    job_id=None,
    seed=None,
    device=-1,
    **kwargs
):
    """ Load the snapshot of a Trainer into a model. """
    # create model
    chainer.config.dtype = get_chainer_dtype(dtype)
    model = L.Classifier(models.__dict__[arch](n_class=n_class))

    if device != -1:
        chainer.cuda.get_device(device).use()
        model.to_gpu()
    if snapshot_name is None:  # will return the initialized model
        return model

    if snapshot_dir is not None:
        train_dir = snapshot_dir
    else:
        assert isinstance(rootdir, str)
        assert os.path.isdir(rootdir)

        train_dir = get_train_dir(dtype, dataset, arch, rootdir, seed=seed, **kwargs)

    fp = os.path.join(train_dir, snapshot_name)
    if not os.path.isdir(train_dir) or not os.path.isfile(fp):
        print("==> Snapshot not found: {}. Downloading ...".format(fp))
        download_from_cluster(
            dtype,
            dataset,
            arch,
            job_id,
            rootdir,
            files=[snapshot_name],
            ignore_job_done=True,
            seed=seed,
            **kwargs
        )

    # create the trainer
    optim = chainer.optimizers.CorrectedMomentumSGD()
    optim.setup(model)

    updator = LoadOnlyUpdator(None, optim, device=0)
    trainer = chainer.training.Trainer(updator)
    chainer.serializers.load_npz(fp, trainer)

    return model


def get_chainer_dtype(dtype):
    if dtype == "float32":
        return np.float32
    if dtype == "float16":
        return np.float16
    if dtype == "mixed16":
        return chainer.mixed16


def get_snapshot_parameters(
    arch,
    dtype,
    snapshot_name=None,
    dataset="imagenet",
    rootdir=None,
    n_class=1000,
    job_id=None,
    device=0,
    seed=None,
    fig=None,
    ax=None,
    savedir=None,
    includes=None,
    **kwargs
):
    """ We will iterate the model and print out all the BN parameters """
    net = load_snapshot_and_create_model(
        arch,
        snapshot_name,
        dataset=dataset,
        dtype=dtype,
        rootdir=rootdir,
        n_class=n_class,
        job_id=job_id,
        seed=seed,
        device=device,
        **kwargs
    )

    if includes is None:
        includes = []

    results = []
    for name, link in net.predictor.namedlinks():
        params = list(link.namedparams())
        results.append((name, params))

    return results
