#!/usr/bin/env python3

import argparse
from functools import partial

import numpy as np
import torch

from controversialstimuli.optimization.torch_transform_image import ActMaxTorchTransfBaseUni
from controversialstimuli.optimization.controversial_objectives import ContrastiveNeuronUnif, ObjectiveIncrease
from controversialstimuli.optimization.stoppers import EMALossStopper

from controversialstimuli.models.clustering_model import TestOnOffModel
from controversialstimuli.models.optimization_clipper import OptimizationClipperNoop
from controversialstimuli.clustering.kmeans import KMeans, DefaultLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run clustering on simple on/off neuron model with "
                    "a two dimensional 18x18 stimulus image")

    parser.add_argument(
        "--num_clusters",
        default=5,
        type=int,
        help="Number of initial clusters to optimize",
    )

    parser.add_argument(
        "--num_neurons",
        default=42,
        type=int,
        help="Number of neurons in on/off model"
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="The device to run on, e.g. cpu or cuda"
    )

    parser.add_argument(
        "--save_data_path",
        default="./kmeans_results",
        help="The folder in which the output of the clustering is stored in (including plots)"
    )

    parser.add_argument(
        "--max_iterations_kmeans",
        default=10,
        type=int,
        help="Number of maximal kmeans iterations",
    )

    parser.add_argument(
        "--max_iterations_optimization",
        default=50,
        type=int,
        help="Max iterations of each stimulus optimization during the optimization step of the clustering algorithm"
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )

    parser.add_argument(
        "--reinitialize_stimuli",
        action="store_true",
        help="Reinitialize stimuli before optimization",
    )

    parser.add_argument(
        "--temperature",
        default=1.6,
        type=float,
        help="Temperature used in the contrastive loss function"
    )

    parser.add_argument(
        "--lr",
        default=100.0,
        type=float,
        help="Learning rate for stimulus optimization",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(args)
    postprocessing = OptimizationClipperNoop()  # do not postprocess the optimized image
    # The shape of the stimulus we optimize, in this case it's a 2d 18x18 pixel input
    # corresponding to a grayscale image
    canvas_size = (18, 18)

    model = TestOnOffModel(args.num_neurons)
    # Values will be overwritten
    objective = ContrastiveNeuronUnif(
        on_clust_idx=np.array([0]),
        off_clust_idc=np.array([1]),
        clust_assignments=np.array([0 for _ in range(args.num_neurons)]),
        temperature=args.temperature,
        device=args.device,
    )

    optimizer_init_fn = partial(torch.optim.SGD, lr=args.lr)
    optim = ActMaxTorchTransfBaseUni(
        model=model,
        seed=args.seed,
        canvas_size=canvas_size,
        verbose=True,
        optimizer_init_fn=optimizer_init_fn,
        stopper=EMALossStopper(verbose=True),
        objective_fct=objective,
        postprocessing=postprocessing,
        max_iter=args.max_iterations_optimization,
        num_imgs=1,
        device=args.device,
    )

    kmeans = KMeans(
        img_optimizer=optim,
        unit_idc=np.arange(args.num_neurons),
        cluster_idc=np.arange(args.num_clusters),
        init_cluster_assignments=None,
        seed=args.seed,
        max_iter=args.max_iterations_kmeans,
        verbose=True,
        output_folder=args.save_data_path,
        disable_progress_bar=True,
        reinitialize_stimuli=args.reinitialize_stimuli,
        plot=True,
        device=args.device,
    )

    # Run initial clustering
    kmeans.run()
    # Split cluster to potentially find a better number of clusters
    kmeans.run_split_cluster(
        args.num_clusters,
        use_rgc_init=False,
        optim_steps_to_estimate_loss=20,
        use_self_plot=True,
    )

    # Further optimize images and evaluate loss
    kmeans._img_optimizer.max_iter = 100
    kmeans._optim_images(reinitialize_stimuli=False)
    final_loss = kmeans.evaluate_training_loss()

    # Safe final kmeans state
    kmeans.save_cluster_results_to_folder(args.save_data_path, "final_kmeans_results.pkl")
    # plot confusion matrices
    kmeans.plot_response_confusion_matrix(
        f"{args.save_data_path}/final_response_confusion_matrix.pdf", title=f"Final loss: {final_loss:.5f}"
    )
    print(f"Finished clustering with {kmeans._num_clusters} clusters and a final loss of: {final_loss:.5f}")


if __name__ == "__main__":
    main()
