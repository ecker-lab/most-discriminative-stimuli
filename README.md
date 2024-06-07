# Most Discriminative Stimuli

Official implementation of the optimization based clustering algorithm described in the ICLR 2024 paper [Most discriminative stimuli for functional cell type clustering](https://openreview.net/forum?id=9W6KaAcYlr).

## Installation
The code is tested using python3.9 and 3.10. Install it as follows:
```commandline
# Optional: create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
# Install repository
pip install .
```

## Example
We provide an example script that clusters neurons of a very simple example model 
containing only on and off cells. We start the clustering by assuming the neurons of the model
can be grouped into 5 clusters, and the algorithm converges to the correct number of 2 clusters.
```commandline
# Print help message
./scripts/run_kmeans_example.py --help
# Run example clustering
./scripts/run_kmeans_example.py --num_clusters 5 --num_neurons 42 --device cpu --save_data_path "kmeans_test_results"
```
You will then find the output including plots of the clustering in the folder `kmeans_test_results`.

We will continuously add code to reproduce the paper figures.

## Code authors
- Max Burg: [Google Scholar](https://scholar.google.com/citations?user=-T_5tc0AAAAJ&hl=de&oi=ao) [GitHub](https://github.com/MaxFBurg)
- Thomas Zenkel: [Google Scholar](https://scholar.google.com/citations?user=jn2QYvoAAAAJ&hl=de&oi=ao) [GitHub](https://github.com/thomasZen)

## Cite as
```
@inproceedings{
burg2024most,
title={Most discriminative stimuli for functional cell type clustering},
author={Max F Burg and Thomas Zenkel and Michaela Vystr{\v{c}}ilov{\'a} and Jonathan Oesterle and Larissa H{\"o}fling and Konstantin Friedrich Willeke and Jan Lause and Sarah M{\"u}ller and Paul G. Fahey and Zhiwei Ding and Kelli Restivo and Shashwat Sridhar and Tim Gollisch and Philipp Berens and Andreas S. Tolias and Thomas Euler and Matthias Bethge and Alexander S Ecker},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=9W6KaAcYlr}
}
```
