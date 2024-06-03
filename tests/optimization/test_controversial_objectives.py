import numpy as np
import torch
from controversialstimuli.optimization.controversial_objectives import (
    ContrastiveNeuronUnif,
)


def test_objective_matches_basic_math():
    # verify analytical finding with toy example
    num_units = 2
    resp1_on = torch.ones(num_units, dtype=torch.float32)
    resp1_off = torch.zeros(num_units, dtype=torch.float32)
    resp2_on = torch.ones(num_units, dtype=torch.float32)
    resp2_off = torch.zeros(num_units, dtype=torch.float32)
    resp3_off = torch.zeros(num_units, dtype=torch.float32)
    resp3_on = torch.ones(num_units, dtype=torch.float32)
    cluster_assignments = torch.cat(
        [torch.zeros(num_units), torch.ones(num_units), 2 * torch.ones(num_units)]
    ).tolist()
    cluster_assignments = [int(c) for c in cluster_assignments]

    t = 1
    # 3 clusters, basic math, on cluster 0
    l = resp1_on
    ll = [l, resp2_off, resp3_off]
    l_mean = torch.mean(l)
    ll_means = torch.tensor([torch.mean(ll_i) for ll_i in ll])

    l_exp = torch.exp(1 / t * l_mean)
    ll_exps = torch.tensor([torch.exp(1 / t * ll_mean) for ll_mean in ll_means])
    ll_exps_mean = torch.mean(ll_exps)
    obj1 = torch.log(l_exp / ll_exps_mean)

    objective_fn = ContrastiveNeuronUnif(
        on_clust_idx=0,
        off_clust_idc=[1, 2],
        clust_assignments=cluster_assignments,
        device="cpu",
    )
    obj2 = objective_fn(torch.cat(ll).unsqueeze(0))
    assert torch.allclose(obj1, obj2)

    # Test: this setting is symmetric, so objective should give same results for all clusters
    t = 1
    # 3 clusters, basic math, on cluster 1
    l = resp2_on
    ll = [resp1_off, l, resp3_off]
    l_mean = torch.mean(l)
    ll_means = torch.tensor([torch.mean(ll_i) for ll_i in ll])

    l_exp = torch.exp(1 / t * l_mean)
    ll_exps = torch.tensor([torch.exp(1 / t * ll_mean) for ll_mean in ll_means])
    ll_exps_mean = torch.mean(ll_exps)
    obj1a = torch.log(l_exp / ll_exps_mean)
    assert torch.allclose(obj1a, obj1)

    objective_fn = ContrastiveNeuronUnif(
        on_clust_idx=1,
        off_clust_idc=[0, 2],
        clust_assignments=cluster_assignments,
        device="cpu",
    )
    obj2a = objective_fn(torch.cat(ll).unsqueeze(0))
    assert torch.allclose(obj2a, obj2)
    assert torch.allclose(obj1a, obj2a)

    t = 1
    # 3 clusters, basic math, on cluster 2
    l = resp3_on
    ll = [resp1_off, resp2_off, resp3_on]
    l_mean = torch.mean(l)
    ll_means = torch.tensor([torch.mean(ll_i) for ll_i in ll])

    l_exp = torch.exp(1 / t * l_mean)
    ll_exps = torch.tensor([torch.exp(1 / t * ll_mean) for ll_mean in ll_means])
    ll_exps_mean = torch.mean(ll_exps)
    obj1a = torch.log(l_exp / ll_exps_mean)
    assert torch.allclose(obj1a, obj1)

    objective_fn = ContrastiveNeuronUnif(
        on_clust_idx=2,
        off_clust_idc=[0, 1],
        clust_assignments=cluster_assignments,
        device="cpu",
    )
    obj2a = objective_fn(torch.cat(ll).unsqueeze(0))
    assert torch.allclose(obj2a, obj2)
    assert torch.allclose(obj1a, obj2a)

    # Test ignore cluster functionality
    t = 1
    # 3 clusters, basic math, on cluster 2, ignore cluster 0
    l = resp3_on
    ll = [resp2_off, resp3_on]
    l_mean = torch.mean(l)
    ll_means = torch.tensor([torch.mean(ll_i) for ll_i in ll])

    l_exp = torch.exp(1 / t * l_mean)
    ll_exps = torch.tensor([torch.exp(1 / t * ll_mean) for ll_mean in ll_means])
    ll_exps_mean = torch.mean(ll_exps)
    obj1_ignore0 = torch.log(l_exp / ll_exps_mean)

    objective_fn = ContrastiveNeuronUnif(
        on_clust_idx=2,
        off_clust_idc=[1],
        clust_assignments=cluster_assignments,
        device="cpu",
    )
    obj2_ignore0a = objective_fn(torch.cat([resp1_off] + ll).unsqueeze(0))
    obj2_ignore0b = objective_fn(torch.cat([resp1_on] + ll).unsqueeze(0))
    assert torch.allclose(obj2_ignore0a, obj2_ignore0b)
    assert torch.allclose(obj1_ignore0, obj2_ignore0a)

    # 2 clusters, basic math
    t = 1.6
    l = torch.cat([resp1_on, resp3_off[: num_units // 2]])
    ll = [l, torch.cat([resp2_off, resp3_off[num_units // 2 :]])]
    l_mean = torch.mean(l)
    ll_means = torch.tensor([torch.mean(ll_i) for ll_i in ll])

    l_exp = torch.exp(1 / t * l_mean)
    ll_exps = torch.tensor([torch.exp(1 / t * ll_mean) for ll_mean in ll_means])
    ll_exps_mean = torch.mean(ll_exps)
    obj2 = torch.log(l_exp / ll_exps_mean)

    cluster_assignments = torch.cat(
        [
            torch.zeros(num_units + num_units // 2),
            torch.ones(num_units + num_units // 2),
        ]
    ).tolist()
    objective_fn = ContrastiveNeuronUnif(
        on_clust_idx=0,
        off_clust_idc=[1],
        clust_assignments=cluster_assignments,
        temperature=t,
        device="cpu",
    )
    obj1 = objective_fn(torch.cat(ll).unsqueeze(0))
    assert torch.allclose(obj1, obj2)

    # test for a little bit more complex case
    clust2 = [1, 5, 7, 8]
    clust1 = [0, 10, 1, 1, 1]
    clust3 = [100]
    t = 1.6

    num = np.exp(np.mean(clust2) / t)
    denom2 = np.exp(np.mean(clust2) / t)
    denom1 = np.exp(np.mean(clust1) / t)
    denom3 = np.exp(np.mean(clust3) / t)
    denom = np.mean([denom1, denom2, denom3])
    obj2 = np.log(num / denom)

    clust1_logits = torch.tensor(clust1, dtype=torch.float64)
    clust2_logits = torch.tensor(clust2, dtype=torch.float64)
    clust3_logits = torch.tensor(clust3, dtype=torch.float64)

    logits = torch.cat([clust1_logits, clust2_logits, clust3_logits])
    cluster_assignments = torch.cat(
        [torch.zeros(5), torch.ones(4), torch.ones(1) * 2]
    ).tolist()

    objective_fn = ContrastiveNeuronUnif(
        on_clust_idx=1,
        off_clust_idc=[0, 2],
        clust_assignments=cluster_assignments,
        temperature=t,
        device="cpu",
    )
    obj1 = objective_fn(logits.unsqueeze(0))
    assert torch.allclose(obj1, torch.tensor(obj2, dtype=torch.float64))


def test_contrastive_neuron_decreases_for_redundant_cluster():
    num_units = 100
    resp1_on = torch.ones(num_units, dtype=torch.float32)
    resp1_off = torch.zeros(num_units, dtype=torch.float32)
    resp2_on = torch.ones(num_units, dtype=torch.float32)
    resp2_off = torch.zeros(num_units, dtype=torch.float32)
    cluster_assignments = torch.cat(
        [torch.zeros(num_units), torch.ones(num_units)]
    ).tolist()

    # 2 clusters
    obj2 = []
    objective_fn = ContrastiveNeuronUnif(
        on_clust_idx=0,
        off_clust_idc=[1],
        clust_assignments=cluster_assignments,
        device="cpu",
    )
    o = objective_fn(torch.cat([resp1_on, resp2_off]).unsqueeze(0))
    obj2.append(o)
    objective_fn = ContrastiveNeuronUnif(
        on_clust_idx=1,
        off_clust_idc=[0],
        clust_assignments=cluster_assignments,
        device="cpu",
    )
    o = objective_fn(torch.cat([resp1_off, resp2_on]).unsqueeze(0))
    obj2.append(o)
    obj2_mean = np.mean(obj2)

    # 3 clusters
    cluster_assignments = torch.cat(
        [
            torch.zeros(num_units),
            torch.ones(num_units // 2),
            2 * torch.ones(num_units // 2),
        ]
    ).tolist()
    obj3 = []
    objective_fn = ContrastiveNeuronUnif(
        on_clust_idx=0,
        off_clust_idc=[1, 2],
        clust_assignments=cluster_assignments,
        device="cpu",
    )
    o = objective_fn(torch.cat([resp1_on, resp2_off]).unsqueeze(0))
    obj3.append(o)
    objective_fn = ContrastiveNeuronUnif(
        on_clust_idx=1,
        off_clust_idc=[0, 2],
        clust_assignments=cluster_assignments,
        device="cpu",
    )
    o = objective_fn(torch.cat([resp1_off, resp2_on]).unsqueeze(0))
    obj3.append(o)
    objective_fn = ContrastiveNeuronUnif(
        on_clust_idx=2,
        off_clust_idc=[1, 0],
        clust_assignments=cluster_assignments,
        device="cpu",
    )
    o = objective_fn(torch.cat([resp1_off, resp2_on]).unsqueeze(0))
    obj3.append(o)
    obj3_mean = np.mean(obj3)

    assert (
        obj3_mean < obj2_mean
    ), "objective with 3 clusters should be lower than objective with 2 clusters because 3rd cluster is redundant for a toy data set only holding 2 clusters"
    print(obj3_mean, obj2_mean)


def test_contrastive_neuron_decreases_when_removing_required_cluster():
    num_units = 100
    resp1_on = torch.ones(num_units, dtype=torch.float32)
    resp1_off = torch.zeros(num_units, dtype=torch.float32)
    resp2_on = torch.ones(num_units, dtype=torch.float32)
    resp2_off = torch.zeros(num_units, dtype=torch.float32)
    resp3_on = torch.ones(num_units, dtype=torch.float32)
    resp3_off = torch.zeros(num_units, dtype=torch.float32)
    cluster_assignments = torch.cat(
        [torch.zeros(num_units), torch.ones(num_units), 2 * torch.ones(num_units)]
    ).tolist()

    # 3 clusters
    obj3 = []
    objective_fn = ContrastiveNeuronUnif(
        on_clust_idx=0,
        off_clust_idc=[1, 2],
        clust_assignments=cluster_assignments,
        device="cpu",
    )
    o = objective_fn(torch.cat([resp1_on, resp2_off, resp3_off]).unsqueeze(0))
    obj3.append(o)
    objective_fn = ContrastiveNeuronUnif(
        on_clust_idx=1,
        off_clust_idc=[0, 2],
        clust_assignments=cluster_assignments,
        device="cpu",
    )
    o = objective_fn(torch.cat([resp1_off, resp2_on, resp3_off]).unsqueeze(0))
    obj3.append(o)
    objective_fn = ContrastiveNeuronUnif(
        on_clust_idx=2,
        off_clust_idc=[1, 0],
        clust_assignments=cluster_assignments,
        device="cpu",
    )
    o = objective_fn(torch.cat([resp1_off, resp2_off, resp3_on]).unsqueeze(0))
    obj3.append(o)
    obj3_mean = np.mean(obj3)

    # 2 clusters
    cluster_assignments = torch.cat(
        [
            torch.zeros(num_units + num_units // 2),
            torch.ones(num_units + num_units // 2),
        ]
    ).tolist()
    obj2 = []
    objective_fn = ContrastiveNeuronUnif(
        on_clust_idx=0,
        off_clust_idc=[1],
        clust_assignments=cluster_assignments,
        device="cpu",
    )
    o = objective_fn(
        torch.cat(
            [
                torch.cat([resp1_on, resp3_off[: num_units // 2]]),
                torch.cat([resp2_off, resp3_off[num_units // 2 :]]),
            ]
        ).unsqueeze(0)
    )
    obj2.append(o)
    objective_fn = ContrastiveNeuronUnif(
        on_clust_idx=1,
        off_clust_idc=[0],
        clust_assignments=cluster_assignments,
        device="cpu",
    )
    o = objective_fn(
        torch.cat(
            [
                torch.cat([resp1_off, resp3_off[: num_units // 2]]),
                torch.cat([resp2_on, resp3_off[num_units // 2 :]]),
            ]
        ).unsqueeze(0)
    )
    obj2.append(o)
    obj2_mean = np.mean(obj2)

    assert (
        obj3_mean > obj2_mean
    ), "objective with 3 clusters should be higher than objective with 2 clusters because toy data contains 3 clusters"
    print(obj3_mean, obj2_mean)


if __name__ == "__main__":
    test_objective_matches_basic_math()
    test_contrastive_neuron_decreases_when_removing_required_cluster()
    test_contrastive_neuron_decreases_for_redundant_cluster()
