"""
ACO-GNN Main Research CLI Driver.
Runs benchmarks, baselines, camouflage stress tests, and ablation studies.
All outputs and figures are saved locally in the research folder.
"""

import argparse
import os
import json
import numpy as np
import torch

from src.data.dataset_loader import load_synthetic_lending, load_yelpchi, inject_camouflage
from src.aco.matrix_ant import VectorizedACOEngine
from src.models.baselines import MLPBaseline, GCNBaseline, GATBaseline
from src.models.aco_gnn import ACORingGNN
from src.training.train_eval import train_and_evaluate
from src.utils.plotting import plot_camouflage_robustness, plot_ablation_study


def task_test_data(args):
    """Test data generation and inspect graph statistics."""
    print("=== Testing Synthetic Multi-Hop Dynamic Lending Network ===")
    data = load_synthetic_lending(
        num_applicants=args.num_nodes,
        fraud_ratio=args.fraud_ratio,
        camouflage_ratio=0.0,
    )
    print(f"Nodes: {data['num_nodes']}")
    print(f"Edges: {data['edge_index'].shape[1]}")
    fraud_count = int(data['y'].sum().item())
    print(f"Fraud Nodes: {fraud_count} ({fraud_count / data['num_nodes'] * 100:.2f}%)")
    print(f"Relation Types: {data['num_relations']}")
    print(f"Train nodes: {int(data['train_mask'].sum().item())}")
    print(f"Val nodes:   {int(data['val_mask'].sum().item())}")
    print(f"Test nodes:  {int(data['test_mask'].sum().item())}")
    print("Data pipeline test passed successfully!")


def task_run_baselines(args):
    """Run classical ML and standard GNN baselines."""
    print("\n=== Running Baseline Models Suite ===")
    data = load_synthetic_lending(
        num_applicants=args.num_nodes,
        fraud_ratio=args.fraud_ratio,
        camouflage_ratio=args.camouflage_ratio,
    )
    in_dim = data["x"].shape[1]

    models = {
        "MLP (Tabular)": MLPBaseline(in_channels=in_dim, hidden_channels=64),
        "GCN": GCNBaseline(in_channels=in_dim, hidden_channels=64),
        "GAT": GATBaseline(in_channels=in_dim, hidden_channels=64),
    }

    results = {}
    for name, model in models.items():
        res = train_and_evaluate(
            model=model,
            data=data,
            aco_engine=None,
            epochs=args.epochs,
            lr=0.005,
            model_name=name,
        )
        results[name] = res["test_metrics"]

    os.makedirs("./experiments/results", exist_ok=True)
    with open("./experiments/results/baselines_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved baseline results to ./experiments/results/baselines_results.json")


def task_train_aco_gnn(args):
    """Train proposed ACO-GNN model."""
    print("\n=== Training Proposed ACO-GNN Framework ===")
    data = load_synthetic_lending(
        num_applicants=args.num_nodes,
        fraud_ratio=args.fraud_ratio,
        camouflage_ratio=args.camouflage_ratio,
    )
    in_dim = data["x"].shape[1]
    num_nodes = data["num_nodes"]
    num_relations = data["num_relations"]

    aco_engine = VectorizedACOEngine(
        num_nodes=num_nodes,
        num_relations=num_relations,
        alpha=1.0,
        beta=1.0,
        rho=0.05,
        temperature=1.0,
    )

    model = ACORingGNN(
        in_channels=in_dim,
        hidden_channels=64,
        out_channels=2,
        num_layers=2,
        num_relations=num_relations,
        lambda_blend=0.5,
        dropout=0.3,
    )

    res = train_and_evaluate(
        model=model,
        data=data,
        aco_engine=aco_engine,
        epochs=args.epochs,
        lr=0.005,
        model_name="ACO-GNN (Proposed)",
    )

    os.makedirs("./experiments/results", exist_ok=True)
    with open("./experiments/results/aco_gnn_results.json", "w") as f:
        json.dump(res["test_metrics"], f, indent=2)
    print("\nSaved ACO-GNN results to ./experiments/results/aco_gnn_results.json")


def task_camouflage_experiment(args):
    """Run camouflage stress tests across 0%, 10%, 20%, 30%, 40%, 50% noise."""
    print("\n=== Running Camouflage Attack Robustness Stress Test ===")
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    models_to_test = ["GCN", "GAT", "ACO-GNN (Proposed)"]
    f1_history = {m: [] for m in models_to_test}

    for noise in noise_levels:
        print(f"\n--- Testing at Camouflage Noise Level: {int(noise*100)}% ---")
        data = load_synthetic_lending(
            num_applicants=args.num_nodes,
            fraud_ratio=args.fraud_ratio,
            camouflage_ratio=noise,
        )
        in_dim = data["x"].shape[1]
        num_nodes = data["num_nodes"]
        num_relations = data["num_relations"]

        # 1. GCN
        gcn = GCNBaseline(in_channels=in_dim, hidden_channels=64)
        res_gcn = train_and_evaluate(gcn, data, aco_engine=None, epochs=args.epochs, model_name=f"GCN_cam{int(noise*100)}")
        f1_history["GCN"].append(res_gcn["test_metrics"]["illicit_f1"])

        # 2. GAT
        gat = GATBaseline(in_channels=in_dim, hidden_channels=64)
        res_gat = train_and_evaluate(gat, data, aco_engine=None, epochs=args.epochs, model_name=f"GAT_cam{int(noise*100)}")
        f1_history["GAT"].append(res_gat["test_metrics"]["illicit_f1"])

        # 3. ACO-GNN
        aco = VectorizedACOEngine(num_nodes=num_nodes, num_relations=num_relations)
        aco_gnn = ACORingGNN(in_channels=in_dim, hidden_channels=64, num_relations=num_relations)
        res_aco = train_and_evaluate(aco_gnn, data, aco_engine=aco, epochs=args.epochs, model_name=f"ACO-GNN_cam{int(noise*100)}")
        f1_history["ACO-GNN (Proposed)"].append(res_aco["test_metrics"]["illicit_f1"])

    os.makedirs("./experiments/results", exist_ok=True)
    with open("./experiments/results/camouflage_experiment.json", "w") as f:
        json.dump(f1_history, f, indent=2)

    plot_camouflage_robustness(
        noise_levels=noise_levels,
        model_results=f1_history,
        save_path="./paper/figures/camouflage_robustness.pdf",
    )


def task_ablation_study(args):
    """Run ablation study isolating individual components."""
    print("\n=== Running ACO-GNN Component Ablation Study ===")
    data = load_synthetic_lending(
        num_applicants=args.num_nodes,
        fraud_ratio=args.fraud_ratio,
        camouflage_ratio=0.20,  # 20% camouflage noise
    )
    in_dim = data["x"].shape[1]
    num_nodes = data["num_nodes"]
    num_relations = data["num_relations"]

    variants = [
        "w/o ACO (Vanilla GAT)",
        "w/o Evaporation (rho=0)",
        "w/o Attn (Pure ACO)",
        "Full ACO-GNN",
    ]
    f1_scores = []
    pr_aucs = []

    # 1. w/o ACO (Standard GAT)
    gat = GATBaseline(in_channels=in_dim, hidden_channels=64)
    res1 = train_and_evaluate(gat, data, aco_engine=None, epochs=args.epochs, model_name="Ablation_No_ACO")
    f1_scores.append(res1["test_metrics"]["illicit_f1"])
    pr_aucs.append(res1["test_metrics"]["pr_auc"])

    # 2. w/o Evaporation (rho=0)
    aco_no_evap = VectorizedACOEngine(num_nodes=num_nodes, num_relations=num_relations, rho=0.0)
    model2 = ACORingGNN(in_channels=in_dim, hidden_channels=64, num_relations=num_relations)
    res2 = train_and_evaluate(model2, data, aco_engine=aco_no_evap, epochs=args.epochs, model_name="Ablation_No_Evap")
    f1_scores.append(res2["test_metrics"]["illicit_f1"])
    pr_aucs.append(res2["test_metrics"]["pr_auc"])

    # 3. w/o Attn (lambda=0, purely pheromone message passing)
    aco_pure = VectorizedACOEngine(num_nodes=num_nodes, num_relations=num_relations)
    model3 = ACORingGNN(in_channels=in_dim, hidden_channels=64, num_relations=num_relations, fixed_lambda=0.0)
    res3 = train_and_evaluate(model3, data, aco_engine=aco_pure, epochs=args.epochs, model_name="Ablation_Pure_ACO")
    f1_scores.append(res3["test_metrics"]["illicit_f1"])
    pr_aucs.append(res3["test_metrics"]["pr_auc"])

    # 4. Full ACO-GNN
    aco_full = VectorizedACOEngine(num_nodes=num_nodes, num_relations=num_relations)
    model4 = ACORingGNN(in_channels=in_dim, hidden_channels=64, num_relations=num_relations, lambda_blend=0.5)
    res4 = train_and_evaluate(model4, data, aco_engine=aco_full, epochs=args.epochs, model_name="Full_ACO-GNN")
    f1_scores.append(res4["test_metrics"]["illicit_f1"])
    pr_aucs.append(res4["test_metrics"]["pr_auc"])

    plot_ablation_study(
        variants=variants,
        f1_scores=f1_scores,
        pr_aucs=pr_aucs,
        save_path="./paper/figures/ablation_study.pdf",
    )


def main():
    parser = argparse.ArgumentParser(description="ACO-GNN Fraud Detection Research Suite")
    parser.add_argument(
        "--task",
        type=str,
        default="test_data",
        choices=["test_data", "run_baselines", "train_aco_gnn", "camouflage_experiment", "ablation_study"],
        help="Task to run",
    )
    parser.add_argument("--num_nodes", type=int, default=3000, help="Number of applicant nodes")
    parser.add_argument("--fraud_ratio", type=float, default=0.05, help="Fraud ratio (default 5%)")
    parser.add_argument("--camouflage_ratio", type=float, default=0.15, help="Camouflage noise ratio")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    args = parser.parse_args()

    if args.task == "test_data":
        task_test_data(args)
    elif args.task == "run_baselines":
        task_run_baselines(args)
    elif args.task == "train_aco_gnn":
        task_train_aco_gnn(args)
    elif args.task == "camouflage_experiment":
        task_camouflage_experiment(args)
    elif args.task == "ablation_study":
        task_ablation_study(args)


if __name__ == "__main__":
    main()
