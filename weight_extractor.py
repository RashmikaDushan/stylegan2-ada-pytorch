import torch
import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle
import dnnlib
import legacy

OLD_MODEL_PATH = "ffhq-256.pkl"
NEW_MODEL_PATH = "ffhq-finetuned.pkl"
DEVICE = torch.device("cuda")

def load_pickle(path):
    print('Loading networks from "%s"...' % path)
    with dnnlib.util.open_url(path) as f:
        network = legacy.load_network_pkl(f)
    return network

def plot_weight_differences(g_old, g_new):
    layer_names = []
    mean_diffs = []
    all_diffs = []

    for (n1, p1), (n2, p2) in zip(g_old.named_parameters(), g_new.named_parameters()):
        diff = torch.abs(p1 - p2).detach().cpu()
        print(n1, diff.mean().item())
        mean_diffs.append(diff.mean().item())
        layer_names.append(n1)
        all_diffs.append(diff.view(-1).numpy()) 

    # Plot 1: Mean difference per layer (Bar)
    plt.figure(figsize=(12, 8))
    # Sort by magnitude for better visibility
    sorted_indices = np.argsort(mean_diffs)[::-1]
    sorted_names = [layer_names[i] for i in sorted_indices]
    sorted_vals = [mean_diffs[i] for i in sorted_indices]
    
    plt.subplot(2, 1, 1)
    plt.bar(sorted_names, sorted_vals)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title("Mean Absolute Difference")
    plt.ylabel("Mean |Delta W|")

    # Plot 2: Global Distribution (Histogram)
    plt.subplot(2, 1, 2)
    plt.hist(np.concatenate(all_diffs), bins=100, log=True, color='orange')
    plt.title("Global Weight Difference Distribution (Log Scale)")
    plt.xlabel("Absolute Difference")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    network_old = load_pickle(OLD_MODEL_PATH)
    network_new = load_pickle(NEW_MODEL_PATH)

    synth_old = network_old['G_ema'].synthesis.to(DEVICE)
    synth_new = network_new['G_ema'].synthesis.to(DEVICE)

    plot_weight_differences(synth_old, synth_new)
