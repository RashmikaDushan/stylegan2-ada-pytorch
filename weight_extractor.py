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
    layerwise_mean_diffs = []
    all_diffs = []

    for (n1, p1), (n2, p2) in zip(g_old.named_parameters(), g_new.named_parameters()):
        diff = torch.abs(p1 - p2).detach().cpu()
        print(n1, diff.mean().item())
        layerwise_mean_diffs.append(diff.mean().item())
        layer_names.append(n1)
        all_diffs.append(diff.view(-1).numpy())

    # Plot 1: Mean difference per layer (Bar)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.bar(layer_names, layerwise_mean_diffs, color='skyblue')
    plt.xticks(rotation=45, ha='right', fontsize=5)
    plt.title("Mean Absolute Difference")
    plt.ylabel("Mean |Delta W|")

    # Plot 3: Global Distribution (Histogram)
    plt.subplot(2, 1, 2)
    plt.hist(np.concatenate(all_diffs), bins=100, log=True, color='sandybrown')
    plt.title("Global Weight Difference Distribution (Log Scale)")
    plt.xlabel("Absolute Difference")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def plot_weight_differences_violin(g_old, g_new):
    layer_names = []
    layerwise_mean_diffs = []
    layerwise_dist_data = []

    for (n1, p1), (n2, p2) in zip(g_old.named_parameters(), g_new.named_parameters()):
        diff = torch.abs(p1 - p2).detach().cpu().flatten().numpy()
        layer_names.append(n1)
        layerwise_mean_diffs.append(np.mean(diff))
        layerwise_dist_data.append(diff)

    plt.figure(figsize=(12, 6))

    # Plot 1: Violin Plot of Weight Differences per Layer
    parts = plt.violinplot(layerwise_dist_data, showmeans=True, showextrema=True)
    
    # Visual styling for the violins
    for pc in parts['bodies']:
        pc.set_facecolor('#2196F3')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    plt.xticks(range(1, len(layer_names) + 1), layer_names, rotation=35, ha='right', fontsize=5)
    plt.title("Distribution of Absolute Weight Differences (Top 15 Layers)")
    plt.ylabel("|Delta W| Distribution")

    plt.tight_layout()
    plt.show()

def compare_and_build_pickle(old_path, new_path, threshold, output_path=None):
    data_old = load_pickle(old_path)
    data_new = load_pickle(new_path)

    result_data = copy.deepcopy(data_old)

    g_old = data_old['G_ema']
    g_new = data_new['G_ema']
    g_result = result_data['G_ema']

    state_old = g_old.state_dict()
    state_result = copy.deepcopy(state_old)  # start from old

    with torch.no_grad():
        for name, param_new in g_new.named_parameters():
            if name in state_old:
                param_old = state_old[name]
                diff = torch.abs(param_new - param_old)
                state_result[name] = torch.where(diff > threshold, param_new, param_old)

    # Write the result back into the module
    g_result.load_state_dict(state_result)

    if output_path:
        with open(output_path, "wb") as f:
            pickle.dump(result_data, f)

    return result_data

if __name__ == "__main__":
    # network_old = load_pickle(OLD_MODEL_PATH)
    # network_new = load_pickle(NEW_MODEL_PATH)

    # synth_old = network_old['G_ema'].synthesis.to(DEVICE)
    # synth_new = network_new['G_ema'].synthesis.to(DEVICE)

    # plot_weight_differences(synth_old, synth_new)
    compare_and_build_pickle(OLD_MODEL_PATH, NEW_MODEL_PATH, 0.5, "ffhq-merged.pkl")
