import edvae.models.edvae as models

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import ttest_ind


def load_model(path: str):
    model = torch.load(path)
    model.eval()
    return model

def train_md(edvae: models.EDVAE, loader, device, batch_size, reporter, num_epochs=1, lr=0.0001):
    print("train_md")
    disc = nn.Sequential(
        nn.Linear(128, 128),
        nn.Sigmoid(),
        nn.Linear(128, 64),
        nn.Sigmoid(),
        nn.Linear(64,32),
        nn.Sigmoid(),
        nn.Linear(32,1),
        nn.Sigmoid()
    )
    disc.to(device)
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=lr)
    edvae.eval()
    for epoch in range(num_epochs):
            batch_iteration = 0
            for data, expert in loader:
                batch_iteration += 1
                data = data.to(device)

                disc.zero_grad()

                _, _, _, _, _, z_star = edvae(data, expert)

                human_label = torch.zeros(batch_size, 1, device=device)
                mouse_label = torch.ones(batch_size, 1, device=device)

                truth = human_label if expert == "human" else mouse_label

                disc_prediction = disc(z_star)

                loss_d = nn.functional.binary_cross_entropy(disc_prediction, truth, reduction="sum")
                loss_d_mean = nn.functional.binary_cross_entropy(disc_prediction, truth, reduction="mean")

                reporter.accumulate_loss("md_disc bce z_star", loss_d_mean)

                loss_d.backward()
                disc_optimizer.step()

            reporter.write_loss("md_disc bce z_star", epoch, divisor=batch_iteration)

            reporter.zero_losses()
    edvae.train()
    return disc


def get_adjusted_latent(disc, edvae, loader, batch_size, device, file_name, alpha=10):
    print("adjusted_latent...")
    edvae.eval()
    data = None
    expert = None
    for x,y in loader:
        data = x
        expert = y
        break
    data = data.to(device)
    x_hat, _, _, _, _, z_star = edvae(data, expert)
    z_star.retain_grad()

    human_label = torch.zeros(batch_size, 1, device=device)
    mouse_label = torch.ones(batch_size, 1, device=device)

    truth = human_label if expert == "human" else mouse_label

    disc_prediction = disc(z_star)
    loss_d = nn.functional.binary_cross_entropy(disc_prediction, truth, reduction="sum")
    loss_d.backward()

    adjusted_z_star = z_star + (alpha * z_star.grad)

    adjusted_x_hat = edvae.decode(adjusted_z_star, "human")

    plot_volcano(x_hat, adjusted_x_hat, file_name)
    return adjusted_x_hat

def odds_ratio(a, b):
    if a >= b:
        return a/b
    else:
        return -b/a
    
    
def plot_volcano(original:torch.Tensor, altered:torch.Tensor, file_name):
    print("Plotting Volcano...")
     #calculate fold_change

    original = original.cpu()
    altered = altered.cpu()

    original_np = original.detach().numpy()
    altered_np = altered.detach().numpy()

    # Calculate fold change and p-values
    genes = original.shape[0]
    fold_changes = []
    p_values = []

    for i in range(genes):
        # Calculate mean expression for the gene across all cells
        mean_original = np.mean(original_np[i, :])
        mean_altered = np.mean(altered_np[i, :])
        
        # Calculate fold change
        # fold_change = odds_ratio(mean_original, mean_altered)
        fold_change = mean_original / mean_altered
        fold_changes.append(fold_change)
        
        # Perform t-test
        t_stat, p_val = ttest_ind(altered_np[i, :], original_np[i, :])
        p_values.append(p_val)

    # Convert to numpy arrays for plotting
    fold_changes = np.array(fold_changes)
    p_values = np.array(p_values)

    # Calculate log2 fold change and -log10 p-values
    log2_fold_changes = np.log2(fold_changes)
    neg_log10_p_values = -np.log10(p_values)

    # Create the volcano plot
    plt.figure(figsize=(10, 6))
    plt.scatter(log2_fold_changes, neg_log10_p_values, c='blue', alpha=0.5)

    # Add labels and title
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 P-value')
    plt.title('Volcano Plot No Odds Ratio')

    # Highlight significant points (e.g., p-value < 0.05)
    significance_threshold = -np.log10(0.05)
    plt.axhline(y=significance_threshold, color='red', linestyle='--')
    # plt.axvline(x=1, color='green', linestyle='--')
    # plt.axvline(x=-1, color='green', linestyle='--')

    # Save the plot as PNG
    plt.savefig(file_name)
