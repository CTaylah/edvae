import data.dataloader as dataloader
import models.discriminator
import utils.utils as utils
from utils.reporter import Reporter

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# def test_dataloaders():
#     species = ["homo_sapiens", "mus_musculus"]
#     train_test_split = {"train": 0.8, "test": 0.2}
#     human_train, human_test= data.dataloader.build_single_species_loaders("homo_sapiens", 200, 16, train_test_split, 1, tag="human")
#     mouse_train, mouse_test= data.dataloader.build_single_species_loaders("mus_musculus", 200, 16, train_test_split, 1, tag="mouse")

#     loader = data.dataloader.SpeciesAlternator(human_train, mouse_train)

#     batch_iteration = 0
   
#     for x_batch, y_batch, expert in loader:
#         batch_iteration += 1
#         print(expert)
#         if batch_iteration % 100 == 0:
#             print(batch_iteration,flush=True)
#     print(batch_iteration)


import models.edvae
import models.vae
from configs import HUMAN_DATA_SIZE, MOUSE_DATA_SIZE


def configure_model():
    human_encoder = nn.Sequential(
        nn.Linear(HUMAN_DATA_SIZE, 512), 
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 448),
        nn.BatchNorm1d(448),
        nn.ReLU(),
        nn.Linear(448, 384),
        nn.BatchNorm1d(384),
        nn.ReLU()
        )

    human_decoder = nn.Sequential(
        nn.Linear(384, 448), 
        nn.BatchNorm1d(448),
        nn.ReLU(),
        nn.Linear(448, 512), 
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, HUMAN_DATA_SIZE), 
        nn.BatchNorm1d(HUMAN_DATA_SIZE),
        nn.ReLU()
        )

    human_expert = models.edvae.Expert(human_encoder, human_decoder)

    mouse_encoder = nn.Sequential(
        nn.Linear(MOUSE_DATA_SIZE, 512), 
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 448),
        nn.BatchNorm1d(448),
        nn.ReLU(),
        nn.Linear(448, 384),
        nn.BatchNorm1d(384),
        nn.ReLU()
        )

    mouse_decoder = nn.Sequential(
        nn.Linear(384, 448), 
        nn.BatchNorm1d(448),
        nn.ReLU(),
        nn.Linear(448, 512), 
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, MOUSE_DATA_SIZE), 
        nn.BatchNorm1d(MOUSE_DATA_SIZE),
        nn.ReLU()
        )


    mouse_expert = models.edvae.Expert(mouse_encoder, mouse_decoder)

    encoder = nn.Sequential(
        nn.Linear(384, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU()
        )

    mu_layer = nn.Linear(128, 128)
    log_var_layer = nn.Linear(128, 128)

    decoder = nn.Sequential(
        nn.Linear(128, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 384),
        nn.BatchNorm1d(384),
        nn.ReLU()
        )

    shared_vae = models.vae.VAE(encoder, decoder, mu_layer, log_var_layer)

    edvae = models.edvae.EDVAE(shared_vae, human_expert, mouse_expert)


    disc = nn.Sequential(
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.Sigmoid(),
        nn.Linear(64, 1),
        nn.Sigmoid()
        )
    
    discriminator = models.discriminator.Discriminator(disc)

    print(edvae)
    return edvae, disc


batch_size = 128
num_samples = 300_000
num_batches = num_samples / batch_size 

def test(epoch_number, edvae, test_loader, device, reporter):
    edvae.eval()
    for data, metadata, expert in test_loader:
        data = data.to(device)
        x_hat, z, mu, log_var = edvae(data, expert)
        recon_mean = torch.nn.functional.mse_loss(x_hat, data, reduction="mean")
        kl_mean = utils.kl_divergence(mu, log_var, reduction="mean")

        if expert == "human":
            reporter.accumulate_loss("TEST human recon", recon_mean.item())
            reporter.accumulate_loss("TEST human KL", kl_mean.item())

        elif expert == "mouse":
            reporter.accumulate_loss("TEST mouse recon", recon_mean.item())
            reporter.accumulate_loss("TEST mouse KL", kl_mean.item())

    reporter.write_losses(epoch_number, num_batches * .2)
    reporter.zero_losses()
    edvae.train()


def train(num_epochs, device):
    edvae, disc = configure_model()
    edvae.to(device)
    disc.to(device)

    train_test_split = {"train": 0.8, "test": 0.2}
    human_train, human_test = dataloader.build_single_species_loaders("homo_sapiens", num_samples/2, batch_size, train_test_split, 1, tag="human")
    mouse_train, mouse_test = dataloader.build_single_species_loaders("mus_musculus", num_samples/2, batch_size, train_test_split, 1, tag="mouse")

    loader = dataloader.SpeciesAlternator(human_train, mouse_train)
    test_loader = dataloader.SpeciesAlternator(human_test, mouse_test)


    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(edvae.parameters(), lr=0.001)
    reporter = Reporter("/mnt/home/taylcard/dev/logs/")

    reporter.add_loss_labels(["human recon",
                             "human KL",
                             "mouse recon",
                             "mouse KL",
                             "disc bce"
                             ])

    for epoch in range(num_epochs):
        batch_iteration = 0
        for data, metadata, expert in loader:
            data = data.to(device)
            print(data)
            edvae.zero_grad()
            disc.zero_grad()

            x_hat, z, mu, log_var = edvae(data, expert)


            label = (torch.zeros(batch_size, 1, device=device) if expert == "human" 
                     else torch.ones(batch_size, 1, device=device))

            disc_prediction = disc(z)

            loss_d = nn.BCELoss(disc_prediction, label)
            loss_d.backward()


            recon_loss = torch.nn.functional.mse_loss(x_hat, data, reduction="sum")
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            # kl_div = utils.kl_divergence(mu, log_var, reduction="sum")

            recon_mean = torch.nn.functional.mse_loss(x_hat, data, reduction="mean")
            # kl_mean = utils.kl_divergence(mu, log_var, reduction="mean")
            kl_mean = kl_div.item() / mu.numel()

            if expert == "human":
                reporter.accumulate_loss("human recon", recon_mean.item())
                reporter.accumulate_loss("human KL", kl_mean)

            elif expert == "mouse":
                reporter.accumulate_loss("mouse recon", recon_mean.item())
                reporter.accumulate_loss("mouse KL", kl_mean)

            total_loss = recon_loss + 0.15 * kl_div.sum()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(edvae.parameters(), 2.0)
            optimizer.step()

            batch_iteration += 1
            print(f"Epoch {epoch+1}, Expert {expert}, Reconstruction Loss: {recon_mean.item():.6f}, KL Divergence: {kl_mean:.6f}", flush=True)
        reporter.write_losses(epoch, (batch_iteration / 2.0))
        reporter.zero_losses()

        # test(epoch, edvae, test_loader, reporter)


if __name__== '__main__':
    print("Hello!")
    start = time.time()
    device = torch.device("cuda")
    train(25, device)
    end = time.time()
    length = end - start

    # Show the results : this can be altered however you like
    print("TEST: 36M header(120_000) 1 worker It took", length, "seconds!")