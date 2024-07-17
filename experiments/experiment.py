import edvae.data.dataloader as dataloader
import edvae.configs.yaml_parser as configs
import edvae.models.edvae as models
from edvae.utils.reporter import Reporter
from mmvae.data.pipes import CellCensusPipeLine
import edvae.eval.latent as latent

import torch
import torch.nn as nn
import torchdata.dataloader2 as dl
import torchdata.datapipes as dp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import umap
import matplotlib.pyplot as plt
import numpy as np


from functools import partial

def configure_models(models_dict):
    human_encoder = models_dict['Human_Expert_encoder']
    human_decoder = models_dict['Human_Expert_decoder']

    mouse_encoder = models_dict['Mouse_Expert_encoder']
    mouse_decoder = models_dict['Mouse_Expert_decoder']

    shared_encoder = models_dict['Shared_VAE_encoder']
    shared_decoder = models_dict['Shared_VAE_decoder']
    
    mu = models_dict['Shared_VAE_mu']
    logvar = models_dict['Shared_VAE_logvar']

    human_expert = models.Expert(human_encoder, human_decoder)
    mouse_expert = models.Expert(mouse_encoder, mouse_decoder)
    shared_vae = models.VAE(shared_encoder, shared_decoder, mu, logvar)

    model = models.EDVAE(shared_vae, human_expert, mouse_expert) 

    discriminator_0: nn.Sequential = models_dict['Discriminator0_model']
    discriminator_1: nn.Sequential = models_dict['Discriminator1_model']
    discriminator_2: nn.Sequential = models_dict['Discriminator2_model']
    print(model)
    
    return model, [discriminator_0, discriminator_1, discriminator_2]


def _preprocess(data, tag):
    return (data[0],tag)

def _build_preprocessing_fn(tag: str):
    return partial(_preprocess, tag=tag)

def configure_species_loader(seed: int, batch_size: int, num_workers: int, tag: str):
    # Convienence wrapper for dataloader config
    if tag == "human":
        # pipeline = CellCensusPipeLine(directory_path="/mnt/projects/debruinz_project/summer_census_data/3m_subset", masks=["3m_human_counts_*.npz"], batch_size=batch_size)
        pipeline = CellCensusPipeLine(directory_path="/mnt/projects/debruinz_project/summer_census_data/3m_subset", masks=["3m_human_counts_15.npz"], batch_size=batch_size)
    elif tag == "mouse":
        # pipeline = CellCensusPipeLine(directory_path="/mnt/projects/debruinz_project/summer_census_data/3m_subset", masks=["3m_mouse_counts_*.npz"], batch_size=batch_size)
        pipeline = CellCensusPipeLine(directory_path="/mnt/projects/debruinz_project/summer_census_data/3m_subset", masks=["3m_mouse_counts_15.npz"], batch_size=batch_size)
    else:
        raise ValueError("dataloader tag must be human or mouse")

    pre_process_fn  = _build_preprocessing_fn(tag)
    tagged_pipe = dp.iter.Mapper(pipeline, pre_process_fn)

    # train_pipe, val_pipe = tagged_pipe.random_split(weights={"train": 0.8, "test": 0.2}, total_length=3_002_880, seed=seed)
    train_pipe, val_pipe = tagged_pipe.random_split(weights={"train": 0.8, "test": 0.2}, total_length=200_192, seed=seed)
    train_loader = dl.DataLoader2(datapipe=train_pipe, datapipe_adapter_fn=None, reading_service=dl.MultiProcessingReadingService(num_workers=num_workers))
    val_loader = dl.DataLoader2(datapipe=val_pipe, datapipe_adapter_fn=None, reading_service=dl.MultiProcessingReadingService(num_workers=num_workers))
    return train_loader, val_loader


def test(epoch_number, edvae, test_loader, device, reporter: Reporter):
    edvae.eval()
    batch_iteration = 0
    human_iterations = 0
    mouse_iterations = 0
    for data, expert in test_loader:
        data = data.to(device)
        with torch.no_grad():
            x_hat, mu, log_var, _, _, _ = edvae(data, expert)
            recon_mean = torch.nn.functional.mse_loss(x_hat, data.to_dense(), reduction="mean")
            kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

            reporter.accumulate_loss("TEST/KL Divergence", kl_div)

            batch_iteration +=  1
            if expert == "human":
                reporter.accumulate_loss("TEST/human recon", recon_mean.item())
                human_iterations += 1

            elif expert == "mouse":
                reporter.accumulate_loss("TEST/mouse recon", recon_mean.item())
                mouse_iterations += 1

    reporter.write_loss("TEST/KL Divergence", epoch_number, divisor=batch_iteration)

    reporter.write_loss("TEST/human recon", epoch_number, divisor=human_iterations)
    reporter.write_loss("TEST/mouse recon", epoch_number, divisor=mouse_iterations)

    reporter.zero_losses()
    edvae.train()

def train_md(edvae, loader, device, batch_size, reporter, num_epochs=1, lr=0.0001):
    disc = nn.Sequential(
        nn.Linear(448, 256),
        nn.Sigmoid(),
        nn.Linear(256, 64),
        nn.Sigmoid(),
        nn.Linear(64,1),
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

                x_hat, mu, log_var, encoder_activations, h, _ = edvae(data, expert)

                human_label = torch.zeros(batch_size, 1, device=device)
                mouse_label = torch.ones(batch_size, 1, device=device)

                truth = human_label if expert == "human" else mouse_label

                disc_prediction = disc(h)

                loss_d = nn.functional.binary_cross_entropy(disc_prediction, truth, reduction="sum")
                loss_d_mean = nn.functional.binary_cross_entropy(disc_prediction, truth, reduction="mean")

                reporter.accumulate_loss("md_disc bce", loss_d_mean)

                loss_d.backward()
                disc_optimizer.step()

            reporter.write_loss("md_disc bce", epoch, divisor=batch_iteration)

            reporter.zero_losses()
    edvae.train()


def train_md_on_output_human(edvae, loader, device, batch_size, reporter, num_epochs=1, lr=0.0001):
    disc_human = nn.Sequential(
        nn.Linear(60_664, 256),
        nn.Sigmoid(),
        nn.Linear(256, 64),
        nn.Sigmoid(),
        nn.Linear(64,1),
        nn.Sigmoid()
    )
    disc_human.to(device)
    disc_human_optimizer = torch.optim.Adam(disc_human.parameters(), lr=lr)
    edvae.eval()
    for epoch in range(num_epochs):
            batch_iteration = 0
            for data, expert in loader:
                batch_iteration += 1
                data = data.to(device)

                human_label = torch.zeros(batch_size, 1, device=device)
                mouse_label = torch.ones(batch_size, 1, device=device)

                truth = human_label if expert == "human" else mouse_label


                disc_human.zero_grad()
                if expert == "human":
                    x_hat, mu, log_var, encoder_activations, h, _ = edvae(data, expert)
                    disc_prediction = disc_human(x_hat)
                else:
                    x_hat, mu, log_var, encoder_activations, h, _ = edvae.cross(data, expert)
                    disc_prediction = disc_human(x_hat)


                loss_d = nn.functional.binary_cross_entropy(disc_prediction, truth, reduction="sum")
                loss_d_mean = nn.functional.binary_cross_entropy(disc_prediction, truth, reduction="mean")

                reporter.accumulate_loss("md_disc bce human", loss_d_mean)

                loss_d.backward()

                disc_human_optimizer.step()

            reporter.write_loss("md_disc bce human", epoch, divisor=batch_iteration)

            reporter.zero_losses()
    edvae.train()

def train_md_on_output_mouse(edvae, loader, device, batch_size, reporter, num_epochs=1, lr=0.0001):
    disc_mouse = nn.Sequential(
        nn.Linear(52_417, 256),
        nn.Sigmoid(),
        nn.Linear(256, 64),
        nn.Sigmoid(),
        nn.Linear(64,1),
        nn.Sigmoid()
    )
    disc_mouse.to(device)
    disc_mouse_optimizer= torch.optim.Adam(disc_mouse.parameters(), lr=lr)
    edvae.eval()
    for epoch in range(num_epochs):
            batch_iteration = 0
            for data, expert in loader:
                batch_iteration += 1
                data = data.to(device)

                human_label = torch.zeros(batch_size, 1, device=device)
                mouse_label = torch.ones(batch_size, 1, device=device)

                truth = human_label if expert == "human" else mouse_label


                disc_mouse.zero_grad()
                if expert == "mouse":
                    x_hat, mu, log_var, encoder_activations, h, _ = edvae(data, expert)
                    disc_prediction = disc_mouse(x_hat)
                else:
                    x_hat, mu, log_var, encoder_activations, h, _ = edvae.cross(data, expert)
                    disc_prediction = disc_mouse(x_hat)


                loss_d = nn.functional.binary_cross_entropy(disc_prediction, truth, reduction="sum")
                loss_d_mean = nn.functional.binary_cross_entropy(disc_prediction, truth, reduction="mean")

                reporter.accumulate_loss("md_disc bce mouse", loss_d_mean)

                loss_d.backward()

                disc_mouse_optimizer.step()

            reporter.write_loss("md_disc bce mouse", epoch, divisor=batch_iteration)

            reporter.zero_losses()
    edvae.train()

def train(yaml_dir: str, device):
    hparams, model_dict = configs.get_configs(yaml_dir)
    edvae, discs = configure_models(model_dict)

    edvae.to(device)
    for disc in discs:
        disc.to(device)

    num_epochs = hparams['num_epochs']
    batch_size = hparams['batch_size']
    gen_weight = hparams['gen_weight']
    disc_warmup = hparams['disc_warmup']
    seed = hparams['seed']

    mmvae_lr = hparams['mmvae_lr']
    disc_lr = hparams['disc_lr']

    human_train, human_test = configure_species_loader(seed, batch_size, 1, "human")
    mouse_train, mouse_test = configure_species_loader(seed, batch_size, 1, "mouse")

    loader = dataloader.SpeciesAlternator(human_train, mouse_train)
    test_loader = dataloader.SpeciesAlternator(human_test, mouse_test)

    disc_optimizers = []
    for disc in discs:
        disc_optimizers.append(torch.optim.Adam(disc.parameters(), lr=disc_lr))

    optimizer = torch.optim.Adam(edvae.parameters(), lr=mmvae_lr)
    reporter = Reporter(hparams['log_directory'], hparams['run_name'])
    reporter.add_loss_labels([
                            "human recon",
                            "mouse recon",
                            "KL Divergence",
                            "disc bce",
                            "md_disc bce z_star",
                            "md_disc bce",
                            "md_disc bce human",
                            "md_disc bce mouse",
                            "gen bce",
                            "TEST/KL Divergence",
                            "TEST/human recon",
                            "TEST/mouse recon"
                             ])

    for epoch in range(num_epochs):
        batch_iteration = 0
        human_iterations = 0
        mouse_iterations = 0
        for data, expert in loader:
            data = data.to(device)

            edvae.zero_grad()
            for disc in discs:
                disc.zero_grad()

            x_hat, mu, log_var, encoder_activations, _, _= edvae(data, expert)

            human_label = torch.zeros(batch_size, 1, device=device)
            mouse_label = torch.ones(batch_size, 1, device=device)

            truth = human_label if expert == "human" else mouse_label
            trick = human_label if expert == "mouse" else mouse_label

            predictions = []

            # predictions.append(discs[0](encoder_activations[0]))
            predictions.append(discs[0](encoder_activations[0]))
            predictions.append(discs[2](encoder_activations[1]))

            loss_d_mean = 0
            for prediction in predictions:
                loss = nn.functional.binary_cross_entropy(prediction, truth, reduction="sum")
                loss_mean = nn.functional.binary_cross_entropy(prediction, truth, reduction="mean")
                loss.backward(retain_graph=True)
                loss_d_mean += loss_mean
            
            loss_mean /= len(predictions)
            for opt in disc_optimizers:
                opt.step()

            reporter.accumulate_loss("disc bce", loss_d_mean)


            predictions = []

            edvae.zero_grad()

            # predictions.append(discs[0](encoder_activations[0]))
            predictions.append(discs[0](encoder_activations[0]))
            predictions.append(discs[2](encoder_activations[1]))

            loss_g = 0
            loss_g_mean = 0
            for prediction in predictions:
                loss = nn.functional.binary_cross_entropy(prediction, trick, reduction="sum")
                loss_mean = nn.functional.binary_cross_entropy(prediction, trick, reduction="mean")
                loss_g += loss
                loss_g_mean += loss_mean

            loss_g_mean /= len(predictions)
            reporter.accumulate_loss("gen bce", loss_g_mean)

            recon_loss = torch.nn.functional.mse_loss(x_hat, data.to_dense(), reduction="sum")
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            recon_mean = torch.nn.functional.mse_loss(x_hat, data.to_dense(), reduction="mean")

            kl_mean = kl_div.item() / mu.numel()

            reporter.accumulate_loss("KL Divergence", kl_mean)

            if expert == "human":
                reporter.accumulate_loss("human recon", recon_mean.item())
                human_iterations += 1

            elif expert == "mouse":
                reporter.accumulate_loss("mouse recon", recon_mean.item())
                mouse_iterations += 1

            if epoch < disc_warmup:
                total_loss = recon_loss + 0.15 * kl_div.sum() + (0 * loss_g)
            else:
                total_loss = recon_loss + 0.15 * kl_div.sum() + (gen_weight * loss_g)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(edvae.parameters(), 2.0)
            optimizer.step()

            batch_iteration += 1

        reporter.write_loss("human recon", epoch, divisor=human_iterations)
        reporter.write_loss("mouse recon", epoch, divisor=mouse_iterations)
        reporter.write_loss("KL Divergence", epoch, divisor=batch_iteration)
        reporter.write_loss("gen bce", epoch, divisor=batch_iteration)
        reporter.write_loss("disc bce", epoch, divisor=batch_iteration)

        reporter.zero_losses()
        test(epoch, edvae, test_loader, device, reporter)

    disc = latent.train_md(edvae, test_loader, device, batch_size, reporter,num_epochs=10)

    human_train, human_test = configure_species_loader(seed, batch_size, 1, "human")

    torch.save(edvae, ".")
    # model = latent.load_model()

    # train_md(edvae, test_loader, device, batch_size, reporter,num_epochs=10)
    # train_md_on_output_human(edvae, test_loader, device, batch_size, reporter,num_epochs=10)
    # train_md_on_output_mouse(edvae, test_loader, device, batch_size, reporter,num_epochs=10)
    # adjusted = latent.get_adjusted_latent(disc, edvae, human_test, batch_size, device, file_name=reporter.file_name)