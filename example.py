import torch.nn as nn

import models
import models.edvae
import models.vae
from .configs import HUMAN_DATA_SIZE, MOUSE_DATA_SIZE

human_encoder = nn.Sequential(HUMAN_DATA_SIZE, 512)
human_decoder = nn.Sequential(512, HUMAN_DATA_SIZE)

human_expert = models.edvae.Expert(human_encoder, human_decoder)


mouse_encoder = nn.Sequential(MOUSE_DATA_SIZE, 512)
mouse_decoder = nn.Sequential(512, MOUSE_DATA_SIZE)

mouse_expert = models.edvae.Expert(mouse_encoder, mouse_decoder)

encoder = nn.Sequential(512, 256, 128)

mu_layer = nn.Linear(128, 64)
log_var_layer = nn.Linear(128, 64)

decoder = nn.Sequential(128, 256, 512)

shared_vae = models.vae.VAE(encoder, decoder, mu_layer, log_var_layer)

edvae = models.edvae.EDVAE(shared_vae, human_expert, mouse_expert)