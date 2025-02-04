from typing import Any, Tuple, Union
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential, mean: nn.Linear, var: nn.Linear) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mean = mean
        self.var = var
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(self, x: torch.Tensor) -> Tuple[Union[Any, torch.Tensor], Any, Any, Tuple[Any]]:
        
        encoder_activations = []
        for name, layer in self.encoder.named_children():
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                encoder_activations.append(x)

        mu = self.mean(x) 
        log_var = self.var(x)
        
        x = self.reparameterize(mu, log_var)
        z_star = x
        x = self.decoder(x)
        return x, mu, log_var, encoder_activations, z_star
