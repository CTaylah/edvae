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
        
        x = self.encoder(x)
        mu = self.mean(x) 
        log_var = self.var(x)
        
        x = self.reparameterize(mu, log_var)
        x = self.decoder(x)
        return x, mu, log_var
