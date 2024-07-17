import torch
import torch.nn as nn

from .vae import VAE

class Expert(nn.Module):
    def __init__(self, encoder: nn.Sequential, decoder:nn.Sequential):
        super(Expert, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


class EDVAE(nn.Module):
    def __init__(self, shared_vae: VAE, human_expert: Expert, mouse_expert: Expert) -> None:
        super(EDVAE, self).__init__()
        self.shared_vae = shared_vae
        self.human_expert = human_expert
        self.mouse_expert = mouse_expert
        

    def forward(self, x: torch.Tensor, expert_flag: str):
        if expert_flag != "mouse" and expert_flag != "human":
            raise ValueError("flag must be either mouse or human")

        intermediate = (self.human_expert.encoder(x) if expert_flag== "human" 
                        else self.mouse_expert.encoder(x))

        vae_output, mu, log_var, encoder_activations, z_star = self.shared_vae(intermediate) 

        x_hat = (self.human_expert.decoder(vae_output) if expert_flag == "human" 
                        else self.mouse_expert.decoder(vae_output))

        return x_hat, mu, log_var, encoder_activations, vae_output, z_star
    

    def decode(self, z_star:torch.Tensor, expert_flag: str):
        shared_decoder_out = self.shared_vae.decoder(z_star)
        x_hat = (self.human_expert.decoder(shared_decoder_out) if expert_flag == "human" 
                        else self.mouse_expert.decoder(shared_decoder_out))
        return x_hat

    def cross(self, x: torch.Tensor, expert_flag: str):
        if expert_flag != "mouse" and expert_flag != "human":
            raise ValueError("flag must be either mouse or human")

        intermediate = (self.human_expert.encoder(x) if expert_flag== "human" 
                        else self.mouse_expert.encoder(x))

        vae_output, mu, log_var, encoder_activations, z_star = self.shared_vae(intermediate) 

        x_hat = (self.human_expert.decoder(vae_output) if expert_flag == "mouse" 
                        else self.mouse_expert.decoder(vae_output))

        return x_hat, mu, log_var, encoder_activations, vae_output, z_star
            