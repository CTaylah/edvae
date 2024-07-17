import yaml
import torch.nn as nn

def create_layers(layer_configs):
    layers = []
    for layer in layer_configs:
        if 'layer' in layer:
            if layer['layer'] == 'Linear':
                layers.append(nn.Linear(layer['in'], layer['out']))
            elif layer['layer'] == 'BatchNorm1d':
                layers.append(nn.BatchNorm1d(layer['in']))
        if 'activation' in layer:
            if layer['activation'] == 'ReLU':
                layers.append(nn.ReLU())
            elif layer['activation'] == 'Sigmoid':
                layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

def get_configs(yaml_directory):
    with open(yaml_directory, 'r') as file:
        config = yaml.safe_load(file)

    hparams = config['HParams']
    del config['HParams']

    models_dict = {}
    for model in config:
        for sub_model in config[model]:
            sequential = create_layers(config[model][sub_model])
            models_dict[model + "_" + sub_model] = sequential

    return hparams, models_dict

