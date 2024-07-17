import time
import torch

import experiments.experiment

def test_loaders():
    import edvae.data.dataloader as dataloader
    num_samples = 3200
    batch_size = 128
    train_test_split = {"train": 0.8, "test": 0.2}
    human_train, human_test = dataloader.build_single_species_loaders("homo_sapiens", num_samples, batch_size, train_test_split, 1, tag="human")
    batch_iteration = 0
    for _ in human_train:
        batch_iteration += 1
    print(batch_iteration)

if __name__== '__main__':
    torch.manual_seed(42)
    start = time.time()
    device = torch.device("cuda")
    experiments.experiment.train('baseline copy.yaml', device)
    test_loaders()
    end = time.time()
    length = end - start
    print("Time taken:", length, "seconds!")