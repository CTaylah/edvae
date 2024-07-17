from functools import partial

import cellxgene_census
import cellxgene_census.experimental.ml as census_ml
import tiledbsoma as soma

import torchdata.datapipes as dp
import torch
from torch.utils.data import DataLoader

def _build_species_data_pipe(species: str, batch_size: int, soma_chunk_size=10_000, shuffle=True, seed=1) -> census_ml.ExperimentDataPipe:
    census = cellxgene_census.open_soma(census_version="2023-12-15")
    experiment = census["census_data"][species]
    filter = 'is_primary_data == True and assay in ["microwell-seq", "10x 3\' v1", "10x 3\' v2", "10x 3\' v3", "10x 3\' transcription profiling", "10x 5\' transcription profiling", "10x 5\' v1", "10x 5\' v2"]'
    return census_ml.ExperimentDataPipe(
        experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_query=soma.AxisQuery(value_filter=filter),
        obs_column_names=["cell_type"],
        batch_size=batch_size,
        shuffle=shuffle,
        soma_chunk_size=soma_chunk_size,
        seed=seed
    )

def _preprocess(data, tag):
    epsilon = 1e-9
    zero_mask = (data[0] == 0.0)
    clipped = torch.clamp(data[0], min=epsilon)
    log_tensor = torch.log(clipped)
    log_tensor[zero_mask] = 0.0
    return (log_tensor, data[1], tag)

def _build_preprocessing_fn(tag: str):
    return partial(_preprocess, tag=tag)

def build_single_species_loaders(species: str, num_samples: int, batch_size: int, train_test_split: dict[str,float], num_workers=1, 
                       soma_chunk_size=10_000, shuffle=True, seed=1, tag=None):

    if(tag == None): 
        tag = species

    experiment_datapipe = _build_species_data_pipe(species, batch_size, soma_chunk_size, shuffle)
    num_batches = num_samples/batch_size

    subset_pipeline = experiment_datapipe.header(num_batches)

    #Remaps data to be of form (data, metadata, expert)
    pre_process_fn =  _build_preprocessing_fn(tag)
    tagged_subset_pipeline = dp.iter.Mapper(subset_pipeline, pre_process_fn)

    train_datapipe, test_datapipe = tagged_subset_pipeline.random_split(total_length=num_batches, weights=train_test_split, seed=seed)

    return (
        census_ml.experiment_dataloader(train_datapipe, num_workers=num_workers, pin_memory=True),
        census_ml.experiment_dataloader(test_datapipe, num_workers=num_workers, pin_memory=True),
    )



class SpeciesAlternator():
    def __init__(self, species1_dataloader, species2_dataloader):
        self.species1_dl = species1_dataloader
        self.species2_dl = species2_dataloader
        self.switch = True
        self.species1_iter = None
        self.species2_iter = None
        self.species1_exhausted = False
        self.species2_exhausted = False

    def __iter__(self):
        self.species1_iter = iter(self.species1_dl)
        self.species2_iter = iter(self.species2_dl)
        self.species1_exhausted = False
        self.species2_exhausted = False
        return self

    def __next__(self):
        if self.species1_exhausted and self.species2_exhausted:
            raise StopIteration

        if self.switch:
            self.switch = not self.switch
            if not self.species1_exhausted:
                try:
                    return next(self.species1_iter)
                except StopIteration:
                    self.species1_exhausted = True
                    if self.species2_exhausted:
                        raise
                    else:
                        return next(self)
        else:
            self.switch = not self.switch
            if not self.species2_exhausted:
                try:
                    return next(self.species2_iter)
                except StopIteration:
                    self.species2_exhausted = True
                    if self.species1_exhausted:
                        raise
                    else:
                        return next(self)