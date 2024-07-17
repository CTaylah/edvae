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


class CombinedDataLoader:
    def __init__(self, species1_dataloader, species2_dataloader):
        self.species1_dl = species1_dataloader
        self.species2_dl = species2_dataloader
        self.species1_iter = iter(self.species1_dl)
        self.species2_iter = iter(self.species2_dl)

    def __iter__(self):
        self.species1_iter = iter(self.species1_dl)
        self.species2_iter = iter(self.species2_dl)
        return self

    def __next__(self):
        try:
            data1 = next(self.species1_iter)
        except StopIteration:
            self.species1_iter = iter(self.species1_dl)
            data1 = next(self.species1_iter)
        
        try:
            data2 = next(self.species2_iter)
        except StopIteration:
            self.species2_iter = iter(self.species2_dl)
            data2 = next(self.species2_iter)
        
        return data1, data2
