import os, sys
import enum

class DatasetType(enum.Enum):
    Flickr30k = "Flickr30k"
    #TODO specify all available datasets here

def load_data_loader(dataset: str, **kwargs):
    """
    Load specified data loader

    Parameters:
        dataset (str): name of dataset to load
        kwargs (dict): additional config arguments to pass to dataset
    Returns:
        DataLoader: instanciated and configured data loader sub-class
    """

    # TODO instanciate data loader sub-class from given dataset name

    raise NotImplementedError

class DataLoader():
    """
    Base class for data loaders
    """
    def __init__(self) -> None:
        pass

    def __iter__(self):
        """
        Iterator for data loader
        """
        self.idx = 0
        return self

    def __next__(self):
        """
        Next item in data loader
        """
        if self.idx < len(self):
            self.idx += 1
            return self[self.idx]
        else:
            raise StopIteration


class Flickr30kDataset(DataLoader):
    def __init__(self) -> None:
        pass