from .camvid_loader import CamVidLoader


def get_loader(dataset_type):
    if dataset_type == 'camvid':
        return CamVidLoader
