from .whurs19 import WHURS19Dataset
from .fgscr42 import FGSCR42
from .cub import CUB
from .aircraft import FGVCAircraft

dataset_list = {
                "whurs19": WHURS19Dataset,
                "cub": CUB,
                "fgscr42": FGSCR42,
                "aircraft": FGVCAircraft
            }


def build_dataset(dataset, root_path, ways, shots, datasplit):
    return dataset_list[dataset](root_path, ways, shots, datasplit)