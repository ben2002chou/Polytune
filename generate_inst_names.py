# Modified from the original implementation: Copyright 2021 kunato
# Repository: https://github.com/kunato/mt3-pytorch
#
# This code is based on the unofficial PyTorch implementation of MT3: Multi-Task Multitrack Music Transcription.
# Adaptations and modifications have been made for use in this project.
#
# The original implementation can be found in the mt3-pytorch repository.
# 
# This software is provided "AS IS", without warranties or conditions of any kind.
from glob import glob
import yaml
import json
import sys

sys.path.insert(0, "..")
# TODO: we don't need this anymore. can we repurpose this?
from contrib.preprocessor import _SLAKH_CLASS_PROGRAMS


_SLAKH_CLASS_PROGRAMS


def _find_inst_name(is_drum, program_num):
    inst = None
    if is_drum:
        return "Drums"
    for i, (k, v) in enumerate(_SLAKH_CLASS_PROGRAMS.items()):
        if program_num >= v:
            inst = k
        else:
            break
    assert inst is not None
    return inst


def main(root_path):
    meta_paths = glob(f"{root_path}/**/metadata.yaml")
    for meta_path in meta_paths:
        with open(meta_path, "r") as f:
            metadata = yaml.safe_load(f)
            inst_names_path = meta_path.replace("metadata.yaml", "inst_names.json")
            inst_names = {}
            for k in metadata["stems"].keys():
                # print(k, metadata['stems'][k]['inst_class'])
                if metadata["stems"][k].get("integrated_loudness", None) is not None:
                    inst_names[k] = _find_inst_name(
                        metadata["stems"][k]["is_drum"],
                        metadata["stems"][k]["program_num"],
                    )
            with open(inst_names_path, "w") as w:
                json.dump(inst_names, w)
    print("done")


if __name__ == "__main__":
    main("/depot/yunglu/data/datasets_ben/MR_MT3/slakh2100_flac_redux/train")
    main("/depot/yunglu/data/datasets_ben/MR_MT3/slakh2100_flac_redux/validation")
    main("/depot/yunglu/data/datasets_ben/MR_MT3/slakh2100_flac_redux/test")
