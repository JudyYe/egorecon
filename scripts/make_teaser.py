import json
import os.path as osp

from tqdm import tqdm


clip_info = 'data/HOT3D-CLIP/clip_definitions.json'
# "1849": {
#     "sequence_id": "P0001_23fa0ee8",
#     "device": "Aria",
#     "per_frame_timestamps_ns": [
#         {
#             "214-1": 54971174871117,
#             "1201-1": 54971174967617,
#             "1201-2": 54971174967629
#         },
#         {
#             "214-1": 54971208198579,
#             "1201-1": 54971208295754,
#             "1201-2": 54971208295766
#         },
#         {
#             "214-1": 54971241525290,
#             "1201-1": 54971241623878,
#             "1201-2": 54971241623865
#         },
#         {
#             "214-1": 54971274859155,
#             "1201-1": 54971274951955,
#             "1201-2": 54971274951967
#         },

split_file = 'data/HOT3D-CLIP/sets/split.json'
split = "test"


def get_candidate_seq():
    with open(split_file, "r") as f:
        split_list = json.load(f)
    
    clip_info = json.load(open(clip_info, "r"))
    

    return candidate_list