import pandas as pd
from torch.utils.data import Dataset
import torch
import os
import random
import numpy as np
from torch import nn
from typing import Dict, Optional, Tuple, List
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, random_split
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score, f1_score, recall_score, classification_report
import time
import math
import matplotlib
matplotlib.rcParams.update({'figure.figsize': (16, 12), 'font.size': 14})
import matplotlib.pyplot as plt
from IPython.display import clear_output



# Example:
# {'directed': True, 'multigraph': False, 'graph': {},
# 'nodes': [{'type': 'QUESTIONS_ENTITY', 'name_': 'Q49', 'id': 0, 'label': 'North America'},
# {'type': 'ANSWER_CANDIDATE_ENTITY', 'name_': 'Q194057', 'id': 1, 'label': 'Mount Rainier'}],
# 'links': [{'name_': 'P30', 'source': 1, 'target': 0, 'label': 'continent'}]}

def linearize_graph_gen(SEP_TOKEN):
    def linearize_graph(graph_dict):
        nodes = sorted((node_dict for node_dict in graph_dict["nodes"]), key=lambda d:d["id"])
        for n_id, node_dict in enumerate(nodes):
            assert n_id == node_dict["id"]
        src_node_id2links = {}
        for link_dict in graph_dict["links"]:
            link_src =  link_dict["source"]
            if src_node_id2links.get(link_src) is None:
                src_node_id2links[link_src] = []
            src_node_id2links[link_src].append(link_dict)
        graph_s = ""
    
        for n_id, node_dict in enumerate(nodes):
            links = src_node_id2links.get(n_id, list())
            start_label = node_dict["label"]
            if node_dict["type"] == "ANSWER_CANDIDATE_ENTITY":
                start_label = f"{SEP_TOKEN} {start_label} {SEP_TOKEN}"
                #start_label = f"[ {start_label} ]"
            for link_dict in links:
                target_label = nodes[link_dict["target"]]["label"]
                if nodes[link_dict["target"]]["type"] == "ANSWER_CANDIDATE_ENTITY":
                    target_label = f"{SEP_TOKEN} {target_label} {SEP_TOKEN}"
                    #target_label = f"[ {target_label} ]"
                link_s = f" {start_label}, {link_dict['label']}, {target_label} "
                graph_s += link_s
    
        return graph_s

    return linearize_graph

def get_graph_num_internal(graph_dict):
    nodes = graph_dict['nodes']
    internal = [node for node in nodes if node['type'] == 'INTERNAL']
    return len(internal)

def get_num_links(graph_dict):
    return len(graph_dict['links'])