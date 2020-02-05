import numpy as np
# import sys,os
# sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from feed_data import RelationEntityBatcher
from grapher import RelationEntityGrapher
#
# from ..data.feed_data import RelationEntityBatcher
# from ..data.grapher import RelationEntityGrapher
from environment import Episode, Env
from collections import defaultdict

from options import read_options
import json
import csv

#TEST MAIN
options = read_options()

data_input_dir="datasets/data_preprocessed/countries_S3/"
vocab_dir="datasets/data_preprocessed/countries_S3/vocab"
total_iterations=1000
path_length=3
hidden_size=2
embedding_size=2
batch_size=128
beta=0.1
Lambda=0.02
use_entity_embeddings=1
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/countries_s3/"
model_load_dir="nothing"
load_model=0
nell_evaluation=0

options['relation_vocab'] = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
options['entity_vocab'] = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))

mid_to_word = {}
print('Total number of entities {}'.format(len(options['entity_vocab'])))
print('Total number of relations {}'.format(len(options['relation_vocab'])))
save_path = ''

params = options
train_environment = Env(params, 'train')
#dev_test_environment = Env(params, 'dev')
#test_test_environment = Env(params, 'test')
#
# for episode in train_environment.get_episodes():
#     # get initial state
#     state = episode.get_state()

# self.grapher = RelationEntityGrapher(triple_store=params['data_input_dir'] + '/' + 'graph.txt',
#                                      max_num_actions=params['max_num_actions'],
#                                      entity_vocab=params['entity_vocab'],
#                                      relation_vocab=params['relation_vocab'])
triple_store = '../datasets/data_preprocessed/countries_S3/graph.txt'
entity_vocab = options['entity_vocab']
relation_vocab = options['relation_vocab']
store = defaultdict(list)
with open(triple_store) as triple_file_raw:
    triple_file = csv.reader(triple_file_raw, delimiter='\t')
    for line in triple_file:
        e1 = entity_vocab[line[0]]
        r = relation_vocab[line[1]]
        e2 = entity_vocab[line[2]]
        store[e1].append((r, e2))

# print(store)

max_num_actions = 200
array_store = np.ones((len(entity_vocab), max_num_actions, 2), dtype=np.dtype('int32'))

for e1 in store:
    num_actions = 1
    array_store[e1, 0, 1] = relation_vocab['NO_OP']
    array_store[e1, 0, 0] = e1
    for r, e2 in store[e1]: # e1, r, e2 (relation before e2)
        if num_actions == array_store.shape[1]: # default: 200
            break
        array_store[e1,num_actions,0] = e2
        array_store[e1,num_actions,1] = r
        num_actions += 1

# count = 0
# for e1 in store:
#     if count > 5:
#          break
#     print(e1)
#     print(store[e1])
#     count += 1

def fine_one_node_neighbor(current_entity, store):
    neighbor_entity = []
    neighbor_relation = []
    for r, e2 in store[current_entity]:
        neighbor_entity.append(e2)
        neighbor_relation.append([e1, r, e2])
    return neighbor_entity, neighbor_relation

#neighbor_entity, neighbor_relation = fine_one_node_neighbor(current_entity, store)

k_degree = 3
vertices = []
adjs = []

current_entity_list = []
current_entity_list.append(2)
print(current_entity_list)

for k in range(k_degree):
    print(k)
    k_vertices = []
    k_adjs = []
    for current_entity in current_entity_list:
        neighbor_entity, neighbor_relation = fine_one_node_neighbor(current_entity, store)
        k_vertices.extend(neighbor_entity)
        k_adjs.extend(neighbor_relation)
    print(k_vertices)
    current_entity_list = k_vertices
    vertices.append(k_vertices)
    adjs.append(k_adjs)
        

current_entity = 2
k1_entity = []
k1_r = []
for r, e2 in store[current_entity]:
    print(store[current_entity])
    k1_entity.append(e2)
    k1_r.append(r)


del store