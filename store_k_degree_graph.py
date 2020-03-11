import numpy as np
from collections import defaultdict
#
# from options import read_options
import json
import csv
#
# # TEST MAIN

DATASET = "grid-world/problem_16_4_2"

data_input_dir = "../datasets/data_preprocessed/" + DATASET
vocab_dir = "../datasets/data_preprocessed/" + DATASET + "/vocab"
triple_store = "../datasets/data_preprocessed/" + DATASET + "/graph.txt"

entity_vocab = json.load(open(vocab_dir + '/entity_vocab.json'))
relation_vocab = json.load(open(vocab_dir + '/relation_vocab.json'))
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
    for r, e2 in store[e1]:  # e1, r, e2 (relation before e2)
        if num_actions == array_store.shape[1]:  # default: 200
            break
        array_store[e1, num_actions, 0] = e2
        array_store[e1, num_actions, 1] = r
        num_actions += 1

def fine_one_node_neighbor(current_entity, store):
    neighbor_entity = []
    neighbor_relation = []
    for r, e2 in store[current_entity]:
        neighbor_entity.append(e2)
        neighbor_relation.append([r, e2])
    return neighbor_entity, neighbor_relation


# neighbor_entity, neighbor_relation = fine_one_node_neighbor(current_entity, store)

#==============store the 3-degree for each node====================
Vertices = []
Adjs = []

Vertices_list = []
Adjs_list = []

k_degree = 3

test_count = 0

for e1 in store:
    current_entity_list = []
    print('e1', e1)
    test_count = test_count + 1

    current_entity_list.append(e1)
    vertices = []
    adjs = []
    for k in range(k_degree):
        # print(k)
        k_vertices = []
        k_adjs = []
        for current_entity in current_entity_list:
            neighbor_entity, neighbor_relation = fine_one_node_neighbor(current_entity, store)
            k_vertices.extend(neighbor_entity)
            k_adjs.extend(neighbor_relation)
        # print(k_vertices)
        current_entity_list = k_vertices
        vertices.extend(k_vertices)
        adjs.extend(k_adjs)
    vertices_array = np.array(vertices).reshape(len(vertices), 1)
    adjs_array = np.array(adjs).reshape(len(adjs), 2)

    print(vertices_array.shape)
    print(adjs_array.shape)

    Vertices.append(vertices_array)
    Adjs.append(adjs_array)

    Vertices_list.append(vertices)
    Adjs_list.append(adjs)

    #
    # if test_count>2:
    #     break

# print(Vertices)
# print(Adjs)

E1s = []
for e1 in store:
    E1s.append(e1)


k_neighbor_dict = {
    'Current_node': E1s,
    'Vertices_list': Vertices_list,
    'Adjs_list': Adjs_list
}

json_dict = json.dumps(k_neighbor_dict)
f = open(data_input_dir + "/k_neighbor_dict.json", "w")
f.write(json_dict)
f.close()

from collections import defaultdict
k_neighbor_dd = defaultdict(list)

for i in range(len(E1s)):
    k_neighbor_dd[E1s[i]].append(Vertices_list[i])
    k_neighbor_dd[E1s[i]].append(Adjs_list[i])

json_dd = json.dumps(k_neighbor_dd)
f = open(data_input_dir + "/k_neighbor_dd.json", "w")
f.write(json_dd)
f.close()
# read as k_neighbor_dd[current_entity_id][0] for Vertices and k_neighbor_dd[current_entity_id][1]

with open(data_input_dir + '/Vertices.txt', 'w') as f:
    for item in Vertices:
        f.write("%s\n" % item)

with open(data_input_dir + '/Adjs.txt', 'w') as f:
    for item in Adjs:
        f.write("%s\n" % item)



# del store