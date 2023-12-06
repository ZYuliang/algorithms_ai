import numpy as np
import torch


def process_model_output(model_predictions, threshold=0):
    # 处理logits ，可以返回每个样本的每个实体的token，根据原文映射返回在原文中的位置
    input_ids_mask = model_predictions[-1]
    ignored_token_ids = [0, -100]
    ignored_token_index_dict = dict()
    for i, j in zip(*np.where(np.isin(input_ids_mask, ignored_token_ids))):
        if i not in ignored_token_index_dict:
            ignored_token_index_dict[i] = {j}
        else:
            ignored_token_index_dict[i].add(j)

    entity_output = model_predictions[0]
    if isinstance(entity_output, torch.Tensor):
        entity_output = entity_output.numpy()

    entity_edges = dict()
    for sample_id, entity_id, x, y in zip(*np.where(entity_output > threshold)):
        if sample_id not in entity_edges:
            entity_edges[sample_id] = dict()
        if entity_id not in entity_edges[sample_id]:
            entity_edges[sample_id][entity_id] = []

        if x <= y:
            entity_edges[sample_id][entity_id].append((x, y))

    max_sample_num, max_entity_num, _, _ = entity_output.shape
    grouped_entities = group_entities(entity_edges, max_sample_num, max_entity_num,
                                      ignored_token_index_dict, return_entity_part=True)
    return grouped_entities


def process_entity_logits(entity_output, ignored_token_index_dict, threshold=0):
    # 处理logits ，可以返回每个样本的每个实体的token，根据原文映射返回在原文中的位置
    if isinstance(entity_output, torch.Tensor):
        entity_output = entity_output.numpy()

    entity_edges = dict()
    for sample_id, entity_id, x, y in zip(*np.where(entity_output > threshold)):
        if sample_id not in entity_edges:
            entity_edges[sample_id] = dict()
        if entity_id not in entity_edges[sample_id]:
            entity_edges[sample_id][entity_id] = []

        if x <= y:
            entity_edges[sample_id][entity_id].append((x, y))

    max_sample_num, max_entity_num, _, _ = entity_output.shape
    grouped_entities = group_entities(entity_edges, max_sample_num, max_entity_num,
                                      ignored_token_index_dict, return_entity_part=False)
    return grouped_entities


def group_entities(entity_edges, max_sample_num, max_entity_num, ignored_token_index_dict, return_entity_part=False):
    # 对于所有样本，对获得的边进行group操作得到每个样本的每个实体类型的每个实体，sample_list[entity_type_list[entity_set(entity_tuple())]]
    grouped_entities = dict()

    for sample_id in range(max_sample_num):
        grouped_entities[sample_id] = dict()
        removed_nodes = ignored_token_index_dict.get(sample_id, [])
        for entity_id in range(max_entity_num):
            if isinstance(entity_edges, dict):
                entity_res = entity_edges.get(sample_id, dict()).get(entity_id, [])
            else:
                entity_res = entity_edges[sample_id][entity_id].tolist()
            entity_res = list(find_cliques(entity_res, removed_nodes))
            if not entity_res:
                grouped_entities[sample_id][entity_id] = set()
            else:
                if return_entity_part:
                    grouped_entities[sample_id][entity_id] = [cut_seq(i) for i in set(entity_res)]
                else:
                    grouped_entities[sample_id][entity_id] = set(entity_res)

    return grouped_entities


def find_cliques(edges, removed_nodes=(0,)):
    # 寻找图中所有最大团,

    nodes = set()
    adj = dict()
    for i, j in edges:
        if i in removed_nodes or j in removed_nodes:
            continue
        if i != j:
            nodes.add(i)
            nodes.add(j)
            if i not in adj:
                adj[i] = {j}
            else:
                adj[i].add(j)

            if j not in adj:
                adj[j] = {i}
            else:
                adj[j].add(i)
        else:
            yield (i,)

    if len(edges) == 0:
        return []
    if len(nodes) == 0:
        return []
    Q = [None]
    subg = set(nodes)
    cand = set(nodes)

    u = max(subg, key=lambda u: len(cand & adj[u]))
    ext_u = cand - adj[u]
    stack = []
    try:
        while True:
            if ext_u:
                q = ext_u.pop()
                cand.remove(q)
                Q[-1] = q
                adj_q = adj[q]
                subg_q = subg & adj_q
                if not subg_q:
                    yield tuple(sorted(Q[:]))
                else:
                    cand_q = cand & adj_q
                    if cand_q:
                        stack.append((subg, cand, ext_u))
                        Q.append(None)
                        subg = subg_q
                        cand = cand_q
                        u = max(subg, key=lambda u: len(cand & adj[u]))
                        ext_u = cand - adj[u]
            else:
                Q.pop()
                subg, cand, ext_u = stack.pop()
    except IndexError:
        pass


def cut_seq(seq):
    # 切序列
    if not seq:
        return []
    if seq[-1] - seq[0] + 1 == len(seq):
        return [seq]
    else:
        refined_seq = []
        index = seq[0]
        sub_seq = [seq[0]]
        for i in seq[1:]:
            if i == index + 1:
                sub_seq.append(i)
            else:
                if sub_seq:
                    refined_seq.append(sub_seq)
                sub_seq = [i]
            index = i
        if sub_seq:
            refined_seq.append(sub_seq)
        return refined_seq

if __name__ == '__main__':
    pass