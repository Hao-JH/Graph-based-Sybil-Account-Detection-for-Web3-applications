import torch
import torch.nn.functional as F

def similarity_loss(node_features, pos_pairs, neg_pairs):
    

    # 获取正样本和负样本的节点特征向量
    pos_node1 = node_features[pos_pairs['node1']]
    pos_node2 = node_features[pos_pairs['node2']]
    neg_node1 = node_features[neg_pairs['node1']]
    neg_node2 = node_features[neg_pairs['node2']]

    # 计算正样本和负样本的相似度
    pos_similarity = F.cosine_similarity(pos_node1, pos_node2)
    neg_similarity = F.cosine_similarity(neg_node1, neg_node2)

    # 计算相似度损失
    similarity_loss = F.relu(neg_similarity - pos_similarity + 1).mean()

    return similarity_loss