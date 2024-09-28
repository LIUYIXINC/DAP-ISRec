import pandas as pd
import networkx as nx
import random
import get_implicit_friends

"""Processing triplet data"""
def getdata(path):
    userset = set()
    itemset =set()
    user_item_rating =[]
    with open(path,"r",encoding="utf_8") as f:
        for line in f.readlines():
            lines = line.strip().split('\t')
            userset.add(int(lines[0]))
            itemset.add(int(lines[1]))
            user_item_rating.append((int(lines[0]),int(lines[1]),int(lines[2])))
    return list(userset),list(itemset),user_item_rating


"""Obtaining edges and entities of the user-attribute graph"""
def readGraphData( path ):
    entity_set = set( )
    pairs = [ ]
    with open(path,'r',encoding='utf_8')as f:
        for line in f.readlines():
            lines = line.strip().split(',')
            if len(lines)!=4:
                continue
            entity_set.add(int(lines[0]))
            entity_set.add(int(lines[1]))
            entity_set.add(int(lines[2]))
            entity_set.add(int(lines[3]))
            pairs.append((int(lines[0]),int(lines[1])))
            pairs.append((int(lines[0]),int(lines[2])))
            pairs.append((int(lines[0]),int(lines[3])))
    return list( entity_set ), list( set( pairs ) )

"""Constructing the user-attribute graph G1"""
def get_graph(pairs):
    G = nx.Graph()
    G.add_edges_from(pairs)
    return G
"""Constructing the Positive sentiment implicit social graph G2"""
def get_Positive_graph(train_data,user_set):
    empty_Positive = []
    pairs,weights = get_implicit_friends.get_positive_friends(train_data)
    G = nx.Graph()
    weighted_edges = [(node1, node2, weight) for (node1, node2), weight in zip(pairs, weights)]
    G.add_weighted_edges_from(weighted_edges)  # Loading data using a weighted edge set
    for user in user_set:
        node_to_check = user
        if G.has_node(node_to_check):
            n = 0
        else:
            empty_Positive.append(node_to_check)
            G.add_edge(node_to_check, node_to_check, weight=1)
    return G,empty_Positive
"""Constructing the Negative sentiment implicit social graph G3"""
def get_Negative_graph(train_data,user_set):
    empty_Negative = []
    pairs,weights = get_implicit_friends.get_negative_friends(train_data)
    G = nx.Graph()
    weighted_edges = [(node1, node2, weight) for (node1, node2), weight in zip(pairs, weights)]
    G.add_weighted_edges_from(weighted_edges)
    for user in user_set:
        node_to_check = user
        if G.has_node(node_to_check):
            n = 0
        else:
            empty_Negative.append(node_to_check)
            G.add_edge(node_to_check, node_to_check, weight=1)
    return G,empty_Negative

"""User-attribute graph sampling"""
def graphSage4RecAdjType( G, items, n_sizes):
    adj_lists = [ ]
    for size in n_sizes:
        target_nodes = items
        neighbor_nodes = []
        items = set( )
        for i in target_nodes:
            neighbors = list( G.neighbors( i ) )
            if len(neighbors) >= size:
                neighbors = random.sample( neighbors, size)
            else:
                neighbors =  [random.choice(neighbors) for _ in range(size)]
            neighbor_nodes.append( neighbors )
            items |= set( neighbors )
        items = list(items)
        adj_lists.append( pd.DataFrame( neighbor_nodes, index = target_nodes ) )
    adj_lists.reverse( )
    return adj_lists
"""User Implicit Social Graph Sampling"""
def graphSage4RecAdjType2( G, items, n_sizes = [5] ):
    adj_lists = [ ]
    for size in n_sizes:
        target_nodes = items
        neighbor_nodes = []
        items = set( )
        for i in target_nodes:
            neighbors = list( G.neighbors( i ) )
            if len(neighbors) >= size:
                neighbors = random.sample( neighbors, size)
            else:
                neighbors =  [random.choice(neighbors) for _ in range(size)]
            neighbor_nodes.append( neighbors )
            items |= set( neighbors )
        items = list(items)
        adj_lists.append( pd.DataFrame( neighbor_nodes, index = target_nodes ) )

    adj_lists.reverse( )

    return adj_lists
"""Obtaining edges and entities of the user-attribute graph (BOOKCrossing)"""
def readGraphData2( path ):
    entity_set = set( )
    pairs = [ ]
    with open(path,'r',encoding='utf_8')as f:
        for line in f.readlines():
            lines = line.strip().split(',')
            if len(lines)!=3:
                continue
            entity_set.add(int(lines[0]))
            entity_set.add(int(lines[1]))
            entity_set.add(int(lines[2]))
            pairs.append((int(lines[0]),int(lines[1])))
            pairs.append((int(lines[0]),int(lines[2])))
    return list( entity_set ), list( set( pairs ) )
if __name__ == '__main__':
    pass