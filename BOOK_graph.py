import utils
import pickle
user_set, item_set, train_data = utils.getdata("Data/BookCrossing/train_set.csv")
print('Constructing graph...')
# Constructing the user-attribute graph G1
entitys, pairs = utils.readGraphData2('Data/BookCrossing/userinfo.csv')
G1 = utils.get_graph(pairs)
# 使用Pickle序列化图数据
with open('Data/BookCrossing/G11.pkl', 'wb') as f:
    pickle.dump(G1, f)
# 从文件中加载图数据
with open('Data/BookCrossing/G11.pkl', 'rb') as f:
    G = pickle.load(f)
# 验证图数据是否加载正确
print(G.edges(data=False))
# exit(0)
# Constructing the Positive sentiment implicit social graph G2
G2, empty_Positive = utils.get_Positive_graph(train_data, user_set)
with open('Data/BookCrossing/G22.pkl', 'wb') as f:
    pickle.dump(G2, f)
# "empty_Positive" refers to the isolated set of users without positive sentiment similar friends.
print(empty_Positive)
# Constructing the Negative sentiment implicit social graph G3
G3, empty_Negative = utils.get_Negative_graph(train_data, user_set)
with open('Data/BookCrossing/G33.pkl', 'wb') as f:
    pickle.dump(G3, f)
# "empty_Negative" refers to the isolated set of users without negative sentiment similar friends.
print(empty_Negative)
