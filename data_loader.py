import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn import tree
import graphviz

def load_credit_card_data():
    data = pd.read_csv('datasets/UCI_Credit_Card.csv')
    dataX = data.drop('default.payment.next.month', axis=1).copy().values
    dataY = data['default.payment.next.month'].copy().values
    return dataX, dataY

def load_mushroom_data(return_cols=False):
    mushrooms = pd.read_csv('datasets/mushrooms.csv')
    mushrooms = mushrooms.drop("veil-type",axis=1)
    # mushrooms = pd.get_dummies(mushrooms,
    #                            columns=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
    #                                     'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
    #                                     'stalk-surface-above-ring', 'stalk-surface-below-ring',
    #                                     'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
    #                                     'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])
    lencoder = LabelEncoder()
    for col in mushrooms.columns:
        mushrooms[col] = lencoder.fit_transform(mushrooms[col])

    # mushrooms["class"] = lencoder.fit_transform(mushrooms["class"])
    mushroomsX = mushrooms.drop('class', axis=1).copy().values
    mushroomsY = mushrooms['class'].copy().values

    if return_cols:
        return mushroomsX, mushroomsY,mushrooms.columns[1:]
    return mushroomsX, mushroomsY


def load_diabetes_data(return_cols=False):
    diabetes = pd.read_csv('datasets/diabetes.csv')
    diabetesX = diabetes.drop('Outcome', axis=1).copy().values
    diabetesY = diabetes['Outcome'].copy().values

    if return_cols:
        return diabetesX, diabetesY, diabetes.columns[:-1]
    return diabetesX, diabetesY



def load_data(dataset,return_cols=False):
    if dataset=="mushrooms":
        return load_mushroom_data(return_cols=return_cols)
    elif dataset=="credit_cards":
        return load_credit_card_data(return_cols=return_cols)
    elif dataset=="diabetes":
        return load_diabetes_data(return_cols=return_cols)


def export_decision_tree(dtree,cols,dataset,label):
    # Export decision tree
    OUTPUT_DIRECTORY = "output"
    graph_data = tree.export_graphviz(dtree, out_file=None, feature_names=cols,class_names=label)
    graph_data = graphviz.Source(graph_data)
    graphviz.render("out[ut/images/{}_dtree.png".format(dataset),view=True)
    # system("dot -Tpng /home/liuwensui/Documents/code/dtree2.dot -o /home/liuwensui/Documents/code/dtree2.png")