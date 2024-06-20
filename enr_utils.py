import torch
import numpy as np
import pandas as pd
import torch_explain as te
from torch_explain import datasets
from torch_explain.nn.concepts import ConceptReasoningLayer
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_explain.nn.semantics import GodelTNorm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def get_dataset(dataset_name, size):
    if dataset_name == 'XOR':
        x, c, y = datasets.xor(size) #n_concepts=2
    elif dataset_name == 'trigonometry':
        x, c, y = datasets.trigonometry(size) #n_concepts=3
    elif dataset_name == 'dot':
        x, c, y = datasets.dot(size) #n_concepts=2
    elif dataset_name == 'IsBinEven':
        x, c, y = datasets.is_bin_even(size) #n_concepts=4
    elif dataset_name == 'mux41':
        x, c, y = datasets.mux41(size) #n_concepts=6
    elif dataset_name == 'Mux41_twoInputs':
        x, c, y = datasets.mux41_two_inputs(size) #n_concepts=5
    elif dataset_name == 'two_Muxes':
        x, c, y = datasets.two_muxes(size) #n_concepts=3
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    y = F.one_hot(y.long().ravel()).float()
    return x, c, y


def load_DCRBase(x, c, y, dataset_name):
    embedding_size = 8
    concept_encoder = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 10),
        torch.nn.LeakyReLU(),
        te.nn.ConceptEmbedding(10, c.shape[1], embedding_size),
    )
    task_predictor = ConceptReasoningLayer(embedding_size, y.shape[1])
    model = torch.nn.Sequential(concept_encoder, task_predictor)

    model.load_state_dict(torch.load(f'Models/DCRBase/model_state_dict_DCRBase_{dataset_name}.pth'))
    cem = model[0]
    dcr = model[1]
    return cem, dcr

def create_df_DCRBase(x, cem, dcr):
    cem.eval()
    c_emb , c_pred = cem(x)
    y_pred_dt, sign_attn_dt, filter_attn_dt = dcr(c_emb, c_pred, return_attn=True)

    logic = GodelTNorm()
    values_dt = c_pred.unsqueeze(-1).repeat(1, 1, len(y_pred_dt[1]))
    sign_terms_dt = logic.iff_pair(sign_attn_dt, values_dt)

    #For each concept we get sign and filter attn
    final_features = []
    for i in range(c_pred.shape[1]):
        sign_concept = np.vstack([sign.detach().numpy() for sign in sign_terms_dt[:,i,:]])
        filter_concept = np.vstack([f.detach().numpy() for f in filter_attn_dt[:,i,:]])
        final_features.extend([sign_concept, filter_concept])
        #filtered_concept = np.vstack([fc.detach().numpy() for fc in filtered_values[:,i,:]])
        #final_features.extend([filtered_concept])

    #Concatenate everything in one single numpy array
    final_features = np.hstack(final_features)

    #Create name of columns
    names = []
    classes = [f"y{i}" for i in range(y_pred_dt.shape[1])]
    for i in range(c_pred.shape[1]):
        for feature_type in ['sign', 'filter']:
            for y in classes:
                names.append(f"{feature_type}_c{i}_{y}")

    df = pd.DataFrame(final_features, columns=names)

    #Identify unique classes
    classes = set(col.split('_')[-1] for col in df.columns if 'y' in col)
    classes = sorted(list(classes))

    #Add a column with the predicted class
    df['class'] = y_pred_dt.argmax(dim=1)
    df['class'] = df['class'].apply(lambda x: 'y0' if x == 0 else 'y1')
    
    return df, classes


def plot_feature_importance(dataset_name, size=1500):
    #Load the dataset
    x, c, y = get_dataset(dataset_name, size)

    #Load the model
    cem, dcr = load_DCRBase(x, c, y, dataset_name)

    #Create the dataframe
    df, classes = create_df_DCRBase(x, cem, dcr)

    X = df.drop(columns=['class'])
    y = df['class']

    clf = RandomForestClassifier(max_depth=int(len(X.columns)/4), random_state=42)
    clf.fit(X, y)
        
    #Aggregate the feature importance by concept
    feature_importances = {}
    for feature_name, imp in zip(clf.feature_names_in_, clf.feature_importances_):
        feature_name = feature_name.split("_")[1] if "_" in feature_name else feature_name
        if feature_name in feature_importances:
            feature_importances[feature_name] += imp
        else:
            feature_importances[feature_name] = imp

    #Sort the feature importance
    feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=False))

    #Plot the feature importance
    y_values = list(feature_importances.keys())
    x_values = list(feature_importances.values())
    plt.barh(y_values, x_values, color='blue')
    plt.xlabel("Feature importance")
    plt.ylabel("Concepts")
    plt.title(f"Feature importance aggregated by concept ({dataset_name})")
    
    #Add numerical values on the bar plot
    #for index, value in enumerate(x_values):
        #plt.text(value - 0.05, index, str(round(value, 3)), color='white', va='center')
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance_per_class(dataset_name, size=1500):
    #Load the dataset
    x, c, y = get_dataset(dataset_name, size)

    #Load the model
    cem, dcr = load_DCRBase(x, c, y, dataset_name)

    #Create the dataframe
    df, classes = create_df_DCRBase(x, cem, dcr)

    n_classes = len(classes)

    #Create a figure with a subplot for each class
    fig, axs = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    fig.suptitle(f"Feature importance aggregated by concept for each class ({dataset_name})")

    X = df.drop(columns=['class'])
    y = df['class']

    clf = RandomForestClassifier(max_depth=int(len(X.columns)/4), random_state=42)
    clf.fit(X, y)

    #Aggregate the feature importance by the categorical variables
    feature_importances = {}
    for feature_name, imp in zip(clf.feature_names_in_, clf.feature_importances_):
        feature_name = feature_name.split("_")[1]+"_"+feature_name.split("_")[2] if "_" in feature_name else feature_name
        if feature_name in feature_importances:
            feature_importances[feature_name] += imp
        else:
            feature_importances[feature_name] = imp

    #Sort the feature importance
    feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=False))

    for ax, class_label in zip(axs, classes):
        feature_importances_c = {}
        for feature_name, imp in feature_importances.items():
            if feature_name.split("_")[1] == class_label:
                feature_name = feature_name.split("_")[0] if "_" in feature_name else feature_name
                if feature_name in feature_importances_c:
                    feature_importances_c[feature_name] += imp
                else:
                    feature_importances_c[feature_name] = imp

        #Plot the feature importance
        y_values = list(feature_importances_c.keys())
        x_values = list(feature_importances_c.values())
        x_values = [x/sum(x_values) for x in x_values]
        ax.barh(y_values, x_values, color='blue')
        ax.set_xlabel("Feature importance")
        ax.set_ylabel("Concepts")
        ax.set_title(f"Class {class_label}")

        #Add numerical values on the bar plot
        #for index, value in enumerate(x_values):
            #ax.text(value - 0.05, index, str(round(value, 3)), color='white', va='center')

    plt.tight_layout()
    plt.show()


def plot_permutation_importance(dataset_name, size=150):
    #Load the dataset
    x, c, y = get_dataset(dataset_name, size)

    #Load the model
    cem, dcr = load_DCRBase(x, c, y, dataset_name)

    #Create the dataframe
    df, classes = create_df_DCRBase(x, cem, dcr)

    X = df.drop(columns=['class'])
    y = df['class']

    clf = RandomForestClassifier(max_depth=int(len(X.columns)/4), random_state=42)
    clf.fit(X, y)

    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42)

    sorted_importances_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns = X.columns[sorted_importances_idx],
    )

    #Aggregate the permutation importance by concept
    importances = importances.T.groupby(importances.T.index.str.split('_').str[1]).mean().T
    
    #Sort the permutation importance
    importances = importances.sort_values(by=0, axis=1)

    #Plot the permutation importance
    ax = importances.plot.box(vert=False, whis=5)
    ax.set_title(f"Permutation Importances aggregated by concept ({dataset_name})")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.set_ylabel("Concepts")

    plt.tight_layout()
    plt.show()


def plot_permutation_importance_per_class(dataset_name, size=150):
    #Load the dataset
    x, c, y = get_dataset(dataset_name, size)

    #Load the model
    cem, dcr = load_DCRBase(x, c, y, dataset_name)

    #Create the dataframe
    df, classes = create_df_DCRBase(x, cem, dcr)

    n_classes = len(classes)

    #Create a figure with a subplot for each class
    fig, axs = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    fig.suptitle(f"Permutation importance aggregated by concept for each class ({dataset_name})")

    X = df.drop(columns=['class'])
    y = df['class']

    clf = RandomForestClassifier(max_depth=int(len(X.columns)/4), random_state=42)
    clf.fit(X, y)

    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42)

    sorted_importances_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns = X.columns[sorted_importances_idx],
    )

    for ax, class_label in zip(axs, classes):
        #Select only the rows of the class (index.split('_')[-1] == class_label), then aggregate by concept (index.split('_')[1])
        importances_per_class = importances[importances.columns[importances.columns.str.split('_').str[-1] == class_label]]

        #Aggregate the permutation importance by concept
        importances_per_class = importances.T.groupby(importances.T.index.str.split('_').str[1]).mean().T

        #Sort the permutation importance
        importances_per_class = importances_per_class.sort_values(by=0, axis=1)

        #Plot the permutation importance
        ax = importances_per_class.plot.box(vert=False, whis=5, ax=ax)
        ax.set_title(f"Class {class_label}")
        ax.axvline(x=0, color="k", linestyle="--")
        ax.set_xlabel("Decrease in accuracy score")
        ax.set_ylabel("Concepts")
        ax.set_xlim(-0.01, 0.15)

    plt.tight_layout()
    plt.show()