from sklearn.tree import DecisionTreeClassifier
from nn.semantics import Logic, GodelTNorm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree



def explain_with_DT(x, concept_encoder: torch.nn.Module, task_predictor: torch.nn.Module, mode = 'single', target_class = 0, logic: Logic = GodelTNorm()):
    c_emb, c_pred = concept_encoder.forward(x)
    y_pred_dt, sign_attn_dt, filter_attn_dt = task_predictor.forward(c_emb, c_pred, return_attn=True)
    values_dt = c_pred.unsqueeze(-1).repeat(1, 1, len(y_pred_dt[1]))
    sign_terms_dt = logic.iff_pair(sign_attn_dt, values_dt)

    # CREATING THE DATASET
    # for each concept I get sign and filter attn
    final_features = []
    for i in range(c_pred.shape[1]):
        sign_concept = np.vstack([sign.detach().numpy() for sign in sign_terms_dt[:,i,:]])
        filter_concept = np.vstack([f.detach().numpy() for f in filter_attn_dt[:,i,:]])
        final_features.extend([sign_concept, filter_concept])
    
    # Concatenate everything in one single numpy array
    final_features = np.hstack(final_features)

    # Create name of columns
    names = []
    classes = [f"y{i}" for i in range(y_pred_dt.shape[1])]
    for i in range(c_pred.shape[1]):
        for feature_type in ['sign', 'filter']:
            for y in classes:
                names.append(f"{feature_type}_c{i}_{y}")

    # Create the DataFrame
    df = pd.DataFrame(final_features, columns=names)

    
    # Now I Train the decision tree
    dt_clf = DecisionTreeClassifier()
    if mode == 'single':
        # A single decision tree for all the classes
        y = [f'Class_{np.argmax(pred)}' for pred in y_pred_dt>0.5]
        dt_clf.fit(df, y)
    elif mode == 'distinct':
        # I train a DT only on the features of that specific class ('target_class')
        df = df.filter(like=f"y{target_class}")
        y = [f'Class_{target_class}' if np.argmax(pred) == target_class else 'Other' for pred in y_pred_dt>0.5]
        dt_clf.fit(df, y)

    fig = plt.figure(figsize=(25,20))
    _ = plot_tree(dt_clf, feature_names=df.columns, class_names=list(set(y)), filled=True)
    plt.show()
    
    return dt_clf