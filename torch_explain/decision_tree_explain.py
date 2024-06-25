from sklearn.tree import DecisionTreeClassifier, _tree
from nn.semantics import Logic, GodelTNorm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sympy.logic.boolalg import to_dnf



def explain_with_DT(x, concept_encoder: torch.nn.Module, task_predictor: torch.nn.Module, mode = 'single', target_class = 0, max_depth = 3, seed = 42, logic: Logic = GodelTNorm()):
    c_emb, c_pred = concept_encoder.forward(x)
    y_pred_dt, sign_attn_dt, filter_attn_dt = task_predictor.forward(c_emb, c_pred, return_attn=True)
    values_dt = c_pred.unsqueeze(-1).repeat(1, 1, len(y_pred_dt[1]))
    sign_terms_dt = logic.iff_pair(sign_attn_dt, values_dt)
    # filtered_values = logic.disj_pair(sign_terms_dt, logic.neg(filter_attn_dt))

    # CREATING THE DATASET
    # for each concept I get sign and filter attn
    final_features = []
    for i in range(c_pred.shape[1]):
        sign_concept = np.vstack([sign.detach().numpy() for sign in sign_terms_dt[:,i,:]])
        filter_concept = np.vstack([f.detach().numpy() for f in filter_attn_dt[:,i,:]])
        final_features.extend([sign_concept, filter_concept])
        # filtered_concept = np.vstack([fc.detach().numpy() for fc in filtered_values[:,i,:]])
        # final_features.extend([filtered_concept])
    
    # Concatenate everything in one single numpy array
    final_features = np.hstack(final_features)

    # Create name of columns
    names = []
    classes = [f"y{i}" for i in range(y_pred_dt.shape[1])]
    for i in range(c_pred.shape[1]):
        for feature_type in ['sign', 'filter']:
            for y in classes:
                names.append(f"{feature_type}_c{i}_{y}")
            # names.append(f"sign_c{i}_{y}")

    # Create the DataFrame
    df = pd.DataFrame(final_features, columns=names)

    
    # Now I Train the decision tree
    dt_clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    if mode == 'single':
        # A single decision tree for all the classes
        y = [f'Class_{np.argmax(pred)}' for pred in y_pred_dt>0.5]
    elif mode == 'distinct':
        # I train a DT only on the features of that specific class ('target_class')
        df = df.filter(like=f"y{target_class}")
        y = [f'Class_{np.argmax(pred)}' for pred in y_pred_dt>0.5] # y = [f'Class_{target_class}' if np.argmax(pred) == target_class else 'Other' for pred in y_pred_dt>0.5]

    dt_clf.fit(df, y)
    fig = plt.figure(figsize=(25,20))
    _ = plot_tree(dt_clf, feature_names=df.columns, class_names=True, filled=True) #
    plt.show()

    
    if mode == 'distinct':
        def get_rules(tree, feature_names):
            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]

            paths = []
            concept_limits = {f'c{i}':[-1,-1] for i in range(c_pred.shape[1])}

            def recurse(node, paths, concept_limits):
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    # p1, p2 = list(path), list(path)
                    # if threshold <= 0.55:
                    # p1 += [f"~{name.split('_')[1]}"] #  [<= {np.round(threshold, 3)}]
                    temp = concept_limits[f'{name.split('_')[1]}'][1]
                    concept_limits[f'{name.split('_')[1]}'][1] = np.round(threshold, 3)
                    recurse(tree_.children_left[node], paths, concept_limits)
                    concept_limits[f'{name.split('_')[1]}'][1] = temp
                    # if threshold >= 0.45:
                    # p2 += [f"{name.split('_')[1]}"] #  [> {np.round(threshold, 3)}]
                    temp = concept_limits[f'{name.split('_')[1]}'][0]
                    concept_limits[f'{name.split('_')[1]}'][0] = np.round(threshold, 3)
                    recurse(tree_.children_right[node], paths, concept_limits)
                    temp = concept_limits[f'{name.split('_')[1]}'][0] = temp
                elif tree_.value[node].argmax() == 0:
                    paths += create_path(concept_limits)

            recurse(0, paths, concept_limits)

            return paths

        rules = get_rules(dt_clf, df.columns)
        rules = "|".join([" & ".join(r) for r in rules])
        print(rules)
        print(to_dnf(rules, simplify=True))


    return dt_clf



def create_path(concept_limits: dict):
    path = []
    for concept, limits in concept_limits.items():
        print(limits)
        if limits[1] == -1:
            if limits[0] == -1:
                continue
            else:
                c = concept
        elif limits[1] <= 0.5:
            c = "~"+concept
        elif limits[0] >= 0.5:
            c = concept
        else:
            c = "~"+concept if (((limits[0] + limits[1])/2) < 0.5) else concept
        path.append(c)
    return [path]