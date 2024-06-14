from typing import List
import torch
from sympy import lambdify, sympify
import copy
import logging
import typing

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()])

logger = logging.getLogger(__name__)


def replace_names(explanation: str, concept_names: List[str]) -> str:
    """
    Replace names of concepts in a formula.
    :param explanation: formula
    :param concept_names: new concept names
    :return: Formula with renamed concepts
    """
    feature_abbreviations = [
        f'feature{i:010}' for i in range(len(concept_names))]
    mapping = []
    for f_abbr, f_name in zip(feature_abbreviations, concept_names):
        mapping.append((f_abbr, f_name))

    for k, v in mapping:
        explanation = explanation.replace(k, v)

    return explanation



############################################################################################################
# Questa funzione testa una formula logica. Prende una formula logica come stringa, un tensore di input x, 
# una soglia per binarizzare le caratteristiche e un flag booleano per il logging. Converte la formula in 
# un’espressione simbolica, poi in una funzione NumPy che può valutare la formula logica sui dati di input. 
# Infine, ritorna le previsioni ottenute valutando la formula sui dati di input.
############################################################################################################
def get_predictions(formula: str, x: torch.Tensor, threshold: float = 0.5, log: bool = False) -> typing.Optional[torch.Tensor]:
    """
    Tests a logic formula.
    :param formula: logic formula
    :param x: input data
    :param target_class: target class
    :return: Accuracy of the explanation and predictions
    """

    if formula in ['True', 'False', ''] or formula is None:
        return None

    else:

        concept_list = [f"feature{i:010}" for i in range(x.shape[1])]
        # get predictions using sympy
        # explanation = to_dnf(formula)
        if log:
            logger.info(f"Formula: {formula}")
            logger.info(f"Concept list: {concept_list}")

        #  convert the formula string into a symbolic expression.
        explanation = sympify(formula)
        if log:
            logger.info(f"Explanation: {explanation}")

        # convert the symbolic expression into a NumPy function that can evaluate the logical formula on the input data
        fun = lambdify(concept_list, explanation, 'numpy')
        if log:
            logger.info(f"Function: {fun}")

        x = x.cpu().detach().numpy()

        # Binarize features and evaluate the logical formula
        predictions = fun(*[x[:, i] > threshold for i in range(x.shape[1])])

        if log:
            logger.info(f"Predictions: {predictions}")

        return predictions



############################################################################################################
# Questa funzione prende un modello, un tensore di input c, un indice di bordo edge_index, una posizione 
# di campione sample_pos, una spiegazione come stringa, una classe target e una lista opzionale di nomi 
# di concetti. Perturba le caratteristiche di input rimuovendo o aggiungendo termini dalla spiegazione e 
# valuta l’effetto sulla previsione del modello. Ritorna due liste di termini: i “buoni” termini che, quando 
# rimossi, peggiorano la previsione, e i “cattivi” termini che, quando rimossi, migliorano la previsione.
############################################################################################################
def get_the_good_and_bad_terms(
    model, c, edge_index, sample_pos, explanation, target_class, concept_names=None, threshold=0.5
):
    def perturb_inputs_rem(inputs, target):
        if threshold == 0.5:
            inputs[:, target] = 0.0
        elif threshold == 0.:
            inputs[:, target] = -1.0
        return inputs

    def perturb_inputs_add(inputs, target):
        # inputs[:, target] += inputs.sum(axis=1) / (inputs != 0).sum(axis=1)
        # inputs[:, target] += inputs.max(axis=1)[0]
        inputs[:, target] = 1
        # inputs[:, target] += 1
        return inputs

    explanation = explanation.split(" & ")

    good, bad = [], []

    if edge_index is None:
        base = model(c)[sample_pos].view(1, -1)
    else:
        base = model(c, edge_index)[sample_pos].view(1, -1)

    for term in explanation:
        atom = term
        remove = True
        if atom[0] == "~":
            remove = False
            atom = atom[1:]

        if concept_names is not None:
            idx = concept_names.index(atom)
        else:
            idx = int(atom[len("feature"):])
        temp_tensor = c[sample_pos].clone().detach().view(1, -1)
        temp_tensor = (
            perturb_inputs_rem(temp_tensor, idx)
            if remove
            else perturb_inputs_add(temp_tensor, idx)
        )
        c2 = copy.deepcopy(c)
        c2[sample_pos] = temp_tensor
        if edge_index is None:
            new_pred = model(c2)[sample_pos].view(1, -1)
        else:
            new_pred = model(c2, edge_index)[sample_pos].view(1, -1)

        if new_pred[:, target_class] >= base[:, target_class]:
            bad.append(term)
        else:
            good.append(term)
        del temp_tensor
    return good, bad


if __name__ == "__main__":
    # Test the functions in this module

    # 1 and 1 returns 1
    # get_predictions("x0 & x1", torch.tensor(
    #     [[0.1, 0.9], [0.4, 0.6]]), log=True)

    # 0 or 0 returns 0
    # get_predictions("y0 | y1", torch.tensor(
    #     [[0.8, 0.2], [0.7, 0.3]]), log=True)

    get_predictions("(x0 & x1)", torch.tensor(
        [[0.1, 0.9], [0.6, 0.4]]), log=True)

    get_predictions("(x0 & ~x1)", torch.tensor(
        [[0.1, 0.9], [0.6, 0.4]]), log=True)
