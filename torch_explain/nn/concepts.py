import torch
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from .semantics import Logic, GodelTNorm
import logging
# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()])

logger = logging.getLogger(__name__)


def softselect(values, temperature):
    """
    Compute soft select scores based on input values and temperature.

    Parameters:
        values (torch.Tensor): Input values with shape (batch_size, num_classes).
        temperature (float): Temperature parameter for softmax function.

    Returns:
        torch.Tensor: Soft select scores with shape (batch_size, num_classes).

    This function computes soft select scores by applying the softmax function to the input values,
    subtracting the mean of the softmax scores multiplied by the temperature, and then applying the
    sigmoid function to obtain the final soft select scores.
    """
    softmax_scores = torch.log_softmax(values, dim=1)
    softscores = torch.sigmoid(
        softmax_scores - temperature * softmax_scores.mean(dim=1, keepdim=True))
    return softscores


class ConceptReasoningLayer(torch.nn.Module):
    def __init__(self, emb_size, n_classes, logic: Logic = GodelTNorm(), temperature: float = 100., log=False):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.logic = logic
        self.filter_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
        )
        self.sign_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
        )
        self.temperature = temperature
        self.log = log

    def forward(self, x, c, return_attn=False, sign_attn=None, filter_attn=None):
        values = c.unsqueeze(-1).repeat(1, 1, self.n_classes)
        if self.log:
            logger.info(f"Values: {values.shape}")
            logger.info(f"Values: {values}")

        if sign_attn is None:
            # compute attention scores to build logic sentence
            # each attention score will represent whether the concept should be active or not in the logic sentence
            sign_attn = torch.sigmoid(self.sign_nn(x))
            if self.log:
                logger.info(f"Sign Attn: {sign_attn.shape}")
                logger.info(f"Sign Attn: {sign_attn}")

        # attention scores need to be aligned with predicted concept truth values (attn <-> values)
        # (not A or V) and (A or not V) <-> (A <-> V)
        sign_terms = self.logic.iff_pair(sign_attn, values)

        if self.log:
            logger.info(f"Sign Terms: {sign_terms.shape}")
            logger.info(f"Sign Terms: {sign_terms}")

        if filter_attn is None:
            # compute attention scores to identify only relevant concepts for each class
            filter_attn = softselect(self.filter_nn(x), self.temperature)

        if self.log:
            logger.info(f"Filter Attn: {filter_attn.shape}")
            logger.info(f"Filter Attn: {filter_attn}")

        # filter value
        # filtered implemented as "or(a, not b)", corresponding to "b -> a"
        filtered_values = self.logic.disj_pair(
            sign_terms, self.logic.neg(filter_attn))

        if self.log:
            logger.info(f"Filtered Values: {filtered_values.shape}")
            logger.info(f"Filtered Values: {filtered_values}")

        # generate minterm
        preds = self.logic.conj(filtered_values, dim=1).squeeze(1).float()

        if self.log:
            logger.info(f"Preds: {preds.shape}")
            logger.info(f"Preds: {preds}")

        if return_attn:
            return preds, sign_attn, filter_attn
        else:
            return preds

    def explain(self, x, c, mode, concept_names=None, class_names=None, filter_attn=None):
        assert mode in ['local', 'global', 'exact']

        if concept_names is None:
            concept_names = [f'c_{i}' for i in range(c.shape[1])]
        if class_names is None:
            class_names = [f'y_{i}' for i in range(self.n_classes)]

        # make a forward pass to get predictions and attention weights
        y_preds, sign_attn_mask, filter_attn_mask = self.forward(
            x, c, return_attn=True, filter_attn=filter_attn)

        explanations = []
        all_class_explanations = {cn: [] for cn in class_names}
        for sample_idx in range(len(x)):
            prediction = y_preds[sample_idx] > 0.5
            active_classes = torch.argwhere(prediction).ravel()

            if len(active_classes) == 0:
                # if no class is active for this sample, then we cannot extract any explanation
                explanations.append({
                    'class': -1,
                    'explanation': '',
                    'attention': [],
                })
            else:
                # else we can extract an explanation for each active class!
                for target_class in active_classes:
                    attentions = []
                    minterm = []
                    for concept_idx in range(len(concept_names)):
                        c_pred = c[sample_idx, concept_idx]
                        sign_attn = sign_attn_mask[sample_idx,
                                                   concept_idx, target_class]
                        filter_attn = filter_attn_mask[sample_idx,
                                                       concept_idx, target_class]

                        # we first check if the concept was relevant
                        # a concept is relevant <-> the filter attention score is lower than the concept probability
                        at_score = 0
                        sign_terms = self.logic.iff_pair(
                            sign_attn, c_pred).item()
                        if self.logic.neg(filter_attn) < sign_terms:
                            if sign_attn >= 0.5:
                                # if the concept is relevant and the sign is positive we just take its attention score
                                at_score = filter_attn.item()
                                if mode == 'exact':
                                    minterm.append(
                                        f'{sign_terms:.3f} ({concept_names[concept_idx]})')
                                else:
                                    minterm.append(
                                        f'{concept_names[concept_idx]}')
                            else:
                                # if the concept is relevant and the sign is positive we take (-1) * its attention score
                                at_score = -filter_attn.item()
                                if mode == 'exact':
                                    minterm.append(
                                        f'{sign_terms:.3f} (~{concept_names[concept_idx]})')
                                else:
                                    minterm.append(
                                        f'~{concept_names[concept_idx]}')
                        attentions.append(at_score)

                    # add explanation to list
                    target_class_name = class_names[target_class]
                    minterm = ' & '.join(minterm)
                    all_class_explanations[target_class_name].append(minterm)
                    explanations.append({
                        'sample-id': sample_idx,
                        'class': target_class_name,
                        'explanation': minterm,
                        'attention': attentions,
                    })

        if mode == 'global':
            # count most frequent explanations for each class
            explanations = []
            for class_id, class_explanations in all_class_explanations.items():
                explanation_count = Counter(class_explanations)
                for explanation, count in explanation_count.items():
                    explanations.append({
                        'class': class_id,
                        'explanation': explanation,
                        'count': count,
                    })

        return explanations


class ConceptReasoningLayerMod(torch.nn.Module):
    def __init__(self, emb_size, n_classes, logic: Logic = GodelTNorm(), temperature: float = 100., log=False):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.logic = logic
        self.filter_sign_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, 2 * n_classes),
        )
        self.temperature = temperature
        self.log = log

    def split(self, x):
        return x[:, :, :2], x[:, :, 2:]

    def forward(self, x, c, return_attn=False, sign_attn=None, filter_attn=None):
        values = c.unsqueeze(-1).repeat(1, 1, self.n_classes)
        if self.log:
            logger.info(f"Values: {values.shape}")
            logger.info(f"Values: {values}")

        x_out = self.filter_sign_nn(x)

        if sign_attn is None:
            # compute attention scores to build logic sentence
            # each attention score will represent whether the concept should be active or not in the logic sentence
            weight_fs = torch.sigmoid(x_out)
            sign_attn = self.split(weight_fs)[0]

            if self.log:
                logger.info(f"Sign Attn: {sign_attn.shape}")
                logger.info(f"Sign Attn: {sign_attn}")

        # attention scores need to be aligned with predicted concept truth values (attn <-> values)
        # (not A or V) and (A or not V) <-> (A <-> V)
        sign_terms = self.logic.iff_pair(sign_attn, values)

        if self.log:
            logger.info(f"Sign Terms: {sign_terms.shape}")
            logger.info(f"Sign Terms: {sign_terms}")

        if filter_attn is None:
            # compute attention scores to identify only relevant concepts for each class
            filter_attn = softselect(self.split(x_out)[1], self.temperature)

        if self.log:
            logger.info(f"Filter Attn: {filter_attn.shape}")
            logger.info(f"Filter Attn: {filter_attn}")

        # filter value
        # filtered implemented as "or(a, not b)", corresponding to "b -> a"
        filtered_values = self.logic.disj_pair(
            sign_terms, self.logic.neg(filter_attn))

        if self.log:
            logger.info(f"Filtered Values: {filtered_values.shape}")
            logger.info(f"Filtered Values: {filtered_values}")

        # generate minterm
        preds = self.logic.conj(filtered_values, dim=1).squeeze(1).float()

        if self.log:
            logger.info(f"Preds: {preds.shape}")
            logger.info(f"Preds: {preds}")

        if return_attn:
            return preds, sign_attn, filter_attn
        else:
            return preds

    def explain(self, x, c, mode, concept_names=None, class_names=None, filter_attn=None):
        assert mode in ['local', 'global', 'exact']

        if concept_names is None:
            concept_names = [f'c_{i}' for i in range(c.shape[1])]
        if class_names is None:
            class_names = [f'y_{i}' for i in range(self.n_classes)]

        # make a forward pass to get predictions and attention weights
        y_preds, sign_attn_mask, filter_attn_mask = self.forward(
            x, c, return_attn=True, filter_attn=filter_attn)

        explanations = []
        all_class_explanations = {cn: [] for cn in class_names}
        for sample_idx in range(len(x)):
            prediction = y_preds[sample_idx] > 0.5
            active_classes = torch.argwhere(prediction).ravel()

            if len(active_classes) == 0:
                # if no class is active for this sample, then we cannot extract any explanation
                explanations.append({
                    'class': -1,
                    'explanation': '',
                    'attention': [],
                })
            else:
                # else we can extract an explanation for each active class!
                for target_class in active_classes:
                    attentions = []
                    minterm = []
                    for concept_idx in range(len(concept_names)):
                        c_pred = c[sample_idx, concept_idx]
                        sign_attn = sign_attn_mask[sample_idx,
                                                   concept_idx, target_class]
                        filter_attn = filter_attn_mask[sample_idx,
                                                       concept_idx, target_class]

                        # we first check if the concept was relevant
                        # a concept is relevant <-> the filter attention score is lower than the concept probability
                        at_score = 0
                        sign_terms = self.logic.iff_pair(
                            sign_attn, c_pred).item()
                        if self.logic.neg(filter_attn) < sign_terms:
                            if sign_attn >= 0.5:
                                # if the concept is relevant and the sign is positive we just take its attention score
                                at_score = filter_attn.item()
                                if mode == 'exact':
                                    minterm.append(
                                        f'{sign_terms:.3f} ({concept_names[concept_idx]})')
                                else:
                                    minterm.append(
                                        f'{concept_names[concept_idx]}')
                            else:
                                # if the concept is relevant and the sign is positive we take (-1) * its attention score
                                at_score = -filter_attn.item()
                                if mode == 'exact':
                                    minterm.append(
                                        f'{sign_terms:.3f} (~{concept_names[concept_idx]})')
                                else:
                                    minterm.append(
                                        f'~{concept_names[concept_idx]}')
                        attentions.append(at_score)

                    # add explanation to list
                    target_class_name = class_names[target_class]
                    minterm = ' & '.join(minterm)
                    all_class_explanations[target_class_name].append(minterm)
                    explanations.append({
                        'sample-id': sample_idx,
                        'class': target_class_name,
                        'explanation': minterm,
                        'attention': attentions,
                    })

        if mode == 'global':
            # count most frequent explanations for each class
            explanations = []
            for class_id, class_explanations in all_class_explanations.items():
                explanation_count = Counter(class_explanations)
                for explanation, count in explanation_count.items():
                    explanations.append({
                        'class': class_id,
                        'explanation': explanation,
                        'count': count,
                    })

        return explanations


class SignRelevanceAttention(nn.Module):
    def __init__(self, num_concepts, num_classes):
        super(SignRelevanceAttention, self).__init__()
        # Learnable parameters for attention scores for both sign and relevance
        self.sign_weights = nn.Parameter(
            torch.randn(num_concepts, num_classes))
        self.relevance_weights = nn.Parameter(
            torch.randn(num_concepts, num_classes))

    def forward(self, sign_tensor, relevance_tensor):
        # Apply softmax to weights
        sign_attention = F.softmax(self.sign_weights, dim=0)
        relevance_attention = F.softmax(self.relevance_weights, dim=0)

        # Apply attention to each tensor
        sign_attended = sign_tensor * \
            sign_attention.unsqueeze(0)  # Adding batch dimension
        relevance_attended = relevance_tensor * \
            relevance_attention.unsqueeze(0)

        # Element-wise multiplication as one way to combine them
        combined = sign_attended * relevance_attended

        return combined


class SignRelevanceNet(nn.Module):
    def __init__(self, num_concepts, num_classes):
        super(SignRelevanceNet, self).__init__()
        # Feature extraction layers
        self.sign_layer = nn.Linear(num_classes, num_classes)
        self.relevance_layer = nn.Linear(num_classes, num_classes)

        # Combination layer: could also be more complex (e.g., a learned mixture)
        self.combine = nn.Linear(2 * num_classes, num_classes)

        # Integration layer
        self.final_layer = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_concepts)  # Batch normalization across concepts
        )

    def forward(self, sign_tensor, relevance_tensor):
        # Process each tensor separately
        sign_features = F.leaky_relu(self.sign_layer(sign_tensor))
        relevance_features = F.leaky_relu(
            self.relevance_layer(relevance_tensor))

        # Combine features
        combined_features = torch.cat(
            (sign_features, relevance_features), dim=2)
        combined_features = F.relu(self.combine(combined_features))

        # Final integration and output
        output = self.final_layer(combined_features)
        return output


class WeightedMerger(nn.Module):
    def __init__(self, num_concepts):
        super(WeightedMerger, self).__init__()
        # Initialize a learnable parameter alpha for each concept
        # Shape: [num_concepts, 1] to broadcast correctly
        self.alpha = nn.Parameter(torch.rand(num_concepts, 1))

    def forward(self, tensor1, tensor2):
        # Apply the concept-wise weighting
        weighted_tensor1 = self.alpha * tensor1
        weighted_tensor2 = (1 - self.alpha) * tensor2
        return weighted_tensor1 + weighted_tensor2


class ReasoningLinearLayer(torch.nn.Module):
    def __init__(self, sign_shape, filter_shape, n_classes, modality="PhDinZurich", bias_comp="post", log=False):
        super().__init__()

        # Sign Shape == Filter Shape == sign_attn.shape[1]

        self.mode = modality
        # Modality are : 1. Attention 2. Weighted 3. nn
        if self.mode == "Attention":
            self.mapper_nn = SignRelevanceAttention(
                sign_shape, n_classes)
        elif self.mode == "Weighted":
            self.mapper_nn = WeightedMerger(sign_shape)
        else:
            self.mapper_nn = SignRelevanceNet(sign_shape, n_classes)

        self.log = log
        self.bias_computation = bias_comp

    def forward(self, sign_attn, filter_attn, c, return_params=False):
        if self.log:
            logger.info(f"sign attention: {sign_attn.shape}")
            logger.info(f"filter attention: {filter_attn.shape}")
            logger.info(f"C: {c.shape}")

        transformed_coeffs = self.mapper_nn(sign_attn, filter_attn)
        if self.log:
            logger.info(f"Transformed: {transformed_coeffs.shape}")
            logger.info(f"Transformed: {transformed_coeffs}")

        # y = \sum{c_i * alpha_i}
        logits = (c.unsqueeze(-1) * transformed_coeffs).sum(dim=1).float()

        if self.log:
            logger.info(f"Logits: {logits.shape}")
            logger.info(f"Logits: {logits}")

        if self.bias_computation == "post":
            bias_vals = self.bias_nn(transformed_coeffs)
            bias_vals = bias_vals.mean(dim=1)
            if self.log:
                logger.info(f"Assigned Biases: {bias_vals.shape}")
                logger.info(f"Assigned Biases: {bias_vals}")
        else:
            bias_vals = self.bias_nn(transformed_coeffs.mean(dim=1))
            if self.log:
                logger.info(f"Assigned Biases: {bias_vals.shape}")
                logger.info(f"Assigned Biases: {bias_vals}")

        logits += bias_vals

        preds = logits
        if return_params:
            return preds, transformed_coeffs, bias_vals
        return preds


class IntpLinearLayer(torch.nn.Module):
    """
    This layer mimics the linear layer over concepts and for each it outputs a weight through a learnable neural network.
    This Layer enhances global explainabality through the use of weights and studying them (posthoc)
    """

    def __init__(self, emb_size, n_classes, bias_computation="post", log=False):
        """
        Initializes an instance of the IntpLinearLayer class.

        Args:
            emb_size (int): The size of the embedding.
            n_classes (int): The number of classes.
            bias_computation (str, optional): The method to compute the bias. Defaults to "post".
            AGGREGATION is POST/PRE
                * "post": Compute the bias by aggregating the output of the NN.
                * "pre": Compute the bias by aggregating the concept values before feeding to the NN.
            log (bool, optional): Whether to log the layer. Defaults to False.

        Returns:
            None
        """
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.bias_computation = bias_computation

        # This NN is responsible to learn the weight of each concept
        self.coeff_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes)
        )

        # This NN is responsible to learn the bias of each concept (here, we have to handle later that each concept has a bias)
        self.bias_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes)
        )

        self.log = log

    def forward(self, x, c, return_params=False, coeff_comp=None):
        if self.log:
            logger.info(f"X: {x.shape}")
            logger.info(f"C: {c.shape}")

        if coeff_comp is None:
            coeff_vals = self.coeff_nn(x)
            if self.log:
                logger.info(f"Assigned Coeffs: {coeff_vals.shape}")
                logger.info(f"Assigned Coeffs: {coeff_vals}")

        # y = \sum{c_i * alpha_i}
        logits = (c.unsqueeze(-1) * coeff_vals).sum(dim=1).float()

        if self.log:
            logger.info(f"Logits: {logits.shape}")
            logger.info(f"Logits: {logits}")

        if self.bias_computation == "post":
            bias_vals = self.bias_nn(x)
            bias_vals = bias_vals.mean(dim=1)
            if self.log:
                logger.info(f"Assigned Biases: {bias_vals.shape}")
                logger.info(f"Assigned Biases: {bias_vals}")
        else:
            bias_vals = self.bias_nn(x.mean(dim=1))

        logits += bias_vals

        preds = logits
        if return_params:
            return preds, coeff_vals, bias_vals
        return preds


# class AttentiveLinearLayer(torch.nn.Module):
#     """
#     This layer mimics the linear layer over concepts and for each it outputs a weight through a learnable neural network.
#     This Layer enhances global explainabality through the use of weights and studying them (posthoc)
#     """

#     def __init__(self, emb_size, n_classes, bias_computation="post", log=False):
#         """
#         Initializes an instance of the IntpLinearLayer class.

#         Args:
#             emb_size (int): The size of the embedding.
#             n_classes (int): The number of classes.
#             bias_computation (str, optional): The method to compute the bias. Defaults to "post".
#             AGGREGATION is POST/PRE
#                 * "post": Compute the bias by aggregating the output of the NN.
#                 * "pre": Compute the bias by aggregating the concept values before feeding to the NN.
#             log (bool, optional): Whether to log the layer. Defaults to False.

#         Returns:
#             None
#         """
#         super().__init__()
#         self.emb_size = emb_size
#         self.n_classes = n_classes
#         self.bias_computation = bias_computation

#         # This NN is responsible to learn the weight of each concept
#         self.coeff_nn = torch.nn.Sequential(
#             torch.nn.Linear(emb_size, emb_size),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(emb_size, n_classes)
#         )

#         # This NN is responsible to learn the bias of each concept (here, we have to handle later that each concept has a bias)
#         self.bias_nn = torch.nn.Sequential(
#             torch.nn.Linear(emb_size, emb_size),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(emb_size, n_classes)
#         )

#         self.log = log

#     def forward(self, x, c, return_params=False, coeff_comp=None):
#         if self.log:
#             logger.info(f"X: {x.shape}")
#             logger.info(f"C: {c.shape}")

#         if coeff_comp is None:
#             coeff_vals = self.coeff_nn(x)
#             if self.log:
#                 logger.info(f"Assigned Coeffs: {coeff_vals.shape}")
#                 logger.info(f"Assigned Coeffs: {coeff_vals}")

#         # y = \sum{c_i * alpha_i}
#         logits = (c.unsqueeze(-1) * coeff_vals).sum(dim=1).float()

#         if self.log:
#             logger.info(f"Logits: {logits.shape}")
#             logger.info(f"Logits: {logits}")

#         if self.bias_computation == "post":
#             bias_vals = self.bias_nn(x)
#             bias_vals = bias_vals.mean(dim=1)
#             if self.log:
#                 logger.info(f"Assigned Biases: {bias_vals.shape}")
#                 logger.info(f"Assigned Biases: {bias_vals}")
#         else:
#             bias_vals = self.bias_nn(x.mean(dim=1))

#         logits += bias_vals

#         preds = logits
#         if return_params:
#             return preds, coeff_vals, bias_vals
#         return preds


class ConceptEmbedding(torch.nn.Module):
    def __init__(
            self,
            in_features,
            n_concepts,
            emb_size,
            active_intervention_values=None,
            inactive_intervention_values=None,
            intervention_idxs=None,
            training_intervention_prob=0.25,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.intervention_idxs = intervention_idxs
        self.training_intervention_prob = training_intervention_prob
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        self.concept_context_generators = torch.nn.ModuleList()
        for i in range(n_concepts):
            self.concept_context_generators.append(torch.nn.Sequential(
                torch.nn.Linear(in_features, 2 * emb_size),
                torch.nn.LeakyReLU(),
            ))
        self.concept_prob_predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, 1),
            torch.nn.Sigmoid(),
        )

        # And default values for interventions here
        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.zeros(n_concepts)

    def _after_interventions(
            self,
            prob,
            concept_idx,
            intervention_idxs=None,
            c_true=None,
            train=False,
    ):
        if train and (self.training_intervention_prob != 0) and (intervention_idxs is None):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(self.ones * self.training_intervention_prob)
            intervention_idxs = torch.nonzero(mask).reshape(-1)
        if (c_true is None) or (intervention_idxs is None):
            return prob
        if concept_idx not in intervention_idxs:
            return prob
        return (c_true[:, concept_idx:concept_idx + 1] * self.active_intervention_values[concept_idx]) + \
            ((c_true[:, concept_idx:concept_idx + 1] - 1) * -
             self.inactive_intervention_values[concept_idx])

    def forward(self, x, intervention_idxs=None, c=None, train=False):
        c_emb_list, c_pred_list = [], []
        # We give precendence to inference time interventions arguments
        used_int_idxs = intervention_idxs
        if used_int_idxs is None:
            used_int_idxs = self.intervention_idxs
        for i, context_gen in enumerate(self.concept_context_generators):
            context = context_gen(x)
            c_pred = self.concept_prob_predictor(context)
            c_pred_list.append(c_pred)
            # Time to check for interventions
            c_pred = self._after_interventions(
                prob=c_pred,
                concept_idx=i,
                intervention_idxs=used_int_idxs,
                c_true=c,
                train=train,
            )

            context_pos = context[:, :self.emb_size]
            context_neg = context[:, self.emb_size:]
            c_emb = context_pos * c_pred + context_neg * (1 - c_pred)
            c_emb_list.append(c_emb.unsqueeze(1))

        return torch.cat(c_emb_list, axis=1), torch.cat(c_pred_list, axis=1)
