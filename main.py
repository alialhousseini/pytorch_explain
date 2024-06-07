import torch.nn.functional as F
from torch_explain.nn.concepts import ConceptReasoningLayer
import torch
import torch_explain as te
from torch_explain import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x, c, y = datasets.xor(500)
x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(
    x, c, y, test_size=0.33, random_state=42)

x_train = x_train[:10]
x_test = x_test[:10]
c_train = c_train[:10]
c_test = c_test[:10]
y_train = y_train[:10]
y_test = y_test[:10]

embedding_size = 8
concept_encoder = torch.nn.Sequential(
    torch.nn.Linear(x.shape[1], 10),
    torch.nn.LeakyReLU(),
    te.nn.ConceptEmbedding(10, c.shape[1], embedding_size),
)


c_emb, c_pred = concept_encoder.forward(x_test)


# -------------------------------------#

y_train = F.one_hot(y_train.long().ravel()).float()
y_test = F.one_hot(y_test.long().ravel()).float()


task_predictor = ConceptReasoningLayer(
    embedding_size, y_train.shape[1], log=True)
model = torch.nn.Sequential(concept_encoder, task_predictor)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
loss_form = torch.nn.BCELoss()
model.train()
for epoch in range(1):
    optimizer.zero_grad()

    # generate concept and task predictions
    c_emb, c_pred = concept_encoder(x_train)
    y_pred = task_predictor(c_emb, c_pred)

    # compute loss
    concept_loss = loss_form(c_pred, c_train)
    task_loss = loss_form(y_pred, y_train)
    loss = concept_loss + 0.5*task_loss

    loss.backward()
    optimizer.step()

local_explanations = task_predictor.explain(c_emb, c_pred, 'local')
global_explanations = task_predictor.explain(c_emb, c_pred, 'global')

# print(local_explanations)
print(global_explanations)


# task_predictor = ConceptReasoningLayer(embedding_size, y_train.shape[1])
# model = torch.nn.Sequential(concept_encoder, task_predictor)

# optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
# loss_form = torch.nn.BCELoss()
# model.train()
# for epoch in range(501):
#     optimizer.zero_grad()

#     # generate concept and task predictions
#     c_emb, c_pred = concept_encoder(x_train)
#     y_pred = task_predictor(c_emb, c_pred)

#     # compute loss
#     concept_loss = loss_form(c_pred, c_train)
#     task_loss = loss_form(y_pred, y_train)
#     loss = concept_loss + 0.5*task_loss

#     loss.backward()
#     optimizer.step()

# local_explanations = task_predictor.explain(c_emb, c_pred, 'local')
# global_explanations = task_predictor.explain(c_emb, c_pred, 'global')

# # print(local_explanations)
# print(global_explanations)
