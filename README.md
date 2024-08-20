## Explainable AI Through Linear Concept Layer: Advancing Deep Concept Reasoner

This repository contains the code and resources for our Concept-based XAI project. This work addresses challenges in the interpretability and trustworthiness of complex AI systems by exploring concept-based models in Explainable AI (XAI). It focus on enhancing Concept-Based Explainable AI (C-XAI), a subfield that seeks to bridge the gap between human understanding and machine learning models through the use of high-level human-like concepts.

### Summary of the paper
The paper introduces a novel variant of the Concept Embedding Model (CEM) that incorporates a linear concept layer to improve the balance between accuracy, interpretability, and computational efficiency. This work builds on the foundation of prior models such as Concept Bottleneck Models (CBMs), which utilize human-specified concepts as intermediate steps in prediction but suffer from limitations like dependency on annotated data and a trade-off between accuracy and interpretability. CEMs improve upon CBMs by embedding concepts into high-dimensional spaces, enhancing accuracy but complicating interpretability.

To address these challenges, we proposed a model that integrates linear equations to simplify the reasoning process while maintaining a high level of interpretability. This approach provides a new method for explainable AI, leveraging linear weights associated with concepts rather than relying on complex logic rules, as seen in the Deep Concept Reasoner (DCR). This alternative model aims to balance the rich representation of concepts in CEMs with clearer and more interpretable predictions.

The paper presents experimental evaluations using datasets like XOR, XNOR, DOT, and MNIST-Addition to demonstrate the efficacy of the proposed method. These datasets test the model's ability to generalize and handle complex non-linear tasks.


_____
Quick start
---------------

You can install ``torch_explain`` along with all its dependencies from
`PyPI <https://pypi.org/project/torch_explain/>`__:

```python
pip install torch-explain
```

Quick tutorial on Deep Concept Reasoning
-----------------------------------------------

Using deep concept reasoning we can solve the same problem as above,
but with an intrinsically interpretable model! In fact, Deep Concept Reasoners (Deep CoRes)
make task predictions by means of interpretable logic rules using concept embeddings.

Using the same example as before, we can just change the task predictor
using a Deep CoRe layer:

```python

    from torch_explain.nn.concepts import ConceptReasoningLayer
    import torch.nn.functional as F

    y_train = F.one_hot(y_train.long().ravel()).float()
    y_test = F.one_hot(y_test.long().ravel()).float()

    task_predictor = ConceptReasoningLayer(embedding_size, y_train.shape[1])
    model = torch.nn.Sequential(concept_encoder, task_predictor)
```

We can now train the network by optimizing the cross entropy loss
on concepts and tasks:

```python

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form = torch.nn.BCELoss()
    model.train()
    for epoch in range(501):
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
```

Once trained the Deep CoRe layer can explain its predictions by
providing both local and global logic rules:

```python
    local_explanations = task_predictor.explain(c_emb, c_pred, 'local')
    global_explanations = task_predictor.explain(c_emb, c_pred, 'global')
```

For global explanations, the reasoner will return a dictionary with entries such as
``{'class': 'y_0', 'explanation': '~c_0 & ~c_1', 'count': 94}``, specifying
for each logic rule, the task it is associated with and the number of samples
associated with the explanation.


Quick tutorial on Concept Linear Linear (named 'LLR v1 and v2')
---------------------------------------------
```python
    embedding_size = 16
    concept_encoder = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], 32),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.LeakyReLU(),
        te.nn.ConceptEmbedding(32, c_train.shape[1], embedding_size),
    )

    task_predictor = IntpLinearLayer3(embedding_size, y_train.shape[1], bias=isBias)
    model = torch.nn.Sequential(concept_encoder, task_predictor)

    c_loss = get_loss_function(wandb.config.loss_function)
    y_loss = get_loss_function(wandb.config.loss_function3)

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.0001, patience=7)
    for epoch in range(101):
        epoch_start_time = time.time()
        model.train()
        train_losses, train_correct = 0, 0
        all_y_true_train, all_y_pred_train = [], []
        all_c_true_train, all_c_pred_train = [], []

        for x_batch, c_batch, y_batch in train_loader:
            optimizer.zero_grad()
            c_emb, c_pred = concept_encoder(x_batch)
            y_pred = task_predictor(c_emb, c_pred)

            concept_loss = c_loss(c_pred, c_batch)
            task_loss = y_loss(y_pred, y_batch)
            loss = concept_loss + 0.5 * task_loss

            loss.backward()
            optimizer.step()

            train_losses += loss.item()
            train_correct += (y_pred.argmax(1) == y_batch.argmax(1)).sum().item()
            all_y_true_train.append(y_batch.cpu().numpy())
            all_y_pred_train.append(y_pred.detach().cpu().numpy())
            all_c_true_train.append(c_batch.cpu().numpy())
            all_c_pred_train.append(c_pred.detach().cpu().numpy())

            wandb.log({
                'train_concept_loss': concept_loss.item(),
                'train_task_loss': task_loss.item(),
                'learning_rate': optimizer.param_groups[0]['lr']
            })


```

Benchmark datasets
-------------------------

We provide a suite of several benchmark datasets to evaluate the performance of our models
in the folder `torch_explain/datasets`. The paper "Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off" proposed these datasets as benchmarks for concept-based models.

Real-world datasets can be downloaded from the links provided in the supplementary material of the paper.


