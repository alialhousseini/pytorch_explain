print("===========================================================")
print("Training Started!")

# Iterate over models
for model_name in models:
    # Each IF-Check refers to a models
    # if model_name == 'DCRBase':
    #     print(f"Training on {model_name} ... ")
    #     print(f"--------------------------------")

    #     # Iterate through datasets on each model
    #     for dataset_name, dataset in zip(dataset_names, datasets):
    #         # Importing dataset
    #         x, c, y = dataset
    #         x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(
    #             x, c, y, test_size=0.2, random_state=42)
    #         print(
    #             f"The following dataset has been loaded successully: {dataset_name}")

    #         # Encode target into one-hode y=[y_0, y_1]
    #         y_train = F.one_hot(y_train.long().ravel()).float()
    #         y_test = F.one_hot(y_test.long().ravel()).float()

    #         # Define the concept_encoder
    #         embedding_size = 8
    #         concept_encoder = torch.nn.Sequential(
    #             torch.nn.Linear(x.shape[1], 10),
    #             torch.nn.LeakyReLU(),
    #             te.nn.ConceptEmbedding(10, c.shape[1], embedding_size),
    #         )

    #         # Define the DCR as task predictor
    #         task_predictor = ConceptReasoningLayer(
    #             embedding_size, y_train.shape[1])

    #         # Create a sequential model (cascaded)
    #         model = torch.nn.Sequential(concept_encoder, task_predictor)

    #         # Split dataset into training and validation
    #         num_val_samples = int(len(x_train) * 0.2)
    #         num_train_samples = len(x_train) - num_val_samples
    #         train_dataset, val_dataset = random_split(list(zip(x_train, c_train, y_train)), [
    #                                                   num_train_samples, num_val_samples])

    #         train_loader = DataLoader(
    #             train_dataset, batch_size=64, shuffle=True)
    #         val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    #         wandb.init(project="pytorch_explain", entity="alih9862",
    #                    name=f"{model_name}_{dataset_name}")

    #         # Define a hyperparameter config
    #         config = {
    #             'lr': 0.0005,
    #             # Initial weight for task loss (to be used later)
    #             'task_loss_weight': 0.5,
    #             'loss_function': 'bce',  # Initial loss function
    #         }
    #         wandb.config.update(config)

    #         loss_form = get_loss_function(wandb.config.loss_function)

    #         # Model and optimizer setup
    #         optimizer = torch.optim.AdamW(
    #             model.parameters(), lr=wandb.config.lr)
    #         scheduler = ReduceLROnPlateau(
    #             optimizer, mode='min', factor=0.001, patience=5)

    #         print(
    #             f'-------------------------- Training {dataset_name} using {model_name} ----------------------')
    #         # Training loop with validation and logging
    #         for epoch in range(51):
    #             model.train()
    #             train_losses, train_correct = 0, 0
    #             for x_batch, c_batch, y_batch in train_loader:
    #                 optimizer.zero_grad()

    #                 c_emb, c_pred = concept_encoder(x_batch)
    #                 y_pred = task_predictor(c_emb, c_pred)

    #                 concept_loss = loss_form(c_pred, c_batch)
    #                 task_loss = loss_form(y_pred, y_batch)

    #                 loss = concept_loss + 0.5 * task_loss

    #                 loss.backward()
    #                 optimizer.step()

    #                 train_losses += loss.item()
    #                 train_correct += (y_pred.argmax(1) ==
    #                                   y_batch.argmax(1)).sum().item()

    #                 wandb.log({
    #                     'train_concept_loss': concept_loss.item(),
    #                     'train_task_loss': task_loss.item(),
    #                     'learning_rate': optimizer.param_groups[0]['lr']
    #                 })

    #             # Validation step
    #             model.eval()
    #             val_losses, val_correct = 0, 0
    #             with torch.no_grad():
    #                 for x_batch, c_batch, y_batch in val_loader:
    #                     c_emb, c_pred = concept_encoder(x_batch)
    #                     y_pred = task_predictor(c_emb, c_pred)

    #                     val_concept_loss = loss_form(c_pred, c_batch)
    #                     val_task_loss = loss_form(y_pred, y_batch)

    #                     val_loss = val_concept_loss + 0.5 * \
    #                         val_task_loss

    #                     val_losses += val_loss.item()
    #                     val_correct += (y_pred.argmax(1) ==
    #                                     y_batch.argmax(1)).sum().item()

    #                     # Log validation losses and learning rate
    #                     wandb.log({
    #                         'val_concept_loss': val_concept_loss.item(),
    #                         'val_task_loss': val_task_loss.item(),
    #                         'val_learning_rate': optimizer.param_groups[0]['lr']
    #                     })

    #             scheduler.step(val_loss)

    #             # Log metrics every epoch!
    #             print(f"Epoch {epoch+1}, Loss: {train_losses/len(train_loader)}, Train Accuracy: {train_correct/len(train_dataset)}, Val Loss: {val_losses/len(val_loader)}, Val Accuracy: {val_correct/len(val_dataset)}")
    #             wandb.log({
    #                 'epoch': epoch + 1,
    #                 'loss': train_losses / len(train_loader),
    #                 'train_accuracy': train_correct / len(train_dataset),
    #                 'val_loss': val_losses / len(val_loader),
    #                 'val_accuracy': val_correct / len(val_dataset)
    #             })

    #         print(
    #             f"\n Training on {dataset_name} using {model_name} has been completed!")

    #         # Save the model
    #         torch.save(model, f'model_{model_name}_{dataset_name}.pth')
    #         torch.save(model.state_dict(),
    #                    f'model_state_dict_{model_name}_{dataset_name}.pth')

    #         # Finish run
    #         wandb.finish()

    #     print(f"===========================================================")

    # if model_name == 'DCRMod':
    #     print(f"Training on {model_name} ... ")
    #     print(f"--------------------------------")

    #     # Iterate through datasets on each model
    #     for dataset_name, dataset in zip(dataset_names, datasets):
    #         # Importing dataset
    #         x, c, y = dataset
    #         x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(
    #             x, c, y, test_size=0.2, random_state=42)
    #         print(
    #             f"The following dataset has been loaded successully: {dataset_name}")

    #         # Encode target into one-hode y=[y_0, y_1]
    #         y_train = F.one_hot(y_train.long().ravel()).float()
    #         y_test = F.one_hot(y_test.long().ravel()).float()

    #         # Define the concept_encoder
    #         embedding_size = 8
    #         concept_encoder = torch.nn.Sequential(
    #             torch.nn.Linear(x.shape[1], 10),
    #             torch.nn.LeakyReLU(),
    #             te.nn.ConceptEmbedding(10, c.shape[1], embedding_size),
    #         )

    #         # Define the DCR as task predictor
    #         task_predictor = ConceptReasoningLayerMod(
    #             embedding_size, y_train.shape[1])

    #         # Create a sequential model (cascaded)
    #         model = torch.nn.Sequential(concept_encoder, task_predictor)

    #         # Split dataset into training and validation
    #         num_val_samples = int(len(x_train) * 0.2)
    #         num_train_samples = len(x_train) - num_val_samples
    #         train_dataset, val_dataset = random_split(list(zip(x_train, c_train, y_train)), [
    #                                                   num_train_samples, num_val_samples])

    #         train_loader = DataLoader(
    #             train_dataset, batch_size=64, shuffle=True)
    #         val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    #         wandb.init(project="pytorch_explain", entity="alih9862",
    #                    name=f"{model_name}_{dataset_name}")

    #         # Define a hyperparameter config
    #         config = {
    #             'lr': 0.0005,
    #             'task_loss_weight': 0.5,  # Initial weight for task loss
    #             'loss_function': 'bce',  # Initial loss function
    #         }
    #         wandb.config.update(config)

    #         loss_form = get_loss_function(wandb.config.loss_function)

    #         # Model and optimizer setup
    #         optimizer = torch.optim.AdamW(
    #             model.parameters(), lr=wandb.config.lr)
    #         scheduler = ReduceLROnPlateau(
    #             optimizer, mode='min', factor=0.1, patience=5)

    #         print(
    #             f'-------------------------- Training {dataset_name} using {model_name} ----------------------')

    #         # Training loop with validation and logging
    #         for epoch in range(51):
    #             model.train()
    #             train_losses, train_correct = 0, 0
    #             for x_batch, c_batch, y_batch in train_loader:
    #                 optimizer.zero_grad()

    #                 c_emb, c_pred = concept_encoder(x_batch)
    #                 y_pred = task_predictor(c_emb, c_pred)

    #                 concept_loss = loss_form(c_pred, c_batch)
    #                 task_loss = loss_form(y_pred, y_batch)

    #                 loss = concept_loss + 0.5 * task_loss

    #                 loss.backward()
    #                 optimizer.step()

    #                 train_losses += loss.item()
    #                 train_correct += (y_pred.argmax(1) ==
    #                                   y_batch.argmax(1)).sum().item()

    #                 wandb.log({
    #                     'train_concept_loss': concept_loss.item(),
    #                     'train_task_loss': task_loss.item(),
    #                     'learning_rate': optimizer.param_groups[0]['lr']
    #                 })

    #             # Validation step
    #             model.eval()
    #             val_losses, val_correct = 0, 0
    #             with torch.no_grad():
    #                 for x_batch, c_batch, y_batch in val_loader:
    #                     c_emb, c_pred = concept_encoder(x_batch)
    #                     y_pred = task_predictor(c_emb, c_pred)

    #                     val_concept_loss = loss_form(c_pred, c_batch)
    #                     val_task_loss = loss_form(y_pred, y_batch)

    #                     val_loss = val_concept_loss + 0.5 * \
    #                         val_task_loss

    #                     val_losses += val_loss.item()
    #                     val_correct += (y_pred.argmax(1) ==
    #                                     y_batch.argmax(1)).sum().item()

    #                     # Log validation losses and learning rate
    #                     wandb.log({
    #                         'val_concept_loss': val_concept_loss.item(),
    #                         'val_task_loss': val_task_loss.item(),
    #                         'val_learning_rate': optimizer.param_groups[0]['lr']
    #                     })

    #             scheduler.step(val_loss)

    #             # Log metrics every epoch!
    #             print(f"Epoch {epoch+1}, Loss: {train_losses/len(train_loader)}, Train Accuracy: {train_correct/len(train_dataset)}, Val Loss: {val_losses/len(val_loader)}, Val Accuracy: {val_correct/len(val_dataset)}")
    #             wandb.log({
    #                 'epoch': epoch + 1,
    #                 'loss': train_losses / len(train_loader),
    #                 'train_accuracy': train_correct / len(train_dataset),
    #                 'val_loss': val_losses / len(val_loader),
    #                 'val_accuracy': val_correct / len(val_dataset)
    #             })

    #         print(
    #             f"\n Training on {dataset_name} using {model_name} has been completed!")

    #         # Save the model
    #         torch.save(model, f'model_{model_name}_{dataset_name}.pth')
    #         torch.save(model.state_dict(),
    #                    f'model_state_dict_{model_name}_{dataset_name}.pth')

    #         # Finish run
    #         wandb.finish()

    #     print(f"===========================================================")

    if model_name == 'LLRAttention':
        print(f"Training on {model_name} ... ")
        print(f"--------------------------------")

        # Iterate through datasets on each model
        for dataset_name, dataset in zip(dataset_names, datasets):
            if dataset_name in ['Mux41_two_inputs', 'Two_Muxes',
                 'Trigonometry', 'Dot']:
              # Importing dataset
              x, c, y = dataset
              x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(
                  x, c, y, test_size=0.2, random_state=42)
              print(
                  f"The following dataset has been loaded successully: {dataset_name}")

              # Encode target into one-hode y=[y_0, y_1]
              y_train = F.one_hot(y_train.long().ravel()).float()
              y_test = F.one_hot(y_test.long().ravel()).float()

              # Define the concept_encoder
              embedding_size = 8
              concept_encoder = torch.nn.Sequential(
                  torch.nn.Linear(x.shape[1], 10),
                  torch.nn.LeakyReLU(),
                  te.nn.ConceptEmbedding(10, c.shape[1], embedding_size),
              )

              # Define the DCR as task predictor
              task_predictor = ConceptReasoningLayer(
                  embedding_size, y_train.shape[1])

              dummy_c_emb, dummy_c_pred = concept_encoder(x_train)
              dummy_y, dummy_sign, dummy_filter = task_predictor(
                  dummy_c_emb, dummy_c_pred, return_attn=True)

              global_linearity = ReasoningLinearLayer(
                  dummy_sign.shape[1], dummy_filter.shape[1], y_train.shape[1], modality='Attention')

              # Create a sequential model (cascaded)
              model = torch.nn.Sequential(
                  concept_encoder, task_predictor, global_linearity)

              # Split dataset into training and validation
              num_val_samples = int(len(x_train) * 0.2)
              num_train_samples = len(x_train) - num_val_samples
              train_dataset, val_dataset = random_split(list(zip(x_train, c_train, y_train)), [
                                                        num_train_samples, num_val_samples])

              train_loader = DataLoader(
                  train_dataset, batch_size=64, shuffle=True)
              val_loader = DataLoader(
                  val_dataset, batch_size=64, shuffle=False)

              wandb.init(project="pytorch_explain", entity="alih9862",
                        name=f"{model_name}_{dataset_name}")

              # Define a hyperparameter config
              config = {
                  'lr': 0.0005,
                  'task_loss_weight': 0.5,  # Initial weight for task loss
                  'loss_function': 'bce',  # Initial loss function
                  'lin_loss_function': 'bceL',
              }
              wandb.config.update(config)

              loss_form = get_loss_function(wandb.config.loss_function)
              linearity_loss = get_loss_function(
                  wandb.config.lin_loss_function)

              # Model and optimizer setup
              optimizer = torch.optim.AdamW(
                  model.parameters(), lr=wandb.config.lr)
              scheduler = ReduceLROnPlateau(
                  optimizer, mode='min', factor=0.1, patience=5)

              print(
                  f'-------------------------- Training {dataset_name} using {model_name} ----------------------')

              # Training loop with validation and logging
              for epoch in range(51):
                  model.train()
                  train_losses, train_correct = 0, 0
                  for x_batch, c_batch, y_batch in train_loader:
                      optimizer.zero_grad()

                      c_emb, c_pred = concept_encoder(x_batch)
                      y_pred, sign_attn, filter_attn = task_predictor(
                          c_emb, c_pred, return_attn=True)
                      y_hat = global_linearity(sign_attn, filter_attn, c_pred)

                      concept_loss = loss_form(c_pred, c_batch)
                      task_loss = loss_form(y_pred, y_batch)
                      glob_loss = linearity_loss(y_hat, y_batch)

                      loss = 0.25 * concept_loss + 0.25 * task_loss + 0.5 * glob_loss

                      loss.backward()
                      optimizer.step()

                      train_losses += loss.item()
                      train_correct += (y_pred.argmax(1) ==
                                        y_batch.argmax(1)).sum().item()

                      wandb.log({
                          'train_concept_loss': concept_loss.item(),
                          'train_task_loss': task_loss.item(),
                          'train_globlinear_loss': glob_loss.item(),
                          'learning_rate': optimizer.param_groups[0]['lr']
                      })

                  # Validation step
                  model.eval()
                  val_losses, val_correct = 0, 0
                  with torch.no_grad():
                      for x_batch, c_batch, y_batch in val_loader:

                          c_emb, c_pred = concept_encoder(x_batch)
                          y_pred, sign_attn, filter_attn = task_predictor(
                              c_emb, c_pred, return_attn=True)
                          y_hat = global_linearity(
                              sign_attn, filter_attn, c_pred)

                          val_concept_loss = loss_form(c_pred, c_batch)
                          val_task_loss = loss_form(y_pred, y_batch)
                          val_glob_loss = linearity_loss(y_hat, y_batch)

                          val_loss = 0.25 * val_concept_loss + 0.25 * val_task_loss + 0.5 * val_glob_loss

                          val_losses += val_loss.item()
                          val_correct += (y_pred.argmax(1) ==
                                          y_batch.argmax(1)).sum().item()

                          # Log validation losses and learning rate
                          wandb.log({
                              'val_concept_loss': val_concept_loss.item(),
                              'val_task_loss': val_task_loss.item(),
                              'val_glob_loss': val_glob_loss.item(),
                              'val_learning_rate': optimizer.param_groups[0]['lr']
                          })

                  scheduler.step(val_loss)

                  # Log metrics every epoch!
                  print(f"Epoch {epoch+1}, Loss: {train_losses/len(train_loader)}, Train Accuracy: {train_correct/len(train_dataset)}, Val Loss: {val_losses/len(val_loader)}, Val Accuracy: {val_correct/len(val_dataset)}")
                  wandb.log({
                      'epoch': epoch + 1,
                      'loss': train_losses / len(train_loader),
                      'train_accuracy': train_correct / len(train_dataset),
                      'val_loss': val_losses / len(val_loader),
                      'val_accuracy': val_correct / len(val_dataset)
                  })

              print(
                  f"\n Training on {dataset_name} using {model_name} has been completed!")

              # Save the model
              torch.save(model, f'model_{model_name}_{dataset_name}.pth')
              torch.save(model.state_dict(),
                        f'model_state_dict_{model_name}_{dataset_name}.pth')

              # Finish run
              wandb.finish()

            print(f"===========================================================")

    # if model_name == 'LLRNN':
    #     print(f"Training on {model_name} ... ")
    #     print(f"--------------------------------")

    #     # Iterate through datasets on each model
    #     for dataset_name, dataset in zip(dataset_names, datasets):
    #         # Importing dataset
    #         x, c, y = dataset
    #         x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(
    #             x, c, y, test_size=0.2, random_state=42)
    #         print(
    #             f"The following dataset has been loaded successully: {dataset_name}")

    #         # Encode target into one-hode y=[y_0, y_1]
    #         y_train = F.one_hot(y_train.long().ravel()).float()
    #         y_test = F.one_hot(y_test.long().ravel()).float()

    #         # Define the concept_encoder
    #         embedding_size = 8
    #         concept_encoder = torch.nn.Sequential(
    #             torch.nn.Linear(x.shape[1], 10),
    #             torch.nn.LeakyReLU(),
    #             te.nn.ConceptEmbedding(10, c.shape[1], embedding_size),
    #         )

    #         # Define the DCR as task predictor
    #         task_predictor = ConceptReasoningLayer(
    #             embedding_size, y_train.shape[1])

    #         dummy_c_emb, dummy_c_pred = concept_encoder(x_train)
    #         dummy_y, dummy_sign, dummy_filter = task_predictor(
    #             dummy_c_emb, dummy_c_pred, return_attn=True)

    #         global_linearity = ReasoningLinearLayer(
    #             dummy_sign.shape[1], dummy_filter.shape[1], y_train.shape[1], modality='NN')

    #         # Create a sequential model (cascaded)
    #         model = torch.nn.Sequential(
    #             concept_encoder, task_predictor, global_linearity)

    #         # Split dataset into training and validation
    #         num_val_samples = int(len(x_train) * 0.2)
    #         num_train_samples = len(x_train) - num_val_samples
    #         train_dataset, val_dataset = random_split(list(zip(x_train, c_train, y_train)), [
    #                                                   num_train_samples, num_val_samples])

    #         train_loader = DataLoader(
    #             train_dataset, batch_size=64, shuffle=True)
    #         val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    #         wandb.init(project="pytorch_explain", entity="alih9862",
    #                    name=f"{model_name}_{dataset_name}")

    #         # Define a hyperparameter config
    #         config = {
    #             'lr': 0.0005,
    #             'task_loss_weight': 0.5,  # Initial weight for task loss
    #             'loss_function': 'bce',  # Initial loss function
    #             'lin_loss_function': 'bceL',
    #         }
    #         wandb.config.update(config)

    #         loss_form = get_loss_function(wandb.config.loss_function)
    #         linearity_loss = get_loss_function(wandb.config.lin_loss_function)

    #         # Model and optimizer setup
    #         optimizer = torch.optim.AdamW(
    #             model.parameters(), lr=wandb.config.lr)
    #         scheduler = ReduceLROnPlateau(
    #             optimizer, mode='min', factor=0.1, patience=5)

    #         print(
    #             f'-------------------------- Training {dataset_name} using {model_name} ----------------------')

    #         # Training loop with validation and logging
    #         for epoch in range(51):
    #             model.train()
    #             train_losses, train_correct = 0, 0
    #             for x_batch, c_batch, y_batch in train_loader:
    #                 optimizer.zero_grad()

    #                 c_emb, c_pred = concept_encoder(x_batch)
    #                 y_pred, sign_attn, filter_attn = task_predictor(
    #                     c_emb, c_pred, return_attn=True)
    #                 y_hat = global_linearity(sign_attn, filter_attn, c_pred)

    #                 concept_loss = loss_form(c_pred, c_batch)
    #                 task_loss = loss_form(y_pred, y_batch)
    #                 glob_loss = linearity_loss(y_hat, y_batch)

    #                 loss = 0.25 * concept_loss + 0.25 * task_loss + 0.5 * glob_loss

    #                 loss.backward()
    #                 optimizer.step()

    #                 train_losses += loss.item()
    #                 train_correct += (y_pred.argmax(1) ==
    #                                   y_batch.argmax(1)).sum().item()

    #                 wandb.log({
    #                     'train_concept_loss': concept_loss.item(),
    #                     'train_task_loss': task_loss.item(),
    #                     'train_globlinear_loss': glob_loss.item(),
    #                     'learning_rate': optimizer.param_groups[0]['lr']
    #                 })

    #             # Validation step
    #             model.eval()
    #             val_losses, val_correct = 0, 0
    #             with torch.no_grad():
    #                 for x_batch, c_batch, y_batch in val_loader:

    #                     c_emb, c_pred = concept_encoder(x_batch)
    #                     y_pred, sign_attn, filter_attn = task_predictor(
    #                         c_emb, c_pred, return_attn=True)
    #                     y_hat = global_linearity(
    #                         sign_attn, filter_attn, c_pred)

    #                     val_concept_loss = loss_form(c_pred, c_batch)
    #                     val_task_loss = loss_form(y_pred, y_batch)
    #                     val_glob_loss = linearity_loss(y_hat, y_batch)

    #                     val_loss = 0.25 * val_concept_loss + 0.25 * val_task_loss + 0.5 * val_glob_loss

    #                     val_losses += val_loss.item()
    #                     val_correct += (y_pred.argmax(1) ==
    #                                     y_batch.argmax(1)).sum().item()

    #                     # Log validation losses and learning rate
    #                     wandb.log({
    #                         'val_concept_loss': val_concept_loss.item(),
    #                         'val_task_loss': val_task_loss.item(),
    #                         'val_glob_loss': val_glob_loss.item(),
    #                         'val_learning_rate': optimizer.param_groups[0]['lr']
    #                     })

    #             scheduler.step(val_loss)

    #             # Log metrics every epoch!
    #             print(f"Epoch {epoch+1}, Loss: {train_losses/len(train_loader)}, Train Accuracy: {train_correct/len(train_dataset)}, Val Loss: {val_losses/len(val_loader)}, Val Accuracy: {val_correct/len(val_dataset)}")
    #             wandb.log({
    #                 'epoch': epoch + 1,
    #                 'loss': train_losses / len(train_loader),
    #                 'train_accuracy': train_correct / len(train_dataset),
    #                 'val_loss': val_losses / len(val_loader),
    #                 'val_accuracy': val_correct / len(val_dataset)
    #             })

    #         print(
    #             f"\n Training on {dataset_name} using {model_name} has been completed!")

    #         # Save the model
    #         torch.save(model, f'model_{model_name}_{dataset_name}.pth')
    #         torch.save(model.state_dict(),
    #                    f'model_state_dict_{model_name}_{dataset_name}.pth')

    #         # Finish run
    #         wandb.finish()

    #     print(f"===========================================================")

    if model_name == 'LLRWeighted':
        print(f"Training on {model_name} ... ")
        print(f"--------------------------------")

        # Iterate through datasets on each model
        for dataset_name, dataset in zip(dataset_names, datasets):
            # Importing dataset
            x, c, y = dataset
            x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(
                x, c, y, test_size=0.2, random_state=42)
            print(
                f"The following dataset has been loaded successully: {dataset_name}")

            # Encode target into one-hode y=[y_0, y_1]
            y_train = F.one_hot(y_train.long().ravel()).float()
            y_test = F.one_hot(y_test.long().ravel()).float()

            # Define the concept_encoder
            embedding_size = 8
            concept_encoder = torch.nn.Sequential(
                torch.nn.Linear(x.shape[1], 10),
                torch.nn.LeakyReLU(),
                te.nn.ConceptEmbedding(10, c.shape[1], embedding_size),
            )

            # Define the DCR as task predictor
            task_predictor = ConceptReasoningLayer(
                embedding_size, y_train.shape[1])

            dummy_c_emb, dummy_c_pred = concept_encoder(x_train)
            dummy_y, dummy_sign, dummy_filter = task_predictor(
                dummy_c_emb, dummy_c_pred, return_attn=True)

            global_linearity = ReasoningLinearLayer(
                dummy_sign.shape[1], dummy_filter.shape[1], y_train.shape[1], modality='Weighted')

            # Create a sequential model (cascaded)
            model = torch.nn.Sequential(
                concept_encoder, task_predictor, global_linearity)

            # Split dataset into training and validation
            num_val_samples = int(len(x_train) * 0.2)
            num_train_samples = len(x_train) - num_val_samples
            train_dataset, val_dataset = random_split(list(zip(x_train, c_train, y_train)), [
                                                      num_train_samples, num_val_samples])

            train_loader = DataLoader(
                train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            wandb.init(project="pytorch_explain", entity="alih9862",
                       name=f"{model_name}_{dataset_name}")

            # Define a hyperparameter config
            config = {
                'lr': 0.0005,
                'task_loss_weight': 0.5,  # Initial weight for task loss
                'loss_function': 'bce',  # Initial loss function
                'lin_loss_function': 'bceL',
            }
            wandb.config.update(config)

            loss_form = get_loss_function(wandb.config.loss_function)
            linearity_loss = get_loss_function(wandb.config.lin_loss_function)

            # Model and optimizer setup
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=wandb.config.lr)
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=5)

            print(
                f'-------------------------- Training {dataset_name} using {model_name} ----------------------')

            # Training loop with validation and logging
            for epoch in range(51):
                model.train()
                train_losses, train_correct = 0, 0
                for x_batch, c_batch, y_batch in train_loader:
                    optimizer.zero_grad()

                    c_emb, c_pred = concept_encoder(x_batch)
                    y_pred, sign_attn, filter_attn = task_predictor(
                        c_emb, c_pred, return_attn=True)
                    y_hat = global_linearity(sign_attn, filter_attn, c_pred)

                    concept_loss = loss_form(c_pred, c_batch)
                    task_loss = loss_form(y_pred, y_batch)
                    glob_loss = linearity_loss(y_hat, y_batch)

                    loss = 0.25 * concept_loss + 0.25 * task_loss + 0.5 * glob_loss

                    loss.backward()
                    optimizer.step()

                    train_losses += loss.item()
                    train_correct += (y_pred.argmax(1) ==
                                      y_batch.argmax(1)).sum().item()

                    wandb.log({
                        'train_concept_loss': concept_loss.item(),
                        'train_task_loss': task_loss.item(),
                        'train_globlinear_loss': glob_loss.item(),
                        'learning_rate': optimizer.param_groups[0]['lr']
                    })

                # Validation step
                model.eval()
                val_losses, val_correct = 0, 0
                with torch.no_grad():
                    for x_batch, c_batch, y_batch in val_loader:

                        c_emb, c_pred = concept_encoder(x_batch)
                        y_pred, sign_attn, filter_attn = task_predictor(
                            c_emb, c_pred, return_attn=True)
                        y_hat = global_linearity(
                            sign_attn, filter_attn, c_pred)

                        val_concept_loss = loss_form(c_pred, c_batch)
                        val_task_loss = loss_form(y_pred, y_batch)
                        val_glob_loss = linearity_loss(y_hat, y_batch)

                        val_loss = 0.25 * val_concept_loss + 0.25 * val_task_loss + 0.5 * val_glob_loss

                        val_losses += val_loss.item()
                        val_correct += (y_pred.argmax(1) ==
                                        y_batch.argmax(1)).sum().item()

                        # Log validation losses and learning rate
                        wandb.log({
                            'val_concept_loss': val_concept_loss.item(),
                            'val_task_loss': val_task_loss.item(),
                            'val_glob_loss': val_glob_loss.item(),
                            'val_learning_rate': optimizer.param_groups[0]['lr']
                        })

                scheduler.step(val_loss)

                # Log metrics every epoch!
                print(f"Epoch {epoch+1}, Loss: {train_losses/len(train_loader)}, Train Accuracy: {train_correct/len(train_dataset)}, Val Loss: {val_losses/len(val_loader)}, Val Accuracy: {val_correct/len(val_dataset)}")
                wandb.log({
                    'epoch': epoch + 1,
                    'loss': train_losses / len(train_loader),
                    'train_accuracy': train_correct / len(train_dataset),
                    'val_loss': val_losses / len(val_loader),
                    'val_accuracy': val_correct / len(val_dataset)
                })

            print(
                f"\n Training on {dataset_name} using {model_name} has been completed!")

            # Save the model
            torch.save(model, f'model_{model_name}_{dataset_name}.pth')
            torch.save(model.state_dict(),
                       f'model_state_dict_{model_name}_{dataset_name}.pth')

            # Finish run
            wandb.finish()

        print(f"===========================================================")

print(f"*********** ALL TRAINING ARE DONE - Check WandB ***********")
