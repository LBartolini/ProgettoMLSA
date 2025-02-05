import torch

def progress_bar(current_value, max_value, size=50):
    prog = (current_value+1)/max_value
    left = '#'*int(prog * size + 0.5) # 0.5 to round up when casting to int
    right = '-'*(size-len(left))
    print('\r[{}{}] {:.1f}%'.format(left, right, prog*100), end='')

def train_model(model, criterion, optimizer, dict_size, train_loader, val_loader=None, tok_k=5, num_epochs=5, DEVICE='cpu'):
    print("\nStarting training...")

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        train_loss = 0.0
        train_accuracy = 0
        valid_loss = 0
        valid_accuracy = 0

        correct_predictions = 0
        total_predictions = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            progress_bar(epoch*len(train_loader)+batch_idx, num_epochs*len(train_loader))

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Clear the gradients from the previous step
            optimizer.zero_grad()

            # Forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(inputs)
            one_hot = torch.nn.functional.one_hot(targets.squeeze(1).to(torch.long), dict_size)

            # Compute the loss
            loss = criterion(outputs, one_hot.to(torch.float))

            # Backward pass: compute gradients of the loss with respect to the model parameters
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Track the loss
            train_loss += loss.item()

            # Track top-k accuracy
            _, top_preds = torch.topk(outputs, tok_k, dim=1, largest=True, sorted=True)
            top_correct = top_preds.eq(targets.view(-1, 1).expand_as(top_preds))
            correct_predictions += top_correct.sum().item()
            total_predictions += targets.size(0)

        train_accuracy = (correct_predictions / total_predictions) * 100
        train_loss = train_loss / len(train_loader)

        if val_loader:
            correct_predictions = 0
            total_predictions = 0
            
            model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                    # Forward pass: compute predicted outputs by passing inputs to the model
                    outputs = model(inputs)
                    one_hot = torch.nn.functional.one_hot(targets.squeeze(1).to(torch.long), dict_size)

                    # Compute the loss
                    loss = criterion(outputs, one_hot.to(torch.float))

                    # Track the loss
                    valid_loss += loss.item()

                    # Track top-k accuracy
                    _, top_preds = torch.topk(outputs, tok_k, dim=1, largest=True, sorted=True)
                    top_correct = top_preds.eq(targets.view(-1, 1).expand_as(top_preds))
                    correct_predictions += top_correct.sum().item()
                    total_predictions += targets.size(0)

                valid_accuracy = (correct_predictions / total_predictions) * 100
                valid_loss = valid_loss / len(val_loader)
        
        print(f" Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%")
    
    return model, (train_loss, train_accuracy, valid_loss, valid_accuracy)

def predict(model, tokenizer, string, input_len, k=5, verbose=False, DEVICE='cpu'):
    input = torch.Tensor(tokenizer.encode(string)[:input_len]).to(torch.long).to(DEVICE)
    if verbose:
        print(f"INPUT: {tokenizer.decode(input)}")

    model.eval()
    with torch.no_grad():
        output = model(input.view(1, -1))

    tokens = torch.topk(output, k, dim=1, largest=True, sorted=True)[1]

    return [tokenizer.decode(t) for t in tokens[0]]