from Parameters import DEVICE
import torch
from torch import nn
# Training Function
from SaveLoad import save_checkpoint, save_metrics


def train(model,
          optimizer,
          train_loader,
          test_loader,
          model_output_file,
          metric_output_file,
          criterion=nn.BCELoss(),
          num_epochs=5,
          eval_every=500,
          best_test_loss=float("Inf")):

    # initialize running values
    running_loss = 0.0
    test_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    test_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):

        for (text, labels), nonefields in train_loader:
            # labels = labels.type(torch.LongTensor)
            labels = labels.to(DEVICE)
            text = text.to(DEVICE)
            result = model(text)
            prediction = torch.argmax(result, 1).float()
            loss = criterion(prediction, labels)
            loss.requires_grad = True

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():

                    # test loop
                    for (tweet_text_test, labels_test), _ in test_loader:
                        tweet_text_test = tweet_text_test.to(DEVICE)
                        labels_test = labels_test.to(DEVICE)
                        result = model(tweet_text_test)
                        prediction = torch.argmax(result, 1).float()
                        loss = criterion(prediction, labels_test)
                        test_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_test_loss = test_running_loss / len(test_loader)
                train_loss_list.append(average_train_loss)
                test_loss_list.append(average_test_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                test_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_test_loss))

                # checkpoint
                if best_test_loss > average_test_loss:
                    best_test_loss = average_test_loss
                    save_checkpoint(model_output_file, model, best_test_loss)
                    save_metrics(metric_output_file, train_loss_list, test_loss_list, global_steps_list)

    save_metrics(metric_output_file, train_loss_list, test_loss_list, global_steps_list)
    print('Finished Training!')
