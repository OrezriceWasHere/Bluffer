import torch.nn as nn
import DatasetPrepare
import Parameters
import torch


# Training Function
from SaveLoad import save_checkpoint, save_metrics


def train(model,
          optimizer,
          criterion=nn.BCELoss(),
          train_loader=DatasetPrepare.train_iter,
          test_loader=DatasetPrepare.test_iter,
          num_epochs=5,
          eval_every=len(DatasetPrepare.train_iter) // 2,
          file_path=Parameters.OUTPUT_FOLDER,
          best_test_loss=float("Inf")):

    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    test_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):

        for (labels, tweet_text), nonefields in train_loader:
            # labels = labels.type(torch.LongTensor)
            labels = labels.to(Parameters.DEVICE)
            tweet_text = tweet_text.to(Parameters.DEVICE)
            output = model(tweet_text, labels)
            loss, _ = output

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

                    # validation loop
                    # id,title,author,text,label
                    for (labels_test, tweet_text_test), _ in test_loader:
                        # labels_test = labels_test.type(torch.LongTensor)
                        tweet_text_test = tweet_text_test.to(Parameters.DEVICE)
                        labels_test = labels_test.to(Parameters.DEVICE)
                        output = model(tweet_text_test, labels_test)
                        loss, _ = output

                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_test_loss = valid_running_loss / len(test_loader)
                train_loss_list.append(average_train_loss)
                test_loss_list.append(average_test_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_test_loss))

                # checkpoint
                if best_test_loss > average_test_loss:
                    best_test_loss = average_test_loss
                    save_checkpoint(Parameters.MODEL_OUTPUT_FILE, model, best_test_loss)
                    save_metrics(Parameters.METRICS_OUTPUT_FILE, train_loss_list, test_loss_list, global_steps_list)

    save_metrics(Parameters.METRICS_OUTPUT_FILE, train_loss_list, test_loss_list, global_steps_list)
    print('Finished Training!')