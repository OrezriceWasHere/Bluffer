import torch
import Parameters


# Save and Load Functions
def save_checkpoint(save_path, model, test_loss):
    if save_path is None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'test_loss': test_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model):
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=Parameters.DEVICE)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['test_loss']


def save_metrics(save_path, train_loss_list, test_loss_list, global_steps_list):
    if save_path is None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'test_loss_list': test_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Metric saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=Parameters.DEVICE)
    print(f'Metric loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['test_loss_list'], state_dict['global_steps_list']
