import copy
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.utils.data
from torch.cuda.amp import autocast
import typing
from tqdm import tqdm
import warnings
import torch.multiprocessing
import wandb

sys.path.append(os.getcwd())

from critical_classification.src import utils, context_handlers, backbone_pipeline
from critical_classification.config import Config, GetConfig

warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy('file_system')
if utils.is_local():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def print_info_for_debug(ground_truths,
                         predictions,
                         video_name_with_time):
    print(utils.blue_text(
        f'ground truth use and count: '
        f'{np.unique(np.array(ground_truths), return_counts=True)}\n'))

    print(utils.blue_text(
        f'prediction use and count: '
        f'{np.unique(np.array(predictions), return_counts=True)}\n'))

    for idx, (video_name, time) in enumerate(video_name_with_time):
        if ground_truths[idx] == predictions[idx]:
            print(utils.green_text(f'correct prediction: '), end='')
        if ground_truths[idx] == 1:
            print(utils.red_text(f'{video_name}{" " * (120 - len(video_name))} '
                                 f'at time {int(int(time) / 60)}:{int(time) - 60 * int(int(time) / 60)}'))
        else:
            print(utils.blue_text(f'{video_name}{" " * (120 - len(video_name))} '
                                  f'at time {int(int(time) / 60)}:{int(time) - 60 * int(int(time) / 60)}'))
        if idx > 20:
            break


def batch_learning_and_evaluating(loaders,
                                  device: torch.device,
                                  fine_tuner: torch.nn.Module,
                                  scheduler: torch.optim.lr_scheduler,
                                  optimizer: torch.optim.Optimizer,
                                  config: Config,
                                  evaluation: bool = False):
    num_batches = len(loaders)
    batches = tqdm(enumerate(loaders, 0),
                   total=num_batches)

    predictions = []
    ground_truths = []
    video_name_with_time = []
    total_running_loss = torch.Tensor([0.0]).to(device)

    if evaluation:
        fine_tuner.eval()
    else:
        fine_tuner.train()

    for batch_num, batch in batches:
        with context_handlers.ClearCache(device=device):
            optimizer.zero_grad()

            X, Y_true, video_name_with_time_batch = batch
            # If X is 1 video only, collapse the dimension (when model is YOLO only
            if X.shape[0] == 1 and 'YOLO' in config.model_name:
                X = torch.squeeze(X)
            X = X.to(device).float()
            Y_true = Y_true.to(device)

            # For debuging:
            # Y_pred = Y_true
            # evaluation = True

            with autocast():
                Y_pred = fine_tuner(X)

            if Y_pred.ndim == 1:
                Y_pred = Y_pred.unsqueeze(dim=1)

            predictions.append(torch.squeeze(torch.where(Y_pred > 0.5, 1, 0)).detach().to('cpu'))
            ground_truths.append(Y_true.detach().to('cpu'))
            video_name_with_time.extend([(os.path.basename(item[0]), item[1])
                                         for item in zip(video_name_with_time_batch[0],
                                                         video_name_with_time_batch[1])])

            criterion = torch.nn.BCEWithLogitsLoss()
            batch_total_loss = criterion(Y_pred, torch.unsqueeze(Y_true, dim=1).float())
            total_running_loss += batch_total_loss.item()

            if evaluation:
                del X, Y_pred, Y_true
                # break  # for debuging
                continue

            batch_total_loss.backward()
            optimizer.step()

            del X, Y_pred, Y_true
            # break  # for debuging

    if not evaluation:
        scheduler.step()

    average_loss = total_running_loss.item() / num_batches

    print(utils.blue_text(f'Current total loss: {total_running_loss.item()}'))

    predictions = [item.unsqueeze(dim=0) if item.ndim == 0 else item for item in predictions]
    predictions = torch.cat(predictions)
    ground_truths = torch.cat(ground_truths)

    print_info_for_debug(ground_truths,
                         predictions,
                         video_name_with_time)

    accuracy = accuracy_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions)
    recall = recall_score(ground_truths, predictions)

    print(f'accuracy: {accuracy}, f1: {f1}, precision: {precision}, recall: {recall}')

    return optimizer, fine_tuner, accuracy, f1, average_loss, get_lr(optimizer)


def fine_tune_combined_model(fine_tuner: torch.nn.Module,
                             device: torch.device,
                             loaders: typing.Dict[str, torch.utils.data.DataLoader],
                             config: Config):
    fine_tuner.to(device)
    fine_tuner.train()

    optimizer = torch.optim.Adam(params=fine_tuner.parameters(),
                                 lr=config.lr)
    if config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=int(config.num_epochs / 2),
                                                               eta_min=1e-07,
                                                               # eta_min=config.lr * 10
                                                               )
    elif config.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.5)
    elif config.scheduler == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.7943)
    else:
        raise NotImplementedError('scheduler not implemented!')
    best_fine_tuner = copy.deepcopy(fine_tuner)

    all_on_device = True
    for name, param in fine_tuner.named_parameters():
        if param.device != device:
            all_on_device = False
            print(f"Parameter '{name}' is on {param.device} instead of {device}")

    if all_on_device:
        print(f"All parameters are on {device}")

    # train_result_dict = {'acc': [], 'f1': [], 'loss': [], 'lr': []}
    # test_result_dict = {'acc': [], 'f1': [], 'loss': [], 'lr': []}
    max_f1_score = 0

    for epoch in range(config.num_epochs):
        with ((context_handlers.TimeWrapper())):
            print('#' * 50 + f'train epoch {epoch}' + '#' * 50)
            optimizer, fine_tuner, train_accuracy, train_f1, train_total_loss, train_lr = \
                batch_learning_and_evaluating(loaders=loaders['train'],
                                              device=device,
                                              optimizer=optimizer,
                                              fine_tuner=fine_tuner,
                                              config=config,
                                              scheduler=scheduler)
            # train_result_dict['acc'].append(train_accuracy)
            # train_result_dict['f1'].append(train_f1)
            # train_result_dict['loss'].append(train_total_loss)
            # train_result_dict['lr'].append(train_lr)
            # utils.plot_figure_and_save_dict(
            #     train_result_dict,
            #     config.get_file_name(current_epoch=0, file_type='fig', additional_info='train'),
            #     config.get_file_name(current_epoch=0, file_type='pickle', additional_info='train'),
            #     train_or_test='train')

            # Log training metrics to W&B
            wandb.log({
                'train/accuracy': train_accuracy,
                'train/f1': train_f1,
                'train/loss': train_total_loss,
                'train/lr': train_lr,
                'epoch': epoch
            })

            # Testing
            print('#' * 50 + f'test epoch {epoch}' + '#' * 50)
            optimizer, fine_tuner, test_accuracy, test_f1, test_total_loss, test_lr = \
                batch_learning_and_evaluating(loaders=loaders['test'],
                                              device=device,
                                              optimizer=optimizer,
                                              fine_tuner=fine_tuner,
                                              scheduler=scheduler,
                                              config=config,
                                              evaluation=True)
            # test_result_dict['acc'].append(test_accuracy)
            # test_result_dict['f1'].append(test_f1)
            # test_result_dict['loss'].append(test_total_loss)
            # test_result_dict['lr'].append(test_lr)
            # utils.plot_figure_and_save_dict(
            #     test_result_dict,
            #     config.get_file_name(current_epoch=0, file_type='fig', additional_info='test'),
            #     config.get_file_name(current_epoch=0, file_type='pickle', additional_info='test'),
            #     train_or_test='test')

            # Log testing metrics to W&B
            wandb.log({
                'test/accuracy': test_accuracy,
                'test/f1': test_f1,
                'test/loss': test_total_loss,
                'test/lr': test_lr,
                'epoch': epoch
            })

            if epoch % 2 == 0:
                print(utils.blue_text(f'save current fine tuner at epoch {epoch}!'))
                torch.save(best_fine_tuner.state_dict(),
                           config.get_file_name(current_epoch=0, file_type='model'))

            if max_f1_score < test_f1:
                max_f1_score = test_f1
                best_fine_tuner = copy.deepcopy(fine_tuner)
                wandb.log({"best_f1_score": max_f1_score})
                print(utils.green_text(f'save best fine tuner at epoch {epoch}!'))

                torch.save(best_fine_tuner.state_dict(),
                           config.get_file_name(current_epoch=epoch, file_type='model', additional_info='best'))

    # Final model
    print('#' * 50 + f'test best fine_tuner' + '#' * 50)
    batch_learning_and_evaluating(loaders=loaders['test'],
                                  device=device,
                                  optimizer=optimizer,
                                  fine_tuner=best_fine_tuner,
                                  scheduler=scheduler,
                                  config=config,
                                  evaluation=True)

    print('#' * 100)

    return best_fine_tuner


def run_combined_fine_tuning_pipeline(config: Config):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="DAAD RISE Germany",

        # track hyperparameters and run metadata
        config=config.get_config(),
    )

    fine_tuner, loaders, device = (
        backbone_pipeline.initiate(config)
    )

    fine_tune_combined_model(
        fine_tuner=fine_tuner,
        device=device,
        loaders=loaders,
        config=config
    )
    print('#' * 100)


if __name__ == '__main__':
    config = GetConfig()
    run_combined_fine_tuning_pipeline(config)
