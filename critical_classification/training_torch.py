import argparse
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

sys.path.append(os.getcwd())

from critical_classification.src import utils, context_handlers, backbone_pipeline
from critical_classification.config import Config

warnings.filterwarnings("ignore")

if utils.is_local():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


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
    pass


def batch_learning_and_evaluating(loaders,
                                  device: torch.device,
                                  fine_tuner: torch.nn.Module,
                                  scheduler: torch.optim.lr_scheduler,
                                  optimizer: torch.optim,
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
            # If X is 1 video only, collapse the dimension
            if X.shape[0] == 1:
                X = torch.squeeze(X)
            X = X.to(device).float()
            Y_true = Y_true.to(device)

            with autocast():
                Y_pred = fine_tuner(X)

            if Y_pred.ndim == 1:
                Y_pred = Y_pred.unsqueeze(dim=0)

            # For debuging:
            # Y_pred = Y_true
            # evaluation = True

            predictions.append(torch.squeeze(torch.where(Y_pred > 0.5, 1, 0)).detach().to('cpu'))
            ground_truths.append(Y_true.detach().to('cpu'))
            video_name_with_time.extend([(os.path.basename(item[0]), item[1])
                                         for item in zip(video_name_with_time_batch[0],
                                                         video_name_with_time_batch[1])])
            if evaluation:
                del X, Y_pred, Y_true
                # break  # for debuging
                continue

            criterion = torch.nn.BCEWithLogitsLoss()
            batch_total_loss = criterion(Y_pred, torch.unsqueeze(Y_true, dim=1).float())
            total_running_loss += batch_total_loss.item() / len(batches)
            # Update progress bar with informative text (without newline)
            if batch_num % (int(num_batches / 2.5)) == 0:
                tqdm.write(f'Current total loss: {total_running_loss.item()}')

            batch_total_loss.backward()
            optimizer.step()

            del X, Y_pred, Y_true
            # break  # for debuging

    if not evaluation:
        scheduler.step()

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

    return optimizer, fine_tuner, accuracy, f1


def fine_tune_combined_model(fine_tuner: torch.nn.Module,
                             device: torch.device,
                             loaders: typing.Dict[str, torch.utils.data.DataLoader],
                             config):
    fine_tuner.to(device)
    fine_tuner.train()

    max_f1_score = 0
    optimizer = torch.optim.Adam(params=fine_tuner.parameters(),
                                 lr=config.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20)
    best_fine_tuner = copy.deepcopy(fine_tuner)

    all_on_device = True
    for name, param in fine_tuner.named_parameters():
        if param.device != device:
            all_on_device = False
            print(f"Parameter '{name}' is on {param.device} instead of {device}")

    if all_on_device:
        print(f"All parameters are on {device}")

    print('#' * 100 + '\n')

    for epoch in range(config.num_epochs):
        with ((context_handlers.TimeWrapper())):
            print('#' * 50 + f'train epoch {epoch}' + '#' * 50)
            optimizer, fine_tuner, train_accuracy, train_f1 = \
                batch_learning_and_evaluating(loaders=loaders['train'],
                                              device=device,
                                              optimizer=optimizer,
                                              fine_tuner=fine_tuner,
                                              scheduler=scheduler)
            # Testing
            print('#' * 50 + f'test epoch {epoch}' + '#' * 50)
            optimizer, fine_tuner, test_accuracy, test_f1 = \
                batch_learning_and_evaluating(loaders=loaders['test'],
                                              device=device,
                                              optimizer=optimizer,
                                              fine_tuner=fine_tuner,
                                              scheduler=scheduler,
                                              evaluation=True)

            if max_f1_score < test_f1:
                max_f1_score = test_f1
                best_fine_tuner = fine_tuner

    if config.save_files:
        torch.save(best_fine_tuner.state_dict(),
                   f"save_models/{best_fine_tuner}_lr{config.lr}_{config.loss}_"
                   f"{config.num_epochs}_{config.additional_saving_info}.pth")

    print('#' * 100)

    return best_fine_tuner


def run_combined_fine_tuning_pipeline(config):
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
    parser = argparse.ArgumentParser(description="Fine-tuning pipeline")
    parser.add_argument('--data_location', type=str, help='Path to the data location',
                        default='critical_classification/dashcam_video/original_video/')
    # Add more arguments as needed

    args = parser.parse_args()
    config = Config()
    config.print_config()
    config.data_location = args.data_location
    run_combined_fine_tuning_pipeline(config)
