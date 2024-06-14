import copy
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.utils.data
import typing
from tqdm import tqdm

sys.path.append(os.getcwd())

from critical_classification.src import utils, context_handlers, backbone_pipeline
from critical_classification import config

if utils.is_local():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def fine_tune_combined_model(fine_tuner: torch.nn.Module,
                             device: torch.device,
                             loaders: typing.Dict[str, torch.utils.data.DataLoader],
                             config,
                             evaluation: bool = False):
    fine_tuner.to(device)
    fine_tuner.train()
    train_loader = loaders['train'] if not evaluation else loaders['test']
    num_batches = len(train_loader)

    prediction = []
    ground_truth = []
    max_f1_score = 0

    optimizer = torch.optim.Adam(params=fine_tuner.parameters(),
                                 lr=config.lr)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
    #                                             step_size=scheduler_step_size,
    #                                             gamma=scheduler_gamma)
    best_fine_tuner = copy.deepcopy(fine_tuner)

    print('#' * 100 + '\n')

    for epoch in range(config.num_epochs):
        with (context_handlers.TimeWrapper()):
            total_running_loss = torch.Tensor([0.0]).to(device)
            batches = tqdm(enumerate(train_loader, 0),
                           total=num_batches)

            for batch_num, batch in batches:
                with context_handlers.ClearCache(device=device):
                    X, Y_true, time = [b.to(device) for b in batch]

                    Y_pred = fine_tuner(X)

                    prediction.append(torch.squeeze(torch.where(Y_pred > 0.5, 1, 0)))
                    ground_truth.append(Y_true)

                    if evaluation:
                        del X, Y_pred, Y_true
                        continue

                    criterion = torch.nn.BCEWithLogitsLoss()
                    batch_total_loss = criterion(Y_pred, torch.unsqueeze(Y_true, dim=1).float())
                    total_running_loss += batch_total_loss.item() / len(batches)
                    print(f'Current total loss: {total_running_loss.item()}')
                    batch_total_loss.backward()
                    optimizer.step()

                del X, Y_pred, Y_true

            print(utils.blue_text(
                f'label use and count: '
                f'{np.unique(np.array(ground_truth), return_counts=True)}'))

            print(f'accuracy: {accuracy_score(ground_truth, prediction)}')
            print(f'f1: {f1_score(ground_truth, prediction)}')

            if max_f1_score < f1_score(ground_truth, prediction):
                max_f1_score = f1_score(ground_truth, prediction)
                best_fine_tuner = fine_tuner

    if config.save_files:
        torch.save(best_fine_tuner.state_dict(),
                   f"save_models/{best_fine_tuner}_lr{config.lr}_{config.loss}_"
                   f"{config.num_epochs}_{config.additional_info}.pth")

    print('#' * 100)

    return best_fine_tuner


def run_combined_fine_tuning_pipeline(config,
                                      debug: bool = utils.is_debug_mode()):
    fine_tuner, loaders, device = (
        backbone_pipeline.initiate(metadata=config.metadata,
                                   batch_size=config.batch_size,
                                   model_name=config.model_name,
                                   pretrained_path=config.pretrained_path,
                                   debug=debug)
    )

    best_fine_tuner = fine_tune_combined_model(
        fine_tuner=fine_tuner,
        device=device,
        loaders=loaders,
        config=config
    )
    print('#' * 100)
    fine_tune_combined_model(
        fine_tuner=best_fine_tuner,
        device=device,
        loaders=loaders,
        config=config,
        evaluation=True
    )
    print('#' * 100)


if __name__ == '__main__':
    run_combined_fine_tuning_pipeline(config)
