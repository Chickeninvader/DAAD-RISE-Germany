import copy
import os
from src import utils, models, context_handlers, backbone_pipeline
import numpy as np
import torch
import torch.utils.data
import typing
from tqdm import tqdm
from critical_classification import config

if utils.is_local():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def fine_tune_combined_model(fine_tuner: models.FineTuner,
                             device: torch.device,
                             loaders: typing.Dict[str, torch.utils.data.DataLoader],
                             num_epochs: int,
                             save_files: bool = True,
                             additional_info: str = None,
                             evaluation: bool = False):
    fine_tuner.to(device)
    fine_tuner.train()
    train_loader = loaders['train'] if not evaluation else loaders['test']
    num_batches = len(train_loader)

    prediction = []
    ground_truth = []

    optimizer = torch.optim.Adam(params=fine_tuner.parameters(),
                                 lr=config.lr)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
    #                                             step_size=scheduler_step_size,
    #                                             gamma=scheduler_gamma)
    best_fine_tuner = copy.deepcopy(fine_tuner)

    print('#' * 100 + '\n')

    for epoch in range(num_epochs):
        with (context_handlers.TimeWrapper()):
            total_running_loss = torch.Tensor([0.0]).to(device)
            batches = tqdm(enumerate(train_loader, 0),
                           total=num_batches)

            for batch_num, batch in batches:
                with context_handlers.ClearCache(device=device):
                    X, Y_true = [b.to(device) for b in batch]

                    Y_pred = fine_tuner(X)

                    prediction.append(1 if Y_pred > 0.5 else 0)
                    ground_truth.append(Y_true)

                    if evaluation:
                        continue

                    criterion = torch.nn.BCEWithLogitsLoss()
                    batch_total_loss = criterion(Y_pred, Y_true.float())
                    total_running_loss += batch_total_loss.item() / len(batches)
                    print(f'Current total loss: {total_running_loss.item()}')
                    batch_total_loss.backward()
                    optimizer.step()

                del X, Y_pred, Y_true

            print(utils.blue_text(
                f'label use and count: '
                f'{np.unique(np.array(ground_truth), return_counts=True)}'))

    if save_files:
        torch.save(best_fine_tuner.state_dict(),
                   f"models/{best_fine_tuner}_lr{config.lr}_{config.loss}_{config.num_epochs}_{additional_info}.pth")
    print('#' * 100)


def run_combined_fine_tuning_pipeline(model_name: str,
                                      metadata,
                                      pretrained_path: str = None,
                                      save_files: bool = True,
                                      debug: bool = utils.is_debug_mode(),
                                      additional_info: str = None,):
    fine_tuner, loaders, device = (
        backbone_pipeline.initiate(metadata=metadata,
                                   batch_size=config.batch_size,
                                   model_name=model_name,
                                   pretrained_path=pretrained_path,
                                   debug=debug)
    )

    fine_tune_combined_model(
        fine_tuner=fine_tuner,
        device=device,
        loaders=loaders,
        num_epochs=config.num_epochs,
        save_files=save_files,
        additional_info=additional_info
    )
    print('#' * 100)
    fine_tune_combined_model(
        fine_tuner=fine_tuner,
        device=device,
        loaders=loaders,
        num_epochs=config.num_epochs,
        save_files=save_files,
        additional_info=additional_info,
        evaluation=True
    )
    print('#' * 100)


if __name__ == '__main__':

    run_combined_fine_tuning_pipeline(model_name=config.model_name,
                                      metadata=config.metadata,
                                      additional_info=config.additional_saving_info)
