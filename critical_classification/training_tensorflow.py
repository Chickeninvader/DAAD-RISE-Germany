import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf
import torch.utils.data
import typing
from tqdm import tqdm
import warnings

sys.path.append(os.getcwd())

from critical_classification.src import utils, context_handlers, backbone_pipeline
from critical_classification import config

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
        if idx > 10:
            break
    pass


def batch_learning_and_evaluating(loaders,
                                  optimizer: torch.optim,
                                  fine_tuner: torch.nn.Module,
                                  evaluation: bool = False,
                                  print_info: bool = True):
    num_batches = len(loaders)
    batches = tqdm(enumerate(loaders, 0),
                   total=num_batches)

    predictions = []
    ground_truths = []
    video_name_with_time = []
    total_running_loss = 0.0
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    for batch_num, batch in batches:
        X, Y_true, video_name_with_time_batch = batch

        X = tf.convert_to_tensor(X)
        Y_true = tf.convert_to_tensor(Y_true, dtype=tf.float32)
        # Y_pred = fine_tuner(X, training=evaluation)
        with tf.GradientTape() as tape:
            Y_pred = fine_tuner(X, training=evaluation)
            assert Y_true.shape == Y_pred.shape, (f"Shape does not match, "
                                                  f"pred shape {Y_pred.shape}, true shape {Y_true.shape}")
            batch_total_loss = criterion(Y_pred, Y_true)

        # Y_pred = Y_true
        # evaluation = True

        # Convert logits to binary predictions (0 or 1)
        y_pred_binary = tf.where(Y_pred > 0.5, 1, 0)

        # Append predictions and ground truths
        predictions.append(tf.squeeze(y_pred_binary).numpy())
        ground_truths.append(Y_true.numpy())

        # Append video names with time
        video_name_with_time.extend([(os.path.basename(item[0]), item[1])
                                     for item in zip(video_name_with_time_batch[0],
                                                     video_name_with_time_batch[1])])

        if evaluation:
            del X, Y_pred, Y_true
            break  # for debug purpose
            # continue

        grads = tape.gradient(batch_total_loss, fine_tuner.trainable_weights)
        optimizer.apply_gradients(zip(grads, fine_tuner.trainable_weights))

        total_running_loss += batch_total_loss / len(batches)
        # Update progress bar with informative text (without newline)
        if batch_num % (int(num_batches / 2.5)) == 0:
            tqdm.write(f'Current total loss: {total_running_loss}')

        print(f'finish iter {batch_num}')

        del X, Y_pred, Y_true
        break  # for debug purpose

    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)

    if print_info:
        print_info_for_debug(ground_truths,
                             predictions,
                             video_name_with_time)

    accuracy = accuracy_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions)
    recall = recall_score(ground_truths, predictions)

    print(f'accuracy: {accuracy}, f1: {f1}, precision: {precision}, recall: {recall}')

    return optimizer, fine_tuner, accuracy, f1


def fine_tune_combined_model(fine_tuner: tf.keras.Model,
                             loaders: typing.Dict[str, torch.utils.data.DataLoader],
                             config):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    max_f1_score = 0

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
    #                                             step_size=scheduler_step_size,
    #                                             gamma=scheduler_gamma)
    # best_fine_tuner = tf.keras.models.clone_model(fine_tuner)

    print('#' * 100 + '\n')

    # fine_tuner.summary()

    for epoch in range(config.num_epochs):
        with ((context_handlers.TimeWrapper())):

            optimizer, fine_tuner, train_accuracy, train_f1 = \
                batch_learning_and_evaluating(loaders=loaders['train'],
                                              optimizer=optimizer,
                                              fine_tuner=fine_tuner)

            # Testing
            optimizer, fine_tuner, test_accuracy, test_f1 = \
                batch_learning_and_evaluating(loaders=loaders['test'],
                                              optimizer=optimizer,
                                              fine_tuner=fine_tuner,
                                              evaluation=True)

            if max_f1_score < test_f1:
                max_f1_score = test_f1
                print(f'best fine tuner will be update later')
                # best_fine_tuner = fine_tuner

    if config.save_files and config.model_name == 'Monocular3D':
        fine_tuner.save_binary_model_weights(
            f"critical_classification/save_models/{config.model_name}_lr{config.lr}_{config.loss}_"
            f"{config.num_epochs}_{config.additional_saving_info}")

    print('#' * 100)

    return fine_tuner


def run_combined_fine_tuning_pipeline(config):
    fine_tuner, loaders, device = (
        backbone_pipeline.initiate(metadata=config.metadata,
                                   batch_size=config.video_batch_size,
                                   model_name=config.model_name,
                                   pretrained_path=config.pretrained_path,
                                   img_representation=config.img_representation,
                                   sample_duration=config.duration,
                                   img_size=config.img_size)
    )

    fine_tune_combined_model(
        fine_tuner=fine_tuner,
        loaders=loaders,
        config=config
    )
    print('#' * 100)


if __name__ == '__main__':
    run_combined_fine_tuning_pipeline(config)
