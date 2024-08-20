import json
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import wandb
from tqdm import tqdm

# from src import keys
from .data import Data
from .metrics import get_micro_metrics, get_macro_metrics

# os.environ["WANDB_API_KEY"] = keys.WANDB_KEY

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.bool_):
#             return bool(obj)
#         if isinstance(obj, (np.floating, np.complexfloating)):
#             return float(obj)
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         if isinstance(obj, np.string_):
#             return str(obj)
#         if isinstance(obj, (datetime, date)):
#             return obj.isoformat()
#         if isinstance(obj, timedelta):
#             return str(obj)
#         return super(NpEncoder, self).default(obj)


class TrainingManager:
    """
    Manages training for a single model.
    """

    def __init__(self, model, train_data, dev_data, test_data,
                 batch_size, eval_size, epochs,
                 lr, transformer_lr_multiplier, lr_decay_epoch, lr_decay,
                 weight_decay, grad_norm, grad_clipping_value,
                 patience, use_micro_f1, use_dwa_loss, dwa_bs,
                 save_dir, use_amp, warmup_epochs=0, combine_batches=False,
                 use_discriminator=False, discriminator_warmup=0, discriminator_alpha=1.,
                 project=None, run_name=None, config=None):
        """
        Initializes the TrainingManager.

        Args:
            model: The pre-iniialized ParsingNet model.
            train_data: Training data.
            dev_data: Development data.
            test_data: Test data.
            batch_size: Batch size for training.
            eval_size: Evaluation batch size.
            epochs: Max number of training epochs.
            lr: Learning rate.
            transformer_lr_multiplier: Multiplier for transformer layers' learning rate.
            lr_decay_epoch: Epoch at which learning rate starts decaying.
            lr_decay: Learning rate decay factor.
            weight_decay: Weight decay for regularization.
            grad_norm: Gradient norm clipping value.
            grad_clipping_value: Gradient clipping value.
            patience: Patience for early stopping.
            use_micro_f1: Whether to use micro F1 score.
            use_dwa_loss: Whether to use dynamic weighting average loss.
            dwa_bs: Batch size for DWA loss computation.
            save_dir: Directory to save model checkpoints.
            use_amp: Whether to use automatic mixed precision.
            warmup_epochs: Number of warmup epochs.
            combine_batches: Whether to combine batches.
            use_discriminator: Whether to use discriminator.
            discriminator_warmup: Warmup epochs for discriminator.
            discriminator_alpha: Alpha value for discriminator.
            project: WandB project name.
            run_name: WandB run name.
            config: Additional configuration.
        """
        self.model = model
        self.train_data = self._merge_data(train_data)
        self.dev_data = self._merge_data(dev_data)
        self.test_data = self._merge_data(test_data)
        self.test_by_dataset = test_data

        self.batch_size = batch_size
        self.combine_batches = combine_batches if self.batch_size == 1 else False
        self.eval_size = eval_size
        self.epochs = epochs
        self.lr = lr
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.grad_norm = grad_norm
        self.grad_clipping_value = grad_clipping_value
        self.patience = patience
        self.use_micro_f1 = use_micro_f1
        self.use_dwa_loss = use_dwa_loss
        self.dwa_bs = dwa_bs // batch_size
        self.warmup_epochs = warmup_epochs
        self.use_discriminator = use_discriminator
        self.discriminator_warmup = discriminator_warmup
        self.discriminator_alpha = discriminator_alpha
        self.use_amp = use_amp

        # self.run = wandb.init(project=project, name=run_name, config=config)
        # run_name = self.run.name if self.run.name else 'tmp'
        self.save_dir = Path(os.path.join(save_dir, run_name))
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(parents=True)
        with open(self.save_dir.joinpath('config.json'), 'w') as f:
            json.dump(config, f)

        if transformer_lr_multiplier:
            transformer_parameters_ids = list(map(id, self.model.encoder.transformer.parameters()))
            other_parameters = filter(lambda p: id(p) not in transformer_parameters_ids, self.model.parameters())
            transformer_parameters = filter(lambda p: id(p) in transformer_parameters_ids, self.model.parameters())

            # segmenter_parameters_ids = list(map(id, self.model.segmenter.parameters()))
            # parser_parameters = filter(lambda p: id(p) not in segmenter_parameters_ids, other_parameters)
            # segmenter_parameters = filter(lambda p: id(p) in segmenter_parameters_ids, other_parameters)

            self.optimizer = optim.AdamW([
                {'params': other_parameters, 'lr': self.lr},
                {'params': transformer_parameters, 'lr': self.lr * transformer_lr_multiplier},
                # {'params': segmenter_parameters, 'lr': self.lr},
                # {'params': parser_parameters, 'lr': self.lr}
            ], weight_decay=self.weight_decay)

        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )

        # Schedule LR based on e2e_val_f1_full
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', min_lr=1e-8,
                                                                 factor=0.5, patience=2, verbose=True, )

        self._cuda_cache_dump_frequency = .1

    def _adjust_lr(self, epoch):
        """
        Default DMRST method for linear learning rate adjustment (deprecated).
        """

        if (epoch % self.lr_decay_epoch == 0) and (epoch != 0):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * self.lr_decay, 1e-9)

    def train(self):
        """
        Trains the model.

        Returns:
            dict: best_metrics
        """

        self.best_epoch = 0
        best_metrics = {
            'epoch': 0,
            'e2e_val_f1_span': 0,
        }
        patience_counter = 0

        label_loss_iter_list = []
        tree_loss_iter_list = []
        edu_loss_iter_list = []

        # w_label, w_tree, w_edu = None, None, None
        dwa_T = 2.0

        batches = self._get_batches(self.train_data, self.batch_size)
        for epoch in range(self.epochs):

            logger.info(f'Epoch {epoch + 1}/{self.epochs}')

            label_loss_iter_list, tree_loss_iter_list, edu_loss_iter_list, dwa_T = self._train_epoch(
                epoch, batches, label_loss_iter_list, tree_loss_iter_list, edu_loss_iter_list, dwa_T)

            metrics_dev, metrics_test, metrics_gs_dev, metrics_gs_test = self._eval()
            metrics_all = {
                'epoch': epoch,
                'step': (epoch + 1) * len(batches),
                'gs_val_f1_span': metrics_gs_dev['f1_span'],
                'gs_val_f1_nuc': metrics_gs_dev['f1_nuclearity'],
                'gs_val_f1_rel': metrics_gs_dev['f1_relation'],
                'gs_val_f1_full': metrics_gs_dev['f1_full'],
                'gs_test_f1_span': metrics_gs_test['f1_span'],
                'gs_test_f1_nuc': metrics_gs_test['f1_nuclearity'],
                'gs_test_f1_rel': metrics_gs_test['f1_relation'],
                'gs_test_f1_full': metrics_gs_test['f1_full'],
                'e2e_val_f1_seg': metrics_dev['f1_seg'],
                'e2e_val_f1_span': metrics_dev['f1_span'],
                'e2e_val_f1_nuc': metrics_dev['f1_nuclearity'],
                'e2e_val_f1_rel': metrics_dev['f1_relation'],
                'e2e_val_f1_full': metrics_dev['f1_full'],
                'e2e_test_f1_seg': metrics_test['f1_seg'],
                'e2e_test_f1_span': metrics_test['f1_span'],
                'e2e_test_f1_nuc': metrics_test['f1_nuclearity'],
                'e2e_test_f1_rel': metrics_test['f1_relation'],
                'e2e_test_f1_full': metrics_test['f1_full'],
            }

            self.lr_scheduler.step(metrics_all['e2e_val_f1_full'])

            # log metrics
            # wandb.log(metrics_all)

            if metrics_all['e2e_test_f1_full'] == 0:
                shutil.rmtree(self.save_dir)
                raise RuntimeError('Zero metrics. Stopping the loop.')

            # save best model
            if metrics_all['e2e_val_f1_span'] > best_metrics['e2e_val_f1_span']:
                print(f'New best result! Saving the model for epoch {epoch}.')
                best_metrics = metrics_all
                self.best_epoch = epoch
                patience_counter = 0
                self._save_model(epoch, metrics_all)
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch}')
                break

        metric_path = self.save_dir / f'best_metrics.json'
        with open(metric_path, 'w') as f:
            m = {k: float(v) for k, v in best_metrics.items()}
            json.dump(m, f, sort_keys=True, indent=4)

        return best_metrics

    def _train_epoch(self, epoch, batches, label_loss_iter_list, tree_loss_iter_list, edu_loss_iter_list, dwa_T):
        """
        Trains the model for one epoch.

        Args:
            epoch: Current epoch number.
            batches: List of training batches.
            label_loss_iter_list: List to store label loss values.
            tree_loss_iter_list: List to store tree loss values.
            edu_loss_iter_list: List to store EDU loss values.
            dwa_T: Temperature for dynamic weighting average.

        Returns:
            Updated label_loss_iter_list, tree_loss_iter_list, edu_loss_iter_list, and dwa_T.
        """

        if self.use_discriminator and epoch == self.discriminator_warmup:
            print(f'Turning on the discriminator')
            self.model.turn_on_discriminator()

        if self.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # self._adjust_lr(epoch)
        self.model.train()

        pbar = tqdm(enumerate(batches), desc=f'Epoch {epoch + 1}/{self.epochs}', total=len(batches))
        for i, batch in pbar:

            batch_input_sentences, batch_sent_breaks, batch_entity_ids, batch_entity_position_ids, \
                batch_edu_breaks, batch_decoder_inputs, batch_relation_labels, \
                batch_parsing_breaks, batch_golden_metrics, batch_dataset_index = batch

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    losses = self.model.training_loss(
                        batch_input_sentences, batch_sent_breaks, batch_entity_ids, batch_entity_position_ids,
                        batch_edu_breaks, batch_relation_labels, batch_parsing_breaks, batch_decoder_inputs,
                        batch_dataset_index)
            else:
                losses = self.model.training_loss(
                    batch_input_sentences, batch_sent_breaks, batch_entity_ids, batch_entity_position_ids,
                    batch_edu_breaks, batch_relation_labels, batch_parsing_breaks, batch_decoder_inputs,
                    batch_dataset_index)

            try:
                loss = self._final_loss(*losses[:3],
                                        label_loss_iter_list=label_loss_iter_list,
                                        tree_loss_iter_list=tree_loss_iter_list,
                                        edu_loss_iter_list=edu_loss_iter_list,
                                        dwa_T=dwa_T)
                if self.model.use_discriminator:
                    loss += losses[3] * self.discriminator_alpha
            except OverflowError:
                loss = torch.tensor(torch.inf)

            label_loss_iter_list.append(losses[0])
            tree_loss_iter_list.append(losses[1])
            edu_loss_iter_list.append(losses[2])

            max_loss_memory = self.dwa_bs * 2
            if len(label_loss_iter_list) > max_loss_memory:
                label_loss_iter_list = label_loss_iter_list[-max_loss_memory:]
                tree_loss_iter_list = tree_loss_iter_list[-max_loss_memory:]
                edu_loss_iter_list = edu_loss_iter_list[-max_loss_memory:]

            if self.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # To avoid exploding gradient
            nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.grad is not None], self.grad_norm)
            nn.utils.clip_grad_value_([p for p in self.model.parameters() if p.grad is not None],
                                      self.grad_clipping_value)

            if self.use_amp:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()

            if random.random() < self._cuda_cache_dump_frequency:
                torch.cuda.empty_cache()

            pbar.set_postfix({'loss': f'{loss.cpu().item():.4f}'})
            if self.use_discriminator and epoch >= self.discriminator_warmup:
                train_loss_AL = losses[3] * self.discriminator_alpha
            else:
                train_loss_AL = 0

            metrics_loss = {
                'step': epoch * len(batches) + i,
                'train_loss_edu': edu_loss_iter_list[-1].cpu().item(),
                'train_loss_tree': tree_loss_iter_list[-1].cpu().item(),
                'train_loss_label': label_loss_iter_list[-1].cpu().item(),
                'train_loss_AL': train_loss_AL
            }
            # wandb.log(metrics_loss)

        return label_loss_iter_list, tree_loss_iter_list, edu_loss_iter_list, dwa_T

    def _final_loss(self, loss_tree_batch, loss_label_batch, loss_segment_batch,
                    label_loss_iter_list, tree_loss_iter_list, edu_loss_iter_list, dwa_T):
        """
        Computes the DWA loss from the loss statistics.

        Args:
            loss_tree_batch: Tree loss for the current batch.
            loss_label_batch: Label loss for the current batch.
            loss_segment_batch: Segment loss for the current batch.
            label_loss_iter_list: List to store label loss values.
            tree_loss_iter_list: List to store tree loss values.
            edu_loss_iter_list: List to store EDU loss values.
            dwa_T: Temperature for dynamic weighting average.

        Returns:
            Final loss value.
        """

        def get_weight(list_losses, k):
            return torch.tensor(list_losses[-k:]).sum() / torch.tensor(list_losses[-2 * k:-k]).sum()

        if self.model.segmenter_type == 'tony' and self.model.segmenter.use_crf:
            loss_segment_batch *= 0.01

        if self.use_dwa_loss:

            if len(label_loss_iter_list) >= 2 * self.dwa_bs:
                r_label = get_weight(label_loss_iter_list, self.dwa_bs)
                r_tree = get_weight(tree_loss_iter_list, self.dwa_bs)
                r_edu = get_weight(edu_loss_iter_list, self.dwa_bs)

                total_r = math.exp(r_label / dwa_T) + math.exp(r_tree / dwa_T) + math.exp(r_edu / dwa_T)

                w_label = 3 * math.exp(r_label / dwa_T) / total_r
                w_tree = 3 * math.exp(r_tree / dwa_T) / total_r
                w_edu = 3 * math.exp(r_edu / dwa_T) / total_r

                label_loss_iter_list.append(loss_label_batch)
                tree_loss_iter_list.append(loss_tree_batch)
                edu_loss_iter_list.append(loss_segment_batch)

                # wandb.log({
                #     'w_label': w_label,
                #     'w_tree': w_tree,
                #     'w_edu': w_edu,
                # })

                return w_label * loss_label_batch + w_tree * loss_tree_batch + w_edu * loss_segment_batch

        return loss_tree_batch + loss_label_batch + loss_segment_batch

    @torch.no_grad()
    def _eval(self):
        """
        Evaluates the model on the development and test data.

        Returns:
            Tuple of metrics for development and test sets on gold and predicted EDUs.
        """

        self.model.eval()

        dev_metrics_gs = self._eval_data(self.dev_data, desc='Validation', use_pred_segmentation=False)
        test_metrics_gs = self._eval_data(self.test_data, desc='Testing', use_pred_segmentation=False)
        print(f"Dev metrics (gold segmentation): {dev_metrics_gs}")
        print(f"Test metrics (gold segmentation): {test_metrics_gs}")

        dev_metrics = self._eval_data(self.dev_data, desc='Validation')
        test_metrics = self._eval_data(self.test_data, desc='Testing')
        print(f"Dev metrics (end-to-end): {dev_metrics}")
        print(f"Test metrics (end-to-end): {test_metrics}")

        return dev_metrics, test_metrics, dev_metrics_gs, test_metrics_gs

    def _eval_data(self, data, desc, use_pred_segmentation=True):
        """
        Evaluates the model on the given data.

        Args:
            data: Data to evaluate.
            desc: Part, from {'dev', 'test'}.
            use_pred_segmentation: Whether to use predicted segmentation.

        Returns:
            Dict of metrics.
        """

        loss_tree_all = []
        loss_label_all = []
        correct_span = 0
        correct_relation = 0
        correct_nuclearity = 0
        correct_full = 0
        no_system = 0
        no_golden = 0
        no_gold_seg = 0
        no_pred_seg = 0
        no_correct_seg = 0

        # Macro
        correct_span_list = []
        correct_relation_list = []
        correct_nuclearity_list = []
        correct_full_list = []
        no_system_list = []
        no_golden_list = []

        batches = self._get_batches(data, self.eval_size)
        pbar = tqdm(enumerate(batches), desc=desc, total=len(batches))
        for i, batch in pbar:
            (loss_tree_batch, loss_label_batch), (
                correct_span_batch, correct_relation_batch, correct_nuclearity_batch, correct_full_batch,
                no_system_batch,
                no_golden_batch, correct_span_batch_list, correct_relation_batch_list, correct_nuclearity_batch_list,
                correct_full_batch_list,
                no_system_batch_list, no_golden_batch_list, segment_results_list
            ) = self.model.eval_loss(batch, use_pred_segmentation=use_pred_segmentation)

            loss_tree_all.append(loss_tree_batch)
            loss_label_all.append(loss_label_batch)

            correct_span += correct_span_batch
            correct_relation += correct_relation_batch
            correct_nuclearity += correct_nuclearity_batch
            correct_full += correct_full_batch
            no_system += no_system_batch
            no_golden += no_golden_batch
            no_gold_seg += segment_results_list[0]
            no_pred_seg += segment_results_list[1]
            no_correct_seg += segment_results_list[2]

            correct_span_list += correct_span_batch_list
            correct_nuclearity_list += correct_nuclearity_batch_list
            correct_relation_list += correct_relation_batch_list
            correct_full_list += correct_full_batch_list

            no_system_list += no_system_batch_list
            no_golden_list += no_golden_batch_list

        span_points, relation_points, nuclearity_points, f1_full, segment_points = get_micro_metrics(correct_span,
                                                                                                     correct_relation,
                                                                                                     correct_nuclearity,
                                                                                                     correct_full,
                                                                                                     no_system,
                                                                                                     no_golden,
                                                                                                     no_gold_seg,
                                                                                                     no_pred_seg,
                                                                                                     no_correct_seg)
        if not self.use_micro_f1:
            span_points, nuclearity_points, relation_points, full_points = get_macro_metrics(
                correct_span_list, correct_nuclearity_list, correct_relation_list, correct_full_list,
                no_system_list, no_golden_list)

            full_pr, full_re, f1_full = full_points

        seg_pr, seg_re, seg_f1 = segment_points
        span_pr, span_re, span_f1 = span_points
        nuc_pr, nuc_re, nuc_f1 = nuclearity_points
        rel_pr, rel_re, rel_f1 = relation_points

        metrics = {
            'loss_tree': np.mean(loss_tree_all),
            'loss_label': np.mean(loss_label_all),
            'f1_seg': seg_f1,
            'f1_span': span_f1,
            'f1_nuclearity': nuc_f1,
            'f1_relation': rel_f1,
            'f1_full': f1_full,
        }

        return metrics

    def _merge_data(self, data: list):
        all_data = {
            'input_sentences': [],
            'edu_breaks': [],
            'decoder_input': [],
            'relation_label': [],
            'parsing_breaks': [],
            'golden_metric': [],
            'dataset_index': []
        }

        for i, d in enumerate(data):
            all_data['input_sentences'] += d.input_sentences
            all_data['edu_breaks'] += d.edu_breaks
            all_data['decoder_input'] += d.decoder_input
            all_data['relation_label'] += d.relation_label
            all_data['parsing_breaks'] += d.parsing_breaks
            all_data['golden_metric'] += d.golden_metric
            all_data['dataset_index'] += [i for _ in range(len(d.input_sentences))]

        return Data(**all_data)

    def _get_batches(self, data: Data, batch_size: int):
        """
        Splits the Data object into batches of given size.

        Args:
            data: Data.
            batch_size: Batch size.

        Returns:
            List of Data batches.
        """

        input_sentences = np.array(data.input_sentences, dtype=object)
        if data.sent_breaks:
            sent_breaks = np.array(data.sent_breaks, dtype=object)
        if data.entity_ids:
            entity_ids = np.array(data.entity_ids, dtype=object)
        if data.entity_position_ids:
            entity_position_ids = np.array(data.entity_position_ids, dtype=object)

        edu_breaks = np.array(data.edu_breaks, dtype=object)
        decoder_inputs = np.array(data.decoder_input, dtype=object)
        relation_labels = np.array(data.relation_label, dtype=object)
        parsing_breaks = np.array(data.parsing_breaks, dtype=object)
        golden_metrics = np.array(data.golden_metric, dtype=object)
        dataset_index = np.array(data.dataset_index, dtype=object)

        batches = []
        indices = list(range(len(data.input_sentences)))
        random.shuffle(indices)

        for i in range(0, len(input_sentences), batch_size):
            batch_indices = indices[i:i + batch_size]

            batch_input_sentences = input_sentences[batch_indices]
            if data.sent_breaks:
                batch_sent_breaks = sent_breaks[batch_indices]
            if data.entity_ids:
                batch_entity_ids = entity_ids[batch_indices]
            if data.entity_position_ids:
                batch_entity_position_ids = entity_position_ids[batch_indices]
            batch_edu_breaks = edu_breaks[batch_indices]
            batch_decoder_inputs = decoder_inputs[batch_indices]
            batch_relation_labels = relation_labels[batch_indices]
            batch_parsing_breaks = parsing_breaks[batch_indices]
            batch_golden_metrics = golden_metrics[batch_indices]
            batch_dataset_index = dataset_index[batch_indices]

            # sort batches by input sentence length
            sorted_idxs = np.argsort([len(x) for x in batch_input_sentences])[::-1]
            batch_input_sentences = batch_input_sentences[sorted_idxs].tolist()
            batch_sent_breaks = batch_sent_breaks[sorted_idxs].tolist() if data.sent_breaks else None
            batch_entity_ids = batch_entity_ids[sorted_idxs].tolist() if data.entity_ids else None
            batch_entity_position_ids = batch_entity_position_ids[sorted_idxs].tolist() \
                if data.entity_position_ids else None
            batch_edu_breaks = batch_edu_breaks[sorted_idxs].tolist()
            batch_decoder_inputs = batch_decoder_inputs[sorted_idxs].tolist()
            batch_relation_labels = batch_relation_labels[sorted_idxs].tolist()
            batch_parsing_breaks = batch_parsing_breaks[sorted_idxs].tolist()
            batch_golden_metrics = batch_golden_metrics[sorted_idxs].tolist()
            batch_dataset_index = batch_dataset_index[sorted_idxs].tolist()

            batch = (
                batch_input_sentences,
                batch_sent_breaks,
                batch_entity_ids,
                batch_entity_position_ids,
                batch_edu_breaks,
                batch_decoder_inputs,
                batch_relation_labels,
                batch_parsing_breaks,
                batch_golden_metrics,
                batch_dataset_index
            )

            batches.append(batch)

        if self.combine_batches:
            batches = self._combine_batches(batches)

        return batches

    def _combine_batches(self, batches, min_edus_number=6):
        """
        Combines batches by appending contents of smaller batches to larger ones.

        Args:
            batches: List of batches to combine.
            min_edus_number: Minimum number of EDUs required for non-trivial batches.

        Returns:
            List of combined batches.
        """

        def merge(sample, batch):
            """ Merges two batches by appending contents of the ``sample`` at the end of the ``batch``. """
            return (batch[i] + sample[i] for i in range(len(batch)))

        result = []
        trivials_stack = []

        # Separate elaborate trees from the trivial ones
        bs = batches[:]
        for batch in sorted(bs):
            num_edus = len(batch[4][0])
            if num_edus < min_edus_number:
                trivials_stack.append(batch)
            else:
                result.append(batch)

        k = 0
        # Put the trivial batches in the batches with elaborates
        while len(trivials_stack):
            for i in range(len(result)):
                if not trivials_stack:
                    break

            k += 1
            merge(sample=trivials_stack.pop(), batch=result[i])

        return result

    def _save_model(self, epoch, metrics):
        """
        Saves the model weights and evaluation metrics.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of evaluation metrics.

        Saves:
            - Model weights to 'best_weights.pt'.
            - Evaluation metrics to 'metrics_epoch_{epoch}.json'.
        """

        model_path = self.save_dir / 'best_weights.pt'
        torch.save(self.model.state_dict(), model_path)

        metric_path = self.save_dir / f'metrics_epoch_{epoch}.json'
        with open(metric_path, 'w') as f:
            m = {k: float(v) for k, v in metrics.items()}
            json.dump(m, f, sort_keys=True, indent=4)
