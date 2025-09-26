import torch
import torch.nn as nn
import torch.nn.functional as F

from .parsing_net import ParsingNet
from .data import nucs_and_rels
from .metrics import get_batch_metrics


class ParsingNetBottomUp(ParsingNet):
    """Bottom-up transition-based parser.

    This module reuses the encoder, segmenters and relation classifiers from
    :class:`ParsingNet` but replaces the top-down pointer-network parser with a
    simple bottom-up shift-reduce style parser. The implementation follows the
    transition system described in `Yu et al., 2020` (CoDI; ACL 2020).
    """

    def __init__(self, *args, pair_hidden_size=256, **kwargs):
        super().__init__(*args, **kwargs)

        # Remove top-down specific modules
        del self.decoder
        del self.pointer

        # Classifier that scores SHIFT vs REDUCE decisions.
        self.action_scorer = nn.Sequential(
            nn.Linear(self.hidden_size * 3, pair_hidden_size, bias=True,
                      device=self._cuda_device),
            nn.ReLU(),
            nn.Linear(pair_hidden_size, 2, bias=True, device=self._cuda_device)
        )


    class _Node:
        def __init__(self, start, end, split, label, left=None, right=None):
            self.start = start
            self.end = end
            self.split = split
            self.label = label
            self.left = left
            self.right = right

    def _build_tree(self, parsing_index, label_index, edu_number, start=0, end=None, idx=0):
        """Reconstructs the gold tree from pre-order traversal."""
        if end is None:
            end = edu_number - 1
        if start == end:
            return self._Node(start, end, None, None), idx

        split = parsing_index[idx]
        label = label_index[idx]
        idx += 1
        left, idx = self._build_tree(parsing_index, label_index, edu_number,
                                     start=start, end=split, idx=idx)
        right, idx = self._build_tree(parsing_index, label_index, edu_number,
                                      start=split + 1, end=end, idx=idx)
        return self._Node(start, end, split, label, left, right), idx

    def _postorder(self, node):
        if node.left is None:
            return []
        ops = []
        ops.extend(self._postorder(node.left))
        ops.extend(self._postorder(node.right))
        ops.append(node)
        return ops

    def _actions(self, node):
        """Return gold transition sequence in postorder."""
        if node.left is None:
            return [("SHIFT", None)]
        actions = []
        actions.extend(self._actions(node.left))
        actions.extend(self._actions(node.right))
        actions.append(("REDUCE", node.label))
        return actions


    def _span_embedding(self, encodings, start, end):
        return torch.mean(encodings[start:end + 1], dim=0, keepdim=True)


    def training_loss(self, input_texts, sent_breaks, entity_ids, entity_position_ids,
                      edu_breaks, label_index, parsing_index, decoder_input_index, dataset_index):
        encoder_outputs, _, total_edu_loss, _ = self.encoder(
            input_texts, entity_ids, entity_position_ids, edu_breaks,
            sent_breaks=sent_breaks, dataset_index=dataset_index)

        label_loss_functions = [nn.NLLLoss(weight=w) for w in self.label_weights]
        struct_loss_fn = nn.NLLLoss()

        loss_label_batch = 0.0
        loss_struct_batch = 0.0
        count_label = 0
        count_struct = 0

        batch_size = len(label_index)
        zero = torch.zeros(1, self.hidden_size, device=self._cuda_device)
        for i in range(batch_size):
            n_edus = len(edu_breaks[i])
            if n_edus == 1:
                continue
            cur_enc = encoder_outputs[i][:n_edus]
            tree, _ = self._build_tree(parsing_index[i], label_index[i], n_edus)
            actions = self._actions(tree)

            stack = []
            buffer = [(j, j, self._span_embedding(cur_enc, j, j))
                      for j in range(n_edus)]

            for act, lbl in actions:
                top1 = stack[-1][2] if len(stack) >= 1 else zero
                top2 = stack[-2][2] if len(stack) >= 2 else zero
                next_buf = buffer[0][2] if buffer else zero
                feat = torch.cat([top2, top1, next_buf], dim=-1)
                log_probs = F.log_softmax(self.action_scorer(feat), dim=-1)
                gold = torch.tensor([0 if act == "SHIFT" else 1], device=self._cuda_device)
                loss_struct_batch += struct_loss_fn(log_probs, gold)
                count_struct += 1

                if act == "SHIFT":
                    stack.append(buffer.pop(0))
                else:  # REDUCE
                    right = stack.pop()
                    left = stack.pop()
                    input_left = left[2]
                    input_right = right[2]
                    cls_idx = self.dataset2classifier[dataset_index[i]]
                    mask = self.dataset_masks[cls_idx]
                    _, log_rel_weights = self.label_classifier(input_left, input_right, mask=mask)
                    loss_label_batch += label_loss_functions[cls_idx](
                        log_rel_weights, torch.tensor([lbl], device=self._cuda_device))
                    count_label += 1
                    new_emb = (input_left + input_right) / 2
                    new_span = (left[0], right[1], new_emb)
                    stack.append(new_span)

        loss_label_batch /= max(1, count_label)
        loss_struct_batch /= max(1, count_struct)

        return loss_struct_batch, loss_label_batch, total_edu_loss


    def testing_loss(self, input_sentence, input_sent_breaks, input_entity_ids, input_entity_position_ids,
                     input_edu_breaks, label_index, parsing_index, generate_tree, use_pred_segmentation, dataset_index):
        encoder_outputs, _, _, _ = self.encoder(
            input_sentence, input_entity_ids, input_entity_position_ids, input_edu_breaks,
            sent_breaks=input_sent_breaks, is_test=True, dataset_index=dataset_index)

        span_batch = []
        label_batch = []
        tree_batch = []

        batch_size = len(input_edu_breaks)
        zero = torch.zeros(1, self.hidden_size, device=self._cuda_device)
        for i in range(batch_size):
            n_edus = len(input_edu_breaks[i])
            if n_edus == 1:
                tree_batch.append([])
                label_batch.append([])
                span_batch.append([])
                continue

            cur_enc = encoder_outputs[i][:n_edus]
            stack = []
            buffer = [(j, j, self._span_embedding(cur_enc, j, j))
                      for j in range(n_edus)]

            cur_tree = []
            cur_labels = []
            cur_span_str = ''

            while buffer or len(stack) > 1:
                top1 = stack[-1][2] if len(stack) >= 1 else zero
                top2 = stack[-2][2] if len(stack) >= 2 else zero
                next_buf = buffer[0][2] if buffer else zero
                feat = torch.cat([top2, top1, next_buf], dim=-1)
                action_scores = self.action_scorer(feat)
                act = int(torch.argmax(action_scores))  # 0=SHIFT, 1=REDUCE

                if act == 0 and buffer:
                    stack.append(buffer.pop(0))
                else:
                    if len(stack) < 2:
                        # force shift if not enough items
                        if buffer:
                            stack.append(buffer.pop(0))
                            continue
                        else:
                            break
                    right = stack.pop()
                    left = stack.pop()
                    input_left = left[2]
                    input_right = right[2]
                    cls_idx = self.dataset2classifier[dataset_index[i]]
                    mask = self.dataset_masks[cls_idx]
                    relation_weights, _ = self.label_classifier(input_left, input_right, mask=mask)
                    label_idx = int(torch.argmax(relation_weights))
                    cur_labels.append(label_idx)
                    cur_tree.append(left[1])

                    if generate_tree:
                        nuc_l, nuc_r, rel_l, rel_r = nucs_and_rels(label_idx, self.relation_tables[cls_idx])
                        span_s = f"({left[0] + 1}:{nuc_l}={rel_l}:{left[1] + 1},{right[0] + 1}:{nuc_r}={rel_r}:{right[1] + 1})"
                        cur_span_str += ' ' + span_s

                    new_emb = (input_left + input_right) / 2
                    new_span = (left[0], right[1], new_emb)
                    stack.append(new_span)

            tree_batch.append(cur_tree)
            label_batch.append(cur_labels)
            span_batch.append([cur_span_str.strip()])

        merged_label_gold = []
        for tmp_i in label_index:
            merged_label_gold.extend(tmp_i)
        merged_label_pred = []
        for tmp_i in label_batch:
            merged_label_pred.extend(tmp_i)

        loss_tree = 0.0
        loss_label = 0.0

        return loss_tree, loss_label, span_batch, (merged_label_gold, merged_label_pred), input_edu_breaks
