"""
Script for multiple runs of experiments.

For monolingual experiments run:
    # Train
    python dmrst_parser/multiple_runs.py --corpus "$CORPUS" --lang "$LANG" --model_type "$TYPE" train
    # Evaluation
    python dmrst_parser/multiple_runs.py --corpus "$CORPUS" --lang "$LANG" --model_type "$TYPE" evaluate

For multilingual experiments:
    # Train
    python dmrst_parser/multiple_runs.py --corpus "$CORPUS" --lang "$LANG" --model_type "$TYPE" train_mixed --mixed 100
"""

import os
import sys
import subprocess
import fire
import json
from glob import glob


class MultipleRunnerGeneral:
    def __init__(self,
                 corpus: str,
                 lang: str,
                 model_type: str,
                 transformer_name: str = 'xlm-roberta-large',
                 emb_size: int = 1024,
                 cuda_device: int = 0,
                 resume_training: bool = False):
        """
        :param corpus: (str)  - 'GUM' or 'RST-DT'
        :param lang: (str)  - 'en' or 'ru'
        :param model_type: (str)  - one of {'default', '+tony', '+tony+trainable_edus', '+tony+trainable_edus+bimpm'}
        :param transformer_name: (str)  - model name or path to the pretrained LM
        :param emb_size: (int)  - LM encodings size
        :param cuda_device: (int)  - number of cuda device
        :param resume_training: (bool)  - whether to rewrite previous saves
        """
        self.corpus = corpus
        self.lang = lang
        self.model_type = model_type
        self.transformer_name = transformer_name
        self.emb_size = emb_size
        self.cuda_device = cuda_device
        self.resume_training = resume_training

        if 'roberta-xl' in self.transformer_name:
            self.freeze_first_n = 28
        else:
            self.freeze_first_n = 0

    def _general_parameters(self):
        overrides = {
            'corpus': self.corpus,
            'lang': self.lang,
            'data_manager_path': f'data/data_manager_{self.corpus.lower()}.pickle',
            'second_lang_fold': 0,
            'second_lang_fraction': 0,
            'transformer_name': self.transformer_name,  # LM name
            'emb_size': self.emb_size,  # LM embedding size
            'freeze_first_n': self.freeze_first_n,  # LM fine-tuning configuration
            'use_crf': 'true',  # ToNy (LSTM-CRF)
            'use_log_crf': 'false',  # [Optional] Logits restriction for ToNy
            'token_bilstm_hidden': 300,  # BiMPM representation hidden size
            'dwa_bs': 12,  # Batch size for DWA computation
            'grad_clipping_value': 10.0,
            'combine_batches': 'false',  # [Optional] Combine batches w/smallest trees (for normalization when bs=1)
            'lr': 0.0001,
            'cuda_device': self.cuda_device,
            'save_path': 'saves/',
        }

        if self.corpus == 'RST-DT':
            overrides.update({
                'batch_size': 2,
                'cross_validation': 'true',
                'hidden_size': 1024,
                'token_bilstm_hidden': 200,
            })

        elif self.corpus == 'GUM':
            overrides.update({
                'batch_size': 1,
                'cross_validation': 'false',
                'hidden_size': 1024,
            })

        elif self.corpus == 'RuRSTB':
            overrides.update({
                'lang': 'ru',
                'batch_size': 6,
                'cross_validation': 'false',
                'hidden_size': 768,
                'dwa_bs': 24,
            })

        # Default parameters
        overrides.update({
            'segmenter_type': 'linear',
            'segmenter_hidden_dim': overrides['hidden_size'],
            'segmenter_dropout': 0.4,
            'lstm_bidirectional': 'true',
            'if_edu_start_loss': 'true',
            'edu_encoding_kind': 'avg',
            'rel_classification_kind': 'default',
            'use_discriminator': 'false',
            'discriminator_warmup': 0,
        })

        if self.model_type != 'default':
            types = self.model_type.split('+')

            if 'tony' in types:
                overrides['segmenter_type'] = 'tony'
                overrides['if_edu_start_loss'] = 'false'
                overrides['segmenter_hidden_dim'] = 200

                if self.corpus == 'RuRSTB':
                    overrides['segmenter_dropout'] = 0.5

            if 'no_crf' in types:
                overrides['use_crf'] = 'false'

            if 'trainable_edus' in types:
                overrides['edu_encoding_kind'] = 'trainable'

            if 'gru_edus' in types:
                overrides['edu_encoding_kind'] = 'gru'

            if 'bigru_edus' in types:
                overrides['edu_encoding_kind'] = 'bigru'

            if 'bilstm_edus' in types:
                overrides['edu_encoding_kind'] = 'bilstm'

            if 'trainable_dus' in types:
                overrides['du_encoding_kind'] = 'trainable'

            if 'bimpm' in types:
                overrides['rel_classification_kind'] = 'with_bimpm'

            if 'al' in types:
                overrides['use_discriminator'] = 'true'
                overrides['discriminator_warmup'] = 3

        return overrides

    def _get_variants(self):
        if self.corpus == 'RST-DT':
            return range(5)  # For there is no fixed dev in RST-DT, we select validation from the train randomly

        elif self.corpus in ['GUM', 'RuRSTB']:
            return range(40, 45)  # There is a fixed split, we just change the nn random seed

    def train(self):
        general_parameters = self._general_parameters()
        for run in self._get_variants():
            if self.corpus == 'RST-DT':
                general_parameters['foldnum'] = run
                general_parameters['seed'] = 43

            elif self.corpus in ['GUM', 'RuRSTB']:
                general_parameters['foldnum'] = 0
                general_parameters['seed'] = run

            general_parameters['run_name'] = f'{self.lang}_{self.corpus}_{self.model_type}_{run}'
            for key, value in general_parameters.items():
                general_parameters[key] = str(value)

            if self.resume_training:
                if os.path.isfile(os.path.join('saves', general_parameters['run_name'], 'best_metrics.json')):
                    continue

            p = subprocess.Popen(
                ['python', 'dmrst_parser/trainer.py',
                 'configs/general_config.jsonnet', json.dumps(general_parameters)],
                stdout=sys.stdout, stderr=sys.stderr
            )
            p.wait()

    def evaluate(self):
        results = {
            'e2e_test_f1_full': [],
            'e2e_test_f1_nuc': [],
            'e2e_test_f1_rel': [],
            'e2e_test_f1_seg': [],
            'e2e_test_f1_span': [],
            'gs_test_f1_full': [],
            'gs_test_f1_nuc': [],
            'gs_test_f1_rel': [],
            'gs_test_f1_span': []
        }
        for run in self._get_variants():
            run_name = f'{self.lang}_{self.corpus}_{self.model_type}_{run}'
            run_path = os.path.join('saves', run_name)
            try:
                all_metrics = glob(os.path.join(run_path, 'metrics_epoch_*.json'))
                best_epoch = sorted([int(os.path.basename(metrics)[14:-5]) for metrics in all_metrics])[-1]
                best_dev_metrics = json.load(open(os.path.join(run_path, f'metrics_epoch_{best_epoch}.json')))
                for key in results:
                    results[key].append(best_dev_metrics[key])
            except:
                print(f'Run {run} is missing.')

        with open(f'{self.lang}_{self.corpus}_{self.model_type}_all_res.json', 'w') as f:
            json.dump(results, f)

    def train_mixed(self, mixed: int):
        """ Running training with second language injection of ``mixed`` % """

        save_path = 'saves_mixed'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        general_parameters = self._general_parameters()
        general_parameters['save_path'] = save_path

        for run in range(5):
            assert self.corpus == 'GUM'  # Cross-lingual training is only for parallel corpus

            general_parameters['foldnum'] = 0
            general_parameters['seed'] = 40
            general_parameters.update({
                'second_lang_fold': run,
                'second_lang_fraction': mixed,
            })

            general_parameters['run_name'] = f'{self.lang}_{mixed}perc_{run}'
            for key, value in general_parameters.items():
                general_parameters[key] = str(value)

            if self.resume_training:
                if os.path.isfile(os.path.join(save_path, general_parameters['run_name'], 'best_metrics.json')):
                    continue

            p = subprocess.Popen(
                ['python', 'dmrst_parser/trainer.py',
                 'configs/general_config.jsonnet', json.dumps(general_parameters)],
                stdout=sys.stdout, stderr=sys.stderr
            )
            p.wait()


if __name__ == '__main__':
    fire.Fire(MultipleRunnerGeneral)
