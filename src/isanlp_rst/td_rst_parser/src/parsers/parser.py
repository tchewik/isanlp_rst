# -*- coding: utf-8 -*-

import os,glob
from datetime import datetime, timedelta

import torch
import torch.distributed as dist

from src.utils import Config, Dataset
from src.utils.field import Field
from src.utils.logging import init_logger, logger
from src.utils.metric import Metric
from src.utils.parallel import DistributedDataParallel as DDP
from src.utils.parallel import is_master
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR,ReduceLROnPlateau


class Parser(object):

    NAME = None
    MODEL = None

    def __init__(self, args, model, transform):
        self.args = args
        self.model = model
        self.transform = transform

    def train(self, train, dev, test,
              buckets=32,
              batch_size=5000,
              lr=8e-4,
              mu=.9,
              nu=.9,
              epsilon=1e-12,
              clip=5.0,
              decay=.75,
              decay_steps=5000,
              step_decay_factor=0.5,
              step_decay_patience = 15,
              epochs=5000,
              patience=100,
              verbose=True,
              **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        if dist.is_initialized():
            args.batch_size = args.batch_size // dist.get_world_size()
        logger.info("Load the data")
        train = Dataset(self.transform, args.train, **args)
        dev = Dataset(self.transform, args.dev)
        test = Dataset(self.transform, args.test)
        train.build(args.batch_size, args.buckets, True, dist.is_initialized())
        dev.build(args.batch_size, args.buckets)
        test.build(args.batch_size, args.buckets)
        logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")

        logger.info(f"{self.model}\n")
        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[dist.get_rank()],
                             find_unused_parameters=True)
        self.optimizer = Adam(self.model.parameters(),
                              args.lr,
                              (args.mu, args.nu),
                              args.epsilon)
        if self.args.learning_rate_schedule=='Exponential':
            self.scheduler = ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps))
        elif self.args.learning_rate_schedule=='Plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', factor=args.step_decay_factor,
                                               patience=args.step_decay_patience, verbose=True)


        elapsed = timedelta()
        best_e, best_metric = 1, Metric()
        best_metric_test = Metric()
        for epoch in range(1, args.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            loss=self._train(train.loader)
            logger.info(f"{'train:':6} - loss: {loss:.4f}")
            loss, dev_metric = self._evaluate(dev.loader)
            logger.info(f"{'dev:':6} - loss: {loss:.4f} - {dev_metric}")
            loss, test_metric = self._evaluate(test.loader)
            logger.info(f"{'test:':6} - loss: {loss:.4f} - {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                dev_metric_name = '_dev_LP_{:.2f}_LR_{:.2f}_LF_{:.2f}.pt'.format(100 * best_metric.lp,
                                                                                 100 * best_metric.lr,
                                                                                 100 * best_metric.lf)
                if is_master():
                    self.save(args.path+dev_metric_name)
                logger.info(f"{t}s elapsed (saved)\n")
                keep_last_n_checkpoint(args.path + '_dev_', n=5)
            else:
                logger.info(f"{t}s elapsed\n")
            elapsed += t
            if self.args.learning_rate_schedule == 'Plateau':
                self.scheduler.step(best_metric.score)

            # if epoch - best_e >= args.patience:
            #     break
        loss, metric = self.load(args.path)._evaluate(test.loader)

        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{'dev:':6} - {best_metric}")
        logger.info(f"{'test:':6} - {metric}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    def evaluate(self, data, buckets=8, batch_size=5000, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        logger.info("Load the data")
        dataset = Dataset(self.transform, data)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Evaluate the dataset")
        start = datetime.now()
        loss, metric = self._evaluate(dataset.loader)
        elapsed = datetime.now() - start
        logger.info(f"loss: {loss:.4f} - {metric}")
        logger.info(f"{elapsed}s elapsed, {len(dataset)/elapsed.total_seconds():.2f} Sents/s")

        return loss, metric

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.eval()
        if args.prob:
            self.transform.append(Field('probs'))

        logger.info("Load the data")
        dataset = Dataset(self.transform, data)
        dataset.build(args.batch_size, args.buckets, shuffle=False)
        logger.info(f"\n{dataset}")

        logger.info("Make predictions on the dataset")
        start = datetime.now()
        preds = self._predict(dataset.loader)
        elapsed = datetime.now() - start

        for name, value in preds.items():
            setattr(dataset, name, value)
        if pred is not None:
            logger.info(f"Save predicted results to {pred}")
            self.transform.save(pred, dataset.sentences)
        logger.info(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")

        return dataset

    def _train(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _evaluate(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _predict(self, loader):
        raise NotImplementedError

    @classmethod
    def build(cls, path, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, path, **kwargs):
        r"""
        Load data fields and model parameters from a pretrained parser.

        Args:
            path (str):
                - a string with the shortcut name of a pre-trained parser defined in supar.PRETRAINED
                  to load from cache or download, e.g., `crf-dep-en`.
                - a path to a directory containing a pre-trained parser, e.g., `./<path>/model`.
            kwargs (dict):
                A dict holding the unconsumed arguments.

        Returns:
            The loaded parser.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if os.path.exists(path):
            state = torch.load(path, map_location=args.device)

        args = state['args'].update(args)
        args.device = 'cpu'

        model = cls.MODEL(**args)

        # print(cls.WORD.embed)
        # model.load_pretrained(cls.WORD.embed).to(args.device)
        # parser = cls.load(**args)
        # parser.model = cls.MODEL(**parser.args)
        # parser.model.load_pretrained(parser.WORD.embed).to(args.device)
        # print(parser.WORD.embed)

        # parser.model.to(args.device)

        # if os.path.exists(path):  # and not args.build:
        #     parser = cls.load(**args)
        #     parser.model = cls.MODEL(**parser.args)
        #     parser.model.load_pretrained(parser.WORD.embed).to(args.device)
        #     return parser

        # parser = cls.load(**args)

        # print(parser.CHART)
        # print(vars(parser.CHART.vocab))

        transform = state['transform']

        if state['pretrained']:
            model.load_pretrained(state['pretrained']).to(args.device)
        else:
            parser = cls(args, model, transform)
            model.load_pretrained(parser.WORD.embed).to(args.device)

        # print(state['state_dict'])

        model.load_state_dict(state['state_dict'])
        model.eval()
        model.to(args.device)

        parser.model = model
        parser.args = args
        parser.transform = transform

        if parser.args.feat in ('char', 'bert'):
            parser.WORD, parser.FEAT = parser.transform.WORD
        else:
            parser.WORD, parser.FEAT = parser.transform.WORD, parser.transform.POS
        parser.EDU_BREAK = parser.transform.EDU_BREAK
        parser.GOLD_METRIC = parser.transform.GOLD_METRIC
        # self.TREE = self.transform.TREE
        try:
            parser.CHART = parser.transform.CHART
            parser.PARSINGORDER = parser.transform.PARSINGORDER
        except:
            print('parser.CHART and parser.PARSINGORDER parameters are not available for this model.')

        return parser


    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args': self.args,
                 'state_dict': model.state_dict(),
                 'pretrained': pretrained,
                 'transform': self.transform}
        torch.save(state, path)

def keep_last_n_checkpoint(checkpoint_dir,n=5):
    checkpoints = glob.glob(checkpoint_dir + '*.pt')
    checkpoints.sort(key=os.path.getmtime)
    num_checkpoints = len(checkpoints)
    if num_checkpoints > n:
        for old_chk in checkpoints[:num_checkpoints - n]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)