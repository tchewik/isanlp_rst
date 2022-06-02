# -*- coding: utf-8 -*-

import torch
from src.utils import Config
from src.utils.logging import init_logger, logger
from src.utils.parallel import init_device
import datetime

def parse(parser):
    parser.add_argument('--conf', '-c', default=None,
                        help='path to config file')
    parser.add_argument('--path', '-p', default=None,
                        help='path to model file')
    parser.add_argument('--predict_output_path', default=None,
                        help='path to model file')
    parser.add_argument('--device', '-d', default='1',
                        help='ID of GPU to use')
    parser.add_argument('--seed', '-s', default=1, type=int,
                        help='seed for generating random numbers')
    parser.add_argument('--threads', '-t', default=16, type=int,
                        help='max num of threads')
    parser.add_argument('--batch-size', default=512, type=int,
                        help='batch size')
    args, unknown = parser.parse_known_args()
    args, _ = parser.parse_known_args(unknown, args)
    args = Config(**vars(args))
    Parser = args.pop('Parser')
    if args.mode =='train':
        time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        args.path = args.path + '/' + time_now + '/model'
    # else:
    #     args.path = args.path + '/model'
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    init_device(args.device)
    # print(f"{args.path}.{args.mode}.log")
    # input()
    init_logger(logger, f"{args.path}.{args.mode}.log")
    logger.info('\n' + str(args))
    if args.mode == 'train':
        parser = Parser.build(**args)
        parser.train(**args)
    elif args.mode == 'evaluate':
        parser = Parser.load(args.path)
        parser.evaluate(**args)
    elif args.mode == 'predict':
        parser = Parser.load(args.path)
        parser.predict(**args)
