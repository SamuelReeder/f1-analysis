import argparse
from . import ARMS, freeze, run
parser = argparse.ArgumentParser()
parser.add_argument('operation', choices=('freeze', 'evaluate', 'compare'))
parser.add_argument('--protocol', required=True)
parser.add_argument('--case'); parser.add_argument('--arm', choices=ARMS, default='gpu64_a')
parser.add_argument('--scope'); parser.add_argument('--output'); parser.add_argument('--cache')
parser.add_argument('--evaluations')
args = parser.parse_args()
if args.operation == 'freeze':
    freeze(args.protocol)
else:
    run(args)
