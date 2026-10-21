import argparse
import sys
from . import freeze,worker,launch,compare,PHASES
parser=argparse.ArgumentParser()
parser.add_argument('operation',choices=('freeze','worker','launch','compare'))
parser.add_argument('--protocol',required=True)
parser.add_argument('--case');parser.add_argument('--phase',choices=PHASES)
parser.add_argument('--scope');parser.add_argument('--output')
args=parser.parse_args()
if args.operation=='freeze':freeze(args.protocol)
else:sys.exit({'worker':worker,'launch':launch,'compare':compare}[args.operation](args))
