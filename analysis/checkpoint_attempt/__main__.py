import argparse
from . import profile

parser = argparse.ArgumentParser(description='Registered synthetic checkpoint fits; no publication')
commands = parser.add_subparsers(dest='operation',required=True)
freeze = commands.add_parser('freeze'); freeze.add_argument('--output',required=True)
run = commands.add_parser('run'); run.add_argument('--protocol',required=True)
worker = commands.add_parser('worker')
for name in ('protocol','directory','case','stage'):worker.add_argument('--'+name,required=True)
worker.add_argument('--attempt',type=int,required=True)
args = parser.parse_args()
if args.operation == 'freeze':
    profile.freeze(args.output)
elif args.operation == 'worker':
    if args.stage not in ('sample_and_archive','diagnostics'):parser.error('Unknown registered stage')
    from .worker import worker as run_worker
    run_worker(args)
else:
    from .runner import run as run_profile
    raise SystemExit(run_profile(args))
