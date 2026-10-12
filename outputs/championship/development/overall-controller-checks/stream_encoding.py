import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

output=Path('outputs/logs/overall-controller-checks/stream-encoding.json')
if output.exists(): raise FileExistsError(output)
addresses=[[2**32,2,0,0],[0,1+2*2**32,0,0]]
states=[np.random.SeedSequence(row).generate_state(8) for row in addresses]
line=next(row for row in Path('/proc/self/cgroup').read_text().splitlines() if row.startswith('0::'))
group=Path('/sys/fs/cgroup')/line.split('::',1)[1].lstrip('/')
value=dict(purpose='out_of_domain_stream_encoding_diagnostic',
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    numpy=np.__version__,addresses=addresses,distinct_addresses=addresses[0]!=addresses[1],
    generated_seed_states_identical=bool(np.array_equal(*states)),
    seed_state_sha256=[hashlib.sha256(state.tobytes()).hexdigest() for state in states],
    maximum_allowed_word=int(np.iinfo(np.uint32).max),
    limits=dict(memory_max=(group/'memory.max').read_text().strip(),cpu_max=(group/'cpu.max').read_text().strip()),
    valid_registered_run_affected=False,
    note='Both example addresses are outside the newly enforced fixed-word domain. This probes seed encoding only; no simulation or model fit was run.')
output.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value,indent=2))
