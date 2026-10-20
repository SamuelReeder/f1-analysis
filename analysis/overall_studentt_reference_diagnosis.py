"""Read-only diagnosis of the failed reference-stability check."""
import argparse
import json
from pathlib import Path

from overall_studentt_numerics import load_reference,previous,numerical_reference_row as reference_row,save


def analytic_reference_row(mp,log_df,residual,digits):
    with mp.workdps(digits):
        z,x=mp.mpf(float(log_df)),mp.mpf(float(residual))
        df=mp.exp(z);u=x*x/df
        value=(mp.loggamma((df+1)/2)-mp.loggamma(df/2)-mp.log(df*mp.pi)/2
               -(df+1)*mp.log1p(u)/2)
        dz=(df*(mp.digamma((df+1)/2)-mp.digamma(df/2))/2-mp.mpf('.5')
            -df*mp.log1p(u)/2+(df+1)*x*x/(2*(df+x*x)))
        dloc=(df+1)*x/(df+x*x)
        dscale=-1+(df+1)*x*x/(df+x*x)
        return [float(value),float(dz),float(dloc),float(dscale)]


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--scope',required=True)
    args=parser.parse_args()
    actual=previous.scope_properties(args.scope,'float32')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    source=json.loads(Path(args.input).read_bytes());mp=load_reference()
    mismatches=[]; analytic_unstable=[];nonzero_derivative_disagreements=[]
    for i,row in enumerate(source['rows']):
        low=reference_row(mp,row['log_df'],row['residual'],80)
        high=reference_row(mp,row['log_df'],row['residual'],120)
        a=analytic_reference_row(mp,row['log_df'],row['residual'],80)
        b=analytic_reference_row(mp,row['log_df'],row['residual'],120)
        if low!=high:mismatches.append(dict(index=i,input=row,numerical80=low,numerical120=high,analytic80=a,analytic120=b))
        if a!=b:analytic_unstable.append(i)
        if any(v!=w and w!=0 for v,w in zip(high,b)):nonzero_derivative_disagreements.append(i)
    result=dict(purpose='post_hoc_reference_stability_diagnosis',input=args.input,input_sha256=previous.sha(args.input),
        actual_scope=actual,rows=len(source['rows']),numerical_reference_mismatches=mismatches,
        analytic_unstable_rows=analytic_unstable,nonzero_derivative_disagreement_rows=nonzero_derivative_disagreements,
        source_sha256={str(Path(__file__).resolve()):previous.sha(__file__)},
        original_arithmetic_failures_preserved=True,mcmc_executed=False,publication_allowed=False)
    save(output/'result.json',result)
    print(json.dumps({k:len(v) if isinstance(v,list) else v for k,v in result.items()
                      if k in ('rows','numerical_reference_mismatches','analytic_unstable_rows','nonzero_derivative_disagreement_rows')}))
