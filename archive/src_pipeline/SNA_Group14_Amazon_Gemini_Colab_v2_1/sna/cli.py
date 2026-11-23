import argparse, json, sys
from .common import load_config
from .data import audit, index_raw
from .profiles import build_prompts, generate
from .embedding import embed
from .package import package
from .experiments import preflight, run_stage, stage_plan, STAGES
from .orchestrator import run_all


def main(argv=None):
    ap=argparse.ArgumentParser(description='SNA Group14 Amazon-book controlled semantic-profile pipeline')
    ap.add_argument('--config',default='configs/amazon_v2.example.json')
    sp=ap.add_subparsers(dest='cmd',required=True)
    for name in ('audit','index-raw','plan','preflight','package'):
        sub=sp.add_parser(name)
        if name=='package':sub.add_argument('--allow-mock',action='store_true')
    for name in ('build-prompts','profiles','embed'):
        sub=sp.add_parser(name);sub.add_argument('--kind',choices=['item','user'],required=True)
        if name=='profiles':
            sub.add_argument('--limit',type=int,default=None)
            sub.add_argument('--dry-run',action='store_true')
    all_=sp.add_parser('run-all');all_.add_argument('--execute',action='store_true')
    sub=sp.add_parser('run-stage')
    sub.add_argument('--stage',required=True,choices=list(STAGES))
    sub.add_argument('--model');sub.add_argument('--seed',type=int);sub.add_argument('--mode')
    sub.add_argument('--layer',type=int);sub.add_argument('--execute',action='store_true')
    args=ap.parse_args(argv);cfg=load_config(args.config)
    if args.cmd=='audit':result=audit(cfg)
    elif args.cmd=='index-raw':result=index_raw(cfg)
    elif args.cmd=='build-prompts':result=build_prompts(cfg,args.kind)
    elif args.cmd=='profiles':result=generate(cfg,args.kind,args.limit,args.dry_run)
    elif args.cmd=='embed':result=embed(cfg,args.kind)
    elif args.cmd=='package':result=package(cfg,args.allow_mock)
    elif args.cmd=='preflight':result=preflight(cfg)
    elif args.cmd=='plan':result=stage_plan(cfg)
    elif args.cmd=='run-stage':result=run_stage(cfg,args.stage,args.model,args.seed,args.mode,args.layer,args.execute)
    elif args.cmd=='run-all':result=run_all(cfg,args.execute)
    else:raise ValueError(args.cmd)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':main()
