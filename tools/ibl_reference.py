#!/usr/bin/env python3
"""Immutable reference capture plans and conservative offline convergence diagnostics.

No GPU is launched unless capture --execute is given. Summaries never qualify a
reference: they diagnose Monte Carlo repeatability, not model/systematic error.
"""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

IDENTITY = ('width', 'height', 'encoding', 'row_stride_bytes', 'roughness', 'view',
            'environment', 'exposure_ev100', 'model', 'source_hash', 'model_hash',
            'build_hash', 'shader_hash', 'dfg_hash', 'material', 'camera_transform', 'camera_projection', 'camera_world_from_view', 'camera_clip_from_view')
HASHES = ('source_hash', 'model_hash', 'build_hash', 'shader_hash', 'dfg_hash')
DEFAULT_CASES = ('furnace_back_50', 'wide_front_20', 'wide_front_50', 'wide_back_50',
                 'narrow_front_20', 'narrow_front_50', 'narrow_back_50')


def read_capture(directory):
    directory = Path(directory)
    m = json.loads((directory / 'manifest.json').read_text())
    if m.get('schema_version') != 1 or m.get('encoding') != 'rgb32float-le':
        raise ValueError('unsupported manifest schema/encoding')
    m = {**m, **m.get('identity', {})}
    n = m['width'] * m['height']
    if n <= 0 or m['row_stride_bytes'] != m['width'] * 12:
        raise ValueError('invalid dimensions/stride')
    if type(m['seed']) is not int or not 0 <= m['seed'] <= 4294967295 or type(m['actual_spp']) is not int or not 0 < m['actual_spp'] <= 16777216:
        raise ValueError('invalid seed/spp')
    def file(name):
        p = directory / m[name]
        if p.resolve().parent != directory.resolve():
            raise ValueError('capture files must be in run directory')
        return p.read_bytes()
    raw = file('reference_file')
    rgb = array.array('f'); rgb.frombytes(raw)
    if sys.byteorder != 'little':
        rgb.byteswap()
    geometry = file('geometry_file')
    if len(raw) != n * 12 or len(geometry) != n or not any(geometry):
        raise ValueError('invalid image/geometry length or empty geometry')
    if any(v not in (0, 1) for v in geometry) or not all(math.isfinite(v) and v >= 0 for v in rgb):
        raise ValueError('invalid geometry/nonfinite radiance')
    return {'manifest': m, 'rgb': rgb, 'geometry': geometry,
            'digest': hashlib.sha256(raw).hexdigest(), 'directory': str(directory)}


def compatible(captures):
    first = captures[0]
    for c in captures[1:]:
        if first['manifest'].get('identity') != c['manifest'].get('identity'):
            raise ValueError('mismatched capture identity')
        for key in IDENTITY:
            if first['manifest'].get(key) != c['manifest'].get(key):
                raise ValueError('mismatched capture metadata: ' + key)
        if first['geometry'] != c['geometry']:
            raise ValueError('mismatched geometry')


def pooled(rgb, geometry, width, height):
    """Origin-anchored 4x4 pooling over geometry, retaining tile population."""
    out = []
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            ids = [j * width + i for j in range(y, min(y+4, height))
                   for i in range(x, min(x+4, width)) if geometry[j*width+i]]
            if ids:
                out.append(([sum(rgb[3*i+k] for i in ids)/len(ids) for k in range(3)], len(ids)))
    return out


def luminance(rgb):
    return sum(a*b for a,b in zip(rgb, (.2126, .7152, .0722)))


def metrics(candidate, reference, peak):
    result = {}
    for region in ('geometry', 'lit', 'core', 'skirt', 'dark'):
        selected = [(c,r,n) for (c,n),(r,_) in zip(candidate, reference)
                    if region == 'geometry' or
                    (region == 'lit' and luminance(r) > .02*peak) or
                    (region == 'core' and luminance(r) > .2*peak) or
                    (region == 'skirt' and .02*peak < luminance(r) <= .2*peak) or
                    (region == 'dark' and luminance(r) <= .02*peak)]
        count = sum(n for _,_,n in selected)
        if not count:
            result[region] = None; continue
        rmse = math.sqrt(sum(n*sum((a-b)**2 for a,b in zip(c,r)) for c,r,n in selected)/(3*count))
        mean = sum(n*luminance(r) for _,r,n in selected)/count
        floor = peak*1e-6
        result[region] = {'population': count, 'rgb_rmse': rmse,
            'relative_rmse': rmse/max(mean,floor) if peak > 0 else None,
            'floor_bound': mean <= floor,
            'bias': sum(n*(luminance(c)-luminance(r)) for c,r,n in selected)/max(count*mean,count*floor) if peak > 0 else None}
    return result


def interval(values):
    """Two-sided Student-t 95% half-width across independent runs."""
    n = len(values)
    if n < 4: return None
    table = {3:3.182446,4:2.776445,5:2.570582,6:2.446912,7:2.364624,8:2.306005,9:2.262158,
             10:2.228139,11:2.200986,12:2.178813,13:2.160369,14:2.144787,15:2.131450,
             16:2.119906,17:2.109816,18:2.100923,19:2.093025,20:2.085964,30:2.042273}
    # Conservative next lower tabulated degrees of freedom.
    df = max(k for k in table if k <= n-1)
    return table[df]*statistics.stdev(values)/math.sqrt(n)


def analyze(captures, reference_seeds, allowed_error=None, claimed_improvement=None):
    compatible(captures)
    high = max(c['manifest']['actual_spp'] for c in captures)
    refs = [c for c in captures if c['manifest']['seed'] in reference_seeds and c['manifest']['actual_spp'] == high]
    if len({c['manifest']['seed'] for c in refs}) != len(refs):
        raise ValueError('duplicate reference seed/budget')
    reasons = ['Monte Carlo diagnostics do not validate systematic/model error; protocol review required']
    if len(refs) < 4: reasons.append('fewer than four independent fixed-reference seeds')
    if not refs: return {'reference_qualified': False, 'unresolved': reasons + ['missing fixed reference'], 'budgets': []}
    m = refs[0]['manifest']; geometry = refs[0]['geometry']
    fixed = [sum(c['rgb'][i] for c in refs)/len(refs) for i in range(len(refs[0]['rgb']))]
    peak = max(luminance(fixed[i:i+3]) for i in range(0,len(fixed),3) if geometry[i//3])
    reference = pooled(fixed, geometry, m['width'], m['height'])
    evaluations = [c for c in captures if c['manifest']['seed'] not in reference_seeds]
    groups = {}
    if not evaluations: reasons.append('missing disjoint evaluation seeds')
    for c in evaluations:
        key = c['manifest']['actual_spp']; groups.setdefault(key, {})
        if c['manifest']['seed'] in groups[key]: raise ValueError('duplicate evaluation seed/budget')
        groups[key][c['manifest']['seed']] = c
    budgets = []
    for budget, runs in sorted(groups.items()):
        reports = [metrics(pooled(c['rgb'], geometry,m['width'],m['height']), reference, peak) for c in runs.values()]
        uncertainty = {}
        for region in ('geometry','lit','core','skirt','dark'):
            uncertainty[region] = {}
            for metric in ('relative_rmse','bias'):
                values = [r[region][metric] for r in reports if r[region] and r[region][metric] is not None]
                half = interval(values)
                limit = min(allowed_error/5, claimed_improvement/3) if allowed_error is not None and claimed_improvement is not None else None
                uncertainty[region][metric] = {'mean': statistics.mean(values) if values else None, 'run_count':len(values),
                    'student_t_95_half_width': half, 'limit': limit, 'gate_pass': half < limit if half is not None and limit is not None else None}
        budgets.append({'spp':budget,'seeds':sorted(runs),'runs':reports,'uncertainty':uncertainty})
        if len(runs)<4: reasons.append(f'{budget} spp: fewer than four independent evaluation seeds')
    successive = []
    ordered = sorted(groups)
    for low, high in zip(ordered,ordered[1:]):
        seeds = sorted(groups[low].keys() & groups[high].keys())
        # Paired budget differences use the same immutable masks/scales; no self-reference.
        deltas = []
        for seed in seeds:
            a,b = [metrics(pooled(groups[s][seed]['rgb'],geometry,m['width'],m['height']), reference,peak) for s in (low,high)]
            if a['lit'] and a['lit']['relative_rmse'] is not None:
                deltas.append(b['lit']['relative_rmse']-a['lit']['relative_rmse'])
        successive.append({'low_spp':low,'high_spp':high,'paired_seeds':seeds,'lit_relative_rmse_change_mean':statistics.mean(deltas) if deltas else None,
                           'student_t_95_half_width':interval(deltas)})
    if any(not m.get(k) for k in HASHES+('material',)): reasons.append('missing provenance hashes/material identity')
    if allowed_error is None or claimed_improvement is None: reasons.append('decision error/improvement thresholds unspecified')
    return {'reference_qualified':False,'unresolved':reasons,'fixed_reference_seeds':sorted(reference_seeds),
            'fixed_reference_spp':max(c['manifest']['actual_spp'] for c in refs),'fixed_reference_capture_hashes':[c['digest'] for c in refs],
            'mask_peak':peak,'pool_size':4,'budgets':budgets,'successive_budgets':successive,
            'uncertainty_scope':'Independent evaluation runs conditional on the fixed pooled reference; reference-pool uncertainty is not included.'}


def capture(args):
    output = Path(args.output).resolve()
    repo = Path(__file__).resolve().parent.parent
    binary = Path(args.binary)
    if not binary.is_absolute(): binary = repo / binary
    plan = []
    for case in args.cases:
        source,view,roughness = case.split('_')
        for spp in args.spp:
            for seed in args.seeds:
                dest = output / case / f'spp{spp}_seed{seed}'
                env = {'IBL_REFERENCE_SPP':str(spp),'IBL_REFERENCE_SEED':str(seed),'IBL_CAPTURE_DIR':str(dest),
                       'BEVY_ASSET_ROOT':str(repo),'IBL_MEASURE':'1','IBL_FILTER':args.filter,'IBL_STRESS':'stress_'+source,'IBL_VIEW':view,'IBL_ROUGHNESS':str(int(roughness)/100)}
                plan.append({'directory':str(dest),'environment':env,'command':[str(binary)]})
    if len({r['directory'] for r in plan}) != len(plan): raise ValueError('duplicate capture runs')
    # Preflight every path before creating anything. Existing results are never reused/overwritten.
    for run in plan:
        if Path(run['directory']).exists(): raise ValueError('run directory already exists: '+run['directory'])
    if args.execute:
        for run in plan:
            dest = Path(run['directory']); dest.mkdir(parents=True, exist_ok=False)
            (dest/'invocation.json').write_text(json.dumps(run,indent=2)+'\n')
            with (dest/'process.log').open('x') as log:
                environment = dict(os.environ)
                environment.pop('IBL_PROFILE', None)
                environment.update(run['environment'])
                subprocess.run(run['command'], env=environment, cwd=repo, stdout=log,stderr=subprocess.STDOUT,check=True, timeout=args.timeout)
            manifests = list(dest.glob('**/manifest.json'))
            if len(manifests) != 1: raise ValueError('expected exactly one capture manifest')
            c = read_capture(manifests[0].parent)
            if c['manifest']['actual_spp'] != int(run['environment']['IBL_REFERENCE_SPP']) or c['manifest']['seed'] != int(run['environment']['IBL_REFERENCE_SEED']):
                raise ValueError('capture sample budget/seed mismatch')
    else: print(json.dumps(plan,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('capture'); c.add_argument('--output',required=True); c.add_argument('--binary',default='target/debug/examples/reflection_probes')
    c.add_argument('--cases',nargs='+',default=DEFAULT_CASES); c.add_argument('--spp',nargs='+',type=int,default=[512,2048,8192]); c.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3]); c.add_argument('--filter',default='fis'); c.add_argument('--timeout',type=float,default=3600,help='maximum seconds per run (default 3600)'); c.add_argument('--execute',action='store_true'); c.add_argument('--dry-run',action='store_true')
    a=sub.add_parser('analyze'); a.add_argument('directories',nargs='+'); a.add_argument('--reference-seeds',nargs='+',type=int,required=True); a.add_argument('--allowed-error',type=float); a.add_argument('--claimed-improvement',type=float)
    args=p.parse_args()
    if args.action=='capture':
        if args.execute and args.dry_run: p.error('--execute and --dry-run conflict')
        if any(v<=0 or v>16777216 for v in args.spp) or any(v<0 or v>4294967295 for v in args.seeds): p.error('invalid spp/seed')
        if not math.isfinite(args.timeout) or args.timeout <= 0: p.error('timeout must be finite and positive')
        capture(args)
    else:
        if any(v is not None and (not math.isfinite(v) or v<=0) for v in (args.allowed_error,args.claimed_improvement)): p.error('thresholds must be finite and positive')
        print(json.dumps(analyze([read_capture(d) for d in args.directories],set(args.reference_seeds),args.allowed_error,args.claimed_improvement),indent=2,allow_nan=False))

if __name__=='__main__': main()
