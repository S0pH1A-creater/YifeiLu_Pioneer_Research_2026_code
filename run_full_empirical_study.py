import sys, json, shutil, time
from pathlib import Path
repo = Path.cwd()
sys.path.insert(0, str(repo / 'V3-Models_result' / 'scripts'))
import run_v3_intraday_hourly_empirical_study as study

# load expiry plan
plan_path = repo / 'V3-Models_result' / 'config' / 'expiry_eval_windows.json'
plan = json.loads(plan_path.read_text())
# set all expiries
study.EXPIRY_FRIDAYS = tuple([p['expiry'] for p in plan['expiry_plan']])
# set tickers and models defaults (use existing defaults if present)
study.TICKERS = getattr(study, 'TICKERS', ('SPY','AAPL','MSFT'))
study.MODELS = getattr(study, 'MODELS', ['GBM','Heston','Merton'])
study.TABLE_MODELS = study.MODELS
study.N_PATHS = 50000
study.SEED = 42

# prepare tmp/cache and results paths
tmp = Path('/tmp/v3_full_run')
results_dir = repo / 'V3-Models_result' / 'results' / 'empirical_study_7d_hourly_full'
for p in (tmp, results_dir):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

study.CACHE = tmp
study.PAYLOAD_JSON = results_dir / 'payload.json'
study.CONTRACTS_JSON = results_dir / 'shared_contracts.json'
study.FILTER_JSON = results_dir / 'filter_funnel.json'

print('Configured run:')
print(' EXPIRY_FRIDAYS len=', len(study.EXPIRY_FRIDAYS))
print(' TICKERS=', study.TICKERS)
print(' MODELS=', study.MODELS)
print(' N_PATHS=', study.N_PATHS)

study.configure_study('7d')

start = time.time()
try:
    payload = study.run_study(only=None)
    elapsed = time.time() - start
    print('\nRun completed in', round(elapsed,1), 'sec')
    # save payload if not saved already
    import json
    (results_dir / 'payload.json').write_text(json.dumps(payload, default=str))
    print('Payload saved to', results_dir / 'payload.json')
    # build notebook and pdf
    nb = study.build_notebook(payload)
    print('Notebook built')
    pdf_path = study.write_pdf(payload)
    print('PDF written to', pdf_path)
    # summary
    print('Summary: cells=', len(payload.get('cells', {})))
    print(' sample table keys:', list(payload.get('tables', {}).keys())[:5])
    print(' shared_contracts keys count:', len(payload.get('shared_contracts', {})))
except Exception:
    import traceback
    traceback.print_exc()
finally:
    # keep tmp for diagnostics
    print('Temp cache at', tmp)
