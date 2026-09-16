"""Fit per-field temperatures on an explicit calibration split, evaluate separately."""
import argparse
import json
from pathlib import Path
from .calibration import fit_temperature, calibration_report


def run(calibration, evaluation):
    # Format: [{id, field, values, logits, label}], label is a class index.
    for rows in (calibration, evaluation):
        if not rows:
            raise ValueError('Both splits must be nonempty')
        keys = [(r['id'], r['field']) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate case/field within split')
    if {r['id'] for r in calibration} & {r['id'] for r in evaluation}:
        raise ValueError('Calibration and evaluation case IDs must be disjoint')
    fields = {r['field'] for r in calibration}
    if fields != {r['field'] for r in evaluation}:
        raise ValueError('Both splits must contain the same fields')
    temperatures, reports = {}, {}
    for field in sorted(fields):
        train = [r for r in calibration if r['field'] == field]
        test = [r for r in evaluation if r['field'] == field]
        values = train[0]['values']
        if len(values) < 2 or len({(type(v).__name__, str(v)) for v in values}) != len(values):
            raise ValueError('Invalid class values')
        signature = json.dumps(values)
        if any(json.dumps(r['values']) != signature or len(r['logits']) != len(values) for r in train + test):
            raise ValueError('Class order and logits width must agree across splits')
        x, y = [r['logits'] for r in train], [r['label'] for r in train]
        ex, ey = [r['logits'] for r in test], [r['label'] for r in test]
        t = fit_temperature(x, y)
        temperatures[field] = t
        reports[field] = {'values': values, 'calibration_n': len(train),
                          'before': calibration_report(ex, ey),
                          'after': calibration_report(ex, ey, t)}
    return {'temperatures': temperatures, 'evaluation': reports,
            'warning': 'Use representative independently labelled splits. Scaling does not establish real-world calibration.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calibration', required=True)
    p.add_argument('--evaluation', required=True)
    p.add_argument('--output', default='reports/calibration.json')
    args = p.parse_args()
    report = run(json.loads(Path(args.calibration).read_text()), json.loads(Path(args.evaluation).read_text()))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['temperatures']))

if __name__ == '__main__':
    main()
