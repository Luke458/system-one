import pytest
import torch
from system_one.calibration import brier_score, expected_calibration_error, reliability_data, fit_temperature, calibration_report

def test_perfect():
    p, y = [[1.,0.], [0.,1.]], [0,1]
    assert brier_score(p,y) == 0
    assert expected_calibration_error(p,y) == 0
    bins = reliability_data(p,y)
    assert sum(x['count'] for x in bins) == 2
    assert bins[-1]['count'] == 2
    assert bins[0]['accuracy'] is None

def test_known_values():
    p,y = [[.8,.2],[.4,.6]], [0,0]
    assert brier_score(p,y) == pytest.approx(.4)
    assert expected_calibration_error(p,y,1) == pytest.approx(.2)

def test_temperature_fit():
    logits = [[5.,0.], [0.,5.], [5.,0.], [0.,5.]]
    y = [0,1,1,0]
    t = fit_temperature(logits,y)
    assert t > 1
    assert calibration_report(logits,y,t)['nll'] < calibration_report(logits,y)['nll']

@pytest.mark.parametrize('p,y', [([],[]), ([[.2,.2]],[0]), ([[1,0]],[2]), ([[1,0]],[.1]), ([[float('nan'),0]],[0])])
def test_invalid(p,y):
    with pytest.raises(ValueError): brier_score(p,y)

def test_split_workflow():
    from system_one.calibrate import run
    train = [{'id': f't{i}', 'field': 'flag', 'values': [False,True], 'logits': [0.,5.], 'label': i % 2} for i in range(4)]
    test = [{'id': f'e{i}', 'field': 'flag', 'values': [False,True], 'logits': [0.,4.], 'label': i % 2} for i in range(4)]
    result = run(train, test)
    assert result['temperatures']['flag'] > 1
    with pytest.raises(ValueError): run(train, train)
    test[0]['values'] = [True,False]
    with pytest.raises(ValueError): run(train,test)
