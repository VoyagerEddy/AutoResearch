import json

import pytest

from autoresearch.services import colab_data


@pytest.fixture
def criteria():
    return {
        'validation': 'scientific_metrics',
        'metrics_file': 'rsi/metrics.json',
        'required_metrics': [
            'B0_eer_percent', 'VQ_eer_percent', 'B0_min_dcf', 'VQ_min_dcf',
            'B0_cross_language_eer_percent', 'VQ_cross_language_eer_percent',
        ],
        'paired_seed_confirmation': {
            'seeds': [42, 43], 'min_cross_eer_drop': 0.03, 'max_min_dcf_increase': 0.005,
        },
    }


@pytest.fixture
def details(criteria):
    return {
        'completion_kind': 'campaign_complete',
        'scientific_metrics_produced': True,
        'metrics_file': criteria['metrics_file'],
        'metrics': {
            'B0_eer_percent': 5.5, 'VQ_eer_percent': 5.0,
            'B0_min_dcf': 0.425, 'VQ_min_dcf': 0.42,
            'B0_cross_language_eer_percent': 7.5, 'VQ_cross_language_eer_percent': 7.0,
        },
        'paired_seeds': {
            '42': {
                'baseline': {'eer': 5.0, 'min_dcf': 0.40, 'cross_eer': 7.0},
                'variant': {'eer': 4.5, 'min_dcf': 0.405, 'cross_eer': 6.5},
                'passed': True,
            },
            '43': {
                'baseline': {'eer': 6.0, 'min_dcf': 0.45, 'cross_eer': 8.0},
                'variant': {'eer': 5.5, 'min_dcf': 0.435, 'cross_eer': 7.5},
                'passed': True,
            },
        },
        'research_effect_validated': True,
    }


def test_report_and_notebook_preserve_confirmed_seed_pairs(tmp_path, criteria, details):
    report = {**details, 'status': 'campaign_complete', 'campaign_complete': True, 'acceptance_complete': False}
    path = tmp_path / criteria['metrics_file']
    path.parent.mkdir()
    path.write_text(json.dumps(report), encoding='utf-8')

    embedded = {}
    exec(colab_data.notebook_helpers_source(), embedded)
    from_notebook = embedded['validate_scientific_result'](tmp_path, criteria)
    assert from_notebook == details
    assert colab_data.validate_completion_details(from_notebook, criteria) == details


@pytest.mark.parametrize(('metric', 'failing_score', 'mean_metric', 'mean'), [
    ('eer', 6.0, 'VQ_eer_percent', 5.25),
    ('cross_eer', 7.99, 'VQ_cross_language_eer_percent', 7.245),
    ('min_dcf', 0.46, 'VQ_min_dcf', 0.4325),
])
def test_one_seed_can_fail_confirmation_despite_a_favorable_mean(criteria, details, metric, failing_score, mean_metric, mean):
    row = details['paired_seeds']['43']
    row['variant'][metric] = failing_score
    details['metrics'][mean_metric] = mean
    with pytest.raises(ValueError, match='Seed 43 confirmation flag'):
        colab_data.validate_completion_details(details, criteria)

    row['passed'] = False
    with pytest.raises(ValueError, match='Overall confirmation flag'):
        colab_data.validate_completion_details(details, criteria)

    details['research_effect_validated'] = False
    validated = colab_data.validate_completion_details(details, criteria)
    assert validated == details
    assert validated['completion_kind'] == 'campaign_complete'


def test_rejects_fabricated_seed_or_mean(criteria, details):
    pair = details['paired_seeds']
    pair['44'] = pair.pop('43')
    with pytest.raises(ValueError, match='exactly the declared confirmation seeds'):
        colab_data.validate_completion_details(details, criteria)

    pair['43'] = pair.pop('44')
    details['metrics']['B0_cross_language_eer_percent'] = 7.6
    with pytest.raises(ValueError, match='Mean B0_cross_language_eer_percent'):
        colab_data.validate_completion_details(details, criteria)


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), -0.1, 1.01])
def test_rejects_invalid_seed_measurement(criteria, details, value):
    details['paired_seeds']['42']['baseline']['min_dcf'] = value
    with pytest.raises(ValueError, match='Seed 42 has an invalid baseline min_dcf'):
        colab_data.validate_completion_details(details, criteria)


def test_confirmation_criteria_require_a_full_campaign_with_six_metrics(criteria):
    criteria['status_value'] = 'acceptance_passed'
    with pytest.raises(ValueError, match='full scientific completion'):
        colab_data.validate_success_criteria(criteria)

    criteria['status_value'] = 'campaign_complete'
    criteria['required_metrics'].remove('VQ_cross_language_eer_percent')
    with pytest.raises(ValueError, match='all six mean metrics'):
        colab_data.validate_success_criteria(criteria)


@pytest.mark.parametrize(('field', 'invalid', 'message'), [
    ('seeds', [42, 42], 'distinct'),
    ('seeds', [True, 43], 'distinct'),
    ('min_cross_eer_drop', float('nan'), 'threshold'),
    ('max_min_dcf_increase', -0.005, 'threshold'),
])
def test_rejects_invalid_success_criterion(criteria, field, invalid, message):
    criteria['paired_seed_confirmation'][field] = invalid
    with pytest.raises(ValueError, match=message):
        colab_data.validate_success_criteria(criteria)
