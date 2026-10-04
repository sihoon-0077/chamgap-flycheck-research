"""Stage 2 of the learning-curve experiment: aggregate per-episode outcomes.

Scopes are chosen here, not at run time:
  all                 every event of the fixed test set
  excl_truth_unknown  events whose evaluator truth is not UNKNOWN
It also reports how truth=UNKNOWN events were handled, per model.

domain=SIM_ONLY_UNCALIBRATED, connectome=DEMO_RANDOM_NOT_BIOLOGICAL.
Run from the repository root:  python -m experiments.analyze_learning_curve
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DOMAIN = 'SIM_ONLY_UNCALIBRATED'
CONNECTOME = 'DEMO_RANDOM_NOT_BIOLOGICAL'
KEY = ['model', 'n_episodes', 'seed']
OUTCOMES = ['auto_correct', 'auto_wrong', 'human', 'timeout']   # a partition of every event
METRICS = ['auto_correct', 'auto_wrong', 'human', 'checks', 'elapsed_s', 'water_ml', 'timeout']
CSV_COLUMNS = ['model', 'n_episodes', 'seed', 'params', 'scope', *METRICS,
               'n_test_events', 'domain', 'connectome', 'structure']
ORDER = ['mlp_h64', 'mlp_h44_matched', 'fly_random', 'fly_rewired']
LABELS = {'mlp_h64': 'MLP 은닉 64', 'mlp_h44_matched': 'MLP 은닉 44 (fly와 크기 맞춤)',
          'fly_random': 'fly 난수 연결', 'fly_rewired': 'fly 차수 보존 재배선'}
# Reference palette slots 1-4 in fixed order; validated (light): CVD dE >= 9.1, normal dE >= 22.9.
# Aqua and yellow are < 3:1 on the surface, so identity also rides on markers + legend + CSV table.
COLORS = {'mlp_h64': '#2a78d6', 'mlp_h44_matched': '#eb6834',
          'fly_random': '#1baf7a', 'fly_rewired': '#eda100'}
MARKERS = {'mlp_h64': 'o', 'mlp_h44_matched': 's', 'fly_random': '^', 'fly_rewired': 'D'}
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'


def load(results):
    ep = pd.read_csv(results / 'learning_curve_episodes.csv.gz')
    runs = pd.read_csv(results / 'learning_curve_runs.csv')
    for frame in (ep, runs):
        if set(frame['domain']) != {DOMAIN} or set(frame['connectome']) != {CONNECTOME}:
            raise ValueError('every row must carry the SIM_ONLY / DEMO_RANDOM provenance labels')
    if not np.allclose(ep[OUTCOMES].sum(axis=1), 1):
        raise ValueError('auto_correct/auto_wrong/human/timeout must partition every event')
    tests = ep.groupby(KEY)['episode'].apply(lambda s: tuple(sorted(s)))
    if tests.nunique() != 1:
        raise ValueError('every condition must be evaluated on the same test events')
    return ep, runs


SCOPES = {'all': lambda ep: ep,
          'excl_truth_unknown': lambda ep: ep[ep['expected'] != 'UNKNOWN']}


def aggregate(ep, runs, scopes=SCOPES):
    parts = []
    for scope, select in scopes.items():
        frame = select(ep)
        agg = frame.groupby(KEY).agg(**{m: (m, 'mean') for m in METRICS},
                                     n_test_events=('episode', 'size')).reset_index()
        agg['scope'] = scope
        parts.append(agg)
    out = pd.concat(parts, ignore_index=True)
    out = out.merge(runs[KEY + ['params', 'structure']], on=KEY, how='left')
    baseline = out['model'] == 'fixed_P0'
    out.loc[baseline, 'params'] = 0
    out.loc[baseline, 'structure'] = 'fixed rule policy P0, no training'
    if out['params'].isna().any():
        raise ValueError('a learned condition has no run metadata')
    out['params'] = out['params'].astype(int)
    out['domain'], out['connectome'] = DOMAIN, CONNECTOME
    rank = {m: i for i, m in enumerate(['fixed_P0'] + ORDER)}
    out = out.sort_values(['scope', 'model', 'n_episodes', 'seed'],
                          key=lambda c: c.map(rank) if c.name == 'model' else c)
    return out[CSV_COLUMNS].round(6)


def unknown_handoff(ep, runs):
    u = ep[ep['expected'] == 'UNKNOWN']
    if u['auto_correct'].any():
        raise ValueError('truth=UNKNOWN can never be an automatic success')
    t = u.assign(auto_as_sensor=u['auto_wrong'].astype(bool) & (u['predicted'] == 'SENSOR'))
    out = t.groupby(KEY).agg(n_truth_unknown=('episode', 'size'), handed_to_human=('human', 'sum'),
                             auto_answered=('auto_wrong', 'sum'), auto_as_sensor=('auto_as_sensor', 'sum'),
                             timeout=('timeout', 'sum')).reset_index()
    out['handoff_rate'] = out['handed_to_human'] / out['n_truth_unknown']
    out['auto_answer_rate'] = out['auto_answered'] / out['n_truth_unknown']
    out['timeout_rate'] = out['timeout'] / out['n_truth_unknown']
    out = out.merge(runs[KEY + ['params']], on=KEY, how='left')
    out['params'] = out['params'].fillna(0).astype(int)
    out['domain'], out['connectome'] = DOMAIN, CONNECTOME
    return out.round(6)


def pct_range(s):
    return f'{s.mean():.1%} [{s.min():.1%}–{s.max():.1%}]'


def table(df, metric, sizes, title):
    lines = [f'\n#### {title}', '', '| 모델 | ' + ' | '.join(f'n={n}' for n in sizes) + ' |',
             '|---|' + '---|' * len(sizes)]
    for m in ORDER:
        cells = [pct_range(df[(df['model'] == m) & (df['n_episodes'] == n)][metric]) for n in sizes]
        lines.append(f'| {LABELS[m]} | ' + ' | '.join(cells) + ' |')
    base = df[df['model'] == 'fixed_P0'][metric]
    if len(base):
        lines.append(f'| 고정 정책 P0 (학습 없음) | {base.iloc[0]:.1%} (모든 n 동일) |' + ' |' * (len(sizes) - 1))
    return '\n'.join(lines)


def paired(curve, a, b, sizes, metric='auto_correct', scope='all'):
    """a - b per data seed (same training data and test set), in percentage points."""
    s = curve[curve['scope'] == scope].set_index(KEY)[metric]
    cells = []
    for n in sizes:
        d = (s.xs((a, n), level=(0, 1)) - s.xs((b, n), level=(0, 1))) * 100
        cells.append(f'{d.mean():+.1f} ({(d > 0).sum()}/{len(d)} 양수)')
    return f'| {LABELS[a]} − {LABELS[b]} | ' + ' | '.join(cells) + ' |'


def plot(curve, out_path, sizes, n_all, n_excl, design):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    if 'Malgun Gothic' not in {f.name for f in font_manager.fontManager.ttflist}:
        raise RuntimeError('Korean font Malgun Gothic not found; labels would render as boxes')
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2,
                         'xtick.color': MUTED, 'ytick.color': MUTED})
    rows = [('auto_correct', '자동 정답률'), ('auto_wrong', '자동 오진률')]
    cols = [('all', f'전체 시험 사건 ({n_all:,}개)'),
            ('excl_truth_unknown', f'정답 UNKNOWN 사건 제외 ({n_excl:,}개)')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), sharex=True, sharey='row', facecolor=SURFACE)
    dodge = dict(zip(ORDER, (-.045, -.015, .015, .045)))   # log10 offsets keep range bars apart
    for r, (metric, ylabel) in enumerate(rows):
        for c, (scope, coltitle) in enumerate(cols):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            sub = curve[curve['scope'] == scope]
            base = sub[sub['model'] == 'fixed_P0'][metric]
            if len(base):
                ax.axhline(base.iloc[0], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1,
                           label='고정 정책 P0 (학습 없음, 기준선)')
            for m in ORDER:
                g = sub[sub['model'] == m].groupby('n_episodes')[metric]
                x = np.array(sizes, dtype=float) * 10 ** dodge[m]
                mean, lo, hi = g.mean().loc[sizes], g.min().loc[sizes], g.max().loc[sizes]
                ax.vlines(x, lo, hi, color=COLORS[m], lw=1.6, alpha=.85, zorder=2)
                params = int(sub[sub['model'] == m]['params'].iloc[0])
                ax.plot(x, mean, color=COLORS[m], lw=2, marker=MARKERS[m], ms=6.5,
                        mec=SURFACE, mew=1.2, solid_capstyle='round', solid_joinstyle='round',
                        zorder=3, label=f'{LABELS[m]} — 학습 파라미터 {params:,}개')
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(FixedLocator(sizes))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(round(v)):,}'))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0%}'))
            ax.grid(axis='y', color=GRID, lw=.8, ls='-')
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color(AXIS)
            ax.tick_params(length=0, pad=6)
            if r == 0:
                ax.set_title(coltitle, fontsize=11, color=INK, loc='left', pad=10)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=11)
            if r == 1:
                ax.set_xlabel('학습에 쓴 가상 사건 수 (로그 축)', fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if labels and labels[0].startswith('고정 정책'):   # models first, reference line last
        handles, labels = handles[1:] + handles[:1], labels[1:] + labels[:1]
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.055, .885), ncol=3,
               frameon=False, fontsize=9.5, handlelength=2.6, columnspacing=1.6)
    fig.text(.055, .975, 'FlyCheck 학습곡선 — 학습 데이터 양에 따른 폐루프 성능',
             fontsize=14, weight='bold', color=INK, va='top')
    fig.text(.055, .943, f'domain={DOMAIN} · connectome={CONNECTOME} (난수 연결, 실제 초파리 커넥톰 아님)',
             fontsize=9, color=INK2, va='top')
    fig.text(.055, .921, f'점 = 시드 {design["n_seeds"]}개 평균, 세로 막대 = 시드 간 최소–최대 · '
             f'시험 세트: 모든 조건에서 같은 가상 사건 {n_all:,}개 (seed {design["test_seeds"]})',
             fontsize=9, color=INK2, va='top')
    fig.subplots_adjust(left=.075, right=.985, top=.78, bottom=.075, wspace=.08, hspace=.16)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE,
                metadata={'Description': f'domain={DOMAIN}; connectome={CONNECTOME}'})
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results', type=Path)
    a = p.parse_args()
    ep, runs = load(a.results)
    curve_all = aggregate(ep, runs)
    learned = curve_all['model'] != 'fixed_P0'
    curve_all[learned].to_csv(a.results / 'learning_curve.csv', index=False)
    curve_all[~learned].to_csv(a.results / 'fixed_baseline.csv', index=False)
    handoff = unknown_handoff(ep, runs)
    handoff.to_csv(a.results / 'unknown_truth_handoff.csv', index=False)
    sizes = sorted(curve_all.loc[learned, 'n_episodes'].unique().tolist())
    n_all = int(curve_all.loc[curve_all['scope'] == 'all', 'n_test_events'].iloc[0])
    n_excl = int(curve_all.loc[curve_all['scope'] == 'excl_truth_unknown', 'n_test_events'].iloc[0])
    manifest = json.loads((a.results / 'learning_curve_manifest.json').read_text(encoding='utf-8'))
    if manifest['domain'] != DOMAIN or manifest['connectome'] != CONNECTOME:
        raise ValueError('manifest provenance labels do not match')
    lo, hi = manifest['design']['test_seeds']
    design = {'n_seeds': curve_all.loc[learned, 'seed'].nunique(), 'test_seeds': f'{lo}–{hi}'}
    plot(curve_all, a.results / 'learning_curve.png', sizes, n_all, n_excl, design)

    by_scope = {s: curve_all[curve_all['scope'] == s] for s in ('all', 'excl_truth_unknown')}
    print(f'domain={DOMAIN} connectome={CONNECTOME}  test events: all={n_all}, '
          f'excl_truth_unknown={n_excl}, truth_unknown={n_all - n_excl}')
    print(table(by_scope['all'], 'auto_correct', sizes, '자동 정답률 — 전체 사건 (평균 [최소–최대])'))
    print(table(by_scope['excl_truth_unknown'], 'auto_correct', sizes, '자동 정답률 — 정답 UNKNOWN 제외'))
    print(table(by_scope['all'], 'auto_wrong', sizes, '자동 오진률 — 전체 사건'))
    print(table(by_scope['excl_truth_unknown'], 'auto_wrong', sizes, '자동 오진률 — 정답 UNKNOWN 제외'))
    print(table(by_scope['all'], 'timeout', sizes, '시간 초과율 — 전체 사건'))
    print(table(handoff, 'handoff_rate', sizes, '정답 UNKNOWN 사건의 인계율'))
    print(table(handoff, 'auto_answer_rate', sizes, '정답 UNKNOWN 사건의 자동 판정률 (= 오진)'))
    print(f'\n#### 같은 시드끼리 짝지은 자동 정답률 차이 (전체 사건, %p, 시드 {design["n_seeds"]}개 평균)\n')
    print('| 비교 | ' + ' | '.join(f'n={n}' for n in sizes) + ' |')
    print('|---|' + '---|' * len(sizes))
    for x, y in (('fly_random', 'fly_rewired'), ('mlp_h44_matched', 'fly_random'),
                 ('mlp_h64', 'mlp_h44_matched')):
        print(paired(curve_all, x, y, sizes))
    print(f'\nwrote {a.results}/learning_curve.csv, fixed_baseline.csv, unknown_truth_handoff.csv, '
          'learning_curve.png')


if __name__ == '__main__':
    main()
