"""Stage 2 of connectome_v1: aggregate, apply the pre-registered decision rules, plot.

Decision rules are copied from experiments/PREREG_connectome_v1.md (sections 3-4); the
pre-registration hash recorded by the runner is compared with the current file.
domain=SIM_ONLY_UNCALIBRATED. Run from chamgap_v4/:
  python -m experiments.analyze_connectome_curve
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analyze_learning_curve import AXIS, GRID, INK, INK2, MUTED, SURFACE

RESULTS = Path('results/connectome_v1')
PREREG = Path('experiments/PREREG_connectome_v1.md')
DOMAIN = 'SIM_ONLY_UNCALIBRATED'
KEY = ['model', 'n_episodes', 'seed']
OUTCOMES = ['auto_correct', 'auto_wrong', 'human', 'timeout']      # a partition of every event
METRICS = ['return', 'auto_correct', 'auto_wrong', 'human', 'timeout', 'checks', 'elapsed_s', 'water_ml']
ORDER = ['fly_real', 'fly_rewired', 'mlp_matched', 'fly_random_fanin']
IDS = {'fly_real': 'P3', 'fly_rewired': 'P2', 'mlp_matched': 'P1', 'fly_random_fanin': 'R', 'fixed_P0': 'P0'}
LABELS = {'fly_real': 'P3 실제 연결 (hemibrain)', 'fly_rewired': 'P2 차수 보존 재배선',
          'mlp_matched': 'P1 MLP (크기 맞춤)', 'fly_random_fanin': 'R 무작위 연결 (입력 수만 맞춤)'}
CONNECTOME = {'mlp_matched': 'NONE', 'fixed_P0': 'NONE',
              'fly_rewired': 'REWIRED_FROM_HEMIBRAIN_NOT_BIOLOGICAL',
              'fly_random_fanin': 'RANDOM_FANIN_MATCHED_NOT_BIOLOGICAL'}   # fly_real: from the manifest
# Reference slots 1, 2, 7, 3. validate_palette --pairs all on #fcfcfb: CVD dE >= 9.2, normal dE >= 16.3.
# Aqua is 2.74:1 on the surface, so identity also rides on markers + legend + the CSV tables.
COLORS = {'fly_real': '#2a78d6', 'fly_rewired': '#eb6834', 'mlp_matched': '#4a3aa7',
          'fly_random_fanin': '#1baf7a'}
MARKERS = {'fly_real': 'o', 'fly_rewired': 'D', 'mlp_matched': 's', 'fly_random_fanin': '^'}
SMALL = (50, 100, 200)
PRIMARY = [('fly_real', 'fly_rewired'), ('fly_real', 'mlp_matched')]
SECONDARY = [('fly_rewired', 'fly_random_fanin'), ('mlp_matched', 'fixed_P0'), ('fly_real', 'fixed_P0')]
ALPHA_PRIMARY = .05 / 6        # Bonferroni over the 6 primary comparisons -> 99.17% intervals
MARGIN = .10                   # equivalence margin in reward per test event
BOOT, BOOT_SEED = 10_000, 20261004


def load(results):
    ep = pd.read_csv(results / 'episodes.csv.gz')
    runs = pd.read_csv(results / 'runs.csv')
    manifest = json.loads((results / 'manifest.json').read_text(encoding='utf-8'))
    expected = {**CONNECTOME, 'fly_real': manifest['real_mask']['label']}
    for frame in (ep, runs):
        if set(frame['domain']) != {DOMAIN}:
            raise ValueError('every row must carry domain=SIM_ONLY_UNCALIBRATED')
        labels = frame.groupby('model')['connectome'].unique()
        bad = {m: list(v) for m, v in labels.items() if list(v) != [expected[m]]}
        if bad:
            raise ValueError(f'connectome labels do not match the model: {bad}')
    if not np.allclose(ep[OUTCOMES].sum(axis=1), 1):
        raise ValueError('auto_correct/auto_wrong/human/timeout must partition every event')
    if ep.groupby(KEY)['episode'].apply(lambda s: tuple(sorted(s))).nunique() != 1:
        raise ValueError('every condition must be evaluated on the same test events')
    if (ep['first_action'] == 3).any():
        raise ValueError('rule A violated: an episode started with REQUEST_INSPECTION')
    return ep, runs, manifest, prereg_status(manifest)


AMENDMENTS = '\n## 7. 변경 기록'.encode('utf-8')


def prereg_status(manifest):
    """'unchanged', 'amended (sections 1-6 unchanged)', or raise if the registered text was edited."""
    data = PREREG.read_bytes()
    recorded = manifest['prereg']['sha256']
    if hashlib.sha256(data).hexdigest() == recorded:
        return 'unchanged'
    cut = data.find(AMENDMENTS)
    if cut >= 0 and hashlib.sha256(data[:cut]).hexdigest() == recorded:
        return 'amended (sections 1-6 unchanged)'
    raise ValueError('pre-registration text differs from the one recorded at run time')


SCOPES = {'all': lambda ep: ep, 'excl_truth_unknown': lambda ep: ep[ep['expected'] != 'UNKNOWN']}


def aggregate(ep, runs):
    parts = []
    for scope, select in SCOPES.items():
        agg = (select(ep).groupby(KEY).agg(**{m: (m, 'mean') for m in METRICS},
                                           n_test_events=('episode', 'size')).reset_index())
        parts.append(agg.assign(scope=scope))
    out = pd.concat(parts, ignore_index=True)
    out = out.merge(runs[KEY + ['params', 'lr', 'connectome']], on=KEY, how='left')
    p0 = out['model'] == 'fixed_P0'
    out.loc[p0, ['params', 'lr', 'connectome']] = [0, 0., 'NONE']
    if out['params'].isna().any():
        raise ValueError('a learned condition has no run metadata')
    out['policy_id'] = out['model'].map(IDS)
    out['domain'] = DOMAIN
    cols = ['policy_id', 'model', 'n_episodes', 'seed', 'lr', 'params', 'scope', *METRICS,
            'n_test_events', 'domain', 'connectome']
    return out[cols].astype({'params': int}).sort_values(['scope', 'model', 'n_episodes', 'seed']).round(6)


def unknown_handoff(ep):
    u = ep[ep['expected'] == 'UNKNOWN']
    if u['auto_correct'].any():
        raise ValueError('truth=UNKNOWN can never be an automatic success')
    out = u.groupby(KEY).agg(n_truth_unknown=('episode', 'size'), handed_to_human=('human', 'sum'),
                             auto_answered=('auto_wrong', 'sum'), timeout=('timeout', 'sum')).reset_index()
    for k in ('handed_to_human', 'auto_answered', 'timeout'):
        out[k.replace('handed_to_human', 'handoff') + '_rate'] = out[k] / out['n_truth_unknown']
    return out.assign(domain=DOMAIN).round(6)


def boot_ci(x, alpha, rng):
    x = np.asarray(x, dtype=float)
    means = x[rng.integers(0, len(x), size=(BOOT, len(x)))].mean(1)
    return np.quantile(means, [alpha / 2, 1 - alpha / 2])


def decide(lo, hi, a, b):
    above, below, inside = lo > 0, hi < 0, lo >= -MARGIN and hi <= MARGIN
    if above and inside:
        return f'{IDS[a]} 높음, 차이 {MARGIN:.2f} 미만'
    if above:
        return f'{IDS[a]} 높음'
    if below and inside:
        return f'{IDS[a]} 낮음, 차이 {MARGIN:.2f} 미만'
    if below:
        return f'{IDS[a]} 낮음'
    if inside:
        return f'동등 (±{MARGIN:.2f})'
    return '판정 불가'


def paired(curve, a, b, n, alpha, rng, metric='return'):
    s = curve[curve['scope'] == 'all'].set_index(KEY)[metric]
    xa = s.xs((a, n), level=(0, 1))
    if b == 'fixed_P0':
        d = xa - float(s.xs(('fixed_P0', 0), level=(0, 1)).iloc[0])
    else:
        xb = s.xs((b, n), level=(0, 1))
        if set(xa.index) != set(xb.index):
            raise ValueError(f'{a} and {b} were not run on the same seeds at n={n}')
        d = xa - xb.loc[xa.index]
    lo, hi = boot_ci(d.values, alpha, rng)
    return {'comparison': f'{IDS[a]} - {IDS[b]}', 'a': a, 'b': b, 'n_episodes': n, 'n_seeds': len(d),
            'mean_diff': d.mean(), 'ci_level': 1 - alpha, 'ci_lo': lo, 'ci_hi': hi,
            'seeds_positive': int((d > 0).sum()), 'decision': decide(lo, hi, a, b)}


def decisions(curve, sizes):
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for a, b in PRIMARY:
        for n in sizes:
            primary = n in SMALL
            rows.append({**paired(curve, a, b, n, ALPHA_PRIMARY if primary else .05, rng), 'primary': primary})
    for a, b in SECONDARY:
        for n in sizes:
            rows.append({**paired(curve, a, b, n, .05, rng), 'primary': False})
    return pd.DataFrame(rows).round(6)


def crossover(curve, sizes):
    s = curve[curve['scope'] == 'all']
    p0 = float(s.loc[s['model'] == 'fixed_P0', 'return'].iloc[0])
    means = s[s['model'] != 'fixed_P0'].groupby(['model', 'n_episodes'])['return'].mean()
    return {m: next((n for n in sizes if means[(m, n)] > p0), None) for m in ORDER}, p0


def plot(curve, dec, out_path, sizes, manifest, n_seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, PercentFormatter

    if 'Malgun Gothic' not in {f.name for f in font_manager.fontManager.ttflist}:
        raise RuntimeError('Korean font Malgun Gothic not found; labels would render as boxes')
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 10,
                         'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED})
    rng = np.random.default_rng(BOOT_SEED)
    s = curve[curve['scope'] == 'all']
    p0 = s[s['model'] == 'fixed_P0'].iloc[0]
    dodge = dict(zip(ORDER, (-.036, -.012, .012, .036)))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), sharex=True, facecolor=SURFACE)

    def style(ax, title, formatter):
        ax.set_facecolor(SURFACE)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator(sizes))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(round(v)):,}'))
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(axis='y', color=GRID, lw=.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(length=0, pad=6)
        ax.set_title(title, fontsize=11, color=INK, loc='left', pad=10)

    def curve_panel(ax, metric, title, formatter, ref_text):
        ax.axhline(p0[metric], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1, label='P0 고정 정책 (학습 없음, 기준선)')
        # Between the first two sizes, where no range bar stands; surface patch keeps it legible over lines.
        ax.text(np.sqrt(sizes[0] * sizes[1]), p0[metric], ref_text, ha='center', va='center', fontsize=8.5,
                color=INK2, zorder=4, bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.5))
        for m in ORDER:
            g = s[s['model'] == m]
            x = np.array(sizes, dtype=float) * 10 ** dodge[m]
            mean = np.array([g.loc[g['n_episodes'] == n, metric].mean() for n in sizes])
            ci = np.array([boot_ci(g.loc[g['n_episodes'] == n, metric].values, .05, rng) for n in sizes])
            ax.vlines(x, ci[:, 0], ci[:, 1], color=COLORS[m], lw=1.6, alpha=.85, zorder=2)
            ax.plot(x, mean, color=COLORS[m], lw=2, marker=MARKERS[m], ms=7, mec=SURFACE, mew=1.4,
                    zorder=3, label=LABELS[m])
        style(ax, title, formatter)

    pct = PercentFormatter(1.0)    # automatic decimals, so 12.5% is not shown as 12%
    curve_panel(axes[0, 0], 'return', '(a) 시험 사건당 평균 보상 — 주 지표',
                FuncFormatter(lambda v, _: f'{v:.2f}'), f'P0 {p0["return"]:.2f}')
    ax = axes[0, 1]
    ax.axhspan(-MARGIN, MARGIN, color=GRID, alpha=.55, lw=0, zorder=0, label=f'동등 범위 ±{MARGIN:.2f}')
    ax.axhline(0, color=AXIS, lw=1, zorder=1)
    for (a, b), off in zip(PRIMARY, (-.02, .02)):
        d = dec[(dec['a'] == a) & (dec['b'] == b)].set_index('n_episodes').loc[sizes]
        x = np.array(sizes, dtype=float) * 10 ** off
        ax.vlines(x, d['ci_lo'], d['ci_hi'], color=COLORS[b], lw=1.6, alpha=.85, zorder=2)
        ax.plot(x, d['mean_diff'], color=COLORS[b], lw=2, marker=MARKERS[b], ms=7, mec=SURFACE, mew=1.4,
                zorder=3, label=f'P3 - {IDS[b]} ({LABELS[b].split(" ", 1)[1]})')   # Malgun Gothic has no U+2212
    style(ax, '(b) 같은 시드끼리 짝지은 보상 차이 — 사전 계획 판정', FuncFormatter(lambda v, _: f'{v:+.2f}'))
    shown = dec[dec['a'].eq('fly_real') & dec['b'].isin([b for _, b in PRIMARY])]
    lo, hi = min(shown['ci_lo'].min(), -MARGIN), max(shown['ci_hi'].max(), MARGIN)
    ax.set_ylim(lo - .1 * (hi - lo) - .07, hi + .05 * (hi - lo))   # free band at the bottom for the legend
    ax.legend(loc='lower center', ncol=3, frameon=False, fontsize=8.5, handlelength=2.2, columnspacing=1.2)
    curve_panel(axes[1, 0], 'auto_wrong', '(c) 자동 오진률', pct, f'P0 {p0["auto_wrong"]:.1%}')
    curve_panel(axes[1, 1], 'human', '(d) 사람 인계율', pct, f'P0 {p0["human"]:.1%}')
    for ax in axes[1]:
        ax.set_xlabel('학습에 쓴 가상 사건 수 (로그 축)', fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()   # one shared legend for (a), (c), (d)
    fig.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], loc='upper left', bbox_to_anchor=(.055, .915),
               ncol=5, frameon=False, fontsize=9, handlelength=2.4, columnspacing=1.4)
    real = manifest['real_mask']
    fig.text(.06, .978, 'FlyCheck 학습곡선 v2 — 실제 hemibrain PN→KC 연결과 대조군',
             fontsize=14, weight='bold', color=INK, va='top')
    fig.text(.06, .949, f'domain={DOMAIN} · P3 connectome={real["label"]} '
             f'(KC {real["shape_kc_pn"][0]:,} × 사구체 {real["shape_kc_pn"][1]}, 선택 규칙 팀 검토 전)',
             fontsize=8.5, color=INK2, va='top')
    fig.text(.06, .928, f'점 = 시드 평균 (n ≤ 200: 시드 {n_seeds[0]}개, n ≥ 500: {n_seeds[1]}개), 막대 = 95% bootstrap 구간 · '
             f'(b) n ≤ 200은 Bonferroni 보정 {1 - ALPHA_PRIMARY:.2%} 구간 · 보상 v2 + 규칙 A · '
             f'학습률은 검증 세트로 모델별 선택 · 시험 사건 {manifest["design"]["test"][1]:,}개 공통',
             fontsize=8.5, color=INK2, va='top')
    fig.subplots_adjust(left=.07, right=.985, top=.85, bottom=.07, wspace=.16, hspace=.2)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE,
                metadata={'Description': f'domain={DOMAIN}; P3 connectome={real["label"]}'})
    plt.close(fig)


def md_table(curve, metric, sizes, title, scope='all', pct=True):
    s = curve[curve['scope'] == scope]
    fmt = (lambda v: f'{v:.1%}') if pct else (lambda v: f'{v:.3f}')
    lines = [f'\n#### {title}', '', '| 모델 | ' + ' | '.join(f'n={n}' for n in sizes) + ' |',
             '|---|' + '---|' * len(sizes)]
    for m in ORDER:
        lines.append(f'| {LABELS[m]} | ' + ' | '.join(
            fmt(s[(s['model'] == m) & (s['n_episodes'] == n)][metric].mean()) for n in sizes) + ' |')
    base = s[s['model'] == 'fixed_P0'][metric]
    lines.append(f'| P0 고정 정책 | {fmt(base.iloc[0])} (모든 n 동일) |' + ' |' * (len(sizes) - 1))
    return '\n'.join(lines)


EXTEND_N = 200


def n200_table(curve, hand):
    head = ('| 모델 | 시드 | 보상 | 자동 정답 | 오진 | 오진 (UNKNOWN 제외) | 인계 | 시간 초과 | UNKNOWN 인계율 | 물 (mL) |\n'
            '|---|---|---|---|---|---|---|---|---|---|')
    lines = [head]
    for m in ORDER + ['fixed_P0']:
        a = curve[(curve['model'] == m) & (curve['scope'] == 'all')]
        e = curve[(curve['model'] == m) & (curve['scope'] == 'excl_truth_unknown')]
        h = hand[hand['model'] == m]
        lines.append(f'| {LABELS.get(m, "P0 고정 정책")} | {a["seed"].nunique()} | {a["return"].mean():.3f} | '
                     f'{a["auto_correct"].mean():.1%} | {a["auto_wrong"].mean():.1%} | {e["auto_wrong"].mean():.1%} | '
                     f'{a["human"].mean():.1%} | {a["timeout"].mean():.1%} | {h["handoff_rate"].mean():.1%} | '
                     f'{a["water_ml"].mean():.2f} |')
    return '\n'.join(lines)


def extension(results):
    """Amendment 1 (PREREG section 7): confirm on the 80 fresh n=200 seeds only; describe all 110."""
    ep_m, runs_m, man_m, status_m = load(results)
    ep_e, runs_e, man_e, status_e = load(results / 'extend')
    if (man_e['real_mask']['sha256'] != man_m['real_mask']['sha256']
            or man_e['design']['lr_choice'] != man_m['design']['lr_choice']):
        raise ValueError('the extension must use the main run\'s real mask and learning rates')
    if set(ep_e['n_episodes']) != {EXTEND_N}:
        raise ValueError(f'the extension must contain n={EXTEND_N} only')
    if set(ep_m.loc[ep_m['n_episodes'] == EXTEND_N, 'seed']) & set(ep_e['seed']):
        raise ValueError('extension seeds reuse main-run seeds')
    p0 = ep_m[ep_m['model'] == 'fixed_P0']
    fresh = aggregate(pd.concat([ep_e, p0]), runs_e)
    both_ep = pd.concat([ep_m[ep_m['n_episodes'] == EXTEND_N], ep_e, p0])
    both = aggregate(both_ep, pd.concat([runs_m[runs_m['n_episodes'] == EXTEND_N], runs_e]))
    rng = np.random.default_rng(BOOT_SEED)
    rows = [{**paired(fresh, a, b, EXTEND_N, .05 / len(PRIMARY), rng),
             'analysis': 'confirm_new80_amendment1_primary'} for a, b in PRIMARY]
    for a, b in PRIMARY + SECONDARY:
        for alpha in ((ALPHA_PRIMARY, .05) if (a, b) in PRIMARY else (.05,)):
            rows.append({**paired(both, a, b, EXTEND_N, alpha, rng), 'analysis': 'describe_all110'})
    dec = pd.DataFrame(rows).round(6)
    out = results / 'extend'
    dec.to_csv(out / 'paired_decisions_extend.csv', index=False)
    both.to_csv(out / 'curve_n200_all110.csv', index=False)
    hand = unknown_handoff(both_ep)
    hand.to_csv(out / 'unknown_truth_handoff_n200_all110.csv', index=False)

    print(f'domain={DOMAIN}  prereg: main run {status_m} / extension {status_e}')
    print(f'\n#### 변경 1 확인 분석 — n={EXTEND_N}, 새 시드 {fresh.loc[fresh["model"] == "fly_real", "seed"].nunique()}개만, '
          f'Bonferroni 2개 → {1 - .05 / len(PRIMARY):.1%} 구간\n')
    print('| 비교 | 평균 차이 | 구간 | 양수 시드 | 판정 |\n|---|---|---|---|---|')
    for r in dec[dec['analysis'].str.startswith('confirm')].itertuples():
        print(f'| {r.comparison} | {r.mean_diff:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | '
              f'{r.seeds_positive}/{r.n_seeds} | **{r.decision}** |')
    print(f'\n#### 기술 분석 — n={EXTEND_N}, 기존 30 + 새 80 = 110개 합산 (판정에 쓰지 않음)\n')
    print('| 비교 | 평균 차이 | 구간 | 양수 시드 | 규칙을 적용하면 |\n|---|---|---|---|---|')
    for r in dec[dec['analysis'] == 'describe_all110'].itertuples():
        print(f'| {r.comparison} | {r.mean_diff:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] ({r.ci_level:.2%}) | '
              f'{r.seeds_positive}/{r.n_seeds} | {r.decision} |')
    print(f'\n#### n={EXTEND_N} 지표 — 110개 시드 평균 (P0는 학습 없음)\n')
    print(n200_table(both, hand))
    print(f'\nwrote {out}/paired_decisions_extend.csv, curve_n200_all110.csv, unknown_truth_handoff_n200_all110.csv')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default=RESULTS, type=Path)
    p.add_argument('--extend', action='store_true', help='amendment 1 analysis (PREREG section 7)')
    a = p.parse_args()
    if a.extend:
        extension(a.results)
        return
    ep, runs, manifest, prereg_ok = load(a.results)
    curve = aggregate(ep, runs)
    curve.to_csv(a.results / 'curve.csv', index=False)
    unknown_handoff(ep).to_csv(a.results / 'unknown_truth_handoff.csv', index=False)
    sizes = sorted(curve.loc[curve['model'] != 'fixed_P0', 'n_episodes'].unique().tolist())
    dec = decisions(curve, sizes)
    dec.to_csv(a.results / 'paired_decisions.csv', index=False)
    cross, p0_return = crossover(curve, sizes)
    seeds = curve[curve['model'] == 'fly_real'].groupby('n_episodes')['seed'].nunique()
    n_seeds = (int(seeds[min(sizes)]), int(seeds[max(sizes)]))
    plot(curve, dec, a.results / 'learning_curve_v2.png', sizes, manifest, n_seeds)

    print(f'domain={DOMAIN}  prereg: {prereg_ok}  lr={manifest["design"]["lr_choice"]}')
    print('\n#### 주 비교 (사전 계획, 보상 차이, Bonferroni 보정 구간)\n')
    print('| 비교 | n | 시드 | 평균 차이 | 구간 | 양수 시드 | 판정 |\n|---|---|---|---|---|---|---|')
    for r in dec[dec['primary']].itertuples():
        print(f'| {r.comparison} | {r.n_episodes} | {r.n_seeds} | {r.mean_diff:+.3f} | '
              f'[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] ({r.ci_level:.2%}) | {r.seeds_positive}/{r.n_seeds} | {r.decision} |')
    print('\n#### 보조 비교 (95% 구간, 판정에 쓰지 않음)\n')
    print('| 비교 | n | 평균 차이 | 95% 구간 | 양수 시드 | 규칙을 적용하면 |\n|---|---|---|---|---|---|')
    for r in dec[~dec['primary']].itertuples():
        print(f'| {r.comparison} | {r.n_episodes} | {r.mean_diff:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | '
              f'{r.seeds_positive}/{r.n_seeds} | {r.decision} |')
    print(md_table(curve, 'return', sizes, '시험 사건당 평균 보상', pct=False))
    print(md_table(curve, 'auto_correct', sizes, '자동 정답률 — 전체'))
    print(md_table(curve, 'auto_correct', sizes, '자동 정답률 — 정답 UNKNOWN 제외', 'excl_truth_unknown'))
    print(md_table(curve, 'auto_wrong', sizes, '자동 오진률 — 전체'))
    print(md_table(curve, 'auto_wrong', sizes, '자동 오진률 — 정답 UNKNOWN 제외', 'excl_truth_unknown'))
    print(md_table(curve, 'human', sizes, '사람 인계율 — 전체'))
    print(md_table(curve, 'timeout', sizes, '시간 초과율 — 전체'))
    print(md_table(curve, 'water_ml', sizes, '사건당 물 (mL, 명령량)', pct=False))
    hand = unknown_handoff(ep).merge(curve[['model', 'n_episodes', 'seed']].drop_duplicates(), on=KEY)
    print(md_table(hand.assign(scope='all'), 'handoff_rate', sizes, '정답 UNKNOWN 사건의 인계율'))
    print(f'\n평균 보상이 P0({p0_return:.3f})를 처음 넘는 n: ' +
          ', '.join(f'{IDS[m]}={cross[m]}' for m in ORDER))
    print(f'\nwrote {a.results}/curve.csv, unknown_truth_handoff.csv, paired_decisions.csv, learning_curve_v2.png')


if __name__ == '__main__':
    main()
