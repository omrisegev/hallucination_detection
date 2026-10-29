const MAIN = {prmbench: 'PRMScore_GAP', socratic: 'macro_f1_GAP', hard2verify: 'balanced_f1_GAP', processbench: 'avg_f1_NC',
                crv_arith: 'macro_f1_GAP', crv_gsm8k: 'macro_f1_GAP', prm800k: 'macro_f1_GAP', deltabench: 'macro_f1_GAP'};
  const COMPARATORS = ['PRM', 'TELE_N16', 'MEAN_N16', 'CHAIN_N16', 'pub:any', 'pub:Trained PRM', 'pub:Supervised on labels', 'pub:Prompted LLM critic'];
  dash.calc('comparisons', {
    title: 'SUB5 against each comparator',
    description: 'For every cell, SUB5 (L-SML) and the plain average of the same channels minus the comparator, per metric; published scores take the best printed value for the benchmark and metric.',
    inputs: ['scores', 'published'],
    fn: (scores, published) => {
      const v = new Map();
      for (const r of scores) v.set([r.benchmark, r.mode, r.prm, r.arm, r.metric].join('|'), Number(r.value));
      const cells = new Map();
      for (const r of scores) cells.set([r.benchmark, r.mode, r.prm].join('|'), r);
      const out = [];
      for (const [ck, c] of cells) {
        for (const arm of ['alone', 'prm']) {
          const oursArm = arm === 'alone' ? 'SUB5 alone' : 'SUB5 +PRM';
          const avgArm = arm === 'alone' ? 'SUB5 mean' : 'SUB5 +PRM mean';
          for (const mk of ['main', 'within_auc', 'crop_kept_a05', 'crop_auroc']) {
            const metric = mk === 'main' ? MAIN[c.benchmark] : mk;
            const ours = v.get([ck, oursArm, metric].join('|'));
            if (ours === undefined) continue;
            const avg = v.get([ck, avgArm, metric].join('|'));
            const prmV = v.get([ck, c.benchmark === 'prmbench' && metric.startsWith('crop') ? 'PRM (1-reward)' : 'PRM', metric].join('|'));
            const pubC = published.filter((p) => p.benchmark === c.benchmark && p.metric === metric && (p.prm === 'any' || p.prm === c.prm)).map((p) => Number(p.value));
            const pubV = pubC.length ? Math.max(...pubC) : undefined;
            const beatsAll = avg !== undefined && prmV !== undefined && pubV !== undefined && ours > avg && ours > prmV && ours > pubV;
            for (const cmp of COMPARATORS) {
              let cv, label, paper = 'Our run (same cell)', paperShort = '';
              if (cmp.startsWith('pub:')) {
                const acc = cmp.slice(4);
                const cand = published.filter((p) => p.benchmark === c.benchmark && p.metric === metric && (p.prm === 'any' || p.prm === c.prm) && (acc === 'any' || p.access === acc));
                if (!cand.length) continue;
                const best = cand.reduce((a, b) => (Number(b.value) > Number(a.value) ? b : a));
                cv = Number(best.value); label = best.competitor; paper = best.paper; paperShort = best.paper_short;
              } else {
                const a = cmp === 'PRM' && c.benchmark === 'prmbench' && metric.startsWith('crop') ? 'PRM (1-reward)' : cmp;
                cv = v.get([ck, a, metric].join('|'));
                if (cv === undefined) continue;
                label = cmp === 'PRM' ? 'The PRM itself' : cmp;
              }
              out.push({key: [ck, arm, mk, cmp].join('|'), benchmark: c.benchmark, benchmark_label: c.benchmark_label, bench_order: Number(c.bench_order),
                        mode: c.mode, prm: c.prm, prm_label: c.prm_label, sample: c.sample, arm, metric_key: mk, metric_label: c['label_' + mk] || metric,
                        cmp, cmp_label: label, cmp_paper: paper, cmp_paper_short: paperShort, ours, avg: avg === undefined ? null : avg, cmp_value: cv,
                        d_ours: ours - cv, d_avg: avg === undefined ? null : avg - cv, beats_all: beatsAll});
            }
          }
        }
      }
      return out;
    },
  });

  