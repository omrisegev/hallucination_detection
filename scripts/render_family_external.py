"""Render audited family-transfer results as publication-ready plots and a local gallery."""
import argparse
import html
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAMES = {
 "B11_lsml": "Bank11 L-SML", "F15_tailtie_lsml": "Family15 tail20 L-SML",
 "F15_cov_lsml": "Family15 continuous L-SML", "K28_cov_lsml": "Filtered28 L-SML",
 "K28_equal": "Filtered28 equal", "F15_equal": "Family15 equal",
 "B11_equal": "Bank11 equal", "B11_partition_equal": "Bank11 partition equal",
 "A48o_equal": "All48 oriented equal", "ct7": "CT7 reference"}
CELL_NAMES = {"hard2verify_qwen3_8b": "Hard2Verify / Qwen3-8B",
 "socratic_qwen3_8b": "Socratic / Qwen3-8B", "socratic_qwq32b": "Socratic / QwQ-32B"}
COLORS = {a: "#D86A22" if a == "F15_tailtie_lsml" else "#266F9B" if "lsml" in a else "#8795A2" if a != "ct7" else "#625D93" for a in NAMES}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "savefig.facecolor": "white", "pdf.fonttype": 42})


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def valid(value):
    return value is not None and np.isfinite(value) and 0 <= value <= 1


def pct(value):
    return 100*value if valid(value) else np.nan


def table_value(value):
    return f"{100*value:.2f}" if valid(value) else "undefined"


def save(fig, directory, name, subtitle=None):
    if subtitle:
        engine=fig.get_layout_engine()
        if engine is not None:
            engine.set(rect=(0, .05, 1, .95))
        fig.text(.01, .008, subtitle, fontsize=9, color="#475569")
    fig.savefig(directory/(name+".png"), dpi=180, bbox_inches="tight")
    fig.savefig(directory/(name+".pdf"), bbox_inches="tight")
    plt.close(fig)


def matrix(ax, values, rows, columns, title, vmin=0, vmax=100, cmap="viridis"):
    array = np.asarray(values, float)
    masked = np.ma.masked_invalid(array)
    im = ax.imshow(masked, vmin=vmin, vmax=vmax, aspect="auto", cmap=cmap)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
    ax.set_title(title)
    for i in range(len(rows)):
        for j in range(len(columns)):
            x = array[i,j]
            ax.text(j,i,f"{x:.1f}" if np.isfinite(x) else "NA",
                    ha="center",va="center",fontsize=8,color="white" if np.isfinite(x) and x < (vmin+vmax)/2 else "black")
    return im


def make_plots(out, metrics, contrasts, literature):
    plotdir = out/"plots"
    plotdir.mkdir(exist_ok=True)
    arms = list(NAMES)
    cells = list(CELL_NAMES)
    labels = [NAMES[a] for a in arms]
    captions = []
    fig, axes = plt.subplots(1,3,figsize=(17,7),sharey=True,layout="constrained")
    for ax,cell in zip(axes,cells):
        m=metrics[cell]; values=[pct(m["arms"][a]["metric"]) for a in arms]
        low=np.array([100*m["arms"][a]["ci95_descriptive"][0] for a in arms])
        high=np.array([100*m["arms"][a]["ci95_descriptive"][1] for a in arms])
        y=np.arange(len(arms))
        ax.barh(y,values,color=[COLORS[a] for a in arms],height=.65)
        ax.errorbar(values,y,xerr=[np.maximum(0,np.array(values)-low),np.maximum(0,high-np.array(values))],
                    fmt="none",ecolor="#18212B",capsize=2,lw=1)
        for i,v in enumerate(values):ax.text(min(97,high[i]+1),i,f"{v:.2f}",va="center",fontsize=9)
        ax.set_yticks(y,labels);ax.set_xlim(0,100);ax.set_title(CELL_NAMES[cell])
        ax.set_xlabel("Balanced F1 (%)" if m["benchmark"]=="hard2verify" else "PRMScore (%)")
        ax.grid(axis="x",alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("All 10 frozen alternatives on complete external populations",fontsize=16)
    save(fig,plotdir,"01_all_methods","Bars are measured; intervals are descriptive 95% source-question bootstrap. Orange: candidate; blue: learned fusion; grey: equal controls.")
    captions.append(("01_all_methods","All alternatives and official primary scores. Metrics differ by benchmark; never averaged."))
    fig,axes=plt.subplots(1,2,figsize=(17,8),sharey=True,layout="constrained")
    for ax,cell in zip(axes,cells[1:]):
        y=np.arange(len(arms))
        for key,offset,color,label in (("f1_correct",-.2,"#367FAC","F1 correct"),("f1_error",.2,"#CE7B36","F1 error")):
            vals=[pct(metrics[cell]["arms"][a][key]) for a in arms]
            bars=ax.barh(y+offset,vals,height=.36,color=color,label=label)
            ax.bar_label(bars,fmt="%.1f",padding=3,fontsize=8)
        ax.set_yticks(y,labels);ax.set_xlim(0,105);ax.set_xlabel("Class F1 (%)")
        ax.set_title(CELL_NAMES[cell]);ax.legend(loc="lower right");ax.grid(axis="x",alpha=.2)
    axes[0].invert_yaxis();fig.suptitle("PRMScore = (F1 correct + F1 error) / 2",fontsize=16)
    save(fig,plotdir,"02_prmscore_components","Both components come from pooled step confusion counts, not an average of answer-level F1 scores.")
    captions.append(("02_prmscore_components","The two class F1 scores that form official PRMScore."))
    fig,axes=plt.subplots(1,2,figsize=(15,8),layout="constrained")
    keys=["precision_correct","recall_correct","precision_error","recall_error"]
    for ax,cell in zip(axes,cells[1:]):
        matrix(ax,[[pct(metrics[cell]["arms"][a][k]) for k in keys] for a in arms],labels,
               ["Correct precision","Correct recall","Error precision","Error recall"],CELL_NAMES[cell])
    fig.suptitle("Why the PRMScore components change",fontsize=16)
    save(fig,plotdir,"03_precision_recall","Values in percent; undefined official sentinel values remain NA in the plot and are retained in METRICS.json.")
    captions.append(("03_precision_recall","Precision and recall for both correct and erroneous steps."))
    fig,axes=plt.subplots(1,2,figsize=(15,8),layout="constrained")
    m=metrics[cells[0]]
    matrix(axes[0],[[pct(m["arms"][a][k]) for k in ("recall_correct","recall_error","metric")] for a in arms],
           labels,["Correct recall","Error recall","Balanced F1"],CELL_NAMES[cells[0]])
    vals=[pct(m["arms"][a]["flawless_false_flag_rate"]) for a in arms]
    axes[1].barh(range(len(arms)),vals,color=[COLORS[a] for a in arms])
    axes[1].set_yticks(range(len(arms)),labels);axes[1].invert_yaxis();axes[1].set_xlim(0,105)
    axes[1].set_xlabel("Entirely correct answers with any error flag (%)")
    axes[1].set_title(f"False alarms on {m['flawless_answers']} entirely correct answers")
    for i,a in enumerate(arms):axes[1].text(vals[i]+.7,i,f"{m['arms'][a]['flawless_false_flag_answers']}/{m['flawless_answers']}",va="center",fontsize=8)
    fig.suptitle("Hard2Verify: harmonic mean of the two class recalls",fontsize=16)
    save(fig,plotdir,"04_hard2_components","Hard2Verify Balanced F1 is not PRMScore. False-alarm panel is a secondary answer-level diagnostic.")
    captions.append(("04_hard2_components","Hard2Verify class recalls and false alarms on completely correct answers."))
    fig,axes=plt.subplots(1,3,figsize=(20,6),sharey=True,layout="constrained")
    for ax,cell in zip(axes,cells):
        selected=[r for r in contrasts if r["cell"]==cell]
        for i,r in enumerate(selected):
            d=100*r["delta"];lo,hi=np.asarray(r["ci_bonferroni"])*100
            ax.plot([lo,hi],[i,i],color="#266F9B",lw=2)
            ax.plot(d,i,"o",color="#D86A22" if r["left"]=="F15_tailtie_lsml" else "#266F9B")
            ax.text(hi+.12,i,f"{d:+.2f}",va="center",fontsize=8)
        ax.axvline(0,color="#475569",lw=1,ls="--")
        ax.set_yticks(range(len(selected)),[NAMES[r["left"]]+"\nminus "+NAMES[r["right"]] for r in selected])
        ax.set_title(CELL_NAMES[cell]);ax.set_xlabel("Difference in official score (percentage points)");ax.grid(axis="x",alpha=.2)
    axes[0].invert_yaxis();fig.suptitle("Six locked contrasts per cell; paired source-question uncertainty",fontsize=16)
    save(fig,plotdir,"05_paired_contrasts","100,000 draws; seed 20260924; Bonferroni correction across all 18 contrasts. Exploratory follow-up, not untouched confirmation.")
    captions.append(("05_paired_contrasts","All 18 registered comparisons with multiplicity-adjusted intervals."))
    for cell in cells:
        categories=list(metrics[cell]["categories"])
        fig,ax=plt.subplots(figsize=(max(9,len(categories)*1.5),8),layout="constrained")
        matrix(ax,[[pct(metrics[cell]["categories"][c]["arms"][a]["metric"]) for c in categories] for a in arms],
               labels,[c+"\nN="+str(metrics[cell]["categories"][c]["answers"]) for c in categories],
               CELL_NAMES[cell]+" / official category score")
        name="06_categories_"+cell
        save(fig,plotdir,name,"Category panels are descriptive. Undefined/degenerate official values are NA, not replaced by zero.")
        captions.append((name,"Official category panels; exact per-class components are retained in METRICS.json."))
    for panel,title,name in (("length_bins","Number of original steps","07_length"),("relative_position","Relative step position quartile","08_position")):
        fig,axes=plt.subplots(1,3,figsize=(19,8),layout="constrained")
        for ax,cell in zip(axes,cells):
            order=[x for x in (["1-4","5-8","9-16","17+"] if panel=="length_bins" else ["Q1","Q2","Q3","Q4"]) if x in metrics[cell][panel]]
            matrix(ax,[[pct(metrics[cell][panel][c]["arms"][a]["metric"]) for c in order] for a in arms],labels,order,CELL_NAMES[cell])
            ax.set_xlabel(title)
        fig.suptitle("Fixed descriptive strata; no target threshold tuning",fontsize=16)
        save(fig,plotdir,name,"Each stratum recomputes the benchmark's own official formula from pooled counts. Class-degenerate strata may be undefined.")
        captions.append((name,title+" diagnostics across every method."))
    height=max(9,.38*(3+max(len(literature[b]["rows"]) for b in ("hard2verify","socratic"))))
    fig,axes=plt.subplots(1,3,figsize=(20,height),layout="constrained")
    for ax,cell in zip(axes,cells):
        bench=metrics[cell]["benchmark"]
        selected=[r for r in literature.get("rows",[]) if r.get("benchmark")==bench and r.get("primary_score") is not None]
        plotlabels=[NAMES[a]+" (measured)" for a in ("F15_tailtie_lsml","B11_lsml","F15_equal")]
        values=[pct(metrics[cell]["arms"][a]["metric"]) for a in ("F15_tailtie_lsml","B11_lsml","F15_equal")]
        colors=[COLORS[a] for a in ("F15_tailtie_lsml","B11_lsml","F15_equal")]
        for row in selected:
            plotlabels.append(row["method"]+" (paper)")
            values.append(float(row["primary_score"])*(100 if row.get("scale","fraction")=="fraction" else 1));colors.append("#C2C9D1")
        bars=ax.barh(range(len(values)),values,color=colors,height=.65)
        for b in list(bars)[3:]:b.set_hatch("///");b.set_edgecolor("#64748B")
        ax.set_yticks(range(len(values)),plotlabels);ax.invert_yaxis();ax.set_xlim(0,105)
        ax.bar_label(bars,fmt="%.2f",padding=3,fontsize=8);ax.set_title(CELL_NAMES[cell])
        ax.set_xlabel("Balanced F1 (%)" if bench=="hard2verify" else "PRMScore (%)")
    fig.suptitle("Measured source-frozen transfer versus paper-reported context",fontsize=16)
    save(fig,plotdir,"09_literature","Hatched bars are published values, not reproduced measurements. Different model sizes, prompts and calibration access; no paired significance comparison.")
    captions.append(("09_literature","Published context is visually separated from this run; unavailable components are never inferred."))
    fig,axes=plt.subplots(1,3,figsize=(21,height),layout="constrained")
    keys=["f1_correct","f1_error","precision_correct","precision_error","recall_correct","recall_error"]
    for ax,cell in zip(axes,cells):
        bench=metrics[cell]["benchmark"]
        measured=("F15_tailtie_lsml","B11_lsml","F15_equal")
        labs=[NAMES[a]+" (measured)" for a in measured]
        values=[[pct(metrics[cell]["arms"][a][k]) for k in keys] for a in measured]
        for row in literature[bench]["rows"]:
            labs.append(row["model"]+" (paper)")
            values.append([row.get(k) if row.get(k) is not None else np.nan for k in keys])
        matrix(ax,values,labs,["F1 correct","F1 error","Precision correct","Precision error","Recall correct","Recall error"],CELL_NAMES[cell])
        ax.axhline(2.5,color="black",lw=2)
    fig.suptitle("Published class components: report available values, preserve missing values",fontsize=16)
    save(fig,plotdir,"10_literature_components","Rows below the black line are paper-reported. Missing class F1/precision values remain NA; they are not inferred from rounded recalls.")
    captions.append(("10_literature_components","Measured and available published class components, with missing literature values explicit."))
    fig,axes=plt.subplots(1,3,figsize=(18,8),sharey=True,layout="constrained")
    for ax,cell in zip(axes,cells):
        m=metrics[cell];y=np.arange(len(arms))
        for offset,color,label,field in ((-.2,"#266F9B",f"Full (N={m['answers']})","full"),
                                         (.2,"#43A38D",f"Observed-disjoint (N={m['disjoint_answers']})","disjoint")):
            values=[pct(m["arms"][a]["metric"] if field=="full" else m["arms"][a]["disjoint"]["metric"]) for a in arms]
            bars=ax.barh(y+offset,values,height=.36,color=color,label=label)
            ax.bar_label(bars,fmt="%.2f",padding=3,fontsize=8)
        ax.set_yticks(y,labels);ax.set_xlim(0,105);ax.set_title(CELL_NAMES[cell])
        ax.set_xlabel("Balanced F1 (%)" if m["benchmark"]=="hard2verify" else "PRMScore (%)")
        ax.legend(loc="lower right",fontsize=8);ax.grid(axis="x",alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Observed-overlap exclusion: sensitivity of every frozen alternative",fontsize=16)
    save(fig,plotdir,"11_disjoint_sensitivity","Sensitivity panel, not an untouched test. Exclusion uses exact normalized questions and source-ID component closure; semantic contamination is not ruled out.")
    captions.append(("11_disjoint_sensitivity","Full versus observed-disjoint official scores for all ten alternatives."))
    diagnostic_path=out/"REPRESENTATION_DIAGNOSTICS.json"
    if diagnostic_path.exists():
        diagnostic=load(diagnostic_path)
        fig,axes=plt.subplots(2,3,figsize=(17,9),layout="constrained")
        for col,cell in enumerate(cells):
            d=diagnostic["cells"][cell]
            for row,(key,ylabel) in enumerate((
                    ("mean_signed_contribution_before_final_z","Mean signed contribution"),
                    ("mean_absolute_contribution_before_final_z","Mean absolute contribution"))):
                ax=axes[row,col];v=np.asarray(d[key])
                for j,(label,color) in enumerate(zip(d["contribution_columns"],("#D86A22","#266F9B"))):
                    ax.plot(range(1,5),v[:,j],marker="o",lw=2,color=color,label=label)
                ax.axhline(0,color="#64748B",lw=.7)
                ax.set_xticks(range(1,5),["Q1","Q2","Q3","Q4"])
                ax.set_xlabel("Relative original step position");ax.set_ylabel(ylabel)
                ax.grid(alpha=.2);ax.legend(fontsize=9)
                if row==0:ax.set_title(CELL_NAMES[cell])
        mass=diagnostic["absolute_weight_mass_cusum"]
        fig.suptitle(f"Candidate score decomposition before final answer-z / CUSUM source weight mass {mass:.4f}",fontsize=15)
        save(fig,plotdir,"12_position_contributions","Descriptive pooled step contributions; no labels or refitting. Weight mass and contributions are not identified group reliability or causal effects.")
        captions.append(("12_position_contributions","CUSUM versus other-family signed and absolute score contributions by relative position."))
        names=list(diagnostic["cells"][cells[0]]["constant_channel_answers"])
        values=[[100*diagnostic["cells"][cell]["constant_channel_answers"][name]/diagnostic["cells"][cell]["n_checked"] for cell in cells] for name in names]
        fig,ax=plt.subplots(figsize=(10,max(10,.27*len(names))),layout="constrained")
        matrix(ax,values,names,[CELL_NAMES[cell] for cell in cells],"Constant channels within an answer (%)",cmap="magma")
        save(fig,plotdir,"13_constant_channels","All 48 channels; constant means within-answer standard deviation <= 1e-12. Descriptive representation property, not an accuracy or reliability estimate.")
        captions.append(("13_constant_channels","Frequency of constant within-answer channels across complete populations; family counts are also in JSON."))
    return captions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=ROOT/"results/family_tail_external_v1")
    args=parser.parse_args();out=args.root
    if not (out/"RED_TEAM.md").exists():
        raise ValueError("Independent RED_TEAM.md required before reporting results")
    metrics=load(out/"METRICS.json");contrasts=load(out/"CONTRASTS.json")
    literature=load(out/"LITERATURE_CONTEXT.json")
    literature["rows"]=[dict(row,benchmark=bench,method=row["model"],
        primary_score=row[literature[bench]["primary_metric"]],scale="percent",
        url=literature["sources"][bench]["url"],protocol_note=literature[bench]["access"])
        for bench in ("hard2verify","socratic") for row in literature[bench]["rows"]]
    captions=make_plots(out,metrics,contrasts,literature)
    md=["# Family15 tail20 external transfer: exploratory follow-up","",
        "All ten methods retain source-frozen weights, calibration and preprocessing. Existing telemetry was reused; no new inference or GPU training.",
        "External benchmarks had already informed the discussion. These results are an exploratory follow-up, not independent confirmation.",
        "The candidate learns within-group tail weights; its two-group between-group split is fixed by normalization. Equal fusion remains a control.","",
        "## Complete-population official scores","",
        "| Method | Hard2Verify / Qwen3 Balanced F1 | Socratic / Qwen3 PRMScore | Socratic / QwQ PRMScore |",
        "|---|---:|---:|---:|"]
    for arm in NAMES:
        md.append("| "+NAMES[arm]+" | "+" | ".join(table_value(metrics[c]["arms"][arm]["metric"]) for c in CELL_NAMES)+" |")
    md += ["","Scores are percentages. Hard2Verify uses the harmonic mean of correct/error recall; Socratic PRMScore is the mean of correct/error F1. They must not be averaged.","",
           "## Registered contrasts","",
           "| Cell | Contrast | Difference (points) | Adjusted interval |","|---|---|---:|---|"]
    for r in contrasts:
        lo,hi=np.asarray(r["ci_bonferroni"])*100
        md.append(f"| {CELL_NAMES[r['cell']]} | {NAMES[r['left']]} minus {NAMES[r['right']]} | {100*r['delta']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    md += ["","100,000 paired source-question bootstrap draws; seed 20260924; Bonferroni across all 18 contrasts. Source-overlap-excluded results are a secondary sensitivity panel in DISJOINT_CONTRASTS.json.","",
           "## Plots",""]
    for name,caption in captions:
        md += [f"### {caption}",f"![{caption}](plots/{name}.png)",f"[Vector PDF](plots/{name}.pdf)",""]
    md += ["## Coverage and safeguards",""]
    for cell,m in metrics.items():
        md.append(f"- {CELL_NAMES[cell]}: {m['answers']} answers, {m['steps']} included steps, {m['groups']} source-question groups; observed-disjoint panel {m['disjoint_answers']} answers. {m['empty_steps']} empty steps retain the locked missing-step decisions.")
    md += ["","All predictions were sealed before annotation access in this follow-up. Every official total/category component is replayed through pinned author code. Descriptive PRMScore strata preserve official undefined sentinels; plots show NA for such components.",
           "Question overlap detection uses exact normalized text and source-ID component closure, not a semantic or pretraining contamination audit.",
           "No result licenses target tuning or selecting a subset of reported arms. Fresh unexposed evaluation is required after any further method changes.","",
           "## Published context","",
           "Published values are not reproduced runs and have different calibration, prompting, model-size and compute conditions. Missing class F1/precision/recall values remain missing.",""]
    for row in literature.get("rows",[]):
        score=row.get("primary_score")
        value="not available" if score is None else str(score)
        md.append(f"- **{row.get('benchmark')} / {row.get('method')}**: {value} ({row.get('scale','fraction')}). {row.get('protocol_note','')} [Source]({row.get('url','')}).")
    md += ["","## Audit and machine-readable evidence","",
           "[Independent audit](RED_TEAM.md), [metrics and all components](METRICS.json), [paired contrasts](CONTRASTS.json), [official replay](OFFICIAL_METRIC_REPLAY.json), [provenance](EVALUATION_PROVENANCE.json).",""]
    he=out/"INTERPRETATION_HE.md"
    if he.exists():md += [he.read_text(encoding="utf8")]
    (out/"REPORT.md").write_text("\n".join(md)+"\n",encoding="utf8",newline="\n")
    hebrew_card=""
    if he.exists():
        he_text=he.read_text(encoding="utf8")
        hebrew_card='<section dir="rtl" lang="he"><div style="white-space:pre-wrap;line-height:1.7">'+html.escape(he_text)+'</div></section>'
    cards="".join(f'<section><h2>{html.escape(caption)}</h2><a href="plots/{name}.pdf">Vector PDF</a><img loading="lazy" src="plots/{name}.png" alt="{html.escape(caption)}"></section>' for name,caption in captions)
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Family15 external transfer</title><style>body{font:16px system-ui;max-width:1550px;margin:auto;padding:32px;background:#f1f5f9;color:#172033}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}a{color:#12649a}h1{font-size:32px}p{max-width:1000px}</style><h1>Family15 tail20: external benchmark comparison</h1><p>Ten frozen alternatives, complete populations. Exploratory follow-up; no new GPU inference. Published comparator bars are context, not reproduced measurements.</p><p><a href="REPORT.md">Full technical report</a> | <a href="RED_TEAM.md">Independent audit</a> | <a href="METRICS.json">Exact metrics</a></p>'+hebrew_card+cards+'</html>'
    (out/"REPORT.html").write_text(page,encoding="utf8",newline="\n")
    print("Rendered",len(captions),"PNG/PDF figure pairs and Markdown/HTML reports")


if __name__=="__main__":
    main()
