"""Target-free deterministic selection from corrected source-question metadata."""
import hashlib

CELLS=('prmbench_qwen3_8b','pb_gsm8k_q8','pb_math_q8','pb_olympiadbench_q8','pb_omnimath_q8')
BINS=((64,255),(256,1023),(1024,2048))


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def select_cohort(release,excluded,namespace,quota=8,cells=CELLS,bins=BINS):
    used=set(excluded);selected=[];support=[]
    if not isinstance(quota,int) or quota<=0:raise ValueError('INVALID_QUOTA')
    for cell in cells:
        rows=release['cells'][cell]['rows']
        if len({r['row_id'] for r in rows})!=len(rows):raise ValueError('DUPLICATE_ROW_ID')
        for lower,upper in bins:
            candidates=[r for r in rows if lower<=r['tokens']<=upper and r['group_id'] not in used]
            candidates.sort(key=lambda r:(digest(namespace+'/'+cell+'/'+r['group_id']),digest(r['row_id'])))
            count=0
            for row in candidates:
                if row['group_id'] in used:continue
                selected.append({**row,'cell':cell,'length_bin':[lower,upper],
                                 'uid':cell+'__'+digest(row['row_id'])[:16]})
                used.add(row['group_id']);count+=1
                if count==quota:break
            support.append({'cell':cell,'length_bin':[lower,upper],
                            'eligible_groups_at_bin_entry':len({r['group_id'] for r in candidates}),
                            'selected':count,'quota':quota,'shortfall':quota-count})
    return selected,support
