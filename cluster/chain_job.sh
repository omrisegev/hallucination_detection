#!/bin/bash
# Extend a running Slurm job into a LINEAR dependency chain of continuation links.
#
# Why this exists
# ---------------
# `--requeue` covers preemption only. A job that catches SIGTERM, checkpoints, and exits 85
# on its own simply ends as `FAILED 85:0`, and nothing resumes it. Any acquisition that will
# not finish inside one wall therefore needs its continuation links submitted UP FRONT.
# CLAUDE.md records three jobs already lost to exactly this (176043, 176044, 177759), and the
# paper_exact runbook's own M2 sizing says "~2 requeues per shard" while the queue held none.
#
# LINEAR, never fan-out
# ---------------------
# Two links that both declare `afterany:<same job>` become eligible together and race on the
# same output directory. Atomic shard replacement prevents corruption but not lost work, so
# link N+1 must depend on link N, not on the original job. This script enforces that: within
# one chain each link depends on the previous link and on nothing else.
#
# Independent chains for independent outputs are fine and are the normal case here: each M2
# shard owns its own `part_NN/` directory exclusively, so 24 parallel linear chains do not
# race with one another.
#
# Safety
# ------
#  * refuses a job that already has a queued successor (no accidental double-chaining)
#  * reuses the job's own recorded Command verbatim — same driver, same --out, same shard, so
#    resume is by stable trace key and the manifest refuses any pinned-field drift
#  * a link whose predecessor already finished the work is a no-op: the driver finds nothing
#    pending and exits 0. Wasted model load, no wasted acquisition.
#  * --dry-run prints the exact sbatch commands and submits nothing
#
# Usage:
#   bash cluster/chain_job.sh --links 2 --dry-run 205333 205334
#   bash cluster/chain_job.sh --links 2 --name-filter pe_m2_        # all matching running jobs
#   bash cluster/chain_job.sh --links 2 205333 205334               # submit for real
set -euo pipefail

REMOTE=${REMOTE:-aircc}
LINKS=2
DRY=0
FILTER=""
JOBS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --links)       LINKS="$2"; shift 2 ;;
        --dry-run)     DRY=1; shift ;;
        --name-filter) FILTER="$2"; shift 2 ;;
        --remote)      REMOTE="$2"; shift 2 ;;
        -h|--help)     sed -n '2,40p' "$0"; exit 0 ;;
        -*)            echo "unknown flag $1" >&2; exit 2 ;;
        *)             JOBS+=("$1"); shift ;;
    esac
done

# The AIRCC login node prints a policy banner on every ssh, on stdout, even for a
# non-interactive command. Captured naively, `jobid=$(ssh ... sbatch --parsable)` would return
# banner text instead of a job id and the chain would be built on garbage. Fence the real
# output between sentinels and keep only what lies between them.
# `printf '\n__END__\n'` rather than `echo __END__`: one of the remote commands pipes through
# `tr '\n' ' '`, which would otherwise pull the closing sentinel onto the content line, leave
# the sed range unterminated, and silently return nothing.
rsh() {
    ssh "$REMOTE" "printf '__BEGIN__\n'; { $1; }; printf '\n__END__\n'" 2>/dev/null \
        | sed -n '/^__BEGIN__$/,/^__END__$/p' | sed '1d;$d'
}

if [ -n "$FILTER" ]; then
    if [ ${#JOBS[@]} -gt 0 ]; then
        echo "give either explicit job ids or --name-filter, not both" >&2; exit 2
    fi
    mapfile -t JOBS < <(rsh "squeue -u \$USER -h -t RUNNING,PENDING -o '%i %j'" \
        | awk -v f="$FILTER" '$2 ~ "^"f {print $1}')
fi

if [ ${#JOBS[@]} -eq 0 ]; then
    echo "no jobs selected" >&2; exit 2
fi

echo "chaining ${#JOBS[@]} job(s) x $LINKS link(s) on $REMOTE  (dry-run=$DRY)"

# One ssh round trip for the whole queue: job id, name, and its dependency string. Used both
# to recover each job's Command and to detect an existing successor.
QUEUE=$(rsh "squeue -u \$USER -h -t RUNNING,PENDING -o '%i|%j|%E'")

for JID in "${JOBS[@]}"; do
    # An existing successor means this chain was already built. Adding another link now would
    # create the fan-out this script exists to prevent.
    if echo "$QUEUE" | grep -q "afterany:${JID}\b"; then
        echo "SKIP $JID: a queued job already depends on it (chain exists)" >&2
        continue
    fi

    INFO=$(rsh "scontrol show job $JID 2>/dev/null | tr '\n' ' '")
    CMD=$(echo "$INFO"     | grep -o 'Command=[^ ]*'    | head -1 | cut -d= -f2-)
    NAME=$(echo "$INFO"    | grep -o 'JobName=[^ ]*'    | head -1 | cut -d= -f2-)
    WORKDIR=$(echo "$INFO" | grep -o 'WorkDir=[^ ]*'    | head -1 | cut -d= -f2-)
    TLIMIT=$(echo "$INFO"  | grep -o 'TimeLimit=[^ ]*'  | head -1 | cut -d= -f2-)
    PART=$(echo "$INFO"    | grep -o 'Partition=[^ ]*'  | head -1 | cut -d= -f2-)
    QOS=$(echo "$INFO"     | grep -o ' QOS=[^ ]*'       | head -1 | cut -d= -f2-)
    STDOUT=$(echo "$INFO"  | grep -o 'StdOut=[^ ]*'     | head -1 | cut -d= -f2-)

    if [ -z "$CMD" ]; then
        echo "SKIP $JID: could not recover Command from scontrol" >&2
        continue
    fi

    # `scontrol show job` reports Command= as the sbatch script path ONLY — the driver and its
    # arguments are POSITIONAL arguments to that script and appear nowhere in scontrol. A
    # continuation built from scontrol alone would run the sbatch with no arguments and die on
    # its own `${1:?...}` guard. The arguments survive in the job's log, which the wrapper
    # echoes as `[sbatch] target=<path> args: <...>` precisely so a resubmit can be
    # reconstructed. Recover them from there.
    ARGLINE=$(rsh "grep -m1 '^\[sbatch\] target=' '$STDOUT' 2>/dev/null || true")
    if [ -z "$ARGLINE" ]; then
        echo "SKIP $JID: no '[sbatch] target=' line in $STDOUT — cannot reconstruct the" >&2
        echo "       driver arguments, and guessing them would launch a different run." >&2
        continue
    fi
    TARGET=$(echo "$ARGLINE" | sed -n 's/^\[sbatch\] target=\([^ ]*\) args:.*/\1/p')
    DARGS=$(echo "$ARGLINE" | sed -n 's/^\[sbatch\] target=[^ ]* args: *//p')
    if [ -z "$TARGET" ]; then
        echo "SKIP $JID: could not parse target from: $ARGLINE" >&2
        continue
    fi

    # Partition and QOS are supplied on the command line, not by the script's #SBATCH block,
    # so a continuation that omits them lands in the default partition with default priority.
    EXTRA=""
    [ -n "$PART" ] && EXTRA="$EXTRA -p $PART"
    [ -n "$QOS" ] && [ "$QOS" != "(null)" ] && EXTRA="$EXTRA --qos=$QOS"

    echo
    echo "$JID ($NAME)  wall=$TLIMIT  partition=$PART  qos=$QOS"
    echo "  target: $TARGET $DARGS"

    PREV="$JID"
    for ((L = 1; L <= LINKS; L++)); do
        SUB="sbatch --parsable --dependency=afterany:$PREV --kill-on-invalid-dep=yes"
        SUB="$SUB -J ${NAME}_c${L} --time=$TLIMIT$EXTRA $CMD $TARGET $DARGS"
        if [ "$DRY" = "1" ]; then
            echo "  link $L: $SUB"
            PREV="<link${L}>"
        else
            # stderr is merged into the fenced region here, unlike every other rsh call. The
            # reason is specific: this project has a purchased GPU-hour cap, and sbatch
            # reports exhausting it ONLY on stderr ("Quota exceeded for project ..."). A
            # helper that discards stderr turns that into a blank job id and an unexplained
            # abort, which is exactly how it first presented.
            OUT=$(rsh "cd '$WORKDIR' && { $SUB; } 2>&1")
            NEW=$(echo "$OUT" | grep -oE '^[0-9]+' | head -1)
            NEW=${NEW%%;*}          # --parsable may append ;cluster
            # Never advance the chain on an unparsed job id: a later link declaring
            # afterany:<garbage> would either be rejected or, worse, held forever.
            if ! [[ "$NEW" =~ ^[0-9]+$ ]]; then
                echo "  ABORT chain for $JID: sbatch did not return a job id." >&2
                echo "$OUT" | sed 's/^/       | /' >&2
                break
            fi
            echo "  link $L -> job $NEW (afterany:$PREV)"
            PREV="$NEW"
        fi
    done
done

echo
if [ "$DRY" = "1" ]; then
    echo "DRY RUN — nothing submitted. Re-run without --dry-run to submit."
else
    echo "done. Verify with: ssh $REMOTE \"squeue -u \\\$USER -o '%i %j %T %E'\""
    echo "Each link depends ONLY on the previous link of its own chain. If you need to repair"
    echo "a fan-out, use: scontrol update jobid=<later> Dependency=afterany:<end-of-chain>"
fi
