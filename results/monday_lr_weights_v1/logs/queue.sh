cd /c/Users/omris/TAU/hallucination_detection/results/monday_lr_weights_v1
while true; do
  if grep -q "N25 done" logs/chain4.log; then break; fi
  n=$(grep -c Traceback logs/chain4.log)
  if [ "$n" -ge 2 ]; then break; fi
  sleep 20
done
for b in N56 N11 N22 N13 N14 N16; do
  if [ ! -f cache/$b.pkl ]; then python run.py bank $b; fi
done
echo QUEUE_FINISHED
