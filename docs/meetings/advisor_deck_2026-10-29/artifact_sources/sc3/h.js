const fs=require('fs');let F;const dash={calc:(id,o)=>{F=o.fn}};
eval(fs.readFileSync(__dirname+'/code.js','utf8').replace(/^const /gm,'var '));
const out=F(JSON.parse(fs.readFileSync(__dirname+'/sc.json')),JSON.parse(fs.readFileSync(__dirname+'/pb.json')));
const m=new Map(), tot=new Set();
for(const r of out){ if(r.metric_key!=='main')continue; tot.add([r.benchmark,r.mode,r.prm,r.arm].join('|')); if(r.beats_all) m.set([r.benchmark,r.mode,r.prm_label,r.arm,r.sample].join(' | '), r.ours.toFixed(2)); }
console.log([...m].join('\n')); console.log('cells',tot.size,'pass',m.size);
