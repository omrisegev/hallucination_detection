const {chromium}=require('playwright');const fs=require('fs');
(async()=>{const b=await chromium.launch();
for(const f of ['prism-result','crop-result','slider-result','ares-result','bracha']){
 const h=fs.readFileSync(`deck2/project/slides/${f}.html`,'utf8');const m=h.match(/<x-embed[^>]*>([\s\S]*)<\/x-embed>/);
 const p=await b.newPage({viewport:{width:1700,height:720}});const errs=[];p.removeAllListeners('pageerror');p.on('pageerror',e=>errs.push(e.message));
 await p.setContent(m[1]);const n=await p.evaluate(()=>document.querySelectorAll('svg *').length);
 await p.screenshot({path:`shot_${f}.png`});console.log(f,n,errs.join('|'));}
await b.close();})();
