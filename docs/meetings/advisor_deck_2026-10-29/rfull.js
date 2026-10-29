const {chromium}=require('playwright');const fs=require('fs');
(async()=>{const b=await chromium.launch();
for(const f of process.argv.slice(2)){
 let h=fs.readFileSync(`deck2/project/slides/${f}.html`,'utf8');
 h=h.replace(/<x-embed([^>]*)>([\s\S]*?)<\/x-embed>/g,(m,a,inner)=>`<iframe${a.replace('style="','style="border:0;')} srcdoc="${inner.replace(/&/g,'&amp;').replace(/"/g,'&quot;')}"></iframe>`).replace(/<aside>[\s\S]*?<\/aside>/,'');
 const p=await b.newPage({viewport:{width:1920,height:1080}});
 await p.setContent(`<html><head><link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=Source+Serif+4&display=swap" rel="stylesheet"><style>*{margin:0;box-sizing:border-box}section{position:relative;width:1920px;height:1080px}</style></head><body>${h}</body></html>`);
 await p.waitForTimeout(800);await p.screenshot({path:`full_${f}.png`});console.log(f);}
await b.close();})();
