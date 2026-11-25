const $=s=>document.querySelector(s);
const motion=matchMedia('(prefers-reduced-motion: reduce)');
const blocks=[...document.querySelectorAll('[data-show]')];
// Measure actual Korean line breaks before applying the stagger.
function splitLines(p){
 const source=p.dataset.source||p.textContent;
 p.dataset.source=source;p.textContent='';
 const tokens=source.split(/\s+/);let groups=[],last=-1;
 for(const word of tokens){
  const part=document.createElement('span');part.textContent=word;p.append(part,document.createTextNode(' '));
  const top=part.offsetTop;
  if(top!==last){groups.push([]);last=top}groups.at(-1).push(word);
 }
 p.replaceChildren();groups.forEach((g,i)=>{const row=document.createElement('span'),inner=document.createElement('span');row.className='wordline';inner.style.setProperty('--delay',i*85+'ms');inner.textContent=g.join(' ')+' ';row.append(inner);p.append(row)});
}
const copy=[...document.querySelectorAll('.profile-intro p,.story p')];copy.forEach(splitLines);
if('IntersectionObserver' in window){document.documentElement.classList.add('js');const obs=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add('seen');obs.unobserve(e.target)}}),{rootMargin:'0px 0px -5% 0px'});blocks.forEach(el=>obs.observe(el))}
const canvas=$('canvas'),ctx=canvas.getContext('2d');let w,h;
function fit(){const box=canvas.getBoundingClientRect();w=box.width;h=box.height;const dpr=Math.min(devicePixelRatio||1,1.75);canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0)}
function paint(){
 ctx.clearRect(0,0,w,h);const mob=w<701,S=mob?w*.56:Math.min(w*.29,h*.60), cx=w*(mob?.78:.755),cy=h*(mob?.625:.50),phase=motion.matches?0:Math.min(scrollY/h,1)*.35;
 // Project a pleated toroidal ribbon, then sort its strands from back to front.
 const lines=[];
 for(let i=0;i<152;i++){const t=i/152*2*Math.PI,pts=[];let depth=0;
 for(let j=0;j<=72;j++){const u=j/72*2*Math.PI,r=0.66+.28*Math.cos(u)+.13*Math.sin(3*t+u+phase);
 const X=r*Math.cos(t),Y=r*Math.sin(t),Z=.37*Math.sin(u)+.24*Math.sin(2*t+phase);
 const v=Y*.52-Z*.85,z=Y*.85+Z*.52;depth+=z;
 pts.push([cx+S*(X*.86+v*.5)*1.23,cy+S*(-X*.5+v*.86)*1.23]);}
 lines.push({pts,depth:depth/73});}
 lines.sort((a,b)=>a.depth-b.depth);for(const {pts,depth:z} of lines){ctx.beginPath();pts.forEach(([x,y],k)=>k?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.strokeStyle='rgba(222,171,126,'+Math.min(.89,.22+(z+1)*.28)+')';ctx.lineWidth=.83;ctx.stroke();}
}
let queued=false;function frame(){queued=false;const root=document.documentElement;const clamp=n=>Math.max(0,Math.min(1,n));$('.progress').style.transform='scaleX('+clamp(scrollY/(root.scrollHeight-innerHeight))+')';const box=$('.story').getBoundingClientRect();root.style.setProperty('--journey',clamp((innerHeight*.55-box.top)/(box.height*.78)));const current=$('#project').getBoundingClientRect().top<innerHeight*.55?'#project':$('#profile').getBoundingClientRect().top<innerHeight*.55?'#profile':'';document.querySelectorAll('nav a').forEach(a=>{if(a.hash===current)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current')});if(scrollY<h)paint()}
function update(){if(!queued){queued=true;requestAnimationFrame(frame)}}
let width=innerWidth;function resize(){if(width!==innerWidth){copy.forEach(splitLines);width=innerWidth}fit();update()}
fit();update();addEventListener('scroll',update,{passive:true});addEventListener('resize',resize);motion.addEventListener('change',update);

document.fonts.ready.then(()=>{copy.forEach(splitLines);update()});
