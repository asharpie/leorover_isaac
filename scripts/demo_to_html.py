#!/usr/bin/env python3
# scripts/demo_to_html.py
"""
Convert a record_demo.py .npz into a SINGLE self-contained interactive 3D HTML
replay. Open the output in any browser (internet needed once, for the three.js
CDN). All recorded controllers replay simultaneously on the identical scenario:
red = hybrid, blue = pure LQR, green = pure PPO (whichever legs the recording
contains). Trails are colored by measured wheel slip (green = rolling,
red = slipping). Terrain is the real raycast height grid, shaded darker where
the soil-softness field is sandier.

Usage:
    python3 scripts/demo_to_html.py evals/demo_<ts>.npz [-o demo.html]

Runs anywhere (laptop or box); needs only numpy.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

LEG_ORDER = ("hybrid", "lqr", "ppo")


def _r(a, nd=3):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def build_payload(npz_path: str) -> str:
    d = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    K = int(meta["num"]); G = int(meta["grid"]); half = float(meta["extent"])
    legs_present = [n for n in LEG_ORDER if f"{n}_pos" in d.files]
    origins = d["origins"]
    SKIP = 3   # drop the spawn-settle frames (rovers start 0.3 m up and drop in ~0.5 s)

    def h_at(e, xy):
        ix = np.clip(np.round((xy[:, 0] - origins[e, 0] + half) / (2 * half) * (G - 1)), 0, G - 1).astype(int)
        iy = np.clip(np.round((xy[:, 1] - origins[e, 1] + half) / (2 * half) * (G - 1)), 0, G - 1).astype(int)
        return d["heights"][e][iy, ix]

    scenarios = []
    for e in range(K):
        legs = {}
        clear = []
        for name in legs_present:
            T = int(d[f"{name}_done"][e]); T = T if T > 0 else d[f"{name}_pos"].shape[0] - 1
            s0 = SKIP if T + 1 > SKIP + 5 else 0
            sl = slice(s0, T + 1)
            p = d[f"{name}_pos"][sl, e]
            q = d[f"{name}_quat"][sl, e]          # isaac (w,x,y,z) -> three (x,y,z,w)
            clear.append(p[:, 2] - h_at(e, p[:, :2]))
            legs[name] = dict(
                pos=_r(p),
                quat=_r(np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], axis=1), 4),
                wheels=_r(d[f"{name}_wheels"][sl, e]),
                slip=_r(d[f"{name}_slip"][sl, e]),
                cte=_r(d[f"{name}_cte"][sl, e]),
            )
        # base-link height above ground: shift the drawn model down by this much so
        # the wheels visually touch the terrain (the physics was always in contact;
        # the recorded pose is the base-link origin, which sits above the contact).
        zoff = float(np.clip(np.median(np.concatenate(clear)), 0.0, 0.35))
        n = int(d["nwp"][e])
        scenarios.append(dict(
            origin=_r(origins[e]),
            zoff=round(zoff, 3),
            heights=_r(d["heights"][e], 2),
            soil=_r(d["soil"][e], 2),
            wps=_r(d["wps"][e, :n] + origins[e, :2]),
            legs=legs))
    return json.dumps(dict(meta=meta, grid=G, half=half, legs=legs_present,
                           scenarios=scenarios), separators=(",", ":"))


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Leo Rover demo replay</title>
<style>
 body{margin:0;background:#0e1116;color:#dde3ea;font:13px/1.4 system-ui,sans-serif;overflow:hidden}
 #hud{position:fixed;top:10px;left:10px;background:rgba(14,17,22,.85);padding:10px 12px;
      border:1px solid #2a3242;border-radius:8px;min-width:250px}
 #hud b{font-size:14px}
 #bar{position:fixed;bottom:10px;left:50%;transform:translateX(-50%);display:flex;gap:8px;
      align-items:center;background:rgba(14,17,22,.85);padding:8px 12px;border:1px solid #2a3242;border-radius:8px}
 button,select{background:#1b2330;color:#dde3ea;border:1px solid #2a3242;border-radius:6px;padding:4px 10px;cursor:pointer}
 button:hover{background:#26324a}
 input[type=range]{width:260px}
 #legend{position:fixed;top:10px;right:10px;background:rgba(14,17,22,.85);padding:10px 12px;
      border:1px solid #2a3242;border-radius:8px;font-size:12px}
 .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
</style></head><body>
<div id="hud"><b>Leo Rover episode replay</b><br>
 <span id="info"></span><br><br>
 <div id="rows"></div>
 t = <span id="time">0.0</span> s</div>
<div id="legend"><div id="legrows"></div>
 <span class="sw" style="background:#caa06a"></span>firm ground<br>
 <span class="sw" style="background:#6b4a2b"></span>soft sand<br>
 trail: <span style="color:#57d977">rolling</span> &rarr; <span style="color:#ff5c5c">slipping</span><br>
 <span style="color:#ffd23f">&#9644;</span> reference path</div>
<div id="bar">
 <select id="scen"></select>
 <button id="play">&#9646;&#9646;</button>
 <button id="restart">&#8634;</button>
 <select id="speed"><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select>
 <input id="scrub" type="range" min="0" max="1000" value="0">
 <label><input id="follow" type="checkbox" checked> follow</label>
 <label><input id="wire" type="checkbox"> heightfield</label>
</div>
<script type="importmap">{"imports":{
 "three":"https://unpkg.com/three@0.160.0/build/three.module.js",
 "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const DATA=/*__DEMO_DATA__*/;
const DT=DATA.meta.dt, G=DATA.grid, HALF=DATA.half;
const COL={hybrid:0xEE6677,lqr:0x4477AA,ppo:0x55BB66};
const CSS={hybrid:'#f2848f',lqr:'#7aa7e0',ppo:'#7fd98a'};
const LBL={hybrid:'Hybrid (LQR+PPO)',lqr:'Pure LQR',ppo:'Pure PPO'};
document.getElementById('info').textContent=
 `terrain ${DATA.meta.level}% | friction ${DATA.meta.friction} | ${DATA.meta.ckpt}`;
const rowsDiv=document.getElementById('rows'), legDiv=document.getElementById('legrows');
for(const k of DATA.legs){
 rowsDiv.insertAdjacentHTML('beforeend',
  `<span style="color:${CSS[k]}">${LBL[k]}</span> CTE <span id="cte_${k}">-</span> m, `+
  `slip <span id="slip_${k}">-</span><br>`);
 legDiv.insertAdjacentHTML('beforeend',
  `<span class="sw" style="background:${CSS[k]}"></span>${LBL[k]}<br>`);}

const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0e1116);
scene.fog=new THREE.Fog(0x0e1116,30,90);
const cam=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,.05,500);
cam.up.set(0,0,1);
const ren=new THREE.WebGLRenderer({antialias:true});
ren.setSize(innerWidth,innerHeight); ren.setPixelRatio(devicePixelRatio);
document.body.appendChild(ren.domElement);
const ctl=new OrbitControls(cam,ren.domElement); ctl.enableDamping=true;
scene.add(new THREE.AmbientLight(0xffffff,.45));
const sun=new THREE.DirectionalLight(0xfff2dd,1.5); sun.position.set(30,-20,50); scene.add(sun);
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();
 ren.setSize(innerWidth,innerHeight);});

function rover(color,zoff){
 // zoff = base-link height above the ground (from the data). The model was built
 // with its origin at ground level, but the recorded pose is the base link, which
 // rides zoff above ground - so shift every part down by zoff to restore contact.
 const dz=(zoff||0);
 const g=new THREE.Group();
 const body=new THREE.Mesh(new THREE.BoxGeometry(.42,.30,.14),
   new THREE.MeshStandardMaterial({color,roughness:.5}));
 body.position.z=.11-dz; g.add(body);
 const mast=new THREE.Mesh(new THREE.BoxGeometry(.03,.03,.16),
   new THREE.MeshStandardMaterial({color:0xffffff}));
 mast.position.set(.12,0,.26-dz); g.add(mast);
 g.wheels=[];
 const wg=new THREE.CylinderGeometry(.0625,.0625,.07,14);
 for(const[dx,dy]of[[.15,.17],[.15,-.17],[-.15,.17],[-.15,-.17]]){
  const w=new THREE.Mesh(wg,new THREE.MeshStandardMaterial({color:0x222831,roughness:.9}));
  w.position.set(dx,dy,.0625-dz); g.add(w); g.wheels.push(w);}
 return g;}

const slipCol=s=>new THREE.Color().setHSL(Math.max(0,.33*(1-Math.min(1,(s-.3)/.5))),.85,.5);
class Trail{constructor(max){this.max=max;this.n=0;
 this.pos=new Float32Array(max*3);this.col=new Float32Array(max*3);
 const g=new THREE.BufferGeometry();
 g.setAttribute('position',new THREE.BufferAttribute(this.pos,3));
 g.setAttribute('color',new THREE.BufferAttribute(this.col,3));
 this.line=new THREE.Line(g,new THREE.LineBasicMaterial({vertexColors:true}));
 this.line.frustumCulled=false;}
 push(p,s){if(this.n>=this.max)return;const c=slipCol(s);
  this.pos.set([p.x,p.y,p.z+.03],this.n*3);this.col.set([c.r,c.g,c.b],this.n*3);this.n++;
  this.line.geometry.setDrawRange(0,this.n);
  this.line.geometry.attributes.position.needsUpdate=true;
  this.line.geometry.attributes.color.needsUpdate=true;}
 reset(){this.n=0;this.line.geometry.setDrawRange(0,0);}}

let world=null, wireMesh=null;
function buildScenario(si){
 if(world)scene.remove(world); world=new THREE.Group(); scene.add(world);
 const S=DATA.scenarios[si], o=S.origin;
 // elevation range for brightness shading (hills read even under the soil tint)
 let zmin=1e9,zmax=-1e9;
 for(const row of S.heights)for(const h of row){if(h<zmin)zmin=h;if(h>zmax)zmax=h;}
 const zrng=Math.max(zmax-zmin,1e-6);
 document.getElementById('info').textContent=
  `terrain ${DATA.meta.level}% (relief ${zrng.toFixed(2)} m) | friction ${DATA.meta.friction} | ${DATA.meta.ckpt}`;
 const geo=new THREE.PlaneGeometry(2*HALF,2*HALF,G-1,G-1);
 const pa=geo.attributes.position, cols=new Float32Array(pa.count*3);
 const firm=new THREE.Color(0xcaa06a), sand=new THREE.Color(0x6b4a2b), c=new THREE.Color();
 for(let i=0;i<pa.count;i++){
  const ix=i%G, iy=Math.floor(i/G), gy=G-1-iy, h=S.heights[gy][ix];
  pa.setZ(i,h-o[2]);
  c.copy(firm).lerp(sand,S.soil[gy][ix]);
  c.multiplyScalar(0.7+0.5*(h-zmin)/zrng);          // higher ground = brighter
  cols.set([c.r,c.g,c.b],i*3);}
 geo.setAttribute('color',new THREE.BufferAttribute(cols,3));
 geo.computeVertexNormals();
 const ter=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({vertexColors:true,roughness:1,metalness:0}));
 ter.position.set(o[0],o[1],o[2]); world.add(ter);
 // the raw heightfield mesh itself, as a toggleable wireframe overlay
 wireMesh=new THREE.Mesh(geo.clone(),new THREE.MeshBasicMaterial(
   {wireframe:true,color:0x9db4d6,transparent:true,opacity:0.28}));
 wireMesh.position.set(o[0],o[1],o[2]+.005);
 wireMesh.visible=document.getElementById('wire').checked;
 world.add(wireMesh);
 const grid=new THREE.GridHelper(2*HALF,16,0x333f55,0x222b3a);
 grid.rotation.x=Math.PI/2; grid.position.set(o[0],o[1],o[2]+.01); world.add(grid);
 const hAt=(x,y)=>{const fx=(x-o[0]+HALF)/(2*HALF)*(G-1),fy=(y-o[1]+HALF)/(2*HALF)*(G-1);
  const ix=Math.min(G-1,Math.max(0,Math.round(fx))),iy=Math.min(G-1,Math.max(0,Math.round(fy)));
  return S.heights[iy][ix];};
 const pp=[];
 for(const w of S.wps)pp.push(new THREE.Vector3(w[0],w[1],hAt(w[0],w[1])+.06));
 const path=new THREE.Line(new THREE.BufferGeometry().setFromPoints(pp),
   new THREE.LineBasicMaterial({color:0xffd23f}));
 path.frustumCulled=false; world.add(path);
 const dotG=new THREE.SphereGeometry(.05,8,8), dotM=new THREE.MeshBasicMaterial({color:0xffd23f});
 for(const p of pp){const m=new THREE.Mesh(dotG,dotM);m.position.copy(p);world.add(m);}
 const goal=new THREE.Mesh(new THREE.SphereGeometry(.11,12,12),
   new THREE.MeshBasicMaterial({color:0x57d977})); goal.position.copy(pp[pp.length-1]); world.add(goal);
 const R={};
 for(const k of DATA.legs){const L=S.legs[k];
  R[k]={mesh:rover(COL[k],S.zoff),trail:new Trail(L.pos.length+4),leg:L,last:-1};
  world.add(R[k].mesh); world.add(R[k].trail.line);}
 cam.position.set(o[0]-7,o[1]-7,o[2]+5.5);
 ctl.target.set(o[0],o[1],o[2]+.6);
 return R;}

const scenSel=document.getElementById('scen');
DATA.scenarios.forEach((s,i)=>{const op=document.createElement('option');
 op.value=i; op.textContent=`scenario ${i+1}`; scenSel.appendChild(op);});
let R=buildScenario(0), t=0, playing=true;
const FOLLOW=DATA.legs[0];
const maxT=si=>DT*Math.max(...DATA.legs.map(k=>DATA.scenarios[si].legs[k].pos.length));
scenSel.onchange=()=>{R=buildScenario(+scenSel.value);t=0;};
document.getElementById('restart').onclick=()=>{t=0;
 for(const k in R){R[k].trail.reset();R[k].last=-1;}};
const playBtn=document.getElementById('play');
playBtn.onclick=()=>{playing=!playing;playBtn.innerHTML=playing?'&#9646;&#9646;':'&#9654;';};
const scrub=document.getElementById('scrub');
scrub.oninput=()=>{t=scrub.value/1000*maxT(+scenSel.value);
 for(const k in R){R[k].trail.reset();R[k].last=-1;}};
document.getElementById('wire').onchange=e=>{if(wireMesh)wireMesh.visible=e.target.checked;};

const qa=new THREE.Quaternion(), qb=new THREE.Quaternion(), yAxis=new THREE.Vector3(0,1,0);
function pose(k,tt){
 const L=R[k].leg, m=R[k].mesh, n=L.pos.length;
 const f=Math.min(tt/DT,n-1.001), i=Math.floor(f), a=f-i, j=Math.min(i+1,n-1);
 m.position.set(
  L.pos[i][0]+(L.pos[j][0]-L.pos[i][0])*a,
  L.pos[i][1]+(L.pos[j][1]-L.pos[i][1])*a,
  L.pos[i][2]+(L.pos[j][2]-L.pos[i][2])*a);
 qa.fromArray(L.quat[i]); qb.fromArray(L.quat[j]); qa.slerp(qb,a); m.quaternion.copy(qa);
 for(let w=0;w<4;w++){m.wheels[w].rotation.set(Math.PI/2,0,0);
  m.wheels[w].rotateOnAxis(yAxis,L.wheels[i][w]%(2*Math.PI));}
 if(i>R[k].last){R[k].trail.push(m.position,L.slip[i]);R[k].last=i;}
 return i;}

let prev=performance.now();
function tick(now){
 requestAnimationFrame(tick);
 const dt=(now-prev)/1000; prev=now;
 const sp=+document.getElementById('speed').value, mT=maxT(+scenSel.value);
 if(playing){t+=dt*sp; if(t>mT+1)t=mT+1; scrub.value=Math.min(1000,t/mT*1000);}
 for(const k of DATA.legs){
  const i=pose(k,Math.min(t,DT*(R[k].leg.pos.length-1)));
  document.getElementById('cte_'+k).textContent=R[k].leg.cte[i].toFixed(3);
  document.getElementById('slip_'+k).textContent=R[k].leg.slip[i].toFixed(2);}
 document.getElementById('time').textContent=t.toFixed(1);
 if(document.getElementById('follow').checked){
  const p=R[FOLLOW].mesh.position;
  ctl.target.lerp(new THREE.Vector3(p.x,p.y,p.z+.4),.06);}
 ctl.update(); ren.render(scene,cam);}
requestAnimationFrame(tick);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="Build a self-contained 3D HTML replay from a demo .npz")
    ap.add_argument("npz")
    ap.add_argument("-o", "--out", default="")
    a = ap.parse_args()
    payload = build_payload(a.npz)
    out = a.out or os.path.splitext(a.npz)[0] + ".html"
    with open(out, "w") as f:
        f.write(TEMPLATE.replace("/*__DEMO_DATA__*/", payload))
    print(f"[demo] wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)  - open in any browser")


if __name__ == "__main__":
    main()
