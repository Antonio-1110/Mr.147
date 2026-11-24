"""A web page that shows what the planner thought about, shot by shot.

    recorder = PlanRecorder(physics, "9-ball, seed 0")
    planner = recorder.watch_planner(planner)       # remembers every search
    cue_arm = recorder.watch_cue_arm(simulated_table)  # notes which shot was played and what really happened
    ... play ...
    recorder.write("outputs/plans.html")              # open it in any browser

For every shot the page draws the table from above with:
  * faint lines:   every shot that potted the ball in the perfect simulation
  * coloured lines: the finalists that were stress tested (click a row to pick one)
  * thick white:   the chosen shot: the cue ball's planned path, the ghost ball
                   and where it should stop; yellow: the object ball to its pocket
  * dashed red:    what really happened when the robot played it (with arm error)

Nothing outside the standard library: the page is one HTML file with the data inside.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path

from mr147.interfaces.cue_arm import Shot
from mr147.physics import PhysicsSimulator
from mr147.shot_planner.shot_planner import PlannedShot, PlanSearch, ShotPlanner, TriedShot
from mr147.table_memory.table_state import TableState


def _xy(p) -> list[float]:
    return [round(float(p[0]), 4), round(float(p[1]), 4)]


def _shot(s: Shot) -> dict:
    return {"dir": round(s.direction_deg, 2), "speed": round(s.speed, 2), "height": round(s.height, 2),
            "side": round(s.side, 2)}


def _tried(t: TriedShot) -> dict:
    return {"target": t.target, "pocket": t.pocket, **_shot(t.shot), "path": [_xy(p) for p in t.cue_ball_path],
            "stop": _xy(t.cue_ball_stops_at), "position": round(t.position_value, 3), "split": t.splits_the_pack}


def _planned(p: PlannedShot, confirmed: PlannedShot | None) -> dict:
    row = {"target": p.target, "pocket": p.pocket, **_shot(p.shot), "path": [_xy(q) for q in p.cue_ball_path],
           "stop": _xy(p.cue_ball_stops_at), "chance": round(p.chance_of_potting, 3),
           "continues": round(p.chance_break_continues, 3), "position": round(p.position_quality, 3),
           "openness": round(p.table_openness, 3), "score": round(p.score, 3), "split": p.splits_the_pack}
    if confirmed is not None:
        row["confirmed"] = {"chance": round(confirmed.chance_of_potting, 3),
                            "continues": round(confirmed.chance_break_continues, 3),
                            "position": round(confirmed.position_quality, 3), "score": round(confirmed.score, 3)}
    return row


@dataclass
class _Played:
    table: TableState
    search: PlanSearch | None
    shot: Shot
    actual_path: list = field(default_factory=list)
    potted: tuple[str, ...] = ()


class _WatchedCueArm:
    """Passes every call through to the real cue arm, and notes what was played."""

    def __init__(self, recorder: "PlanRecorder", arm):
        self._recorder, self._arm = recorder, arm

    def play(self, shot: Shot) -> None:
        self._recorder._played_shot(self._arm, shot)

    def __getattr__(self, name):
        return getattr(self._arm, name)


class PlanRecorder:
    def __init__(self, physics: PhysicsSimulator, title: str = "Shot plans"):
        self.physics = physics
        self.title = title
        self.shots: list[_Played] = []
        self._pending: list[PlanSearch] = []  # searches since the last shot (ball in hand tries several spots)

    def watch_planner(self, planner: ShotPlanner) -> ShotPlanner:
        plan_for = planner.plan_for

        def recording_plan_for(table, goal):
            plan = plan_for(table, goal)
            if planner.last_search is not None:
                self._pending.append(planner.last_search)
            return plan

        planner.plan_for = recording_plan_for  # plan() calls self.plan_for, so snooker is covered too
        return planner

    def watch_cue_arm(self, arm):
        """`arm` is the real cue arm; on the simulated table it also tells us what really happened."""
        return _WatchedCueArm(self, arm)

    def _played_shot(self, arm, shot: Shot) -> None:
        search = next((s for s in reversed(self._pending) if s.chosen is not None and s.chosen.shot == shot), None)
        before = getattr(arm, "true_table", None) or (search.table if search else TableState({}))
        self._pending.clear()
        arm.play(shot)
        played = _Played(before, search, shot)
        result = getattr(arm, "last_result", None)
        if result is not None:
            played.actual_path = [_xy(p) for p in result.cue_ball_path]
            played.potted = tuple(result.potted_ball_ids)
        self.shots.append(played)

    # ---- the page ------------------------------------------------------------------

    def data(self) -> dict:
        t = self.physics.table
        shots = []
        for n, p in enumerate(self.shots, 1):
            shots.append(self._shot_data(n, p.table, p.search, p.shot, p.actual_path, p.potted))
        if self._pending:  # the run ended because no shot was found
            s = self._pending[-1]
            shots.append(self._shot_data(len(shots) + 1, s.table, s, None, [], ()))
        return {"title": self.title, "game": t.simulator_style, "length": t.length, "width": t.width,
                "ball": t.ball_diameter, "cue": t.cue_ball, "pockets": [_xy(p) for p in self.physics.pockets],
                "shots": shots}

    def _shot_data(self, n, table, search, shot, actual_path, potted) -> dict:
        d = {"n": n, "balls": {k: _xy(v) for k, v in table.positions.items()}, "played": _shot(shot) if shot else None,
             "actual": actual_path, "potted": list(potted), "kind": "shot"}
        if search is None:
            d["kind"] = "break" if n == 1 else "other"
            return d
        confirmed = {c.shot: c for c in search.confirmed}
        finalists = sorted(search.finalists, key=lambda f: (confirmed[f.shot].score if f.shot in confirmed else -1,
                                                            f.score), reverse=True)
        d.update(targets=list(search.targets), tried=[_tried(x) for x in search.tried],
                 finalists=[_planned(f, confirmed.get(f.shot)) for f in finalists],
                 chosen=0 if search.chosen is not None else None)
        if shot is None:
            d["kind"] = "none"
        return d

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".html":
            path = path / "plans.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        page = _PAGE.replace("__TITLE__", html.escape(self.title))
        page = page.replace("__DATA__", json.dumps(self.data(), separators=(",", ":")).replace("</", "<\\/"))
        path.write_text(page, encoding="utf-8")
        return path


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f6f3;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#ddd;--hi:#fff6c9}
@media (prefers-color-scheme:dark){:root{--bg:#18181a;--fg:#ececec;--muted:#9a9a9a;--card:#232326;--line:#3a3a3e;--hi:#3c3720}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,-apple-system,sans-serif}
header{padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;border-bottom:1px solid var(--line)}
h1{font-size:17px;margin:0 12px 0 0}button{font:inherit;padding:4px 10px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;cursor:pointer}
html,body{overflow-x:hidden}main{display:grid;grid-template-columns:minmax(0,11fr) minmax(0,9fr);gap:16px;padding:16px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
svg{width:100%;height:auto;display:block}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:4px 6px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
tbody tr{cursor:pointer}tbody tr.sel{background:var(--hi)}.muted{color:var(--muted)}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.legend span{margin-right:14px;white-space:nowrap}.legend i{display:inline-block;width:22px;height:4px;outline:1px solid #888;vertical-align:middle;margin-right:5px}
#summary{margin:0 0 10px}label{white-space:nowrap}
</style></head><body>
<header><h1>__TITLE__</h1>
<button id="prev">&larr; Prev</button><span id="where"></span><button id="next">Next &rarr;</button>
<label><input type="checkbox" id="showTried" checked> all shots that potted</label>
<label><input type="checkbox" id="showActual" checked> what really happened</label>
<span class="muted">Arrow keys move between shots; click a row to see that option.</span></header>
<main><div class="card"><svg id="table"></svg>
<div class="legend muted" style="margin-top:8px">
<span><i style="background:#fff"></i>chosen cue-ball path</span><span><i style="background:#ffd23f"></i>object ball to pocket</span>
<span><i style="background:repeating-linear-gradient(90deg,#ff4d4d 0 5px,transparent 5px 8px)"></i>what really happened</span>
<span><i style="background:rgba(255,255,255,.35)"></i>other shots that potted</span><span>&#9675; where the cue ball stops</span></div></div>
<div class="card"><p id="summary"></p><div style="overflow-x:auto"><table><thead><tr><th>#</th><th>Pot</th><th>Speed</th><th>Spin</th><th title="share of replays with arm and camera error that went in">Pots</th><th title="of those, share that left a next ball to pot">Next ball</th><th title="how easy the next pot is, 0-1">Position</th><th>Score</th></tr></thead><tbody id="rows"></tbody></table></div>
<p class="muted" style="margin-top:10px">Score = pots &times; next ball &times; (0.6 + 0.4 &times; position). The top three are re-tested with fresh replays; those numbers are shown (marked &#10003;) because picking the best of many short tests flatters the winner.</p></div></main>
<script>
const D=__DATA__;
const L=D.length,W=D.width,R=D.ball/2,M=0.09,S=1000;
const svg=document.getElementById('table');
svg.setAttribute('viewBox',`${-(L/2+M)*S} ${-(W/2+M)*S} ${(L+2*M)*S} ${(W+2*M)*S}`);
const cloth=D.game==='pool'?'#2f6fb5':'#1f7a3a';
const SNK={red:'#c62828',yellow:'#f2d21b',green:'#1b7f3b',brown:'#6d4423',blue:'#1f5bd1',pink:'#f48fb1',black:'#111',white:'#f7f5ee'};
const POOL=['#f2c200','#1f4fb8','#d32f2f','#6a2c91','#f57c00','#1b7f3b','#7b2e1e','#111'];
let i=0,sel=0;
const X=x=>x*S,Y=y=>-y*S; // y up on the table, down on screen
const el=(t,a,parent)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',t);if(t==='line'||t==='polyline'||(a.fill==='none'))a['vector-effect']='non-scaling-stroke';for(const k in a)e.setAttribute(k,a[k]);(parent||svg).appendChild(e);return e};
const pts=p=>p.map(q=>`${X(q[0]).toFixed(1)},${Y(q[1]).toFixed(1)}`).join(' ');
function ballColour(id){if(id===D.cue)return SNK.white;if(D.game==='pool'){const n=+id;return POOL[(n-1)%8]}return SNK[id.split('_')[0]]||'#888'}
function pocketName(k){const p=D.pockets[k];const end=Math.abs(p[0])<0.1?'middle':(p[0]<0?'left':'right');const side=p[1]>0?'top':'bottom';return end==='middle'?`${side} middle`:`${side} ${end}`}
function spin(s){const h=s.height>0.05?`follow ${s.height}`:s.height<-0.05?`screw ${-s.height}`:'plain';const sd=Math.abs(s.side)>0.05?(s.side>0?' + right':' + left'):'';return h+sd}
function speedWord(v){return v<1.2?'soft':v<2.5?'medium':v<4?'firm':'hard'}
function ballLabel(id){return id===D.cue?'cue ball':(D.game==='pool'?id+'-ball':id.replace('_',' '))}
function drawBall(id,p,shot){const c=ballColour(id),cx=X(p[0]),cy=Y(p[1]),r=R*S;
 if(D.game==='pool'&&+id>8){const cp='c'+shot+'_'+id;const g=el('clipPath',{id:cp});el('circle',{cx,cy,r},g);el('circle',{cx,cy,r,fill:'#f7f5ee'});el('rect',{x:cx-r,y:cy-r*0.55,width:2*r,height:r*1.1,fill:c,'clip-path':`url(#${cp})`})}
 else el('circle',{cx,cy,r,fill:c});
 el('circle',{cx,cy,r,fill:'none',stroke:'rgba(0,0,0,.45)','stroke-width':0.8});
 if(D.game==='pool'&&id!==D.cue){el('circle',{cx,cy,r:r*0.45,fill:'#fff'});const t=el('text',{x:cx,y:cy+r*0.28,'text-anchor':'middle','font-size':r*0.8,'font-weight':700,fill:'#111'});t.textContent=id}}
function draw(){const s=D.shots[i];svg.innerHTML='';
 el('rect',{x:X(-L/2-M),y:Y(W/2+M),width:(L+2*M)*S,height:(W+2*M)*S,rx:40,fill:'#5a3a22'});
 el('rect',{x:X(-L/2),y:Y(W/2),width:L*S,height:W*S,fill:cloth});
 D.pockets.forEach(p=>el('circle',{cx:X(p[0]),cy:Y(p[1]),r:R*S*1.9,fill:'#0b0b0b'}));
 const showTried=document.getElementById('showTried').checked,showActual=document.getElementById('showActual').checked;
 if(showTried&&s.tried)for(const t of s.tried){el('polyline',{points:pts(t.path),fill:'none',stroke:'rgba(255,255,255,.18)','stroke-width':1});el('circle',{cx:X(t.stop[0]),cy:Y(t.stop[1]),r:R*S*0.35,fill:'rgba(255,255,255,.35)'})}
 for(const id in s.balls)drawBall(id,s.balls[id],s.n);
 const f=s.finalists&&s.finalists[sel];
 if(f){const ob=s.balls[f.target],pk=D.pockets[f.pocket];
  el('line',{x1:X(ob[0]),y1:Y(ob[1]),x2:X(pk[0]),y2:Y(pk[1]),stroke:'#ffd23f','stroke-width':2,'stroke-dasharray':'6 4'});
  const a=Math.atan2(pk[1]-ob[1],pk[0]-ob[0]);const gh=[ob[0]-D.ball*Math.cos(a),ob[1]-D.ball*Math.sin(a)];
  el('circle',{cx:X(gh[0]),cy:Y(gh[1]),r:R*S,fill:'none',stroke:'#fff','stroke-width':1.5,'stroke-dasharray':'3 2'});
  el('polyline',{points:pts(f.path),fill:'none',stroke:'#fff','stroke-width':3,'stroke-linejoin':'round'});
  el('circle',{cx:X(f.stop[0]),cy:Y(f.stop[1]),r:R*S*1.25,fill:'none',stroke:'#fff','stroke-width':2.5})}
 if(s.played&&!s.finalists){const c=s.balls[D.cue];if(c){const a=s.played.dir*Math.PI/180;el('line',{x1:X(c[0]),y1:Y(c[1]),x2:X(c[0]+0.8*Math.cos(a)),y2:Y(c[1]+0.8*Math.sin(a)),stroke:'#fff','stroke-width':3})}}
 if(showActual&&s.actual&&s.actual.length){el('polyline',{points:pts(s.actual),fill:'none',stroke:'#ff4d4d','stroke-width':2,'stroke-dasharray':'6 4'});const e=s.actual[s.actual.length-1];el('circle',{cx:X(e[0]),cy:Y(e[1]),r:R*S*0.6,fill:'#ff4d4d'})}
 document.getElementById('where').textContent=`Shot ${s.n} of ${D.shots.length}`;
 summary(s);rows(s)}
function summary(s){const out=document.getElementById('summary');const pot=s.potted.length?`Went down: ${s.potted.map(ballLabel).join(', ')}.`:(s.played?'Nothing went down.':'');
 if(s.kind==='break'){out.textContent=`The break: ${s.played.speed} m/s, ${spin(s.played)}. ${pot}`;return}
 if(s.kind==='none'){out.textContent=`No shot found: nothing could pot ${s.targets.map(ballLabel).join(' / ')} in the simulator. The run ends here.`;return}
 if(!s.finalists){out.textContent=pot;return}
 const n=s.tried.length,on=s.targets.length>3?`any of ${s.targets.length} balls`:s.targets.map(ballLabel).join(' / ');
 out.innerHTML=`Ball on: <b>${on}</b>. ${n} shots potted it in the perfect simulation; the best ${s.finalists.length} were replayed with arm and camera error. ${pot}`}
function rows(s){const tb=document.getElementById('rows');tb.innerHTML='';if(!s.finalists)return;
 s.finalists.forEach((f,k)=>{const c=f.confirmed||f,tr=document.createElement('tr');if(k===sel)tr.className='sel';
  const tick=f.confirmed?' &#10003;':'';const chosen=k===0?' <b>(chosen)</b>':'';
  tr.innerHTML=`<td>${k+1}${chosen}</td><td>${ballLabel(f.target)} &rarr; ${pocketName(f.pocket)}${f.split?' <span title="cue ball runs into the pack">split</span>':''}</td><td>${f.speed} <span class="muted">${speedWord(f.speed)}</span></td><td>${spin(f)}</td><td>${Math.round(c.chance*100)}%</td><td>${Math.round(c.continues*100)}%</td><td>${c.position.toFixed(2)}</td><td>${c.score.toFixed(2)}${tick}</td>`;
  tr.onclick=()=>{sel=k;draw()};tb.appendChild(tr)})}
function go(d){i=Math.max(0,Math.min(D.shots.length-1,i+d));sel=0;draw()}
document.getElementById('prev').onclick=()=>go(-1);document.getElementById('next').onclick=()=>go(1);
document.getElementById('showTried').onchange=draw;document.getElementById('showActual').onchange=draw;
document.addEventListener('keydown',e=>{if(e.key.startsWith('Arrow'))e.preventDefault();if(e.key==='ArrowLeft')go(-1);if(e.key==='ArrowRight')go(1);if(e.key==='ArrowDown'&&D.shots[i].finalists){sel=Math.min(sel+1,D.shots[i].finalists.length-1);draw()}if(e.key==='ArrowUp'){sel=Math.max(sel-1,0);draw()}});
if(D.shots.length)draw();else document.getElementById('summary').textContent='No shots were played.';
</script></body></html>
"""
