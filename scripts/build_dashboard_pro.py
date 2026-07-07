import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, skew, kurtosis
import plotly.offline as pyo

from src.data import demo_series, load_prices_yf, log_returns
from src.regimes import (
    rolling_windows,
    labels_to_series,
    transition_matrix,
    distance_matrix,
    embed_2d,
    silhouette,
)
from src.wkmeans import WassersteinKMeans, MomentKMeans
from src.mmd import cluster_mmd


W = 21
STEP = 21
K_RANGE = [2, 3, 4, 5]
K_SCAN = [2, 3, 4, 5, 6]
P_RANGE = [1, 2]

NAMES = {
    2: ["Calm", "Stress"],
    3: ["Calm", "Choppy", "Stress"],
    4: ["Calm", "Mild", "Choppy", "Stress"],
    5: ["Calm", "Mild", "Choppy", "Elevated", "Stress"],
}


def r(x, nd=5):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r(v, nd) for v in x]
    return round(float(x), nd)


def cluster_kde(returns, wlabels, starts, k, xs):
    curves = []
    for c in range(k):
        chunks = [returns[s:s + W] for lab, s in zip(wlabels, starts) if lab == c]
        if not chunks:
            curves.append([0.0] * len(xs))
            continue
        pooled = np.concatenate(chunks)
        if pooled.std() < 1e-12:
            curves.append([0.0] * len(xs))
            continue
        curves.append(gaussian_kde(pooled)(xs).tolist())
    return curves


def cluster_stats(returns, wlabels, starts, k):
    out = []
    for c in range(k):
        chunks = [returns[s:s + W] for lab, s in zip(wlabels, starts) if lab == c]
        pooled = np.concatenate(chunks) if chunks else np.array([0.0])
        out.append({
            "mean": r(pooled.mean() * 252 * 100, 2),
            "vol": r(pooled.std() * np.sqrt(252) * 100, 2),
            "skew": r(float(skew(pooled)), 2),
            "kurt": r(float(kurtosis(pooled)), 2),
            "worst": r(pooled.min() * 100, 2),
            "best": r(pooled.max() * 100, 2),
        })
    return out


def get_series(args):
    if not args:
        dates, prices, returns, _ = demo_series()
        return list(dates), np.asarray(prices), np.asarray(returns), "SYNTHETIC DEMO SERIES"
    if args[0].lower() == "csv":
        path = args[1]
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        price_col = cols.get("close") or cols.get("price") or df.columns[-1]
        prices = df[price_col].astype(float).dropna().values
        date_col = cols.get("date") or cols.get("time") or cols.get("datetime")
        if date_col:
            dates = pd.to_datetime(df[date_col]).iloc[-len(prices):].tolist()
        else:
            dates = pd.bdate_range(end=pd.Timestamp.today(), periods=len(prices)).tolist()
        returns = log_returns(prices)
        name = os.path.splitext(os.path.basename(path))[0].upper()
        return dates[1:], prices[1:], returns, name + " (CSV) · DAILY"
    ticker = args[0].upper()
    start = args[1] if len(args) > 1 else "2015-01-01"
    end = args[2] if len(args) > 2 else None
    px = load_prices_yf(ticker, start, end)
    prices = px.values.astype(float)
    dates = list(px.index)
    returns = log_returns(prices)
    label = f"{ticker} · {dates[0].strftime('%Y')}-{dates[-1].strftime('%Y')} · DAILY"
    return dates[1:], prices[1:], returns, label


def build_payload(dates, prices, returns, source):
    windows, starts = rolling_windows(returns, W, STEP)

    vol = np.full(returns.size, np.nan)
    for i in range(20, returns.size):
        vol[i] = returns[i - 20:i + 1].std() * np.sqrt(252) * 100
    vol = np.nan_to_num(vol, nan=float(np.nanmean(vol)))

    lo = np.percentile(returns, 0.5)
    hi = np.percentile(returns, 99.5)
    xs = np.linspace(lo, hi, 220)

    dist = {}
    mds = {}
    fits = {}
    for p in P_RANGE:
        dist[str(p)] = r(distance_matrix(windows, p=p), 6)
        mds[str(p)] = r(embed_2d(np.array(dist[str(p)]), random_state=0), 5)
        for k in K_SCAN:
            fits[(k, p)] = WassersteinKMeans(k=k, p=p, random_state=0).fit(windows)

    mom_fits = {k: MomentKMeans(k=k, random_state=0).fit(windows) for k in K_SCAN}

    configs = {}
    for p in P_RANGE:
        for k in K_RANGE:
            m = fits[(k, p)]
            wl = m.labels_.astype(int)
            series = labels_to_series(wl, starts, W, returns.size).astype(int)
            wk = cluster_mmd(windows, wl)
            mm = cluster_mmd(windows, mom_fits[k].labels_)
            configs[f"k{k}_p{p}"] = {
                "labels": wl.tolist(),
                "series": series.tolist(),
                "names": NAMES[k],
                "trans": r(transition_matrix(wl, k), 4),
                "sil": r(silhouette(np.array(dist[str(p)]), wl), 3),
                "wk": {"between": r(wk["between"]), "within": r(wk["within"]),
                       "ratio": r(wk["ratio"], 2)},
                "mom": {"between": r(mm["between"]), "within": r(mm["within"]),
                        "ratio": r(mm["ratio"], 2)},
                "kde": r(cluster_kde(returns, wl, starts, k, xs), 4),
                "stats": cluster_stats(returns, wl, starts, k),
                "shares": r([float((series == c).mean() * 100) for c in range(k)], 1),
            }

    kscan = {}
    for p in P_RANGE:
        inert = [r(fits[(k, p)].inertia_, 5) for k in K_SCAN]
        sils = [r(silhouette(np.array(dist[str(p)]), fits[(k, p)].labels_), 3) for k in K_SCAN]
        kscan[str(p)] = {"ks": K_SCAN, "inertia": inert, "sil": sils}

    return {
        "dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates],
        "prices": r(prices, 3),
        "returns": r(returns, 6),
        "vol": r(vol, 2),
        "w": W, "step": STEP,
        "starts": starts.tolist(),
        "windows": r([w.tolist() for w in windows], 6),
        "kdeX": r(xs, 6),
        "dist": dist,
        "mds": mds,
        "configs": configs,
        "kscan": kscan,
        "source": source,
    }


APP_JS = r"""
const C = {bg:'#F2F5F9',card:'#FFFFFF',ink:'#1C2534',sub:'#5B6675',faint:'#8A94A6',
  line:'#E5E9F0',grid:'#EDF0F5',blue:'#2F8FD4',navy:'#1F3A5F',maroon:'#7C1D2E'};
const PAL = {
  2:['#4E9BD4','#B03A52'],
  3:['#4E9BD4','#8E86C9','#B03A52'],
  4:['#4E9BD4','#8BBCE3','#8E86C9','#B03A52'],
  5:['#4E9BD4','#8BBCE3','#8E86C9','#D77A96','#B03A52']
};
const state = {k:3, p:2, order:'time', win:null};
const cfg = () => DATA.configs['k'+state.k+'_p'+state.p];
const col = i => PAL[state.k][i];
const PC = {displayModeBar:false, responsive:true};

function rgba(hex,a){const h=hex.replace('#','');
  return 'rgba('+parseInt(h.substr(0,2),16)+','+parseInt(h.substr(2,2),16)+','+parseInt(h.substr(4,2),16)+','+a+')';}

function base(extra){
  return Object.assign({
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:'Sora, sans-serif', size:11.5, color:C.sub},
    margin:{l:52,r:18,t:12,b:38},
    hoverlabel:{bgcolor:C.ink,bordercolor:C.ink,font:{family:'JetBrains Mono',color:'#FFFFFF',size:12}},
    legend:{orientation:'h',y:1.08,x:0,bgcolor:'rgba(0,0,0,0)',font:{size:11,color:C.sub}},
    xaxis:{gridcolor:C.grid,zeroline:false,linecolor:C.line,tickcolor:C.line},
    yaxis:{gridcolor:C.grid,zeroline:false,linecolor:C.line,tickcolor:C.line}
  }, extra||{});
}

function segments(series){
  const out=[]; let a=0;
  for(let i=1;i<series.length;i++){
    if(series[i]!==series[a]){out.push([a,i-1,series[a]]); a=i;}
  }
  out.push([a,series.length-1,series[a]]);
  return out;
}

function drawHero(){
  const c=cfg(), d=DATA.dates, n=d.length;
  const shapes=segments(c.series).map(s=>({type:'rect',xref:'x',yref:'paper',
    x0:d[s[0]],x1:d[Math.min(s[1]+1,n-1)],y0:0,y1:1,
    fillcolor:rgba(col(s[2]),0.14),line:{width:0},layer:'below'}));
  const traces=[
    {x:d,y:DATA.prices,mode:'lines',name:'Price',yaxis:'y',
     line:{color:C.navy,width:1.7},
     hovertemplate:'%{x|%b %d, %Y}<br>px %{y:.2f}<extra></extra>'},
    {x:d,y:DATA.vol,mode:'lines',name:'Realised vol (ann. %)',yaxis:'y2',
     line:{color:C.blue,width:1},fill:'tozeroy',fillcolor:rgba(C.blue,0.12),
     hovertemplate:'%{x|%b %d, %Y}<br>vol %{y:.1f}%<extra></extra>'}
  ];
  c.names.forEach((nm,i)=>traces.push({x:[null],y:[null],mode:'markers',name:nm,
    marker:{size:10,symbol:'square',color:col(i)}}));
  const layout=base({height:360,shapes:shapes,
    yaxis:{domain:[0.32,1],title:{text:'Index',font:{size:11}},gridcolor:C.grid,linecolor:C.line},
    yaxis2:{domain:[0,0.24],title:{text:'Vol %',font:{size:11}},gridcolor:C.grid,linecolor:C.line},
    xaxis:{gridcolor:C.grid,linecolor:C.line}});
  Plotly.react('hero',traces,layout,PC);
  document.getElementById('hero').on('plotly_click',ev=>{
    const idx=ev.points[0].pointIndex;
    selectWindow(Math.min(Math.floor(idx/DATA.step),DATA.windows.length-1));
  });
}

function drawRidge(){
  const c=cfg(), xs=DATA.kdeX, traces=[];
  for(let i=0;i<state.k;i++){
    const dens=c.kde[i], m=Math.max(...dens,1e-9);
    const nd=dens.map(v=>v/m+i);
    traces.push({x:xs,y:xs.map(()=>i),mode:'lines',hoverinfo:'skip',
      line:{width:0},showlegend:false});
    traces.push({x:xs,y:nd,mode:'lines',name:c.names[i],fill:'tonexty',
      fillcolor:rgba(col(i),0.32),line:{color:col(i),width:1.8},
      hovertemplate:'ret %{x:.3f}<extra>'+c.names[i]+'</extra>'});
  }
  const layout=base({height:330,showlegend:false,
    xaxis:{title:{text:'Daily log-return'},tickformat:'.1%',gridcolor:C.grid,linecolor:C.line},
    yaxis:{tickvals:[...Array(state.k).keys()].map(i=>i+0.4),
      ticktext:cfg().names,gridcolor:'rgba(0,0,0,0)',linecolor:C.line}});
  Plotly.react('ridge',traces,layout,PC);
}

function drawHeat(){
  const c=cfg(), D=DATA.dist[String(state.p)], n=D.length;
  let order=[...Array(n).keys()];
  if(state.order==='regime') order.sort((a,b)=>c.labels[a]-c.labels[b]||a-b);
  const z=order.map(i=>order.map(j=>D[i][j]));
  const scale=[[0,'#FFFFFF'],[0.4,'#BBD8EE'],[0.7,'#4E9BD4'],[1,'#7C1D2E']];
  const custom=order.map(i=>order.map(j=>[i,j,c.names[c.labels[i]],c.names[c.labels[j]]]));
  const traces=[{z:z,type:'heatmap',colorscale:scale,customdata:custom,
    colorbar:{thickness:9,len:0.85,outlinecolor:C.line,tickcolor:C.line,tickfont:{color:C.faint,size:10}},
    hovertemplate:'w%{customdata[0]} (%{customdata[2]}) vs w%{customdata[1]} (%{customdata[3]})'+
      '<br>W'+state.p+' = %{z:.4f}<extra></extra>'}];
  const layout=base({height:330,
    xaxis:{title:{text:state.order==='regime'?'Window (grouped by regime)':'Window (chronological)'},
      showgrid:false,linecolor:C.line},
    yaxis:{autorange:'reversed',showgrid:false,linecolor:C.line}});
  Plotly.react('heat',traces,layout,PC);
  document.getElementById('heat').on('plotly_click',ev=>{
    selectWindow(ev.points[0].customdata[0]);
  });
}

function drawMds(){
  const c=cfg(), M=DATA.mds[String(state.p)], traces=[];
  for(let g=0;g<state.k;g++){
    const idx=[...c.labels.keys()].filter(i=>c.labels[i]===g);
    traces.push({x:idx.map(i=>M[i][0]),y:idx.map(i=>M[i][1]),mode:'markers',
      name:c.names[g],customdata:idx,
      marker:{size:11,color:col(g),opacity:0.9,line:{color:'#FFFFFF',width:1.5}},
      hovertemplate:'w%{customdata} · '+c.names[g]+'<extra></extra>'});
  }
  const layout=base({height:330,
    xaxis:{title:{text:'MDS-1'},showticklabels:false,gridcolor:C.grid,linecolor:C.line},
    yaxis:{title:{text:'MDS-2'},showticklabels:false,gridcolor:C.grid,linecolor:C.line}});
  Plotly.react('mds',traces,layout,PC);
  document.getElementById('mds').on('plotly_click',ev=>{
    selectWindow(ev.points[0].customdata);
  });
}

function drawTrans(){
  const c=cfg(), scale=[[0,'#FFFFFF'],[0.5,'#BBD8EE'],[1,'#2F8FD4']];
  const text=c.trans.map(row=>row.map(v=>Math.round(v*100)+'%'));
  const traces=[{z:c.trans,x:c.names,y:c.names,type:'heatmap',colorscale:scale,
    showscale:false,text:text,texttemplate:'%{text}',
    textfont:{family:'JetBrains Mono',color:C.ink,size:13},
    hovertemplate:'from %{y} to %{x}: %{z:.2f}<extra></extra>'}];
  const layout=base({height:330,
    xaxis:{title:{text:'To'},showgrid:false,linecolor:C.line},
    yaxis:{title:{text:'From'},autorange:'reversed',showgrid:false,linecolor:C.line}});
  Plotly.react('trans',traces,layout,PC);
}

function drawDonut(){
  const c=cfg();
  const traces=[{values:c.shares,labels:c.names,type:'pie',hole:0.62,
    marker:{colors:c.names.map((_,i)=>col(i)),line:{color:'#FFFFFF',width:3}},
    textinfo:'none',sort:false,direction:'clockwise',
    hovertemplate:'%{label}: %{value:.1f}% of days<extra></extra>'}];
  const layout=base({height:300,showlegend:true,
    legend:{orientation:'v',x:1.02,y:0.5,font:{size:12,color:C.sub}},
    margin:{l:10,r:10,t:10,b:10},
    annotations:[{text:'REGIME<br>MIX',showarrow:false,
      font:{family:'JetBrains Mono',size:12,color:C.faint}}]});
  Plotly.react('donut',traces,layout,PC);
}

function drawAllocTable(){
  const c=cfg();
  let rows='';
  for(let i=0;i<state.k;i++){
    const s=c.stats[i];
    rows+='<tr><td><span class="dot" style="background:'+col(i)+'"></span>'+c.names[i]+'</td>'+
      '<td>'+c.shares[i]+'%</td><td>'+s.vol+'%</td><td>'+s.mean+'%</td><td>'+s.skew+'</td></tr>';
  }
  document.getElementById('alloc-table').innerHTML=
    '<table class="tbl"><thead><tr><th>Regime</th><th>Share</th><th>Ann. vol</th>'+
    '<th>Ann. ret</th><th>Skew</th></tr></thead><tbody>'+rows+'</tbody></table>';
}

function drawKeyInfo(){
  const c=cfg();
  const uplift=(c.wk.ratio/c.mom.ratio).toFixed(1);
  const rows=[
    ['Method','Wasserstein k-means'],
    ['Foundation','arXiv:2110.11848 (2021)'],
    ['Distance metric','W'+state.p+' optimal transport'],
    ['Rolling window','21 trading days'],
    ['Regimes (k)',String(state.k)],
    ['Windows clustered',String(DATA.windows.length)],
    ['Separation (WK / moment)',c.wk.ratio+'x / '+c.mom.ratio+'x'],
    ['Uplift vs baseline',uplift+'x'],
    ['Silhouette',String(c.sil)],
    ['Data source',DATA.source.toLowerCase()]
  ];
  document.getElementById('keyinfo').innerHTML=rows.map(x=>
    '<div class="krow"><span>'+x[0]+'</span><b>'+x[1]+'</b></div>').join('');
}

function drawCmp(){
  const c=cfg();
  const traces=[
    {x:['Between-cluster','Within-cluster'],y:[c.wk.between,c.wk.within],type:'bar',
     name:'WK-means',marker:{color:C.blue},hovertemplate:'%{y:.4f}<extra>WK-means</extra>'},
    {x:['Between-cluster','Within-cluster'],y:[c.mom.between,c.mom.within],type:'bar',
     name:'Moment k-means',marker:{color:'#B9C2CF'},hovertemplate:'%{y:.4f}<extra>Moment</extra>'}
  ];
  const layout=base({height:330,barmode:'group',bargap:0.35,bargroupgap:0.12,
    yaxis:{title:{text:'MMD'},gridcolor:C.grid,linecolor:C.line},
    annotations:[{xref:'paper',yref:'paper',x:0.98,y:0.95,showarrow:false,align:'right',
      text:'sep ratio  WK <b style="color:#2F8FD4">'+c.wk.ratio+'x</b>  vs  moment <b>'+c.mom.ratio+'x</b>',
      font:{family:'JetBrains Mono',size:12,color:C.sub}}]});
  Plotly.react('cmp',traces,layout,PC);
}

function drawKsel(){
  const s=DATA.kscan[String(state.p)];
  const traces=[
    {x:s.ks,y:s.inertia,mode:'lines+markers',name:'Inertia',
     line:{color:C.blue,width:2},marker:{size:8}},
    {x:s.ks,y:s.sil,mode:'lines+markers',name:'Silhouette',yaxis:'y2',
     line:{color:C.maroon,width:2,dash:'dot'},marker:{size:8}}
  ];
  const layout=base({height:330,
    xaxis:{title:{text:'k'},dtick:1,gridcolor:C.grid,linecolor:C.line},
    yaxis:{title:{text:'Inertia'},gridcolor:C.grid,linecolor:C.line},
    yaxis2:{title:{text:'Silhouette'},overlaying:'y',side:'right',gridcolor:'rgba(0,0,0,0)'},
    shapes:[{type:'line',x0:state.k,x1:state.k,yref:'paper',y0:0,y1:1,
      line:{color:C.maroon,width:1.5,dash:'dash'}}]});
  Plotly.react('ksel',traces,layout,PC);
}

function selectWindow(i){
  state.win=i;
  drawInspector();
  document.getElementById('inspector-card').classList.add('flash');
  setTimeout(()=>document.getElementById('inspector-card').classList.remove('flash'),450);
}

function drawInspector(){
  const c=cfg();
  if(state.win===null) state.win=DATA.windows.length-1;
  const i=state.win, wr=DATA.windows[i], lab=c.labels[i];
  const start=DATA.starts[i];
  const d0=DATA.dates[start], d1=DATA.dates[Math.min(start+DATA.w-1,DATA.dates.length-1)];
  const traces=[
    {x:wr,type:'histogram',histnorm:'probability density',nbinsx:12,
     marker:{color:rgba(col(lab),0.55),line:{color:col(lab),width:1}},
     name:'window',hovertemplate:'%{x:.3f}<extra></extra>'},
    {x:DATA.kdeX,y:c.kde[lab],mode:'lines',name:c.names[lab]+' cluster',
     line:{color:C.ink,width:1.6,dash:'dot'},hoverinfo:'skip'}
  ];
  const layout=base({height:250,showlegend:false,margin:{l:44,r:12,t:8,b:34},
    xaxis:{title:{text:'Daily log-return'},tickformat:'.1%',gridcolor:C.grid,linecolor:C.line},
    yaxis:{title:{text:'Density'},gridcolor:C.grid,linecolor:C.line}});
  Plotly.react('inspect',traces,layout,PC);

  const mean=wr.reduce((a,b)=>a+b,0)/wr.length;
  const sd=Math.sqrt(wr.reduce((a,b)=>a+(b-mean)*(b-mean),0)/wr.length);
  const st=c.stats[lab];
  document.getElementById('insp-meta').innerHTML=
    '<span class="pill" style="background:'+rgba(col(lab),0.12)+';color:'+col(lab)+
    ';border-color:'+rgba(col(lab),0.45)+'">'+c.names[lab]+'</span>'+
    '<span class="mono">window '+i+' · '+d0+' → '+d1+'</span>';
  document.getElementById('insp-stats').innerHTML=
    row('window ann. vol',(sd*Math.sqrt(252)*100).toFixed(1)+'%')+
    row('window mean (daily)',(mean*100).toFixed(3)+'%')+
    row('cluster ann. return',st.mean+'%')+
    row('cluster ann. vol',st.vol+'%')+
    row('cluster skew / kurt',st.skew+' / '+st.kurt)+
    row('cluster worst / best day',st.worst+'% / '+st.best+'%');
}

function row(k,v){return '<div class="srow"><span>'+k+'</span><b>'+v+'</b></div>';}

function drawKpis(){
  const c=cfg();
  const uplift=(c.wk.ratio/c.mom.ratio).toFixed(1);
  const dom=c.shares.indexOf(Math.max(...c.shares));
  const cells=[
    ['windows',DATA.windows.length,C.ink],
    ['regimes (k)',state.k,col(0)],
    ['WK separation',c.wk.ratio+'x',C.blue],
    ['moment baseline',c.mom.ratio+'x',C.faint],
    ['uplift',uplift+'x',C.blue],
    ['silhouette',c.sil,C.maroon],
    ['dominant regime',c.names[dom]+' '+c.shares[dom]+'%',col(dom)]
  ];
  document.getElementById('kpis').innerHTML=cells.map(x=>
    '<div class="metric"><span class="dot big" style="background:'+x[2]+'"></span>'+
    '<div><div class="mval">'+x[1]+'</div><div class="mlab">'+x[0]+'</div></div></div>').join('');
}

function renderAll(){
  drawKpis();drawHero();drawRidge();drawHeat();drawMds();drawTrans();
  drawDonut();drawAllocTable();drawKeyInfo();drawCmp();drawKsel();drawInspector();
  document.querySelectorAll('[data-k]').forEach(b=>b.classList.toggle('on',+b.dataset.k===state.k));
  document.querySelectorAll('[data-p]').forEach(b=>b.classList.toggle('on',+b.dataset.p===state.p));
  document.querySelectorAll('[data-order]').forEach(b=>b.classList.toggle('on',b.dataset.order===state.order));
}

function png(id,name){Plotly.downloadImage(id,{format:'png',scale:2,width:1200,height:520,filename:name});}

document.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('[data-k]').forEach(b=>b.onclick=()=>{state.k=+b.dataset.k;renderAll();});
  document.querySelectorAll('[data-p]').forEach(b=>b.onclick=()=>{state.p=+b.dataset.p;renderAll();});
  document.querySelectorAll('[data-order]').forEach(b=>b.onclick=()=>{state.order=b.dataset.order;drawHeat();
    document.querySelectorAll('[data-order]').forEach(x=>x.classList.toggle('on',x.dataset.order===state.order));});
  document.querySelectorAll('.banner').forEach(b=>b.onclick=()=>{
    const t=document.getElementById(b.dataset.target);
    if(t) t.scrollIntoView({behavior:'smooth',block:'start'});});
  renderAll();
});
"""


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>P04 · Regime Analytics</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Sora:wght@300;400;600;800&display=swap" rel="stylesheet">
<script>__PLOTLYJS__</script>
<style>
:root{--bg:#F2F5F9;--card:#FFFFFF;--ink:#1C2534;--sub:#5B6675;--faint:#8A94A6;
  --line:#E5E9F0;--blue:#2F8FD4;--maroon:#7C1D2E;}
*{box-sizing:border-box;}
body{margin:0;background:linear-gradient(180deg,#F7F9FC 0%,#EEF2F7 100%);
  color:var(--ink);font-family:'Sora',sans-serif;padding:0 28px 56px;}
.kick{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1.6px;
  color:var(--faint);text-transform:uppercase;}
.mono{font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--sub);}
.topbar{position:sticky;top:0;z-index:50;display:flex;justify-content:space-between;
  align-items:center;gap:18px;padding:15px 28px 13px;margin:0 -28px 22px;
  background:rgba(255,255,255,0.88);backdrop-filter:blur(10px);
  border-bottom:1px solid var(--line);}
.h1{font-weight:800;font-size:21px;letter-spacing:-0.4px;line-height:1.1;color:var(--ink);}
.sub{color:var(--sub);font-size:12px;font-weight:300;}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;}
.ctl{display:flex;align-items:center;gap:8px;}
.ctl label{font-family:'JetBrains Mono',monospace;font-size:9.5px;color:var(--faint);
  letter-spacing:1.2px;text-transform:uppercase;}
.seg{display:flex;border:1px solid #D7DEE8;border-radius:9px;overflow:hidden;background:#FFFFFF;}
.seg button{all:unset;cursor:pointer;padding:6px 12px;font-family:'JetBrains Mono',monospace;
  font-size:12px;color:var(--sub);transition:all .15s;}
.seg button:hover{color:var(--ink);background:#F3F6FA;}
.seg button.on{background:rgba(47,143,212,0.10);color:var(--blue);
  box-shadow:inset 0 0 0 1px rgba(47,143,212,0.45);font-weight:500;}
.badge{font-family:'JetBrains Mono',monospace;font-size:9.5px;letter-spacing:1.3px;
  color:var(--maroon);border:1px solid #E8D5D9;border-radius:999px;padding:6px 13px;
  background:#FBF5F6;white-space:nowrap;}
.metrics{display:grid;grid-template-columns:repeat(7,1fr);gap:14px;margin-bottom:22px;}
.metric{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:15px 15px;display:flex;gap:11px;align-items:flex-start;
  box-shadow:0 8px 24px rgba(23,32,55,0.06);transition:transform .15s,box-shadow .15s;}
.metric:hover{transform:translateY(-2px);box-shadow:0 12px 30px rgba(23,32,55,0.10);}
.mval{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;
  line-height:1;color:var(--ink);}
.mlab{color:var(--faint);font-size:9.5px;margin-top:7px;letter-spacing:0.5px;
  font-family:'JetBrains Mono',monospace;text-transform:uppercase;}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px;}
.dot.big{width:10px;height:10px;margin-top:4px;flex:none;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;}
.grid3{display:grid;grid-template-columns:1.1fr 1fr 1fr;gap:20px;margin-bottom:20px;}
.gridInsp{display:grid;grid-template-columns:1.6fr 1fr;gap:20px;margin-bottom:20px;}
.full{margin-bottom:20px;}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;
  padding:18px 18px 10px;box-shadow:0 10px 30px rgba(23,32,55,0.07);
  transition:box-shadow .2s;}
.card:hover{box-shadow:0 14px 36px rgba(23,32,55,0.10);}
.card.flash{box-shadow:0 0 0 2px rgba(47,143,212,0.45),0 10px 30px rgba(23,32,55,0.07);}
.card-head{display:flex;justify-content:space-between;align-items:flex-start;
  border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:8px;}
.ctitle{font-weight:600;font-size:15px;color:var(--ink);margin-top:3px;}
.chint{color:var(--faint);font-size:11px;font-weight:300;margin-top:2px;}
.pngbtn{all:unset;cursor:pointer;font-family:'JetBrains Mono',monospace;font-size:9.5px;
  color:var(--faint);border:1px solid var(--line);border-radius:8px;padding:5px 10px;
  letter-spacing:1px;transition:all .15s;background:#FBFCFE;}
.pngbtn:hover{color:var(--blue);border-color:rgba(47,143,212,0.45);}
.banner{cursor:pointer;background:var(--maroon);color:#FFFFFF;border-radius:8px;
  display:flex;justify-content:space-between;align-items:center;
  padding:13px 18px;margin:6px 0 20px;font-family:'JetBrains Mono',monospace;
  font-size:11.5px;letter-spacing:1.8px;text-transform:uppercase;
  box-shadow:0 8px 22px rgba(124,29,46,0.25);transition:transform .15s,box-shadow .15s;}
.banner:hover{transform:translateY(-1px);box-shadow:0 12px 26px rgba(124,29,46,0.32);}
.banner .arrow{font-size:15px;}
.pill{font-family:'JetBrains Mono',monospace;font-size:11px;border:1px solid;
  border-radius:999px;padding:4px 12px;margin-right:10px;}
#insp-meta{display:flex;align-items:center;margin:6px 0 4px;}
#insp-stats{padding:6px 2px 10px;}
.srow{display:flex;justify-content:space-between;padding:7px 2px;
  border-bottom:1px solid var(--line);font-size:12.5px;color:var(--sub);}
.srow b{color:var(--ink);font-family:'JetBrains Mono',monospace;font-weight:500;}
.krow{display:flex;justify-content:space-between;gap:12px;padding:8px 2px;
  border-bottom:1px solid var(--line);font-size:12px;color:var(--sub);}
.krow b{color:var(--ink);font-family:'JetBrains Mono',monospace;font-weight:500;
  text-align:right;}
.tbl{width:100%;border-collapse:collapse;margin:4px 0 12px;}
.tbl th{font-family:'JetBrains Mono',monospace;font-size:9.5px;letter-spacing:1px;
  text-transform:uppercase;color:var(--faint);text-align:right;padding:8px 6px;
  border-bottom:1px solid var(--ink);}
.tbl th:first-child{text-align:left;}
.tbl td{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--ink);
  text-align:right;padding:9px 6px;border-bottom:1px solid var(--line);}
.tbl td:first-child{text-align:left;font-family:'Sora',sans-serif;font-size:12.5px;}
.objective{font-size:12.5px;color:var(--sub);line-height:1.65;font-weight:300;
  padding:4px 2px 12px;}
.foot{display:flex;gap:22px;flex-wrap:wrap;border-top:1px solid var(--line);
  margin-top:18px;padding-top:18px;color:var(--faint);
  font-family:'JetBrains Mono',monospace;font-size:10.5px;}
.warn{color:var(--maroon);margin-left:auto;}
@media(max-width:1100px){.metrics{grid-template-columns:repeat(3,1fr);}
  .grid2,.grid3,.gridInsp{grid-template-columns:1fr;}
  .topbar{flex-direction:column;align-items:flex-start;}}
</style>
</head>
<body>

<div class="topbar">
  <div>
    <div class="kick">QUANT-LAB // P04 // MARKET REGIME ANALYTICS</div>
    <div class="h1">Market Regime Clustering</div>
    <div class="sub">Wasserstein k-means on empirical return distributions · Horvath, Issa &amp; Muguruza (2021)</div>
  </div>
  <div class="controls">
    <div class="ctl"><label>Regimes k</label>
      <div class="seg">
        <button data-k="2">2</button><button data-k="3">3</button>
        <button data-k="4">4</button><button data-k="5">5</button>
      </div>
    </div>
    <div class="ctl"><label>Distance</label>
      <div class="seg">
        <button data-p="1">W1</button><button data-p="2">W2</button>
      </div>
    </div>
    <div class="badge">__SOURCE__</div>
  </div>
</div>

<div class="metrics" id="kpis"></div>

<div class="full">
  <div class="card">
    <div class="card-head">
      <div><span class="kick">Price · Regime · Volatility</span>
        <div class="ctitle">Regime-shaded timeline</div>
        <div class="chint">Click anywhere on the price line to inspect that window below</div></div>
      <button class="pngbtn" onclick="png('hero','p04_timeline')">PNG</button>
    </div>
    <div id="hero"></div>
  </div>
</div>

<div class="banner" data-target="explorer">Regime Explorer <span class="arrow">→</span></div>

<div class="gridInsp" id="explorer">
  <div class="card" id="inspector-card">
    <div class="card-head">
      <div><span class="kick">Window Inspector</span>
        <div class="ctitle">Window vs its cluster</div>
        <div class="chint">Histogram of the selected window against its cluster's density (dotted)</div></div>
      <button class="pngbtn" onclick="png('inspect','p04_window')">PNG</button>
    </div>
    <div id="insp-meta"></div>
    <div id="inspect"></div>
  </div>
  <div class="card">
    <div class="card-head">
      <div><span class="kick">Statistics</span>
        <div class="ctitle">Window &amp; cluster stats</div></div>
    </div>
    <div id="insp-stats"></div>
  </div>
</div>

<div class="grid2">
  <div class="card">
    <div class="card-head">
      <div><span class="kick">Distributions</span>
        <div class="ctitle">Per-cluster return ridgelines</div></div>
      <button class="pngbtn" onclick="png('ridge','p04_ridgelines')">PNG</button>
    </div>
    <div id="ridge"></div>
  </div>
  <div class="card">
    <div class="card-head">
      <div><span class="kick">Optimal Transport</span>
        <div class="ctitle">Wasserstein distance matrix</div>
        <div class="chint" style="display:flex;gap:10px;align-items:center;">Order
          <span class="seg" style="display:inline-flex;">
            <button data-order="time">time</button><button data-order="regime">regime</button>
          </span> · click a cell to inspect</div></div>
      <button class="pngbtn" onclick="png('heat','p04_distance_matrix')">PNG</button>
    </div>
    <div id="heat"></div>
  </div>
</div>

<div class="grid2">
  <div class="card">
    <div class="card-head">
      <div><span class="kick">Embedding</span>
        <div class="ctitle">2D regime map (MDS)</div>
        <div class="chint">Click a point to inspect that window</div></div>
      <button class="pngbtn" onclick="png('mds','p04_regime_map')">PNG</button>
    </div>
    <div id="mds"></div>
  </div>
  <div class="card">
    <div class="card-head">
      <div><span class="kick">Dynamics</span>
        <div class="ctitle">Regime transition matrix</div></div>
      <button class="pngbtn" onclick="png('trans','p04_transitions')">PNG</button>
    </div>
    <div id="trans"></div>
  </div>
</div>

<div class="banner" data-target="report">Regime Report <span class="arrow">→</span></div>

<div class="grid3" id="report">
  <div class="card">
    <div class="card-head">
      <div><span class="kick">Regime Allocation</span>
        <div class="ctitle">Share of days per regime</div></div>
      <button class="pngbtn" onclick="png('donut','p04_regime_mix')">PNG</button>
    </div>
    <div id="donut"></div>
  </div>
  <div class="card">
    <div class="card-head">
      <div><span class="kick">Allocation Detail</span>
        <div class="ctitle">Regime characteristics</div></div>
    </div>
    <div id="alloc-table"></div>
  </div>
  <div class="card">
    <div class="card-head">
      <div><span class="kick">Key Information</span>
        <div class="ctitle">Model factsheet</div></div>
    </div>
    <div id="keyinfo"></div>
  </div>
</div>

<div class="full">
  <div class="card">
    <div class="card-head">
      <div><span class="kick">Research Objective</span>
        <div class="ctitle">What this model does</div></div>
    </div>
    <div class="objective">
      Standard regime detection compresses each slice of market history into a few
      scalar features and clusters those, discarding the shape of the distribution:
      skew, fat tails, bimodality. This model keeps the entire empirical return
      distribution as the object of study and measures the distance between market
      states with the p-Wasserstein metric from optimal transport, clustering
      directly on the space of probability measures with Wasserstein barycenters as
      centroids. Separation quality is validated with the maximum mean discrepancy
      between and within clusters against a moment-based k-means baseline, following
      Horvath, Issa &amp; Muguruza (2021).
    </div>
  </div>
</div>

<div class="banner" data-target="validation">Validation Report <span class="arrow">→</span></div>

<div class="grid2" id="validation">
  <div class="card">
    <div class="card-head">
      <div><span class="kick">Validation</span>
        <div class="ctitle">WK-means vs moment baseline (MMD)</div></div>
      <button class="pngbtn" onclick="png('cmp','p04_validation')">PNG</button>
    </div>
    <div id="cmp"></div>
  </div>
  <div class="card">
    <div class="card-head">
      <div><span class="kick">Model Selection</span>
        <div class="ctitle">k by inertia and silhouette</div></div>
      <button class="pngbtn" onclick="png('ksel','p04_k_selection')">PNG</button>
    </div>
    <div id="ksel"></div>
  </div>
</div>

<div class="foot">
  <span>Method: Horvath, Issa &amp; Muguruza (2021), arXiv:2110.11848.</span>
  <span>1D optimal transport distances; centroids are Wasserstein barycenters.</span>
  <span class="warn">Not financial advice.</span>
</div>

<script>const DATA = __PAYLOAD__;</script>
<script>__APPJS__</script>
</body>
</html>"""


def build(args=None):
    dates, prices, returns, source = get_series(args or [])
    print("series:", source, "|", len(returns), "returns")
    payload = build_payload(dates, prices, returns, source)
    html = (TEMPLATE
            .replace("__PLOTLYJS__", pyo.get_plotlyjs())
            .replace("__SOURCE__", payload["source"])
            .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
            .replace("__APPJS__", APP_JS))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "output", "P04_regime_terminal.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    c = payload["configs"]["k3_p2"]
    print("configs", len(payload["configs"]), "| k3_p2 wk", c["wk"]["ratio"],
          "mom", c["mom"]["ratio"], "sil", c["sil"])
    print("wrote", out)


if __name__ == "__main__":
    build(sys.argv[1:])
