"""Premium Streamlit experience for selecting and assessing source incidents."""

from __future__ import annotations

import os
import re
import time
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from incidentrag.approval.ui.api_client import IncidentRAGAPIClient, IncidentRAGAPIError
from incidentrag.approval.ui.briefing import render_briefing_html
from incidentrag.approval.ui.graph_payload import analysis_payload, assessment_payload
from incidentrag.approval.ui.network_engine import render_network

load_dotenv()
API_URL = os.getenv("INCIDENTRAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("INCIDENTRAG_API_KEY", "")
client = IncidentRAGAPIClient(API_URL, API_KEY)

st.set_page_config(page_title="IncidentRAG", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")


def _neutral(value: object) -> str:
    """Preserve source values internally, but never surface source product branding."""
    return re.sub(r"argo[\s-]*cd|argocd|agrocd", "source system", str(value or ""), flags=re.I)


def _issue(issues: list[dict[str, Any]], number: object) -> dict[str, Any] | None:
    return next((item for item in issues if str(item.get("number")) == str(number)), None)


def _metric(label: str, value: str, detail: str) -> str:
    return f"<div class='metric'><b>{value}</b><div>{label}</div><span>{detail}</span></div>"


def _metrics(result: dict[str, Any]) -> str:
    hypotheses = result.get("ai_hypotheses") or {}
    count = int(bool(hypotheses.get("likely_diagnosis"))) + len(hypotheses.get("possible_contributing_factors") or [])
    confidence = result.get("confidence")
    confidence_text = f"{float(confidence):.0%}" if isinstance(confidence, (int, float)) else "—"
    severity = str((result.get("facts_extracted") or {}).get("severity") or "Unspecified").upper()
    return "".join((
        _metric("Confidence", confidence_text, "overall assessment"),
        _metric("Risk", severity, "returned severity"),
        _metric("Evidence", f"{len(result.get('retrieved_evidence') or []):02d}", "supporting items"),
        _metric("Hypotheses", f"{count:02d}", "ranked explanations"),
        _metric("Authorization", "REVIEW" if result.get("human_approval_required") else "CLEAR", "human gate status"),
    ))


def _error(title: str, exc: IncidentRAGAPIError) -> None:
    code = f" · HTTP {exc.status_code}" if exc.status_code else ""
    st.markdown(
        f"<div class='interruption'><small>ANALYSIS INTERRUPTED{code}</small><strong>{title}</strong><p>{_neutral(exc)}</p></div>",
        unsafe_allow_html=True,
    )


st.markdown("""
<style>
:root{--ice:#9ee9ff;--cyan:#55c9eb;--ink:#edf5ff;--muted:#8b9ab5}
.stApp{background:radial-gradient(900px 520px at 18% -8%,rgba(58,100,167,.25),transparent 66%),radial-gradient(760px 560px at 90% 34%,rgba(82,64,145,.15),transparent 70%),#05070b;color:var(--ink)}
.stApp:before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.38;background-image:linear-gradient(rgba(142,190,225,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(142,190,225,.035) 1px,transparent 1px);background-size:48px 48px;mask-image:radial-gradient(ellipse at center,black,transparent 82%)}
.main .block-container{max-width:1280px;padding:1.1rem 2.2rem 3.5rem;position:relative;z-index:1}[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],header{display:none!important}[data-testid="stToolbar"]{right:1rem}
.irag-nav{display:flex;align-items:center;justify-content:space-between;margin:0 auto 7.6vh;padding:5px 0}.wordmark{font-weight:750;font-size:1.12rem;letter-spacing:-.03em;color:#f4f8ff}.wordmark i{color:var(--cyan);font-style:normal}.nav-status{color:#a8bad1;font:600 .67rem/1 ui-monospace,Consolas,monospace;letter-spacing:.13em}.nav-status:before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:8px;background:#5ee5bd;box-shadow:0 0 16px #5ee5bd}
.eyebrow{color:#8ecfe6;font:600 .68rem/1 ui-monospace,Consolas,monospace;letter-spacing:.2em}.hero{max-width:790px;margin:0 auto;text-align:center;padding-bottom:2.6rem}.hero h1{font-size:clamp(3.2rem,7.4vw,6.3rem);letter-spacing:-.065em;line-height:.9;margin:.78rem 0 1.3rem;color:#f5f8ff}.hero p{margin:0 auto;max-width:530px;color:#aab5c8;line-height:1.65;font-size:1rem}.glass-label{color:#c2ecfa;font:600 .69rem/1 ui-monospace,Consolas,monospace;letter-spacing:.17em}.glass-note,.irag-credit{color:#6e7d98;font-size:.78rem;letter-spacing:.06em}
div[data-testid="stForm"]{border:1px solid rgba(181,224,255,.15);border-radius:22px;padding:1.15rem 1.15rem .8rem;margin:0 auto 1.5rem;background:linear-gradient(125deg,rgba(255,255,255,.075),rgba(255,255,255,.025));box-shadow:0 28px 90px rgba(0,0,0,.28),inset 0 1px rgba(255,255,255,.08);backdrop-filter:blur(20px)}div[data-testid="stForm"]:hover{border-color:rgba(145,221,255,.3)}
label,[data-testid="stWidgetLabel"] p{color:#b6c6db!important;font-size:.72rem!important;letter-spacing:.07em}input,[data-baseweb="select"]>div{background:rgba(4,8,16,.52)!important;border-color:rgba(164,205,239,.17)!important;border-radius:11px!important;color:#eaf5ff!important}input:focus{border-color:#68d7f7!important;box-shadow:0 0 0 3px rgba(88,207,241,.12)!important}
.stButton>button,.stFormSubmitButton>button{min-height:44px;border-radius:12px;border:1px solid rgba(166,233,255,.42);color:#f3fcff;font-weight:650;letter-spacing:.03em;background:linear-gradient(105deg,rgba(59,174,213,.9),rgba(98,95,201,.86))!important;box-shadow:0 12px 30px rgba(46,140,207,.16);transition:transform .18s ease,box-shadow .18s ease}.stButton>button:hover,.stFormSubmitButton>button:hover{transform:translateY(-2px) scale(1.006);box-shadow:0 17px 38px rgba(65,166,232,.3);border-color:#d0f7ff}
.results-head{margin:3.2rem 0 1rem;color:#edf5ff;font-size:1.5rem;letter-spacing:-.03em}.incident-number{color:#77d9f4;font:600 .71rem/1 ui-monospace,Consolas,monospace;letter-spacing:.13em}.incident-title{color:#edf5ff;font-size:1.08rem;font-weight:650;margin:.28rem 0}.incident-meta{color:#8292ae;font-size:.78rem}.selection-strip{border-left:2px solid rgba(102,215,244,.68);margin:0 0 .35rem;padding:.3rem 0 .45rem .95rem;background:linear-gradient(90deg,rgba(67,173,209,.09),transparent 60%)}
.analysis-bar{display:flex;justify-content:space-between;align-items:center;margin:0 auto 1rem;max-width:1180px;color:#bdd1e8;font:600 .72rem/1 ui-monospace,Consolas,monospace;letter-spacing:.14em}.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;margin:2rem 0 1.4rem;border:1px solid rgba(180,220,255,.12);background:rgba(180,220,255,.12)}.metric{min-height:115px;padding:1.15rem;background:rgba(8,13,24,.77)}.metric b{display:block;color:#edf7ff;font-size:1.85rem;letter-spacing:-.05em}.metric div{color:#99dcef;margin-top:.45rem;font:600 .68rem/1 ui-monospace,Consolas,monospace;letter-spacing:.13em;text-transform:uppercase}.metric span{display:block;color:#697994;margin-top:.45rem;font-size:.72rem}.briefing{border:1px solid rgba(180,220,255,.14);border-radius:20px;padding:1.45rem 1.5rem 1.7rem;background:linear-gradient(180deg,rgba(10,16,28,.88),rgba(7,11,20,.82));margin-bottom:1.2rem}.briefing>.brief-kicker:first-child{margin-top:0}.briefing h2{margin:.2rem 0 .85rem;font-size:1.55rem;letter-spacing:-.04em;line-height:1.25;color:#f4f8ff}.brief-kicker{margin:1.15rem 0 .4rem;color:#8ecfe6;font:600 .68rem/1 ui-monospace,Consolas,monospace;letter-spacing:.16em;text-transform:uppercase}.brief-copy{font-size:1.05rem;line-height:1.6;color:#e7eef8}.brief-note,.brief-empty{color:#8b9ab5;font-size:.9rem}.brief-chips{display:flex;flex-wrap:wrap;gap:.45rem;margin:.2rem 0 1rem}.brief-chip{border:1px solid rgba(158,233,255,.22);border-radius:999px;padding:.28rem .7rem;color:#c5d6ea;font-size:.72rem;letter-spacing:.06em}.brief-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:.8rem}.brief-card{padding:1rem;border:1px solid rgba(180,220,255,.12);border-radius:14px;background:rgba(8,13,24,.55)}.brief-card strong{display:block;margin:.2rem 0 .45rem;color:#edf5ff}.brief-card p{margin:0;color:#c9d4e4;line-height:1.5;font-size:.92rem}.brief-card span,.brief-card code{display:block;margin-top:.55rem;color:#7f91b3;font-size:.75rem}.brief-card code{white-space:pre-wrap;color:#9ee9ff}.brief-steps{list-style:none;padding:0;margin:0}.brief-steps li{display:flex;gap:.75rem;align-items:flex-start;padding:.55rem 0;border-bottom:1px solid rgba(180,220,255,.08);color:#d7e2f0}.brief-steps b{color:#77d9f4;font:600 .78rem/1.4 ui-monospace,Consolas,monospace}.brief-block ul{margin:.2rem 0 0;padding-left:1.1rem;color:#d0dbea;line-height:1.55}.brief-alert{margin:1rem 0;padding:.85rem 1rem;border:1px solid rgba(255,107,138,.4);border-radius:12px;background:rgba(48,12,22,.45);color:#ffd0da}.interruption{max-width:680px;margin:1.2rem auto;padding:1.3rem 1.4rem;border:1px solid rgba(255,151,119,.36);border-radius:16px;background:rgba(54,16,20,.5)}.interruption small{display:block;color:#ffad92;letter-spacing:.14em;margin-bottom:.5rem}.interruption strong{font-size:1.1rem}.interruption p{color:#d9bac0;margin:.5rem 0 0}
@media(max-width:700px){.main .block-container{padding:1rem 1rem 2.5rem}.irag-nav{margin-bottom:3.8rem}.hero{text-align:left}.hero h1{font-size:3.35rem}.metrics{grid-template-columns:repeat(2,1fr)}.analysis-bar{padding:0 .4rem;font-size:.62rem}.briefing{padding:1.1rem}}
</style>
""", unsafe_allow_html=True)

try:
    health = client.health()
    client.github_status()  # Preserve readiness integration without surfacing repository details.
    online = health.get("status") == "ok"
except IncidentRAGAPIError:
    online = False

st.markdown(f"<nav class='irag-nav'><div class='wordmark'>Incident<i>RAG</i></div><div class='nav-status'>{'SYSTEM ONLINE' if online else 'SYSTEM OFFLINE'}</div></nav>", unsafe_allow_html=True)
phase = st.session_state.get("phase", "idle")
issues: list[dict[str, Any]] = st.session_state.get("issues", [])
selected = _issue(issues, st.session_state.get("selected_issue"))

if phase in {"idle", "searching", "selected"}:
    st.markdown("<section class='hero'><div class='eyebrow'>INCIDENT INTELLIGENCE</div><h1>See the signal.<br>Follow the evidence.</h1><p>Find an engineering incident, inspect its context, then launch an evidence-grounded assessment.</p></section><div class='glass-label'>DISCOVER / SOURCE INCIDENTS</div>", unsafe_allow_html=True)
    with st.form("issue-search", border=False):
        cols = st.columns([2.2, 1, .55])
        keyword = cols[0].text_input("Search terms", value=st.session_state.get("keyword", "repo-server"), placeholder="timeout, sync, TLS…")
        component = cols[1].text_input("Component label", value=st.session_state.get("component", ""), placeholder="optional")
        limit = cols[2].selectbox("Results", [1, 2], index=1)
        search = st.form_submit_button("Find incidents  →", use_container_width=True)
    if search:
        st.session_state.update(keyword=keyword, component=component, phase="searching")
        st.session_state.pop("assessment", None)
        try:
            response = client.find_issues(keyword=keyword.strip(), component=component.strip(), limit=limit)
            st.session_state["issues"] = response.get("issues", [])
            st.session_state.pop("selected_issue", None)
            st.session_state["phase"] = "selected" if st.session_state["issues"] else "idle"
            st.rerun()
        except IncidentRAGAPIError as exc:
            st.session_state.update(issues=[], phase="idle")
            _error("Incident search could not complete", exc)
    if issues:
        st.markdown("<div class='results-head'>Matching incidents</div>", unsafe_allow_html=True)
        for item in issues:
            number = item.get("number")
            is_selected = selected is not None and str(selected.get("number")) == str(number)
            label_text = _neutral(" · ".join(item.get("labels") or []) or "Open engineering issue")
            st.markdown(f"<div class='selection-strip'><div class='incident-number'>INCIDENT / #{number}</div><div class='incident-title'>{_neutral(item.get('title'))}</div><div class='incident-meta'>{label_text}</div></div>", unsafe_allow_html=True)
            if st.button("Selected" if is_selected else "Select incident  →", key=f"select-{number}", use_container_width=True):
                st.session_state.update(selected_issue=number, phase="selected")
                st.rerun()
    selected = _issue(st.session_state.get("issues", []), st.session_state.get("selected_issue"))
    if selected:
        labels = _neutral(" · ".join(selected.get("labels") or []) or "Open engineering issue")
        st.markdown(f"<div class='results-head'>Origin selected</div><div class='selection-strip'><div class='incident-number'>#{selected.get('number')} / READY</div><div class='incident-title'>{_neutral(selected.get('title'))}</div><div class='incident-meta'>{labels}</div></div>", unsafe_allow_html=True)
        preview = _neutral(selected.get("body", "").strip())
        if preview:
            with st.expander("Incident context"):
                st.write(preview)
        left, right = st.columns([1, 2.2])
        if selected.get("html_url"):
            left.link_button("View source context", selected["html_url"], use_container_width=True)
        if right.button("Run analysis  →", key="analyze", type="primary", use_container_width=True):
            st.session_state.pop("assessment", None)
            st.session_state["phase"] = "analyzing"
            st.rerun()
    elif not issues:
        st.markdown("<p class='glass-note'>The incident field is waiting for a search query.</p>", unsafe_allow_html=True)

if phase == "analyzing" and selected:
    st.markdown("<div class='analysis-bar'><span>ANALYSIS ACTIVE</span><span>CONSTRUCTING EVIDENCE NETWORK</span></div>", unsafe_allow_html=True)
    render_network(analysis_payload(selected), height=740)
    try:
        st.session_state["assessment"] = client.analyze_issue(int(selected["number"]))
        st.session_state["phase"] = "result_transition"
        st.rerun()
    except IncidentRAGAPIError as exc:
        st.session_state.update(analysis_error=exc, phase="error")
        st.rerun()

assessment = st.session_state.get("assessment")
if phase == "result_transition" and assessment:
    st.markdown("<div class='analysis-bar'><span>NETWORK CONVERGING</span><span>BINDING RETURNED ENTITIES</span></div>", unsafe_allow_html=True)
    render_network(assessment_payload(assessment), height=760)
    time.sleep(.75)
    st.session_state["phase"] = "ready"
    st.rerun()
if phase == "ready" and assessment:
    st.markdown("<div class='analysis-bar'><span>ASSESSMENT READY</span><span>EXPLORE THE KNOWLEDGE GRAPH</span></div>", unsafe_allow_html=True)
    render_network(assessment_payload(assessment), height=760)
    st.markdown(f"<section class='metrics'>{_metrics(assessment)}</section>", unsafe_allow_html=True)
    st.markdown(render_briefing_html(assessment, sanitize=_neutral), unsafe_allow_html=True)
    if assessment.get("disclaimer"):
        st.caption(_neutral(assessment["disclaimer"]))
if phase == "error" and selected:
    render_network(analysis_payload(selected), height=620)
    error = st.session_state.get("analysis_error")
    if isinstance(error, IncidentRAGAPIError):
        _error("The assessment stopped before completion", error)
    if st.button("Retry analysis  →", type="primary", use_container_width=True):
        st.session_state["phase"] = "analyzing"
        st.rerun()

st.markdown("<div class='irag-credit'>Created by SG</div>", unsafe_allow_html=True)
