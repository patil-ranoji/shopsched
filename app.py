import pandas as pd
import plotly.express as px
import streamlit as st
from ortools.sat.python import cp_model

st.set_page_config(page_title="ShopSched", layout="wide")
S = st.session_state

# ---------------- state + sample data ----------------
if "init" not in S:
    S.machines = pd.DataFrame({"Machine": ["Lathe-1", "Lathe-2", "Mill-1", "Mill-2", "Drill-1", "Grinder-1"],
                               "Type": ["Lathe", "Lathe", "Mill", "Mill", "Drill", "Grinder"]})
    S.jobs = pd.DataFrame({"Job": ["J1", "J2", "J3", "J4", "J5"], "Due": [14, 12, 16, 15, 18]})
    S.ops = pd.DataFrame(
        [("J1", 1, "Lathe", 4), ("J1", 2, "Mill", 3), ("J1", 3, "Drill", 2),
         ("J2", 1, "Mill", 5), ("J2", 2, "Grinder", 3),
         ("J3", 1, "Lathe", 6), ("J3", 2, "Drill", 2), ("J3", 3, "Grinder", 2),
         ("J4", 1, "Drill", 3), ("J4", 2, "Lathe", 4), ("J4", 3, "Mill", 2),
         ("J5", 1, "Lathe", 5), ("J5", 2, "Mill", 4), ("J5", 3, "Grinder", 3)],
        columns=["Job", "Op", "Type", "Hours"])
    S.rows, S.status, S.downs, S.log, S.init = [], {}, [], [], True

machines = dict(zip(S.machines.dropna().Machine, S.machines.dropna().Type))
ops_ok = S.ops.dropna()
jobs = {r.Job: {"due": int(r.Due),
                "ops": [(o.Type, int(o.Hours)) for o in ops_ok[ops_ok.Job == r.Job].sort_values("Op").itertuples()]}
        for r in S.jobs.dropna().itertuples()}


# ---------------- scheduling engines ----------------
def rule_schedule(name):
    key = {"FCFS": lambda j: list(jobs).index(j),
           "SPT": lambda j: sum(t for _, t in jobs[j]["ops"]),
           "EDD": lambda j: jobs[j]["due"]}[name]
    free = {m: 0 for m in machines}
    rows = []
    for j in sorted(jobs, key=key):
        ready = 0
        for k, (typ, p) in enumerate(jobs[j]["ops"]):
            m = min((m for m, t in machines.items() if t == typ), key=lambda m: max(free[m], ready))
            s = max(free[m], ready)
            free[m] = ready = s + p
            rows.append(dict(Job=j, Op=k + 1, Machine=m, Start=s, End=s + p))
    return rows


def cpsat(w_late=5, limit=10, now=0, fixed=None, downs=()):
    """fixed = {(job, op): row} for done/running ops; only the rest are re-planned from `now`."""
    fixed = fixed or {}
    m = cp_model.CpModel()
    H = int(now + sum(t for j in jobs.values() for _, t in j["ops"]) + sum(d[2] for d in downs)
            + max([r["End"] for r in fixed.values()] or [0]) + 1)
    pm = {k: [] for k in machines}
    ends, starts, choice, lates = {}, {}, {}, []
    span = m.NewIntVar(0, H, "span")
    for j, d in jobs.items():
        for k, (typ, p) in enumerate(d["ops"]):
            if (j, k + 1) in fixed:  # finished or running: keep as is
                r = fixed[(j, k + 1)]
                if r["Machine"] in pm and r["End"] > now:
                    pm[r["Machine"]].append(m.NewIntervalVar(now, r["End"] - now, r["End"], f"f_{j}_{k}"))
                ends[(j, k)] = int(r["End"])
                continue
            s = m.NewIntVar(now, H, f"s_{j}_{k}")
            e = m.NewIntVar(now, H, f"e_{j}_{k}")
            m.Add(e == s + p)
            lits = []
            for mk, mt in machines.items():
                if mt == typ:  # compatibility
                    x = m.NewBoolVar(f"x_{j}_{k}_{mk}")
                    pm[mk].append(m.NewOptionalIntervalVar(s, p, e, x, f"i_{j}_{k}_{mk}"))
                    choice[(j, k, mk)] = x
                    lits.append(x)
            m.AddExactlyOne(lits)
            if k > 0:  # precedence
                m.Add(s >= ends[(j, k - 1)])
            starts[(j, k)], ends[(j, k)] = s, e
        last = ends[(j, len(d["ops"]) - 1)]
        m.Add(span >= last)
        L = m.NewIntVar(0, H, f"late_{j}")
        m.Add(L >= last - d["due"])
        lates.append(L)
    for mk, t0, dur in downs:  # breakdowns block the machine
        if t0 + dur > now:
            a = max(t0, now)
            pm[mk].append(m.NewIntervalVar(a, t0 + dur - a, t0 + dur, f"down_{mk}_{t0}"))
    for ivs in pm.values():  # machine capacity
        m.AddNoOverlap(ivs)
    m.Minimize(w_late * sum(lates) + span)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit
    if solver.Solve(m) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return [dict(Job=j, Op=k + 1, Machine=mk, Start=solver.Value(starts[(j, k)]), End=solver.Value(ends[(j, k)]))
            for (j, k, mk), x in choice.items() if solver.Value(x)]


def generate(method, w_late=5):
    rows = cpsat(w_late) if method == "CP-SAT" else rule_schedule(method)
    if rows is None:
        st.error("No feasible schedule. Check that every operation type has a machine.")
        return
    S.rows, S.status, S.downs, S.log = rows, {}, [], []


def replan(now):
    fixed = {(r["Job"], r["Op"]): r for r in S.rows if S.status.get((r["Job"], r["Op"]), "Planned") != "Planned"}
    new = cpsat(now=now, fixed=fixed, downs=S.downs)
    if new is None:
        st.error("No feasible re-plan.")
        return
    S.rows = list(fixed.values()) + new


def metrics(rows):
    df = pd.DataFrame(rows)
    span = df.End.max()
    busy = (df.End - df.Start).sum()
    total = span * len(machines)
    last = df.groupby("Job").End.max()
    late = sum(max(0, last.get(j, 0) - jobs[j]["due"]) for j in jobs)
    return {"Makespan": int(span), "Total lateness": int(late),
            "Idle time": int(total - busy), "Utilisation %": round(100 * busy / total, 1)}


def gantt(rows):
    df = pd.DataFrame(rows)
    df["Status"] = [S.status.get((r.Job, r.Op), "Planned") for r in df.itertuples()]
    base = pd.Timestamp("2026-01-01 08:00")
    df["S"] = base + pd.to_timedelta(df.Start, unit="h")
    df["F"] = base + pd.to_timedelta(df.End, unit="h")
    fig = px.timeline(df, x_start="S", x_end="F", y="Machine", color="Job", text="Job", hover_data=["Op", "Status"])
    fig.update_yaxes(autorange="reversed")
    return fig


# ---------------- pages ----------------
page = st.sidebar.radio("ShopSched", ["Home", "Jobs", "Machines", "Schedule", "Operator", "Reports"])

if page == "Home":
    st.title("Today")
    mt = metrics(S.rows) if S.rows else None
    a, b, c, d = st.columns(4)
    a.metric("Active jobs", len(jobs))
    b.metric("Machines", len(machines))
    c.metric("Utilisation", f"{mt['Utilisation %']}%" if mt else "—")
    d.metric("Idle time (h)", mt["Idle time"] if mt else "—")
    if st.button("Generate schedule (CP-SAT)", type="primary"):
        generate("CP-SAT")
        st.rerun()
    if S.rows:
        df = pd.DataFrame(S.rows)
        last = df.groupby("Job").End.max()
        risk = [f"{j} (ends {int(last[j])} h, due {jobs[j]['due']} h)" for j in jobs if last.get(j, 0) > jobs[j]["due"]]
        st.write("**Late-risk jobs:** " + (", ".join(risk) if risk else "none"))
        st.plotly_chart(gantt(S.rows), use_container_width=True)
    else:
        st.info("No schedule yet. Click the button above.")

elif page == "Jobs":
    st.title("Jobs")
    st.caption("Job cards (due time) and routing (ordered operations with machine type and hours).")
    S.jobs = st.data_editor(S.jobs, num_rows="dynamic", key="je")
    S.ops = st.data_editor(S.ops, num_rows="dynamic", key="oe")

elif page == "Machines":
    st.title("Machines")
    S.machines = st.data_editor(S.machines, num_rows="dynamic", key="me")
    if S.downs:
        st.write("**Breakdowns this run**")
        st.table(pd.DataFrame(S.downs, columns=["Machine", "From hour", "Hours down"]))

elif page == "Schedule":
    st.title("Schedule")
    a, b = st.columns(2)
    method = a.selectbox("Method", ["FCFS", "SPT", "EDD", "CP-SAT"], index=3)
    w = b.slider("CP-SAT weight on lateness (makespan weight = 1)", 0, 10, 5)
    if st.button("Run", type="primary"):
        generate(method, w)
        st.rerun()
    if S.rows:
        st.plotly_chart(gantt(S.rows), use_container_width=True)
        st.table(pd.DataFrame([metrics(S.rows)]))
        st.caption("Dynamic re-planning happens from the Operator page when you start, complete, delay or report a breakdown.")

elif page == "Operator":
    st.title("Operator")
    if not S.rows:
        st.info("Generate a schedule first.")
    else:
        mk = st.selectbox("Machine", list(machines))
        now = int(st.number_input("Current hour", 0, 200, 0))
        q = sorted([r for r in S.rows if r["Machine"] == mk and S.status.get((r["Job"], r["Op"]), "Planned") != "Done"],
                   key=lambda r: r["Start"])
        if not q:
            st.success("No pending operations on this machine.")
        else:
            r = q[0]
            key = (r["Job"], r["Op"])
            stt = S.status.get(key, "Planned")
            st.subheader(f"{r['Job']} · Op {r['Op']}")
            st.write(f"Status: **{stt}** · planned {r['Start']}–{r['End']} h")
            c1, c2, c3 = st.columns(3)
            if c1.button("Start", disabled=stt == "Running", type="primary"):
                if stt == "Planned":
                    d = r["End"] - r["Start"]
                    r["Start"], r["End"] = now, now + d
                S.status[key] = "Running"
                S.log.append(dict(Hour=now, Machine=mk, Job=r["Job"], Event="Start"))
                replan(now)
                st.rerun()
            if c2.button("Pause", disabled=stt != "Running"):
                S.status[key] = "Paused"
                S.log.append(dict(Hour=now, Machine=mk, Job=r["Job"], Event="Pause"))
                st.rerun()
            if c3.button("Complete", disabled=stt not in ("Running", "Paused")):
                r["End"] = max(now, r["Start"])
                S.status[key] = "Done"
                S.log.append(dict(Hour=now, Machine=mk, Job=r["Job"], Event="Complete"))
                replan(now)
                st.rerun()
            with st.expander("Report delay or breakdown"):
                kind = st.radio("Type", ["Delay on current operation", "Machine breakdown"])
                hrs = int(st.number_input("Hours", 1, 100, 2))
                if st.button("Apply and re-plan"):
                    if kind.startswith("Delay"):
                        if stt in ("Running", "Paused"):
                            r["End"] += hrs
                        else:
                            st.warning("Start the operation first.")
                            st.stop()
                    else:
                        S.downs.append((mk, now, hrs))
                    S.log.append(dict(Hour=now, Machine=mk, Job=r["Job"], Event=f"{kind} ({hrs} h)"))
                    replan(now)
                    st.rerun()
            st.write("**Next on this machine**")
            st.table(pd.DataFrame(q[1:])[["Job", "Op", "Start", "End"]] if len(q) > 1 else pd.DataFrame({"": ["none"]}))

elif page == "Reports":
    st.title("Reports")
    if st.button("Compare all methods on current data"):
        res = {n: metrics(rule_schedule(n)) for n in ["FCFS", "SPT", "EDD"]}
        cp = cpsat()
        if cp:
            res["CP-SAT"] = metrics(cp)
        st.table(pd.DataFrame(res).T)
        st.caption("Lower is better except utilisation. Copy this table into the report.")
    if S.rows:
        df = pd.DataFrame(S.rows)
        span = df.End.max()
        util = (df.assign(B=df.End - df.Start).groupby("Machine").B.sum() / span * 100).reindex(list(machines)).fillna(0)
        st.write("**Utilisation by machine (current schedule)**")
        st.bar_chart(util)
        st.download_button("Download schedule (CSV)", df.to_csv(index=False), "schedule.csv")
    if S.log:
        st.write("**Event log**")
        st.table(pd.DataFrame(S.log))
