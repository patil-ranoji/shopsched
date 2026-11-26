from __future__ import annotations

import io
import math
import os
import sqlite3
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from ortools.sat.python import cp_model

try:
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except Exception:
    SQLALCHEMY_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

st.set_page_config(page_title="ShopSched 2.0", page_icon="🏭", layout="wide")
st.title("🏭 ShopSched 2.0")
st.caption("Manufacturing production planning, finite-capacity scheduling and shop-floor tracking")

# ---------- Database: SQLite locally; managed PostgreSQL when DATABASE_URL is configured ----------
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
USE_POSTGRES = bool(DATABASE_URL and SQLALCHEMY_AVAILABLE)
DB_PATH = os.getenv("SHOPSHED_DB_PATH", "shopsched.db")

def db_connect():
    if USE_POSTGRES:
        return create_engine(DATABASE_URL, pool_pre_ping=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def db_exec(sql: str, params=()):
    if USE_POSTGRES:
        with db_connect().begin() as con:
            con.execute(text(sql), dict(params) if isinstance(params, dict) else {})
    else:
        with db_connect() as con:
            con.execute(sql, params)
            con.commit()

def db_read(sql: str) -> pd.DataFrame:
    if USE_POSTGRES:
        with db_connect().connect() as con:
            return pd.read_sql(text(sql), con)
    with db_connect() as con:
        return pd.read_sql_query(sql, con)

def init_db():
    stmts = [
        """CREATE TABLE IF NOT EXISTS machines (
            machine TEXT PRIMARY KEY, machine_type TEXT NOT NULL,
            shift_start TEXT NOT NULL DEFAULT '08:00', shift_end TEXT NOT NULL DEFAULT '17:00',
            available_days TEXT NOT NULL DEFAULT '0,1,2,3,4,5', active INTEGER NOT NULL DEFAULT 1)""",
        """CREATE TABLE IF NOT EXISTS jobs (
            job TEXT PRIMARY KEY, due_hour INTEGER NOT NULL DEFAULT 24,
            quantity INTEGER NOT NULL DEFAULT 1, batch_size INTEGER NOT NULL DEFAULT 1,
            priority INTEGER NOT NULL DEFAULT 1, notes TEXT DEFAULT '')""",
        """CREATE TABLE IF NOT EXISTS operations (
            id INTEGER PRIMARY KEY, job TEXT NOT NULL, op INTEGER NOT NULL,
            machine_type TEXT NOT NULL, hours REAL NOT NULL,
            UNIQUE(job, op))""",
        """CREATE TABLE IF NOT EXISTS downtimes (
            id INTEGER PRIMARY KEY, machine TEXT NOT NULL, start_hour REAL NOT NULL,
            duration REAL NOT NULL, reason TEXT DEFAULT 'Breakdown', kind TEXT DEFAULT 'Breakdown')""",
        """CREATE TABLE IF NOT EXISTS event_log (
            id INTEGER PRIMARY KEY, event_time TEXT NOT NULL, machine TEXT, job TEXT,
            event TEXT NOT NULL, details TEXT DEFAULT '')"""
    ]
    for s in stmts: db_exec(s)

def seed_if_empty():
    if db_read("SELECT * FROM machines").empty:
        rows = [
            ("Lathe-1","Lathe","08:00","17:00","0,1,2,3,4,5",1),
            ("Lathe-2","Lathe","08:00","17:00","0,1,2,3,4,5",1),
            ("Mill-1","Mill","08:00","17:00","0,1,2,3,4,5",1),
            ("Mill-2","Mill","08:00","17:00","0,1,2,3,4,5",1),
            ("Drill-1","Drill","08:00","17:00","0,1,2,3,4,5",1),
            ("Grinder-1","Grinder","08:00","17:00","0,1,2,3,4,5",1),
        ]
        for r in rows:
            db_exec("INSERT INTO machines(machine,machine_type,shift_start,shift_end,available_days,active) VALUES(?,?,?,?,?,?)",r)
    if db_read("SELECT * FROM jobs").empty:
        for r in [("J1",14,100,25,1,""),("J2",12,80,20,1,""),("J3",16,120,30,1,""),("J4",15,60,20,1,""),("J5",18,100,25,1,"")]:
            db_exec("INSERT INTO jobs(job,due_hour,quantity,batch_size,priority,notes) VALUES(?,?,?,?,?,?)",r)
    if db_read("SELECT * FROM operations").empty:
        rows = [
            ("J1",1,"Lathe",4),("J1",2,"Mill",3),("J1",3,"Drill",2),
            ("J2",1,"Mill",5),("J2",2,"Grinder",3),
            ("J3",1,"Lathe",6),("J3",2,"Drill",2),("J3",3,"Grinder",2),
            ("J4",1,"Drill",3),("J4",2,"Lathe",4),("J4",3,"Mill",2),
            ("J5",1,"Lathe",5),("J5",2,"Mill",4),("J5",3,"Grinder",3)]
        for r in rows:
            db_exec("INSERT INTO operations(job,op,machine_type,hours) VALUES(?,?,?,?)",r)

init_db()
seed_if_empty()

def load_data():
    m = db_read("SELECT machine AS Machine, machine_type AS Type, shift_start AS Shift_Start, shift_end AS Shift_End, available_days AS Available_Days, active AS Active FROM machines")
    j = db_read("SELECT job AS Job, due_hour AS Due, quantity AS Quantity, batch_size AS Batch_Size, priority AS Priority, notes AS Notes FROM jobs")
    o = db_read("SELECT job AS Job, op AS Op, machine_type AS Type, hours AS Hours FROM operations ORDER BY job, op")
    d = db_read("SELECT machine AS Machine, start_hour AS Start, duration AS Hours, reason AS Reason, kind AS Type FROM downtimes")
    return m,j,o,d

machines_df, jobs_df, ops_df, downs_df = load_data()

def replace_table(table: str, df: pd.DataFrame, columns: list[str], placeholders: str, insert_sql: str):
    # Validate required columns and replace atomically for a table.
    missing = set(columns) - set(df.columns)
    if missing: raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if USE_POSTGRES:
        with db_connect().begin() as con:
            con.execute(text(f"DELETE FROM {table}"))
            for row in df[columns].itertuples(index=False, name=None):
                con.execute(text(insert_sql), dict(zip(columns, row)))
    else:
        with db_connect() as con:
            con.execute(f"DELETE FROM {table}")
            con.executemany(insert_sql, [tuple(row) for row in df[columns].itertuples(index=False, name=None)])
            con.commit()

# ---------- Core data normalization ----------
def valid_inputs(machines: pd.DataFrame, jobs: pd.DataFrame, ops: pd.DataFrame):
    if machines.empty or jobs.empty or ops.empty:
        raise ValueError("Add at least one machine, one job and one operation before scheduling.")
    if machines.Machine.duplicated().any() or jobs.Job.duplicated().any():
        raise ValueError("Machine names and job IDs must be unique.")
    if (jobs[["Due","Quantity","Batch_Size"]] < 0).any().any() or (jobs.Quantity < 1).any() or (jobs.Batch_Size < 1).any():
        raise ValueError("Due hours cannot be negative; quantity and batch size must be at least 1.")
    if (ops.Hours <= 0).any() or (ops.Op < 1).any():
        raise ValueError("Operation numbers and processing hours must be positive.")
    types = set(machines.loc[machines.Active.astype(bool), "Type"].dropna())
    unsupported = sorted(set(ops.Type.dropna()) - types)
    if unsupported:
        raise ValueError("No active machine for operation type(s): " + ", ".join(unsupported))
    if not set(ops.Job).issubset(set(jobs.Job)):
        raise ValueError("Every operation must refer to an existing job.")

def build_tasks(machines, jobs, ops):
    # One task chain per production batch. Each task retains the operation sequence.
    tasks = {}
    jobmeta = {}
    for jr in jobs.itertuples():
        j = str(jr.Job)
        qty, batch = int(jr.Quantity), int(jr.Batch_Size)
        count = math.ceil(qty / batch)
        jobmeta[j] = {"due": int(jr.Due), "quantity": qty, "batch_size": batch,
                      "batches": count, "priority": int(getattr(jr, "Priority", 1))}
        route = ops[ops.Job == j].sort_values("Op")
        for b in range(1, count + 1):
            this_qty = min(batch, qty - (b - 1) * batch)
            key = f"{j}-B{b}"
            tasks[key] = {
                "job": j, "batch": b, "quantity": this_qty,
                "ops": [(str(r.Type), max(1, int(math.ceil(float(r.Hours) * this_qty / batch * 100)))) for r in route.itertuples()]
            }
    return tasks, jobmeta

def hours_from_units(value): return int(round(value * 100))
def units_from_hours(value): return value / 100.0

def schedule_rules(method, machines, jobs, ops):
    valid_inputs(machines,jobs,ops)
    tasks, meta = build_tasks(machines,jobs,ops)
    active = dict(zip(machines[machines.Active.astype(bool)].Machine, machines[machines.Active.astype(bool)].Type))
    if method == "FCFS": order = sorted(tasks, key=lambda k: (list(meta).index(tasks[k]["job"]), tasks[k]["batch"]))
    elif method == "SPT": order = sorted(tasks, key=lambda k: (sum(p for _,p in tasks[k]["ops"]), tasks[k]["job"], tasks[k]["batch"]))
    else: order = sorted(tasks, key=lambda k: (meta[tasks[k]["job"]]["due"], tasks[k]["job"], tasks[k]["batch"]))
    free = {m: 0 for m in active}
    rows=[]
    for task in order:
        ready=0
        for oi,(typ,p100) in enumerate(tasks[task]["ops"],1):
            candidates=[m for m,t in active.items() if t==typ]
            if not candidates: raise ValueError(f"No machine supports {typ}.")
            chosen=min(candidates,key=lambda m:max(free[m],ready))
            start=max(free[chosen],ready)
            end=start+p100
            rows.append({"Task":task,"Job":tasks[task]["job"],"Batch":tasks[task]["batch"],
                         "Batch_Qty":tasks[task]["quantity"],"Op":oi,"Machine":chosen,
                         "Start":start/100,"End":end/100,"Status":"Planned"})
            free[chosen]=ready=end
    return rows,meta

def schedule_cpsat(machines,jobs,ops,downs,weight=5,time_limit=10):
    valid_inputs(machines,jobs,ops)
    tasks, meta = build_tasks(machines,jobs,ops)
    active = dict(zip(machines[machines.Active.astype(bool)].Machine, machines[machines.Active.astype(bool)].Type))
    model=cp_model.CpModel()
    total=sum(p for t in tasks.values() for _,p in t["ops"])
    horizon=max(1000,total+500)
    machine_intervals={m:[] for m in active}
    starts,ends,choices={}, {}, {}
    lateness=[]
    span=model.NewIntVar(0,horizon,"makespan")
    for task,data in tasks.items():
        previous=None
        for oi,(typ,dur) in enumerate(data["ops"],1):
            s=model.NewIntVar(0,horizon,f"s_{task}_{oi}")
            e=model.NewIntVar(0,horizon,f"e_{task}_{oi}")
            model.Add(e==s+dur)
            picks=[]
            for mk,mt in active.items():
                if mt==typ:
                    x=model.NewBoolVar(f"x_{task}_{oi}_{mk}")
                    interval=model.NewOptionalIntervalVar(s,dur,e,x,f"iv_{task}_{oi}_{mk}")
                    machine_intervals[mk].append(interval)
                    choices[(task,oi,mk)]=x
                    picks.append(x)
            if not picks: raise ValueError(f"No active machine for operation type {typ}.")
            model.AddExactlyOne(picks)
            if previous is not None: model.Add(s>=previous)
            starts[(task,oi)],ends[(task,oi)]=s,e
            previous=e
        last=ends[(task,len(data["ops"]))]
        due=hours_from_units(meta[data["job"]]["due"])
        late=model.NewIntVar(0,horizon,f"late_{task}")
        model.Add(late>=last-due)
        lateness.append(late)
        model.Add(span>=last)
    # Downtime reservations are translated to hundredths of an hour.
    for r in downs.itertuples():
        if r.Machine in machine_intervals and float(r.Hours)>0:
            stt=max(0,hours_from_units(float(r.Start)))
            dur=hours_from_units(float(r.Hours))
            machine_intervals[r.Machine].append(model.NewIntervalVar(stt,dur,stt+dur,f"down_{r.Machine}_{len(machine_intervals[r.Machine])}"))
    for intervals in machine_intervals.values(): model.AddNoOverlap(intervals)
    model.Minimize(weight*sum(lateness)+span)
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=float(time_limit)
    status=solver.Solve(model)
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        raise ValueError("CP-SAT could not find a feasible schedule. Review machine availability and downtime.")
    rows=[]
    for (task,oi,mk),lit in choices.items():
        if solver.Value(lit):
            data=tasks[task]
            rows.append({"Task":task,"Job":data["job"],"Batch":data["batch"],"Batch_Qty":data["quantity"],
                         "Op":oi,"Machine":mk,"Start":solver.Value(starts[(task,oi)])/100,
                         "End":solver.Value(ends[(task,oi)])/100,"Status":"Planned"})
    return rows,meta

def make_schedule(method,weight):
    if method=="CP-SAT": return schedule_cpsat(machines_df,jobs_df,ops_df,downs_df,weight)
    return schedule_rules(method,machines_df,jobs_df,ops_df)

def schedule_metrics(rows,meta,machines):
    if not rows: return {}
    df=pd.DataFrame(rows)
    makespan=float(df.End.max())
    busy=float((df.End-df.Start).sum())
    machine_count=max(1,int(machines[machines.Active.astype(bool)].shape[0]))
    available=makespan*machine_count
    last=df.groupby("Job").End.max()
    late=sum(max(0,float(last.get(j,0))-v["due"]) for j,v in meta.items())
    return {"Makespan (h)":round(makespan,2),"Total lateness (h)":round(late,2),
            "Idle time (h)":round(max(0,available-busy),2),
            "Utilisation (%)":round(100*busy/available,1) if available else 0,
            "Scheduled operation batches":len(df)}

def chart(rows):
    df=pd.DataFrame(rows).copy()
    if df.empty: return None
    df["Start_Label"]=df.Start.map(lambda x:f"{x:.2f} h")
    fig=px.timeline(df,x_start="Start",x_end="End",y="Machine",color="Job",text="Task",
                    hover_data=["Op","Batch","Batch_Qty","Status"])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(xaxis_title="Production time (hours from schedule origin)",yaxis_title="Machine")
    return fig

def persist_event(machine,job,event,details=""):
    db_exec("INSERT INTO event_log(event_time,machine,job,event,details) VALUES(?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"),machine,job,event,details))

def pdf_report(rows,metrics):
    if not REPORTLAB_AVAILABLE: return None
    buff=io.BytesIO()
    doc=SimpleDocTemplate(buff,pagesize=landscape(A4),rightMargin=24,leftMargin=24,topMargin=24,bottomMargin=24)
    styles=getSampleStyleSheet()
    story=[Paragraph("ShopSched 2.0 — Production Schedule Report",styles["Title"]),
           Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}",styles["Normal"]),Spacer(1,12)]
    data=[["Metric","Value"]]+[[str(k),str(v)] for k,v in metrics.items()]
    t=Table(data,repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#243447")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                           ("GRID",(0,0),(-1,-1),.4,colors.grey),("PADDING",(0,0),(-1,-1),6)]))
    story += [t,Spacer(1,16),Paragraph("Operation schedule",styles["Heading2"])]
    cols=["Task","Job","Batch","Batch_Qty","Op","Machine","Start","End","Status"]
    sdf=pd.DataFrame(rows)
    data=[cols]+[[str(getattr(r,c)) for c in cols] for r in sdf[cols].itertuples(index=False,name="Row")]
    tab=Table(data,repeatRows=1)
    tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#243447")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                             ("GRID",(0,0),(-1,-1),.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("PADDING",(0,0),(-1,-1),4)]))
    story.append(tab)
    doc.build(story)
    return buff.getvalue()

# ---------- Navigation ----------
page=st.sidebar.radio("ShopSched menu",["Dashboard","Jobs & Routing","Machines & Shifts","Maintenance & Downtime","Schedule","Operator","Reports","Data & Backup"])
st.sidebar.caption("Database: PostgreSQL" if USE_POSTGRES else "Database: local SQLite")
st.sidebar.caption("V2 · Batch-aware production scheduling")

if "schedule_rows" not in st.session_state: st.session_state.schedule_rows=[]
if "schedule_meta" not in st.session_state: st.session_state.schedule_meta={}
if "operator_status" not in st.session_state: st.session_state.operator_status={}
if "active_method" not in st.session_state: st.session_state.active_method="CP-SAT"

if page=="Dashboard":
    st.header("Production dashboard")
    jobs_df,machines_df,ops_df,downs_df=load_data()
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Active jobs",len(jobs_df))
    c2.metric("Active machines",int(machines_df.Active.astype(bool).sum()))
    c3.metric("Production quantity",int(jobs_df.Quantity.sum()) if not jobs_df.empty else 0)
    c4.metric("Downtime records",len(downs_df))
    if st.session_state.schedule_rows:
        mm=schedule_metrics(st.session_state.schedule_rows,st.session_state.schedule_meta,machines_df)
        cols=st.columns(len(mm))
        for col,(k,v) in zip(cols,mm.items()): col.metric(k,v)
        st.plotly_chart(chart(st.session_state.schedule_rows),use_container_width=True)
    else: st.info("No schedule has been generated yet. Open Schedule and run a scheduling method.")
    log=db_read("SELECT event_time,machine,job,event,details FROM event_log ORDER BY id DESC LIMIT 10")
    if not log.empty:
        st.subheader("Recent shop-floor events")
        st.dataframe(log,use_container_width=True,hide_index=True)

elif page=="Jobs & Routing":
    st.header("Jobs, production quantity and routing")
    st.caption("Edit in the table or add a job through the form. One job can be split into multiple production batches.")
    with st.form("job_form",clear_on_submit=True):
        a,b,c=st.columns(3)
        job=a.text_input("Job ID")
        due=b.number_input("Due time (hours from start)",min_value=0,value=24)
        qty=c.number_input("Production quantity",min_value=1,value=100)
        d,e,f=st.columns(3)
        batch=d.number_input("Batch size",min_value=1,value=25)
        priority=e.number_input("Priority",min_value=1,value=1)
        notes=f.text_input("Notes")
        add=st.form_submit_button("Add job")
        if add:
            try:
                db_exec("INSERT INTO jobs(job,due_hour,quantity,batch_size,priority,notes) VALUES(?,?,?,?,?,?)",
                        (job.strip(),int(due),int(qty),int(batch),int(priority),notes))
                st.success("Job added.")
                st.rerun()
            except Exception as ex: st.error(f"Could not add job: {ex}")
    jobs_df,_,ops_df,_=load_data()
    edited=st.data_editor(jobs_df,use_container_width=True,num_rows="dynamic",key="jobs_editor")
    if st.button("Save job table"):
        try:
            clean=edited.rename(columns={"Job":"job","Due":"due_hour","Quantity":"quantity","Batch_Size":"batch_size","Priority":"priority","Notes":"notes"})
            replace_table("jobs",clean,["job","due_hour","quantity","batch_size","priority","notes"],"","INSERT INTO jobs(job,due_hour,quantity,batch_size,priority,notes) VALUES(:job,:due_hour,:quantity,:batch_size,:priority,:notes)" if USE_POSTGRES else "INSERT INTO jobs(job,due_hour,quantity,batch_size,priority,notes) VALUES(?,?,?,?,?,?)")
            st.success("Jobs saved."); st.rerun()
        except Exception as ex: st.error(f"Validation/save error: {ex}")
    st.subheader("Operation routing")
    edited_ops=st.data_editor(ops_df,use_container_width=True,num_rows="dynamic",key="ops_editor")
    if st.button("Save routing"):
        try:
            clean=edited_ops.rename(columns={"Job":"job","Op":"op","Type":"machine_type","Hours":"hours"})
            replace_table("operations",clean,["job","op","machine_type","hours"],"","INSERT INTO operations(job,op,machine_type,hours) VALUES(:job,:op,:machine_type,:hours)" if USE_POSTGRES else "INSERT INTO operations(job,op,machine_type,hours) VALUES(?,?,?,?)")
            st.success("Routing saved."); st.rerun()
        except Exception as ex: st.error(f"Validation/save error: {ex}")

elif page=="Machines & Shifts":
    st.header("Machine register and shift calendars")
    st.caption("Shift start/end times are machine-local daily windows. The current scheduling engine validates and displays shift definitions; shift-window enforcement in the solver is a planned follow-on.")
    with st.form("machine_form",clear_on_submit=True):
        a,b,c=st.columns(3)
        name=a.text_input("Machine name")
        typ=b.selectbox("Machine type",sorted(set(machines_df.Type.dropna())|{"Lathe","Mill","Drill","Grinder","Other"}))
        active=c.checkbox("Active",True)
        d,e=st.columns(2)
        shift_start=d.time_input("Shift start",time(8,0))
        shift_end=e.time_input("Shift end",time(17,0))
        days=st.multiselect("Working days",["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],default=["Mon","Tue","Wed","Thu","Fri","Sat"])
        if st.form_submit_button("Add machine"):
            try:
                if shift_end<=shift_start: raise ValueError("Shift end must be later than shift start.")
                db_exec("INSERT INTO machines(machine,machine_type,shift_start,shift_end,available_days,active) VALUES(?,?,?,?,?,?)",
                        (name.strip(),typ,shift_start.strftime("%H:%M"),shift_end.strftime("%H:%M"),",".join(str(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(x)) for x in days),int(active)))
                st.success("Machine added."); st.rerun()
            except Exception as ex: st.error(f"Could not add machine: {ex}")
    machines_df,_,_,_=load_data()
    edited=st.data_editor(machines_df,use_container_width=True,num_rows="dynamic",key="machines_editor")
    if st.button("Save machine table"):
        try:
            clean=edited.rename(columns={"Machine":"machine","Type":"machine_type","Shift_Start":"shift_start","Shift_End":"shift_end","Available_Days":"available_days","Active":"active"})
            clean["active"]=clean["active"].astype(int)
            replace_table("machines",clean,["machine","machine_type","shift_start","shift_end","available_days","active"],"","INSERT INTO machines(machine,machine_type,shift_start,shift_end,available_days,active) VALUES(:machine,:machine_type,:shift_start,:shift_end,:available_days,:active)" if USE_POSTGRES else "INSERT INTO machines(machine,machine_type,shift_start,shift_end,available_days,active) VALUES(?,?,?,?,?,?)")
            st.success("Machines saved."); st.rerun()
        except Exception as ex: st.error(f"Could not save machines: {ex}")
    st.info("Scheduling currently uses machine active/inactive status. Shift-window constraints are recorded but not yet enforced by FCFS/SPT/EDD/CP-SAT in this release.")

elif page=="Maintenance & Downtime":
    st.header("Machine breakdowns and planned maintenance")
    with st.form("downtime_form",clear_on_submit=True):
        a,b,c=st.columns(3)
        machine=a.selectbox("Machine",machines_df.Machine.tolist() if not machines_df.empty else [])
        start=b.number_input("Start hour from schedule origin",min_value=0.0,value=0.0,step=0.5)
        duration=c.number_input("Duration (hours)",min_value=0.25,value=2.0,step=0.25)
        d,e=st.columns(2)
        kind=d.selectbox("Type",["Breakdown","Planned maintenance"])
        reason=e.text_input("Reason",value=kind)
        if st.form_submit_button("Record downtime"):
            db_exec("INSERT INTO downtimes(machine,start_hour,duration,reason,kind) VALUES(?,?,?,?,?)",(machine,float(start),float(duration),reason,kind))
            persist_event(machine,"",kind,f"{duration} h from hour {start}: {reason}")
            st.success("Downtime saved."); st.rerun()
    _,_,_,downs_df=load_data()
    if not downs_df.empty:
        st.dataframe(downs_df,use_container_width=True,hide_index=True)
        selected=st.selectbox("Remove downtime record",downs_df.index.tolist(),format_func=lambda i:f"{downs_df.iloc[i].Machine} · {downs_df.iloc[i].Type} · {downs_df.iloc[i].Start} h")
        if st.button("Delete selected downtime"):
            did=int(db_read("SELECT id FROM downtimes ORDER BY id").iloc[selected,0])
            db_exec("DELETE FROM downtimes WHERE id=?", (did,))
            st.rerun()
    else: st.info("No breakdown or maintenance windows recorded.")

elif page=="Schedule":
    st.header("Production scheduling")
    method=st.selectbox("Scheduling method",["FCFS","SPT","EDD","CP-SAT"],index=3)
    weight=st.slider("CP-SAT lateness penalty weight",1,20,5)
    limit=st.slider("CP-SAT solver time limit (seconds)",1,30,10)
    if st.button("Generate production schedule",type="primary"):
        machines_df,jobs_df,ops_df,downs_df=load_data()
        try:
            valid_inputs(machines_df,jobs_df,ops_df)
            if method=="CP-SAT": rows,meta=schedule_cpsat(machines_df,jobs_df,ops_df,downs_df,weight,limit)
            else: rows,meta=schedule_rules(method,machines_df,jobs_df,ops_df)
            st.session_state.schedule_rows=rows
            st.session_state.schedule_meta=meta
            st.session_state.active_method=method
            st.success(f"{method} schedule created with {len(rows)} operation-batch tasks.")
        except Exception as ex: st.error(f"Scheduling failed: {ex}")
    if st.session_state.schedule_rows:
        rows=st.session_state.schedule_rows
        st.plotly_chart(chart(rows),use_container_width=True)
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.json(schedule_metrics(rows,st.session_state.schedule_meta,machines_df))
        st.download_button("Download schedule CSV",pd.DataFrame(rows).to_csv(index=False),"shopsched_schedule.csv","text/csv")
    st.caption("Each production batch follows its job's operation order. Machine capacity and recorded downtime are enforced by CP-SAT. Rule-based methods use active machine availability.")

elif page=="Operator":
    st.header("Operator workbench")
    rows=st.session_state.schedule_rows
    if not rows: st.info("Generate a schedule first.")
    else:
        df=pd.DataFrame(rows)
        task=st.selectbox("Select scheduled operation",df.apply(lambda r:f"{r.Task} · Op {r.Op} · {r.Machine} ({r.Start:.2f}–{r.End:.2f} h)",axis=1).tolist())
        idx=df.index[df.apply(lambda r:f"{r.Task} · Op {r.Op} · {r.Machine} ({r.Start:.2f}–{r.End:.2f} h)"==task,axis=1)][0]
        row=rows[int(idx)]
        key=f"{row['Task']}|{row['Op']}"
        current=st.session_state.operator_status.get(key,row.get("Status","Planned"))
        st.subheader(f"{row['Task']} · Operation {row['Op']}")
        st.write(f"Machine: **{row['Machine']}** · Batch quantity: **{row['Batch_Qty']}** · Planned: **{row['Start']:.2f}–{row['End']:.2f} h**")
        st.write(f"Status: **{current}**")
        a,b,c=st.columns(3)
        if a.button("Start",disabled=current in ["Running","Done"]):
            st.session_state.operator_status[key]="Running"
            persist_event(row["Machine"],row["Job"],"Start",row["Task"])
            st.rerun()
        if b.button("Pause",disabled=current!="Running"):
            st.session_state.operator_status[key]="Paused"
            persist_event(row["Machine"],row["Job"],"Pause",row["Task"])
            st.rerun()
        if c.button("Complete",disabled=current not in ["Running","Paused"]):
            st.session_state.operator_status[key]="Done"
            persist_event(row["Machine"],row["Job"],"Complete",row["Task"])
            st.rerun()
        with st.expander("Report delay or machine incident"):
            kind=st.selectbox("Event",["Operation delay","Machine breakdown"])
            h=st.number_input("Additional duration (hours)",min_value=0.25,value=1.0,step=.25)
            if st.button("Record event and request re-plan"):
                if kind=="Machine breakdown":
                    db_exec("INSERT INTO downtimes(machine,start_hour,duration,reason,kind) VALUES(?,?,?,?,?)",
                            (row["Machine"],float(row["Start"]),float(h),"Operator-reported breakdown","Breakdown"))
                persist_event(row["Machine"],row["Job"],kind,f"{h} hours")
                st.warning("Event recorded. Generate a new CP-SAT schedule on the Schedule page to calculate the revised plan.")
        st.subheader("Operation status")
        status_df=pd.DataFrame([{"Task":r["Task"],"Op":r["Op"],"Machine":r["Machine"],
                                 "Status":st.session_state.operator_status.get(f"{r['Task']}|{r['Op']}","Planned")} for r in rows])
        st.dataframe(status_df,use_container_width=True,hide_index=True)

elif page=="Reports":
    st.header("Production reports and method comparison")
    if st.session_state.schedule_rows:
        rows=st.session_state.schedule_rows
        metrics=schedule_metrics(rows,st.session_state.schedule_meta,machines_df)
        st.subheader("Current schedule KPIs")
        st.dataframe(pd.DataFrame([metrics]),use_container_width=True,hide_index=True)
        df=pd.DataFrame(rows)
        util=(df.assign(Busy=df.End-df.Start).groupby("Machine").Busy.sum()).reindex(machines_df.Machine).fillna(0)
        st.subheader("Machine load (processing hours)")
        st.bar_chart(util)
        col1,col2=st.columns(2)
        col1.download_button("Export Excel workbook",data=(lambda b:(df.to_excel(b,index=False),b.getvalue())[1])(io.BytesIO()),
                             file_name="shopsched_report.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if REPORTLAB_AVAILABLE:
            pdf=pdf_report(rows,metrics)
            col2.download_button("Download PDF report",pdf,"shopsched_report.pdf","application/pdf")
        else: col2.warning("Install reportlab to enable PDF downloads.")
    else: st.info("Generate a schedule to create production reports.")
    if st.button("Compare FCFS, SPT, EDD and CP-SAT"):
        records=[]
        for meth in ["FCFS","SPT","EDD","CP-SAT"]:
            try:
                if meth=="CP-SAT": r,meta=schedule_cpsat(machines_df,jobs_df,ops_df,downs_df,5,10)
                else: r,meta=schedule_rules(meth,machines_df,jobs_df,ops_df)
                records.append({"Method":meth,**schedule_metrics(r,meta,machines_df)})
            except Exception as ex: records.append({"Method":meth,"Error":str(ex)})
        st.dataframe(pd.DataFrame(records),use_container_width=True,hide_index=True)

elif page=="Data & Backup":
    st.header("Excel import/export and database backup")
    machines_df,jobs_df,ops_df,downs_df=load_data()
    st.subheader("Export all production data to Excel")
    workbook=io.BytesIO()
    with pd.ExcelWriter(workbook,engine="openpyxl") as writer:
        machines_df.to_excel(writer,sheet_name="Machines",index=False)
        jobs_df.to_excel(writer,sheet_name="Jobs",index=False)
        ops_df.to_excel(writer,sheet_name="Operations",index=False)
        downs_df.to_excel(writer,sheet_name="Downtime",index=False)
    st.download_button("Download data workbook",workbook.getvalue(),"shopsched_data.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.subheader("Import jobs and operations from Excel")
    st.caption("Workbook must contain sheets named Jobs and Operations with columns matching the exported workbook.")
    uploaded=st.file_uploader("Choose Excel workbook",type=["xlsx"])
    if uploaded and st.button("Import Jobs and Operations"):
        try:
            xls=pd.ExcelFile(uploaded)
            imp_jobs=pd.read_excel(xls,"Jobs")
            imp_ops=pd.read_excel(xls,"Operations")
            # Validate before replacing; only expected columns are written.
            valid_inputs(machines_df,imp_jobs,imp_ops)
            cleanj=imp_jobs[["Job","Due","Quantity","Batch_Size","Priority","Notes"]].rename(columns={"Job":"job","Due":"due_hour","Quantity":"quantity","Batch_Size":"batch_size","Priority":"priority","Notes":"notes"})
            cleano=imp_ops[["Job","Op","Type","Hours"]].rename(columns={"Job":"job","Op":"op","Type":"machine_type","Hours":"hours"})
            replace_table("jobs",cleanj,["job","due_hour","quantity","batch_size","priority","notes"],"","INSERT INTO jobs(job,due_hour,quantity,batch_size,priority,notes) VALUES(:job,:due_hour,:quantity,:batch_size,:priority,:notes)" if USE_POSTGRES else "INSERT INTO jobs(job,due_hour,quantity,batch_size,priority,notes) VALUES(?,?,?,?,?,?)")
            replace_table("operations",cleano,["job","op","machine_type","hours"],"","INSERT INTO operations(job,op,machine_type,hours) VALUES(:job,:op,:machine_type,:hours)" if USE_POSTGRES else "INSERT INTO operations(job,op,machine_type,hours) VALUES(?,?,?,?)")
            st.success("Imported jobs and operations."); st.rerun()
        except Exception as ex: st.error(f"Import failed; check sheet names and required columns. Details: {ex}")
    st.subheader("Storage configuration")
    if USE_POSTGRES:
        st.success("Managed PostgreSQL is configured through DATABASE_URL.")
    else:
        st.warning("Using local SQLite file shopsched.db. Streamlit Community Cloud's local filesystem is not a reliable permanent database. Configure DATABASE_URL with a managed PostgreSQL provider (for example Supabase or Neon) before using this app for persistent production data.")
    st.code("""# Streamlit Cloud → App settings → Secrets
DATABASE_URL = "postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE""")

    if not USE_POSTGRES:
        st.caption("For a local backup, download the Excel workbook. The SQLite file is stored beside app.py when running locally.")

st.sidebar.divider()
st.sidebar.caption("ShopSched 2.0 · Validate data and review the generated plan before shop-floor use.")
