import streamlit as st, sqlite3, json, datetime as dt, requests, io, html
import pandas as pd
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB = "governai.db"
ROLES = ["submitter", "model_owner", "privacy_reviewer", "legal_reviewer",
         "risk_reviewer", "review_board", "governance_admin"]
STAGES = ["Ideation & Intake", "Impact Assessment & Requirements", "Data Acquisition & Preparation",
          "Design & Development", "Testing & Independent Validation", "Deployment & Go-Live Approval",
          "Production Monitoring & Periodic Review", "Change Management", "AI Incident Management",
          "Retirement & Decommissioning"]
CHECKLIST = {
    1: ["Use case registered in inventory", "Accountable business owner named", "Non-AI alternatives considered"],
    2: ["Impact assessment completed", "Requirements and success measures agreed", "Applicable laws confirmed"],
    3: ["Data sources documented", "Lawful basis / consent confirmed", "Data quality and bias checks done"],
    4: ["Model design documented", "Human oversight designed", "Security review of architecture"],
    5: ["Performance tests passed", "Fairness tests passed", "Independent validation report (Medium/High)"],
    6: ["Monitoring operational", "Incident response plan confirmed", "Go-live sign-offs complete"],
    7: ["Monitoring dashboard reviewed", "Periodic review completed"],
    8: ["Change impact assessed", "Re-testing completed"],
    9: ["Incidents contained", "Root causes closed"],
    10: ["Users notified", "Data retained or deleted per policy", "System archived"],
}
GATE_STAGES = {2, 5, 6}

def approvers(stage, tier):
    if tier == "prohibited":
        return []
    req = ["model_owner"]
    if tier in ("medium", "high") and stage in GATE_STAGES:
        req += ["privacy_reviewer", "risk_reviewer"]
    if tier == "high" and stage in GATE_STAGES:
        req += ["legal_reviewer", "review_board"]
    return req

REGS = [
    ("EU AI Act", "Art. 5", "Prohibited AI practices: social scoring, manipulation, exploiting vulnerabilities, untargeted facial scraping, emotion recognition at work or school, predictive policing on profiling."),
    ("EU AI Act", "Art. 6 & Annex III", "High-risk AI includes employment and recruitment, credit scoring, life and health insurance pricing, education, essential services, biometrics."),
    ("EU AI Act", "Art. 9", "High-risk systems need a continuous risk management system across the lifecycle."),
    ("EU AI Act", "Art. 10", "Training data must be relevant, representative, examined for bias, with data governance practices."),
    ("EU AI Act", "Art. 13-14", "Transparency to deployers and effective human oversight for high-risk systems."),
    ("EU AI Act", "Art. 50", "Users must be told they are interacting with an AI chatbot; AI-generated content must be labelled."),
    ("GDPR", "Art. 5", "Personal data principles: lawfulness, purpose limitation, data minimisation, accuracy, storage limitation."),
    ("GDPR", "Art. 9", "Special category data such as health or biometric data requires an explicit legal condition."),
    ("GDPR", "Art. 22", "Right not to be subject to solely automated decisions with legal or significant effects; right to human review."),
    ("GDPR", "Art. 35", "Data protection impact assessment required for high-risk processing such as large-scale profiling."),
    ("UK GDPR", "Art. 22", "UK equivalent safeguards on solely automated decision-making and profiling."),
    ("DPDP Act 2023", "Sec. 4-7", "Personal data processed only for lawful purpose with notice and free, specific, informed consent or legitimate use."),
    ("DPDP Act 2023", "Sec. 8", "Data fiduciary must ensure accuracy, security safeguards, breach notification and erasure."),
    ("DPDP Act 2023", "Sec. 9", "Children's data needs verifiable parental consent; no tracking or targeted advertising to children."),
    ("HIPAA", "Privacy Rule", "Protected health information use and disclosure limited; minimum necessary standard."),
    ("NIST AI RMF", "Govern/Map/Measure/Manage", "Voluntary framework: govern culture, map context and risks, measure trustworthiness, manage and monitor risks."),
    ("EU AI Act", "Annex III 5(c)", "AI used for risk assessment and pricing of natural persons in life and health insurance is high-risk."),
    ("EU AI Act", "Art. 26-27", "Deployers of high-risk AI must follow instructions, assign human oversight, keep logs, inform affected people and, for insurance and credit, run a fundamental rights impact assessment."),
    ("FCA Consumer Duty", "PRIN 2A", "Firms must deliver good outcomes for retail customers: fair value, products that meet needs, understandable communications and support; avoid foreseeable harm, especially to vulnerable customers."),
    ("FCA Consumer Duty", "Pricing practices (ICOBS 6B)", "Home and motor renewal prices must not exceed the equivalent new business price; no price walking of loyal customers."),
    ("Equality Act 2010", "s.19 & s.29", "Indirect discrimination: a pricing, claims or eligibility rule that disadvantages people with a protected characteristic (age, disability, race, sex, religion etc.) is unlawful unless objectively justified. Proxy variables such as postcode can cause it."),
    ("ISO/IEC 42001", "AI management system", "Management system standard for AI: policy, roles, risk assessment, impact assessment, continual improvement."),
]

PROHIBITED = ["Social scoring", "Manipulative or subliminal techniques", "Exploiting vulnerabilities",
              "Untargeted facial image scraping", "Emotion recognition at work or school",
              "Real-time remote biometric ID in public", "Predictive policing based on profiling"]
TRIGGERS = ["Material decisions about individuals", "Biometric or sensitive-trait inference",
            "Employment, credit, life/health insurance, health, education or public services",
            "Safety-critical or essential service", "Agent can act externally (send, transact, delete, approve)",
            "Vulnerable groups or children affected"]
DIMS = ["Impact", "People", "Data", "Automation", "Scale", "Reversibility", "Technical uncertainty", "Misuse", "Dependency"]

# ---------- database ----------
def db():
    c = sqlite3.connect(DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def now():
    return dt.datetime.now().isoformat(timespec="seconds")

def log(actor, action, uc=None, detail=""):
    c = db()
    c.execute("insert into audit values(?,?,?,?,?)", (now(), actor, action, uc, detail))
    c.commit()

# ---------- deterministic risk engine (AI never sets the tier) ----------
def classify(prohibited, triggers, scores, geo_pts=0):
    total = sum(scores.values()) + geo_pts
    if prohibited:
        return "prohibited", total, [f"Prohibited practice: {p}" for p in prohibited]
    if triggers:
        return "high", total, [f"Escalation trigger: {t}" for t in triggers]
    tier = "low" if total <= 14 else "medium" if total <= 20 else "high"
    why = [f"Score {total} -> {tier}"]
    if geo_pts: why.append(f"includes +{geo_pts} for operating in more than one jurisdiction")
    return tier, total, why

def auto_triggers(trig, data, geos):
    t = list(trig)
    if "Biometric" in data and TRIGGERS[1] not in t: t.append(TRIGGERS[1])
    if "Children" in data and TRIGGERS[5] not in t: t.append(TRIGGERS[5])
    return t

def applicable_laws(geos, data_types, domain=""):
    laws = ["NIST AI RMF", "ISO/IEC 42001"]
    if "EU" in geos: laws += ["GDPR", "EU AI Act"]
    if "UK" in geos:
        laws += ["UK GDPR", "Equality Act 2010"]
        if domain in ("Insurance", "Banking"): laws += ["FCA Consumer Duty"]
    if "India" in geos: laws += ["DPDP Act 2023"]
    if "US" in geos and "Health" in data_types: laws += ["HIPAA"]
    return laws

def insert_use_case(c, name, bu, intake, proh, trig, scores, actor="submitter"):
    t = auto_triggers(trig, intake.get("data", []), intake.get("geos", []))
    geo_pts = min(3, max(0, len(intake.get("geos", [])) - 1))
    tier, total, reasons = classify(proh, t, scores, geo_pts)
    laws = applicable_laws(intake.get("geos", []), intake.get("data", []), intake.get("domain", ""))
    n = (c.execute("select max(id) from use_cases").fetchone()[0] or 0) + 1
    code = f"AI-{dt.date.today().year}-{n:04d}"
    cur = c.execute("insert into use_cases(code,name,bu,intake,tier,risk,regs,status,created) values(?,?,?,?,?,?,?,?,?)",
              (code, name, bu, json.dumps(intake), tier, json.dumps(dict(scores=scores, total=total, reasons=reasons)),
               json.dumps(laws), "blocked" if tier == "prohibited" else "in_review", dt.date.today().isoformat()))
    c.commit()
    log(actor, "registered", cur.lastrowid, f"{code} tier={tier}")
    return cur.lastrowid, code, tier, total, reasons, laws

def init_db():
    c = db()
    c.executescript("""
    create table if not exists use_cases(id integer primary key, code text, name text, bu text, intake text,
      tier text, risk text, regs text, stage int default 1, status text default 'in_review',
      next_review text, created text);
    create table if not exists checklist(uc int, stage int, item text, done int, primary key(uc,stage,item));
    create table if not exists approvals(uc int, stage int, role text, outcome text, comment text, at text,
      primary key(uc,stage,role));
    create table if not exists audit(at text, actor text, action text, uc int, detail text);
    create table if not exists incidents(id integer primary key, uc int, title text, severity text,
      root_cause text, actions text, status text, at text);
    create table if not exists changes(id integer primary key, uc int, description text, material int, at text);
    create table if not exists screenings(id integer primary key, uc int, note text, at text);
    create table if not exists actions(id integer primary key, uc int, finding text, action text, owner text,
      due text, status text default 'open', at text);
    """)
    c.commit()
    if c.execute("select count(*) from use_cases").fetchone()[0] == 0:
        seed(c)

def S(*v):
    return dict(zip(DIMS, v))

DEMOS = [
    # name, bu, intake, prohibited, triggers, scores, stage, status, review_offset_days
    ("Motor claims triage model", "Claims", dict(domain="Insurance", purpose="Route motor claims to fast-track or investigation", build="build", model="XGBoost v3", geos=["UK"], data=["Personal", "Financial"], automation="human-in-loop", misuse="Unfair denial of valid claims"),
     [], ["Material decisions about individuals"], S(3,2,2,2,3,2,2,2,1), 4, "in_review", None),
    ("Home insurance pricing model", "Pricing", dict(domain="Insurance", purpose="Price home insurance premiums", build="build", model="GLM + GBM", geos=["UK"], data=["Personal", "Financial", "Behavioural"], automation="fully automated", misuse="Proxy discrimination by postcode"),
     [], ["Material decisions about individuals"], S(3,2,3,3,3,2,2,2,1), 6, "in_review", None),
    ("Customer support chatbot", "Customer Operations", dict(domain="Insurance", purpose="Answer policy FAQs", build="buy", model="Vendor LLM", geos=["UK"], data=["Personal"], automation="advisory", misuse="Hallucinated policy advice"),
     [], [], S(2,2,2,1,3,1,3,2,3), 7, "production", 120),
    ("Internal document summariser", "Operations", dict(domain="Insurance", purpose="Summarise internal reports", build="configure", model="Gemini", geos=["India"], data=["None"], automation="advisory", misuse="Leaking confidential text"),
     [], [], S(1,1,2,1,1,1,2,1,2), 7, "production", -15),
    ("HR video interview analyser", "HR", dict(domain="Insurance", purpose="Score candidate emotions in interviews", build="buy", model="Vendor CV model", geos=["EU", "UK"], data=["Personal", "Biometric"], automation="fully automated", misuse="Discriminatory hiring"),
     ["Emotion recognition at work or school"], [], S(3,3,3,3,2,3,3,3,3), 1, "blocked", None),
    ("CV screening assistant", "HR", dict(domain="Insurance", purpose="Shortlist CVs for recruiters", build="buy", model="Vendor NLP", geos=["India", "UK"], data=["Personal"], automation="human-in-loop", misuse="Bias against protected groups"),
     [], ["Employment, credit, life/health insurance, health, education or public services"], S(3,2,2,2,2,2,2,2,3), 2, "in_review", None),
    ("Fraud detection scoring", "Counter Fraud", dict(domain="Insurance", purpose="Flag suspicious claims", build="build", model="Isolation Forest", geos=["UK"], data=["Personal", "Financial"], automation="human-in-loop", misuse="Wrongful fraud accusation"),
     [], ["Material decisions about individuals"], S(3,2,3,2,3,2,2,2,1), 5, "in_review", None),
    ("Call-centre speech analytics", "Customer Operations", dict(domain="Insurance", purpose="Detect complaint themes in calls", build="buy", model="Vendor ASR", geos=["UK"], data=["Personal", "Behavioural"], automation="advisory", misuse="Covert monitoring of staff"),
     [], [], S(2,2,3,1,3,1,2,2,3), 3, "in_review", None),
    ("Marketing email copy generator", "Marketing", dict(domain="Insurance", purpose="Draft campaign emails", build="configure", model="Gemini", geos=["UK"], data=["None"], automation="human-in-loop", misuse="Misleading claims"),
     [], [], S(1,1,1,2,2,1,2,2,2), 7, "production", 400),
    ("Legacy churn prediction model", "Retention", dict(domain="Insurance", purpose="Predict policy lapse", build="build", model="Logistic regression", geos=["UK"], data=["Personal", "Behavioural"], automation="advisory", misuse="Targeting vulnerable customers"),
     [], [], S(2,2,2,1,2,1,1,2,1), 10, "in_review", None),
]

def seed(c):
    for name, bu, intake, proh, trig, scores, stage, status, rev in DEMOS:
        uid, *_ = insert_use_case(c, name, bu, intake, proh, trig, scores, actor="seed")
        nr = (dt.date.today() + dt.timedelta(days=rev)).isoformat() if rev is not None else None
        c.execute("update use_cases set stage=?, status=?, next_review=? where id=?", (stage, status, nr, uid))
        tier = c.execute("select tier from use_cases where id=?", (uid,)).fetchone()[0]
        for s in range(1, stage):
            for role in approvers(s, tier):
                c.execute("insert or replace into approvals values(?,?,?,?,?,?)", (uid, s, role, "approved", "Seeded demo approval", now()))
        if stage > 1 and status != "blocked":
            c.execute("insert or replace into checklist values(?,?,?,?)", (uid, stage, CHECKLIST[stage][0], 1))
    # a few incidents and a conditional approval for realism
    c.execute("insert into incidents(uc,title,severity,root_cause,actions,status,at) values(3,'Chatbot quoted wrong excess amount','medium','model','Prompt fixed, retrieval source updated','open',?)", (dt.date.today().isoformat(),))
    c.execute("insert into incidents(uc,title,severity,root_cause,actions,status,at) values(4,'Summary included restricted data','high','data','Source folder access restricted','open',?)", (dt.date.today().isoformat(),))
    c.execute("insert or replace into approvals values(2,6,'privacy_reviewer','approved_with_conditions','Run postcode proxy test - owner: Pricing lead, due in 30 days',?)", (now(),))
    for uc, f, a, o, d in [(2, "Equality Act 2010 s.19", "Test postcode and occupation factors for proxy discrimination", "Pricing lead", 30),
                           (1, "GDPR Art. 22 / UK GDPR Art. 22", "Confirm a human reviews every declined claim", "Claims ops manager", -5)]:
        c.execute("insert into actions(uc,finding,action,owner,due,status,at) values(?,?,?,?,?,?,?)",
                  (uc, f, a, o, (dt.date.today() + dt.timedelta(days=d)).isoformat(), "open", now()))
    c.commit()

# ---------- findings -> actions ----------
CONTROLS = {
    "Art. 5": "Confirm no prohibited practice; stop the project if one applies",
    "Art. 6 & Annex III": "Treat as high-risk: complete conformity checklist before go-live",
    "Annex III 5(c)": "Treat insurance pricing as high-risk under the EU AI Act",
    "Art. 9": "Set up a documented risk management process for the whole lifecycle",
    "Art. 10": "Document training data sources and run bias checks on data",
    "Art. 13-14": "Design and document human oversight and user instructions",
    "Art. 26-27": "Assign human oversight, keep logs and complete a fundamental rights impact assessment",
    "Art. 50": "Tell users they are interacting with AI and label generated content",
    "Art. 22": "Add human review and an appeal route for significant automated decisions",
    "Art. 35": "Complete a data protection impact assessment (DPIA)",
    "Sec. 4-7": "Issue a DPDP notice and capture valid consent or legitimate use",
    "Sec. 8": "Confirm security safeguards, breach process and erasure handling",
    "Sec. 9": "Obtain verifiable parental consent; block tracking of children",
    "PRIN 2A": "Run a Consumer Duty outcomes and fair value assessment, incl. vulnerable customers",
    "Pricing practices (ICOBS 6B)": "Test that renewal prices never exceed equivalent new business price",
    "s.19 & s.29": "Test outputs for proxy discrimination across protected characteristics",
    "Privacy Rule": "Apply HIPAA minimum necessary rule to health data",
}

def suggest_actions(hits):
    out, seen = [], set()
    for _, (law, art, _txt) in hits:
        a = CONTROLS.get(art)
        if a and a not in seen:
            seen.add(a); out.append((f"{law} {art}", a))
    return out

# ---------- RAG + Gemini screening ----------
def retrieve(query, laws, k=6):
    docs = [r for r in REGS if r[0] in laws]
    if not docs: return []
    texts = [f"{a} {b} {c}" for a, b, c in docs]
    vec = TfidfVectorizer(stop_words="english").fit(texts + [query])
    sims = cosine_similarity(vec.transform([query]), vec.transform(texts))[0]
    ranked = sorted(zip(sims, range(len(docs))), reverse=True)[:k]
    return [(round(float(s), 3), docs[i]) for s, i in ranked if s > 0]

def secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

def screening_note(uc, hits):
    if not hits or hits[0][0] < 0.05:
        return "Insufficient basis - refer to legal."
    clauses = "\n".join(f"- {d[0]} {d[1]}: {d[2]}" for _, d in hits)
    key = secret("GEMINI_API_KEY")
    model = secret("GEMINI_MODEL", "gemini-2.0-flash")
    prompt = ("You are an AI compliance screener. Using ONLY the clauses below, list laws in scope, obligations, "
              "potential violations with exact citations, severity and recommended controls. If the clauses do not "
              "support a point, say 'insufficient basis - refer to legal'.\n\nUSE CASE:\n"
              + json.dumps(uc, indent=1) + "\n\nCLAUSES:\n" + clauses)
    if key:
        try:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                              json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=60)
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"AI unavailable ({type(e).__name__}). Retrieved clauses for manual review:\n\n" + clauses
    return "No Gemini key set - retrieved clauses for manual review:\n\n" + clauses

# ---------- gates ----------
def required_items(u):
    return [i for i in CHECKLIST[u["stage"]] if not (u["tier"] == "low" and "Medium/High" in i)]

def blockers(u):
    c = db(); b = []
    if u["tier"] == "prohibited": b.append("Prohibited practice - cannot proceed")
    if u["status"] == "returned": b.append("Returned for remediation")
    if u["status"] == "retired": b.append("System is retired")
    if b: return b
    for item in required_items(u):
        r = c.execute("select done from checklist where uc=? and stage=? and item=?", (u["id"], u["stage"], item)).fetchone()
        if not r or not r["done"]: b.append(f"Checklist incomplete: {item}")
    for role in approvers(u["stage"], u["tier"]):
        r = c.execute("select outcome from approvals where uc=? and stage=? and role=?", (u["id"], u["stage"], role)).fetchone()
        if not r or r["outcome"] == "rejected": b.append(f"Approval required: {role}")
    return b

def advance(u, actor):
    b = blockers(u)
    if b: return b
    c = db()
    if u["stage"] == 10:
        c.execute("update use_cases set status='retired' where id=?", (u["id"],))
    elif u["stage"] == 6:
        months = {"high": 6, "medium": 12}.get(u["tier"], 24)
        nxt = (dt.date.today() + dt.timedelta(days=30 * months)).isoformat()
        c.execute("update use_cases set stage=7, status='production', next_review=? where id=?", (nxt, u["id"]))
    else:
        c.execute("update use_cases set stage=stage+1 where id=?", (u["id"],))
    c.commit(); log(actor, "stage_advance", u["id"], f"from stage {u['stage']}")
    return []


# ---------- use-case-specific questions ----------
QBANK = {
    "pricing": (["pricing", "premium", "underwriting", "rate"], [
        "Which rating factors does the model use, and could any act as a proxy for a protected trait (e.g. postcode for ethnicity)?",
        "How will you show fair value to customers under FCA Consumer Duty?",
        "Can a customer get an explanation of how their price was set?"]),
    "claims": (["claim", "triage", "settlement"], [
        "Can the model decline or delay a claim without a human reviewing it?",
        "How are vulnerable customers identified and routed?",
        "What is the appeal route if a claim is wrongly flagged?"]),
    "fraud": (["fraud", "suspicious", "anomaly"], [
        "What happens to a customer once flagged - is any action automatic?",
        "How do you measure false positives across customer groups?",
        "Who reviews flags before any customer contact?"]),
    "hr": (["cv", "recruit", "hiring", "interview", "candidate", "employee", "hr"], [
        "Does the system rank, filter or reject candidates without human review?",
        "Has it been tested for bias by gender, age, ethnicity and disability?",
        "Are candidates told AI is used, and can they request human review?"]),
    "genai": (["chatbot", "llm", "gpt", "gemini", "generate", "assistant", "summar", "copy"], [
        "Are users clearly told they are talking to AI or reading AI-generated content?",
        "How do you prevent and detect hallucinated answers?",
        "Can prompts or outputs contain personal or confidential data, and where is it sent?"]),
    "health": (["health", "medical", "patient", "diagnos"], [
        "Is health data processed, and under which legal condition?",
        "Could outputs influence clinical or coverage decisions?",
        "How is accuracy validated across patient groups?"]),
    "biometric": (["face", "voice", "biometric", "emotion", "speech"], [
        "Does the system identify people or infer emotions or traits?",
        "Is it used on employees or in public spaces?",
        "Have you checked EU AI Act Art. 5 prohibitions?"]),
}
GENERIC_Q = ["Who is affected by the output and how significant is the effect?",
             "What happens if the system is wrong, and can it be reversed?",
             "Who can override or switch off the system?"]

def bank_questions(text):
    t = text.lower(); qs = []
    for _, (keys, q) in QBANK.items():
        if any(k in t for k in keys): qs += q
    return (qs or GENERIC_Q)[:6]

def generate_questions(desc, domain):
    key = secret("GEMINI_API_KEY")
    if key:
        model = secret("GEMINI_MODEL", "gemini-2.0-flash")
        prompt = ("You are an AI governance reviewer. For the AI use case below, write 5 specific follow-up questions "
                  "a risk reviewer must ask before approval, covering fairness, data, automation, transparency and "
                  "applicable law. Return ONLY a JSON array of strings.\n\nDomain: " + domain + "\nUse case: " + desc)
        try:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                              json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=45)
            r.raise_for_status()
            txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            txt = txt[txt.find("["): txt.rfind("]") + 1]
            qs = [str(q) for q in json.loads(txt) if str(q).strip()]
            if qs: return qs[:6], "Gemini"
        except Exception:
            pass
    return bank_questions(desc + " " + domain), "built-in question bank"

  # ---------- look and feel ----------
TIER_COL = {"low": "#16a34a", "medium": "#d97706", "high": "#dc2626", "prohibited": "#111827"}
CSS = """
<style>
.block-container {padding-top: 2rem;}
.hero {background: linear-gradient(135deg,#0f2a4a,#0d9488); color:white; padding:3rem 2.5rem; border-radius:18px;}
.hero h1 {color:white; font-size:2.8rem; margin-bottom:.3rem;}
.hero p {font-size:1.15rem; opacity:.92;}
.feat {background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px; padding:1.2rem; height:100%;}
.feat h4 {margin:.2rem 0 .4rem 0;}
.card {border:1px solid #e2e8f0; border-radius:14px; padding:1rem 1.1rem; margin-bottom:.4rem; background:white;}
.card h5 {margin:0 0 .3rem 0;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600; color:white;}
.low {background:#16a34a;} .medium {background:#d97706;} .high {background:#dc2626;} .prohibited {background:#111827;}
.track {display:flex; gap:4px; margin:.6rem 0 1rem 0;}
.step {flex:1; text-align:center; font-size:.68rem; padding:6px 2px; border-radius:6px; background:#e5e7eb; color:#374151;}
.step.done {background:#99f6e4;} .step.now {background:#0f766e; color:white; font-weight:700;}
.alert {background:#fef2f2; border-left:5px solid #dc2626; padding:.6rem 1rem; border-radius:8px; margin-bottom:.4rem;}
</style>
"""

def E(x):
    return html.escape(str(x if x is not None else ""))

def badge(tier):
    return f"<span class='badge {tier}'>{tier.upper()}</span>"

def tracker(stage):
    cells = []
    for i, s in enumerate(STAGES, 1):
        cls = "done" if i < stage else "now" if i == stage else ""
        cells.append(f"<div class='step {cls}' title='{s}'>{i}. {s.split(' ')[0]}</div>")
    return "<div class='track'>" + "".join(cells) + "</div>"

def report_html(u, c):
    risk = json.loads(u["risk"]); intake = json.loads(u["intake"])
    appr = c.execute("select stage,role,outcome,comment,at from approvals where uc=? order by stage", (u["id"],)).fetchall()
    inc = c.execute("select title,severity,status,at from incidents where uc=?", (u["id"],)).fetchall()
    scr = c.execute("select note from screenings where uc=? order by id desc limit 1", (u["id"],)).fetchone()
    acts = c.execute("select finding,action,owner,due,status from actions where uc=? order by due", (u["id"],)).fetchall()
    rows = "".join(f"<tr><td>{a['stage']}</td><td>{E(a['role'])}</td><td>{E(a['outcome'])}</td><td>{E(a['comment'])}</td></tr>" for a in appr)
    irows = "".join(f"<tr><td>{E(i['at'])}</td><td>{E(i['title'])}</td><td>{E(i['severity'])}</td><td>{E(i['status'])}</td></tr>" for i in inc)
    arows = "".join(f"<tr><td>{E(a['finding'])}</td><td>{E(a['action'])}</td><td>{E(a['owner'])}</td><td>{E(a['due'])}</td><td>{E(a['status'])}</td></tr>" for a in acts)
    fu = "".join(f"<li><b>{E(q)}</b><br>{E(a)}</li>" for q, a in intake.get("follow_up", {}).items())
    return f"""<html><head><meta charset='utf-8'><title>{E(u['code'])}</title>
<style>body{{font-family:Arial;margin:40px;color:#1f2937}} td,th{{border:1px solid #ccc;padding:6px;text-align:left}} table{{border-collapse:collapse;width:100%}}
.cover{{background:#0f2a4a;color:white;padding:28px;border-radius:12px}} .tier{{display:inline-block;padding:4px 14px;border-radius:999px;color:white;font-weight:bold;background:{TIER_COL.get(u['tier'],'#555')}}}</style></head><body>
<div class='cover'><small>GovernAI AI System Assessment</small><h1>{E(u['code'])} - {E(u['name'])}</h1><span class='tier'>{E(u['tier']).upper()}</span></div>
<p><b>Business unit:</b> {E(u['bu'])} | <b>Model owner:</b> {E(intake.get('model_owner') or 'not named')} | <b>Tier:</b> {u['tier'].upper()} (score {risk['total']}) | <b>Stage:</b> {u['stage']} {STAGES[u['stage']-1]} | <b>Status:</b> {u['status']} | <b>Next review:</b> {u['next_review'] or '-'}</p>
<h2>Purpose</h2><p>{E(intake.get('purpose',''))}</p>
<h2>Why this tier</h2><ul>{''.join('<li>'+E(r)+'</li>' for r in risk['reasons'])}</ul>
<h2>Laws in scope</h2><p>{', '.join(json.loads(u['regs']))}</p>
<h2>Use-case questions</h2><ul>{fu or '<li>None recorded</li>'}</ul>
<h2>Approvals</h2><table><tr><th>Stage</th><th>Role</th><th>Outcome</th><th>Comment</th></tr>{rows}</table>
<h2>Actions</h2><table><tr><th>Finding</th><th>Action</th><th>Owner</th><th>Due</th><th>Status</th></tr>{arows}</table>
<h2>Incidents</h2><table><tr><th>Date</th><th>Title</th><th>Severity</th><th>Status</th></tr>{irows}</table>
<h2>Latest AI screening</h2><pre style='white-space:pre-wrap'>{E(scr['note']) if scr else 'Not run'}</pre>
<h2>Sign-off</h2><table><tr><th>Role</th><th>Name</th><th>Signature</th><th>Date</th></tr><tr><td>Business owner</td><td></td><td></td><td></td></tr><tr><td>Risk reviewer</td><td></td><td></td><td></td></tr></table>
<p><i>AI-assisted risk screening, not legal advice. Generated {now()}.</i></p></body></html>"""

# ---------- pages ----------
def landing():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("<div class='hero'><h1>GovernAI</h1><p>Register every AI system, classify its risk with transparent rules, "
                "screen it against GDPR, the EU AI Act, UK GDPR and India's DPDP Act, and move it through approval gates "
                "with a full audit trail.</p></div>", unsafe_allow_html=True)
    st.write("")
    cols = st.columns(4)
    feats = [("Register", "Describe a use case and get questions tailored to it."),
             ("Classify", "Deterministic tiering: Prohibited, High, Medium or Low, with reasons."),
             ("Screen", "Retrieves relevant legal clauses; AI explains, never decides."),
             ("Govern", "10 lifecycle stages, approvals, incidents, reviews and audit log.")]
    for col, (h, t) in zip(cols, feats):
        col.markdown(f"<div class='feat'><h4>{h}</h4>{t}</div>", unsafe_allow_html=True)
    st.write("")
    if st.button("Enter GovernAI", type="primary"):
        st.session_state["entered"] = True; st.rerun()
    st.caption("AI-assisted risk screening, not legal advice.")

def page_dashboard(c, ucs):
    st.title("Dashboard")
    df = pd.DataFrame(ucs)
    if df.empty:
        st.info("No systems yet - register one."); return
    today = dt.date.today().isoformat()
    over = [u for u in ucs if u["next_review"] and u["next_review"] < today and u["status"] == "production"]
    hi_inc = c.execute("select i.title, u.code from incidents i join use_cases u on u.id=i.uc where i.status='open' and i.severity in ('high','critical')").fetchall()
    for u in over:
        st.markdown(f"<div class='alert'>Review overdue: <b>{E(u['code'])} {E(u['name'])}</b> (due {u['next_review']})</div>", unsafe_allow_html=True)
    for i in hi_inc:
        st.markdown(f"<div class='alert'>Open high-severity incident on <b>{E(i['code'])}</b>: {E(i['title'])}</div>", unsafe_allow_html=True)
    open_inc = c.execute("select count(*) from incidents where status='open'").fetchone()[0]
    k = st.columns(5)
    k[0].metric("AI systems", len(df))
    k[1].metric("High / prohibited", int(df["tier"].isin(["high", "prohibited"]).sum()))
    k[2].metric("In production", int((df["status"] == "production").sum()))
    k[3].metric("Overdue reviews", len(over))
    k[4].metric("Open incidents", open_inc)
    late = c.execute("select a.action, a.owner, a.due, u.code from actions a join use_cases u on u.id=a.uc where a.status='open' and a.due < ?", (today,)).fetchall()
    for x in late:
        st.markdown(f"<div class='alert'>Action overdue on <b>{E(x['code'])}</b>: {E(x['action'])} (owner {E(x['owner'])}, due {E(x['due'])})</div>", unsafe_allow_html=True)
    tiers = list(TIER_COL)
    a, b = st.columns(2)
    td = df["tier"].value_counts().rename_axis("tier").reset_index(name="count")
    a.subheader("Risk tiers")
    a.altair_chart(alt.Chart(td).mark_arc(innerRadius=60).encode(
        theta="count:Q", color=alt.Color("tier:N", scale=alt.Scale(domain=tiers, range=[TIER_COL[t] for t in tiers])),
        tooltip=["tier", "count"]), use_container_width=True)
    sd = df["stage"].value_counts().rename_axis("stage").reset_index(name="count")
    sd["stage_name"] = sd["stage"].map(lambda i: f"{i}. {STAGES[int(i)-1]}")
    b.subheader("Systems by lifecycle stage")
    b.altair_chart(alt.Chart(sd).mark_bar(color="#0d9488").encode(
        x="count:Q", y=alt.Y("stage_name:N", sort=[f"{i}. {s}" for i, s in enumerate(STAGES, 1)], title=None),
        tooltip=["stage_name", "count"]), use_container_width=True)
    st.subheader("Business unit x risk tier")
    hm = df.groupby(["bu", "tier"]).size().reset_index(name="count")
    st.altair_chart(alt.Chart(hm).mark_rect().encode(
        x=alt.X("tier:N", sort=tiers), y=alt.Y("bu:N", title="Business unit"),
        color=alt.Color("count:Q", scale=alt.Scale(scheme="teals")), tooltip=["bu", "tier", "count"]), use_container_width=True)

def page_register(c, role):
    st.title("Register an AI use case")
    st.caption("Step 1: describe it. Step 2: answer questions generated for this use case. Step 3: classify.")
    name = st.text_input("System name")
    bu = st.text_input("Business unit")
    domain = st.selectbox("Domain", ["Insurance", "Banking", "HR", "Healthcare", "Customer service", "Marketing", "Operations", "Other"])
    desc = st.text_area("Describe what the AI does, who it affects and what decisions it supports")
    if st.button("Generate use-case questions"):
        if not str(desc).strip():
            st.error("Describe the use case first.")
        else:
            qs, src = generate_questions(desc, domain)
            st.session_state["qs"] = qs; st.session_state["qsrc"] = src
    qs = st.session_state.get("qs", [])
    if not qs:
        return
    st.info(f"Questions from {st.session_state.get('qsrc')}")
    with st.form("intake"):
        answers = {q: st.text_area(q, key="q_" + str(i)) for i, q in enumerate(qs)}
        build = st.selectbox("Classification", ["build", "buy", "configure", "integrate"])
        model = st.text_input("Model / provider / version")
        owner = st.text_input("Model owner (name of accountable person)")
        geos = st.multiselect("Geographies", ["EU", "UK", "India", "US", "Singapore"])
        data = st.multiselect("Personal data types", ["None", "Personal", "Health", "Biometric", "Children", "Financial", "Behavioural"])
        auto = st.selectbox("Degree of automation", ["advisory", "human-in-loop", "fully automated"])
        misuse = st.text_area("Foreseeable misuse")
        st.markdown("**Prohibited practices (EU AI Act Art. 5)**")
        proh = [p for p in PROHIBITED if st.checkbox(p, key="p_" + p)]
        st.markdown("**Escalation triggers**")
        trig = [t for t in TRIGGERS if st.checkbox(t, key="t_" + t)]
        st.markdown("**Risk dimensions (1 = low, 3 = high)**")
        scores = {d: st.slider(d, 1, 3, 1) for d in DIMS}
        ok = st.form_submit_button("Submit and classify")
    if ok:
        if not str(name).strip():
            st.error("System name is required.")
        else:
            intake = dict(purpose=desc, domain=domain, build=build, model=model, geos=geos, data=data,
                          automation=auto, misuse=misuse, follow_up=answers, model_owner=str(owner).strip())
            _, code, tier, total, reasons, laws = insert_use_case(c, name.strip(), bu, intake, proh, trig, scores, role)
            st.markdown(f"### {E(code)} registered {badge(tier)}", unsafe_allow_html=True)
            st.write(f"Score {total}")
            for r in reasons: st.write("- " + r)
            st.write("Laws in scope: " + ", ".join(laws))
            st.session_state["qs"] = []

def register_excel(c, items):
    rows = []
    for u in items:
        it = json.loads(u["intake"]); rk = json.loads(u["risk"])
        rows.append({"Code": u["code"], "Name": u["name"], "Business unit": u["bu"], "Domain": it.get("domain", ""),
                     "Tier": u["tier"], "Score": rk.get("total"), "Stage": f"{u['stage']}. {STAGES[u['stage']-1]}",
                     "Status": u["status"], "Next review": u["next_review"], "Laws": ", ".join(json.loads(u["regs"])),
                     "Automation": it.get("automation", ""), "Geographies": ", ".join(it.get("geos", [])), "Created": u["created"]})
    ids = [u["id"] for u in items]
    q = ",".join("?" * len(ids))
    acts = pd.read_sql(f"select u.code, a.finding, a.action, a.owner, a.due, a.status from actions a join use_cases u on u.id=a.uc where a.uc in ({q})", c, params=ids)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame(rows).to_excel(w, sheet_name="Register", index=False)
        acts.to_excel(w, sheet_name="Actions", index=False)
    return buf.getvalue()

def page_register_list(c, ucs, role):
    sel = st.session_state.get("open_id")
    if sel:
        row = c.execute("select * from use_cases where id=?", (sel,)).fetchone()
        if row:
            return page_detail(c, dict(row), role)
    st.title("AI Systems Register")
    st.caption("Every AI system in the organisation, with its risk tier, lifecycle stage and status. Open one to work on it.")
    f1, f2, f3 = st.columns(3)
    tiers = f1.multiselect("Risk tier", ["prohibited", "high", "medium", "low"])
    stats = f2.multiselect("Status", ["in_review", "production", "blocked", "returned", "retired"])
    q = f3.text_input("Search")
    items = [u for u in ucs if (not tiers or u["tier"] in tiers) and (not stats or u["status"] in stats)
             and (not str(q).strip() or str(q).lower() in (u["name"] + u["bu"] + u["code"]).lower())]
    if not items:
        st.info("No systems match."); return
    st.download_button("Export register to Excel", register_excel(c, items), file_name="ai_register.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    cols = st.columns(3)
    for i, u in enumerate(items):
        with cols[i % 3]:
            st.markdown(f"<div class='card'><h5>{E(u['name'])}</h5>{badge(u['tier'])} &nbsp; <small>{E(u['code'])} | {E(u['bu'])}</small>"
                        f"<br><small>Stage {u['stage']}: {STAGES[u['stage']-1]} | {u['status']}</small></div>", unsafe_allow_html=True)
            if st.button("Open", key=f"open_{u['id']}"):
                st.session_state["open_id"] = u["id"]; st.rerun()

def page_detail(c, u, role):
    if st.button("Back to register"):
        st.session_state["open_id"] = None; st.rerun()
    risk = json.loads(u["risk"]); intake = json.loads(u["intake"])
    st.markdown(f"## {E(u['name'])} {badge(u['tier'])}", unsafe_allow_html=True)
    st.caption(f"{u['code']} | {u['bu']} | model owner {intake.get('model_owner') or 'not named'} | status {u['status']} | score {risk['total']} | next review {u['next_review'] or '-'}")
    st.markdown(tracker(u["stage"]), unsafe_allow_html=True)
    st.download_button("Download assessment report", report_html(u, c), file_name=f"{u['code']}.html", mime="text/html")
    with st.expander("Why this tier and use-case answers"):
        for r in risk["reasons"]: st.write("- " + r)
        sc = pd.DataFrame({"dimension": list(risk["scores"]), "score": list(risk["scores"].values())})
        st.altair_chart(alt.Chart(sc).mark_bar().encode(
            x=alt.X("score:Q", scale=alt.Scale(domain=[0, 3])), y=alt.Y("dimension:N", sort=DIMS, title=None),
            color=alt.Color("score:Q", scale=alt.Scale(domain=[1, 3], range=["#16a34a", "#dc2626"]), legend=None)),
            use_container_width=True)
        for qq, aa in intake.get("follow_up", {}).items():
            st.write(f"**{qq}**  \n{aa}")
    t1, t2, t5, t3, t4, t6 = st.tabs(["Stage gate", "AI screening", "Actions", "Change / incident", "History", "Edit / delete"])
    with t1:
        st.subheader(f"Stage {u['stage']}: {STAGES[u['stage']-1]}")
        for item in required_items(u):
            r = c.execute("select done from checklist where uc=? and stage=? and item=?", (u["id"], u["stage"], item)).fetchone()
            cur = bool(r and r["done"])
            val = st.checkbox(item, value=cur, key=f"ck_{u['id']}_{u['stage']}_{item}")
            if val != cur:
                c.execute("insert or replace into checklist values(?,?,?,?)", (u["id"], u["stage"], item, int(val)))
                c.commit(); log(role, "checklist", u["id"], f"{item}={val}"); st.rerun()
        st.write("Required approvers: " + (", ".join(approvers(u["stage"], u["tier"])) or "none"))
        b = blockers(u)
        if b:
            st.warning("Blocking items:")
            for x in b: st.write("- " + x)
            pend = [x.split(": ", 1)[1] for x in b if x.startswith("Approval required")]
            if pend:
                st.info("To give an approval, switch 'Acting as' in the sidebar to " + ", ".join(pend)
                        + ", open 'My review queue' and submit the decision. Then come back here and advance the stage.")
        else:
            st.success("All gate criteria met.")
        if st.button("Advance stage", disabled=bool(b)):
            advance(u, role); st.rerun()
    with t2:
        if st.button("Run AI screening"):
            laws = json.loads(u["regs"])
            query = " ".join(str(v) for v in intake.values()) + " " + " ".join(risk["reasons"])
            hits = retrieve(query, laws)
            note = screening_note(dict(name=u["name"], tier=u["tier"], **intake), hits)
            c.execute("insert into screenings(uc,note,at) values(?,?,?)", (u["id"], note, now()))
            existing = {r[0] for r in c.execute("select action from actions where uc=?", (u["id"],))}
            for f, a in suggest_actions(hits):
                if a not in existing:
                    c.execute("insert into actions(uc,finding,action,owner,due,status,at) values(?,?,?,?,?,?,?)",
                              (u["id"], f, a, "Unassigned", (dt.date.today() + dt.timedelta(days=30)).isoformat(), "open", now()))
            c.commit(); log(role, "ai_output", u["id"], "screening note")
        last = c.execute("select note from screenings where uc=? order by id desc limit 1", (u["id"],)).fetchone()
        if last:
            st.caption("AI-assisted risk screening, not legal advice.")
            st.markdown(last["note"])
    with t5:
        acts = c.execute("select * from actions where uc=? order by status desc, due", (u["id"],)).fetchall()
        if not acts: st.info("No actions yet. Run AI screening or add one below.")
        today = dt.date.today().isoformat()
        for a in acts:
            late = a["status"] == "open" and a["due"] < today
            with st.container(border=True):
                st.markdown(f"**{E(a['action'])}**  \n<small>{E(a['finding'])} | owner {E(a['owner'])} | due {E(a['due'])}"
                            f"{' | <b style=color:#dc2626>OVERDUE</b>' if late else ''} | {E(a['status'])}</small>", unsafe_allow_html=True)
                if a["status"] == "open":
                    x1, x2, x3 = st.columns([2, 2, 1])
                    own = x1.text_input("Owner", a["owner"], key=f"ow{a['id']}")
                    due = x2.date_input("Due", dt.date.fromisoformat(a["due"]), key=f"du{a['id']}")
                    if x3.button("Save", key=f"sv{a['id']}"):
                        c.execute("update actions set owner=?, due=? where id=?", (own.strip() or "Unassigned", due.isoformat(), a["id"]))
                        c.commit(); log(role, "action_updated", u["id"], a["action"]); st.rerun()
                    if x3.button("Done", key=f"dn{a['id']}"):
                        c.execute("update actions set status='done' where id=?", (a["id"],))
                        c.commit(); log(role, "action_done", u["id"], a["action"]); st.rerun()
        with st.form("act"):
            fa = st.text_input("Finding (e.g. GDPR Art. 22)")
            aa = st.text_input("Action")
            oa = st.text_input("Owner")
            da = st.date_input("Due date", dt.date.today() + dt.timedelta(days=30))
            if st.form_submit_button("Add action") and str(aa).strip():
                c.execute("insert into actions(uc,finding,action,owner,due,status,at) values(?,?,?,?,?,?,?)",
                          (u["id"], fa, aa.strip(), str(oa).strip() or "Unassigned", da.isoformat(), "open", now()))
                c.commit(); log(role, "action_added", u["id"], aa); st.rerun()
    with t6:
        with st.form("edit"):
            nn = st.text_input("System name", u["name"])
            nb = st.text_input("Business unit", u["bu"])
            npur = st.text_area("Description / purpose", intake.get("purpose", ""))
            nm = st.text_input("Model / provider / version", intake.get("model", ""))
            no = st.text_input("Model owner (name)", intake.get("model_owner", ""))
            st.caption("Risk answers (data, automation, geography, triggers) change through a material change request so the tier is re-assessed.")
            if st.form_submit_button("Save details"):
                if not str(nn).strip():
                    st.error("Name cannot be empty.")
                else:
                    intake["purpose"] = npur; intake["model"] = nm; intake["model_owner"] = str(no).strip()
                    c.execute("update use_cases set name=?, bu=?, intake=? where id=?", (nn.strip(), nb, json.dumps(intake), u["id"]))
                    c.commit(); log(role, "edited", u["id"], "details updated"); st.rerun()
        if role == "governance_admin":
            st.markdown("**Delete system**")
            sure = st.checkbox(f"I understand this permanently deletes {u['code']}", key=f"del_ok_{u['id']}")
            if st.button("Delete", disabled=not sure, key=f"del_{u['id']}"):
                for t in ["checklist", "approvals", "incidents", "changes", "screenings", "actions"]:
                    c.execute(f"delete from {t} where uc=?", (u["id"],))
                c.execute("delete from use_cases where id=?", (u["id"],))
                c.commit(); log(role, "deleted", u["id"], f"{u['code']} {u['name']}")
                st.session_state["open_id"] = None; st.rerun()
        else:
            st.caption("Only governance_admin can delete a system.")
    with t3:
        with st.form("chg"):
            d = st.text_area("Change description")
            mat = st.checkbox("Material change (model, data, prompt, users, geography or automation)")
            sub = st.form_submit_button("Submit change")
        if sub and u["status"] in ("retired", "blocked"):
            st.error(f"A {u['status']} system cannot take change requests.")
        elif sub and str(d).strip():
            c.execute("insert into changes(uc,description,material,at) values(?,?,?,?)", (u["id"], d, int(mat), dt.date.today().isoformat()))
            if mat:
                c.execute("update use_cases set stage=2, status='in_review' where id=?", (u["id"],))
                c.execute("delete from approvals where uc=?", (u["id"],))
                c.execute("delete from checklist where uc=? and stage>=2", (u["id"],))
            c.commit(); log(role, "change_request", u["id"], f"material={mat}"); st.rerun()
        with st.form("inc"):
            ti = st.text_input("Incident title")
            sev = st.selectbox("Severity", ["low", "medium", "high", "critical"])
            rc = st.selectbox("Root cause", ["model", "data", "design", "human", "process", "vendor"])
            act = st.text_area("Containment and corrective actions")
            sub2 = st.form_submit_button("Report incident")
        if sub2 and str(ti).strip():
            c.execute("insert into incidents(uc,title,severity,root_cause,actions,status,at) values(?,?,?,?,?,?,?)",
                      (u["id"], ti, sev, rc, act, "open", dt.date.today().isoformat()))
            c.commit(); log(role, "incident", u["id"], f"{sev}: {ti}"); st.rerun()
        inc = pd.read_sql("select id,title,severity,root_cause,status,at from incidents where uc=?", c, params=(u["id"],))
        st.dataframe(inc, use_container_width=True)
        if role in ("risk_reviewer", "governance_admin"):
            for _, row in inc[inc["status"] == "open"].iterrows():
                if st.button(f"Approve closure of incident {row['id']}", key=f"close_{row['id']}"):
                    c.execute("update incidents set status='closed' where id=?", (int(row["id"]),))
                    c.commit(); log(role, "incident_closed", u["id"], str(row["id"])); st.rerun()
    with t4:
        st.dataframe(pd.read_sql("select stage,role,outcome,comment,at from approvals where uc=? order by stage", c, params=(u["id"],)), use_container_width=True)
        st.dataframe(pd.read_sql("select * from audit where uc=? order by at desc", c, params=(u["id"],)), use_container_width=True)

def page_queue(c, ucs, role):
    st.title(f"Review queue - {role}")
    mine = [u for u in ucs if u["status"] in ("in_review", "production") and role in approvers(u["stage"], u["tier"])
            and not c.execute("select 1 from approvals where uc=? and stage=? and role=? and outcome!='rejected'", (u["id"], u["stage"], role)).fetchone()]
    if not mine: st.info("Nothing waiting for you.")
    for u in mine:
        with st.expander(f"{u['code']} - {u['name']} | stage {u['stage']} | {u['tier']}"):
            out = st.radio("Decision", ["approved", "approved_with_conditions", "rejected"], key=f"o{u['id']}")
            com = st.text_area("Comment / conditions (owner + due date)", key=f"c{u['id']}")
            if st.button("Submit decision", key=f"b{u['id']}"):
                if out == "approved_with_conditions" and not str(com).strip():
                    st.error("Conditions are required.")
                else:
                    c.execute("insert or replace into approvals values(?,?,?,?,?,?)", (u["id"], u["stage"], role, out, com, now()))
                    if out == "rejected": c.execute("update use_cases set status='returned' where id=?", (u["id"],))
                    c.commit(); log(role, f"approval_{out}", u["id"], com); st.rerun()
    if role == "governance_admin":
        st.subheader("Returned systems")
        for u in [x for x in ucs if x["status"] == "returned"]:
            if st.button(f"Reopen {u['code']} after remediation", key=f"r{u['id']}"):
                c.execute("update use_cases set status='in_review' where id=?", (u["id"],))
                c.execute("delete from approvals where uc=? and stage=? and outcome='rejected'", (u["id"], u["stage"]))
                c.commit(); log(role, "reopened", u["id"]); st.rerun()

def main():
    st.set_page_config(page_title="GovernAI", layout="wide")
    init_db()
    if not st.session_state.get("entered"):
        return landing()
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.title("GovernAI")
    role = st.sidebar.selectbox("Acting as (demo role)", ROLES)
    page = st.sidebar.radio("Go to", ["Dashboard", "Register a use case", "AI Systems Register", "My review queue", "Audit log"])
    if st.sidebar.button("Back to home"):
        st.session_state["entered"] = False; st.rerun()
    st.sidebar.caption("AI-assisted risk screening, not legal advice.")
    c = db()
    ucs = [dict(r) for r in c.execute("select * from use_cases").fetchall()]
    if page != "AI Systems Register":
        st.session_state["open_id"] = None
    if page == "Dashboard": page_dashboard(c, ucs)
    elif page == "Register a use case": page_register(c, role)
    elif page == "AI Systems Register": page_register_list(c, ucs, role)
    elif page == "My review queue": page_queue(c, ucs, role)
    else:
        st.title("Audit log (append-only)")
        st.dataframe(pd.read_sql("select * from audit order by at desc", c), use_container_width=True)

main()
