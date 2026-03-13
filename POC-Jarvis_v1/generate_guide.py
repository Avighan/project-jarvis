"""
Generates the Jarvis 2-Week Onboarding Guide PDF.
Run: python3 POC/POC-Jarvis_v1/generate_guide.py

Fix: all table cells use Paragraph objects so text wraps correctly.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import pathlib

OUTPUT = pathlib.Path(__file__).parent / "Jarvis_2Week_Guide.pdf"

# A4 usable width = 21cm - 2*2.2cm margins = 16.6cm
PAGE_W = 16.6 * cm

PURPLE     = colors.HexColor("#7c6aff")
DARK_BG    = colors.HexColor("#1a1a2e")
LIGHT_GREY = colors.HexColor("#f4f4f8")
MID_GREY   = colors.HexColor("#d8d8e8")
TEXT       = colors.HexColor("#1a1a2e")
MUTED      = colors.HexColor("#6b6b80")
RED        = colors.HexColor("#c0392b")
GREEN_C    = colors.HexColor("#1a7a40")
STRIPE     = colors.HexColor("#eeeef6")


# ── Style factory ─────────────────────────────────────────────────────────────

def S(name, **kw):
    """Create a named ParagraphStyle with defaults."""
    defaults = dict(fontName="Helvetica", fontSize=9, leading=13,
                    textColor=TEXT, spaceAfter=0, spaceBefore=0)
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)


TITLE    = S("title",    fontName="Helvetica-Bold", fontSize=26, textColor=PURPLE, leading=32, spaceAfter=2)
SUBTITLE = S("subtitle", fontSize=12, textColor=MUTED, spaceAfter=16)
H1       = S("h1",       fontName="Helvetica-Bold", fontSize=15, textColor=TEXT,   spaceBefore=14, spaceAfter=5, leading=19)
H2       = S("h2",       fontName="Helvetica-Bold", fontSize=11, textColor=PURPLE, spaceBefore=10, spaceAfter=3)
H3       = S("h3",       fontName="Helvetica-Bold", fontSize=10, textColor=TEXT,   spaceBefore=6,  spaceAfter=2)
BODY     = S("body",     fontSize=10, leading=15, spaceAfter=4)
CENTER   = S("center",   fontSize=9,  textColor=MUTED, alignment=TA_CENTER)
CALLOUT  = S("callout",  fontSize=10, leading=15, textColor=TEXT, backColor=LIGHT_GREY,
              leftIndent=10, rightIndent=10, spaceBefore=6, spaceAfter=6,
              borderPadding=(6, 8, 6, 8))

# Table cell styles — no indent/backColor so they render cleanly inside cells
TC       = S("tc",       fontSize=9,  leading=13, textColor=TEXT)
TC_BOLD  = S("tc_bold",  fontName="Helvetica-Bold", fontSize=9, leading=13, textColor=TEXT)
TC_HDR   = S("tc_hdr",   fontName="Helvetica-Bold", fontSize=9, leading=13, textColor=colors.white)
TC_CODE  = S("tc_code",  fontName="Courier", fontSize=8, leading=12, textColor=colors.HexColor("#2d2d6b"))
TC_RED   = S("tc_red",   fontSize=9, leading=13, textColor=RED)
TC_GRN   = S("tc_grn",   fontSize=9, leading=13, textColor=GREEN_C)
TC_PURP  = S("tc_purp",  fontName="Helvetica-Bold", fontSize=9, leading=13, textColor=PURPLE)
TC_MUT   = S("tc_muted", fontSize=9, leading=13, textColor=MUTED)
CODE_BLK = S("code_blk", fontName="Courier", fontSize=9, leading=13,
              textColor=colors.HexColor("#2d2d6b"), backColor=LIGHT_GREY,
              leftIndent=10, rightIndent=10, spaceBefore=3, spaceAfter=3,
              borderPadding=(4, 6, 4, 6))


def p(text, style=None):
    return Paragraph(text, style or BODY)


def br(text):
    """Replace literal newlines with <br/> for Paragraph."""
    return text.replace("\n", "<br/>")


def hr(story, color=MID_GREY, thick=0.5):
    story.append(Spacer(1, 3))
    story.append(HRFlowable(width="100%", thickness=thick, color=color))
    story.append(Spacer(1, 5))


def table(data, col_widths, stripe=True, header_bg=DARK_BG):
    """Build a styled table. data rows must already be lists of Paragraphs."""
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND",   (0, 0), (-1, 0),  header_bg),
        ("GRID",         (0, 0), (-1, -1), 0.35, MID_GREY),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]
    if stripe:
        for i in range(1, len(data)):
            bg = STRIPE if i % 2 == 0 else colors.white
            style.append(("BACKGROUND", (0, i), (-1, i), bg))
    t.setStyle(TableStyle(style))
    return t


# ── Build ─────────────────────────────────────────────────────────────────────

def build_pdf():
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2*cm,    bottomMargin=2*cm,
        title="Jarvis — 2-Week Onboarding Guide",
    )
    story = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 0.8*cm),
        p("Jarvis", TITLE),
        p("2-Week Onboarding Guide", SUBTITLE),
        p("From first launch to investor-ready demo — a day-by-day playbook.", BODY),
    ]
    hr(story, PURPLE, 1.5)

    # ── What Jarvis Does ──────────────────────────────────────────────────────
    story += [
        p("What Jarvis Does", H1),
        p("Jarvis is a local-first memory layer that makes AI assistants remember you. "
          "Every preference you teach it, every goal you log, every work pattern it "
          "observes — all injected silently into every future conversation. "
          "No cloud. No reset. Runs entirely on your machine.", BODY),
        Spacer(1, 4),
        p("The next two weeks have one purpose: <b>fill the memory store with enough "
          "high-quality context that Jarvis gives a demonstrably better answer than any "
          "cold AI session.</b> That moment — when it uses a memory from Week 1 to answer "
          "a question in Week 2 — is your investor demo.", BODY),
        Spacer(1, 10),
    ]

    # ── Quick Start ───────────────────────────────────────────────────────────
    story += [p("Quick Start (Day 1, ~15 minutes)", H1)]
    hr(story)

    steps = [
        ("Step 1", "Make sure Ollama is running",         "ollama serve"),
        ("Step 2", "Initialise the database (one-time)",  "python3 POC/core/setup_db.py"),
        ("Step 3", "Start the web app",                   "cd POC/POC-Jarvis_v1/web-app<br/>python3 main.py"),
        ("Step 4", "Open in browser",                     "http://localhost:8000"),
        ("Step 5", "Seed your first memories",            "Use the + Add tab in the sidebar"),
    ]
    for step, label, cmd in steps:
        story += [
            p(f"<b>{step} — {label}</b>", H3),
            p(cmd, CODE_BLK),
            Spacer(1, 2),
        ]
    story.append(Spacer(1, 8))

    # ── CLI Reference ─────────────────────────────────────────────────────────
    story += [p("CLI Quick Reference", H1)]
    hr(story)

    W = PAGE_W
    cli_rows = [
        [p("Command", TC_HDR),        p("What it does", TC_HDR)],
        [p('python3 POC/core/jarvis_cli.py "question"', TC_CODE),
         p("Ask with memory injection (default)", TC)],
        [p('... --no-memory "question"', TC_CODE),
         p("Ask without memory — use for baseline comparison", TC)],
        [p('... --add-memory "fact"<br/>  --category preference', TC_CODE),
         p("Manually seed one memory", TC)],
        [p('... --add-memory "fact"<br/>  --category goal --confidence 0.95', TC_CODE),
         p("Seed a goal with high confidence", TC)],
        [p('... --extract "User: ...<br/>Assistant: ..."', TC_CODE),
         p("Extract + save facts from conversation text via Claude Haiku", TC)],
        [p('... --extract-only "..."', TC_CODE),
         p("Preview extracted facts without saving (dry run)", TC)],
        [p('... --show-memories', TC_CODE),
         p("List all stored memories with category and confidence", TC)],
        [p('... --rate &lt;id&gt; &lt;1-5&gt;', TC_CODE),
         p("Rate a previous response (builds quality signal over time)", TC)],
    ]
    story.append(table(cli_rows, [W * 0.50, W * 0.50]))
    story.append(Spacer(1, 10))

    # ── 5 Memory Categories ───────────────────────────────────────────────────
    story += [
        p("What to Seed — The 5 Memory Categories", H1),
    ]
    hr(story)
    story += [
        p("The quality of Jarvis's answers is directly proportional to the quality of "
          "your memories. Poor seeds = generic answers. Specific seeds = aha moments.", BODY),
        Spacer(1, 6),
    ]

    categories = [
        ("preference", "How you like to work and receive information",
         ['"Prefers direct answers without preamble or filler"',
          '"Prefers bullet points over prose for technical topics"',
          '"Dislikes over-engineering — minimal solutions always first"']),
        ("goal", "What you are trying to achieve right now",
         ['"Building Jarvis — local-first AI memory layer, pre-seed stage"',
          '"Target: 3–5 design partners using Jarvis daily within 6 weeks"',
          '"Need investor-ready demo with MCP integration in 4 weeks"']),
        ("expertise", "What you know and at what depth",
         ['"Senior Python engineer, ~10 years. Strongest in backend/systems"',
          '"Intermediate in ML/AI — know enough to build, not to research"',
          '"New to React/frontend — frame explanations in backend terms"']),
        ("pattern", "How you actually work day to day",
         ['"Works best in 2-hour focused blocks before 11am — protect this time"',
          '"Meeting-heavy weeks kill deep work. Never schedule architecture after 2pm"',
          '"Needs 3+ uninterrupted hours to make real progress on hard problems"']),
        ("general", "Project state and operational context",
         ['"14 experiments complete. Extraction: 74% recall via Claude Haiku (Option D)"',
          '"Web app at localhost:8000. DB at ~/.jarvis/jarvis_poc.db"',
          '"Config: TOP_N=4, INJECTION_FORMAT=json, CONFIDENCE_WEIGHT=True"']),
    ]

    for cat, title, examples in categories:
        story.append(KeepTogether([
            p(f"<b>{cat}</b> — {title}", H2),
            p("<br/>".join(examples), CODE_BLK),
            Spacer(1, 4),
        ]))
    story.append(Spacer(1, 8))

    # ── Week 1 ────────────────────────────────────────────────────────────────
    story += [p("Week 1 — Build the Memory Store", H1)]
    hr(story)
    story += [
        p("<b>Goal:</b> 25–40 high-signal memories across all 5 categories. "
          "This is the foundation everything else is built on.", BODY),
        Spacer(1, 6),
    ]

    w1_rows = [
        [p("Day", TC_HDR), p("Task", TC_HDR), p("Time", TC_HDR)],
        [p("1", TC_BOLD),
         p("Seed 10 core memories across all 5 categories using the web app + Add tab "
           "or CLI --add-memory. Focus on what is most specific and surprising about you.", TC),
         p("20 min", TC_MUT)],
        [p("2", TC_BOLD),
         p("Use Jarvis for one real work task. Rate the response 1–5. "
           "Note what memory chips appeared below the answer.", TC),
         p("5 min", TC_MUT)],
        [p("3", TC_BOLD),
         p("After a real work session, paste the conversation into --extract or the web app "
           "Extract tab. Review what Claude Haiku found. Correct or supplement manually.", TC),
         p("10 min", TC_MUT)],
        [p("4", TC_BOLD),
         p("Add 5 more memories based on what Jarvis got wrong or missed yesterday. "
           "Every failure is a product requirement.", TC),
         p("10 min", TC_MUT)],
        [p("5", TC_BOLD),
         p("Use Jarvis for planning or research. Run the same question with and without "
           "--no-memory. Note where memory made a visible difference.", TC),
         p("10 min", TC_MUT)],
        [p("6–7", TC_BOLD),
         p("Weekend: reflect on the week. Add any patterns or goals that emerged. "
           "Seed anything about your working style not yet captured.", TC),
         p("15 min", TC_MUT)],
    ]
    story.append(table(w1_rows, [W * 0.08, W * 0.77, W * 0.15]))
    story.append(Spacer(1, 10))

    # ── Week 2 ────────────────────────────────────────────────────────────────
    story += [p("Week 2 — Build Toward the Demo", H1)]
    hr(story)
    story += [
        p("<b>Goal:</b> Hit 40–60 memories, find your aha moment, and build toward "
          "showing it to someone else.", BODY),
        Spacer(1, 6),
    ]

    w2_rows = [
        [p("Day", TC_HDR), p("Task", TC_HDR), p("Time", TC_HDR)],
        [p("8", TC_BOLD),
         p("Review all memories. Delete weak ones (vague, redundant, low-confidence). "
           "Re-seed the same slots with more specific versions.", TC),
         p("15 min", TC_MUT)],
        [p("9", TC_BOLD),
         p("Aha test: ask Jarvis something you discussed in Week 1 without any extra "
           "context. Does it answer as if it already knows your situation?", TC),
         p("5 min", TC_MUT)],
        [p("10", TC_BOLD),
         p("Add 10 project-specific memories — current work, current blockers, "
           "current decisions. Make Jarvis operationally aware, not just personally aware.", TC),
         p("15 min", TC_MUT)],
        [p("11", TC_BOLD),
         p("Use Jarvis on a real planning or research task. Note the 1–2 moments "
           "where it surprised you with contextual accuracy. Those are your demo clips.", TC),
         p("10 min", TC_MUT)],
        [p("12", TC_BOLD),
         p("Dry-run demo with yourself: open the web app cold, ask 3 questions a "
           "stranger might ask, see how it performs without any prompting from you.", TC),
         p("10 min", TC_MUT)],
        [p("13–14", TC_BOLD),
         p("Show someone else. Friend, colleague — anyone. Watch their face when "
           "it knows something it shouldn't. That reaction is the product.", TC),
         p("30 min", TC_MUT)],
    ]
    story.append(table(w2_rows, [W * 0.08, W * 0.77, W * 0.15]))
    story.append(Spacer(1, 10))

    # ── Good vs Bad Memories ──────────────────────────────────────────────────
    story += [p("What Makes a Good Memory", H1)]
    hr(story)

    gb_rows = [
        [p("WEAK — too vague", TC_HDR), p("STRONG — specific and actionable", TC_HDR)],
        [p('"Likes concise answers"', TC_RED),
         p('"Prefers bullet points over prose for technical topics. '
           'Gets frustrated by long preambles — just give the answer."', TC_GRN)],
        [p('"Working on an AI project"', TC_RED),
         p('"Building Project Jarvis: local-first memory layer. Pre-seed stage. '
           'Goal: 3 design partners in 6 weeks, investor demo in 8 weeks."', TC_GRN)],
        [p('"Good at Python"', TC_RED),
         p('"Senior Python engineer, ~10 years. Strongest in SQLite, async systems, '
           'data pipelines. Weak on React/frontend — explain in backend terms."', TC_GRN)],
        [p('"Busy schedule"', TC_RED),
         p('"Morning blocks before 11am are protected deep work. Afternoons are meetings. '
           'Never schedule architecture decisions after 2pm."', TC_GRN)],
    ]
    story.append(table(gb_rows, [W * 0.38, W * 0.62]))
    story.append(Spacer(1, 10))

    # ── Rating guide ──────────────────────────────────────────────────────────
    story += [p("Rate Every Response — It Takes 3 Seconds", H1)]
    hr(story)
    story += [
        p("Every rating builds toward the DPO fine-tuning dataset (Phase 4). "
          "Use the 1–5 buttons in the chat UI or --rate in the CLI. Target: 5+ ratings per day.", BODY),
        Spacer(1, 6),
    ]

    r_rows = [
        [p("★", TC_HDR), p("Meaning", TC_HDR), p("Action", TC_HDR)],
        [p("5", TC_BOLD),
         p("Memory made a visible difference. It knew something.", TC),
         p("Note which memory chips fired. Add more like them.", TC)],
        [p("4", TC_BOLD),
         p("Good answer, slightly better than a cold AI.", TC),
         p("Fine — keep going.", TC)],
        [p("3", TC_BOLD),
         p("Same as any AI. Memory didn't visibly help.", TC),
         p("Check: were injected memories relevant? Seed more specific ones.", TC)],
        [p("2", TC_BOLD),
         p("Worse than expected — irrelevant memory injected.", TC),
         p("Delete the memory that caused the confusion.", TC)],
        [p("1", TC_BOLD),
         p("Wrong or hallucinated. Actively harmful.", TC),
         p("Delete the memory. Note the pattern to avoid repeating it.", TC)],
    ]
    story.append(table(r_rows, [W * 0.07, W * 0.40, W * 0.53]))
    story.append(Spacer(1, 10))

    # ── Web App sections ──────────────────────────────────────────────────────
    story += [p("Web App — What Each Section Does", H1)]
    hr(story)

    sections = [
        ("Chat",
         "Main interface. Type a question, press Enter or ↑. The purple chips below each "
         "response show exactly which memories were injected — this is your evidence the "
         "memory layer is working. Rate every response using the 1–5 buttons."),
        ("Memories",
         "Full list of everything Jarvis knows about you, sorted by confidence then "
         "access count. High access count = frequently used memory. "
         "Low access + low confidence = candidate for deletion or refinement."),
        ("+ Add",
         "Manual seeding. Choose content, category, and confidence. This is your primary "
         "workflow for Week 1. The Extract sub-section lets you paste any conversation — "
         "Claude Haiku extracts facts, you review them, then save the correct ones."),
        ("Stats",
         "Total memories, recent interaction count, average rating. "
         "Watch total memories climb toward 60 across the two weeks."),
    ]
    for tab_name, desc in sections:
        story.append(KeepTogether([
            p(f"<b>{tab_name}</b>", H3),
            p(desc, BODY),
            Spacer(1, 3),
        ]))
    story.append(Spacer(1, 8))

    # ── Memory targets ────────────────────────────────────────────────────────
    story += [p("Memory Targets by Day", H1)]
    hr(story)

    t_rows = [
        [p("Day",  TC_HDR), p("Target", TC_HDR), p("Focus", TC_HDR)],
        [p("1",  TC_BOLD), p("10",  TC_PURP), p("Core identity: preferences, expertise, top 2–3 goals", TC)],
        [p("3",  TC_BOLD), p("18",  TC_PURP), p("Work patterns, project context, current blockers", TC)],
        [p("5",  TC_BOLD), p("25",  TC_PURP), p("First extraction session from a real conversation", TC)],
        [p("7",  TC_BOLD), p("30",  TC_PURP), p("Week 1 review: delete weak memories, re-seed stronger versions", TC)],
        [p("10", TC_BOLD), p("40",  TC_PURP), p("Operational context: current decisions, project state, active blockers", TC)],
        [p("12", TC_BOLD), p("50",  TC_PURP), p("Aha test: can Jarvis answer a real question without you prompting it?", TC)],
        [p("14", TC_BOLD), p("60+", TC_PURP), p("Demo-ready. Show someone else. Watch their reaction.", TC)],
    ]
    story.append(table(t_rows, [W * 0.08, W * 0.12, W * 0.80]))
    story.append(Spacer(1, 10))

    # ── What comes next ───────────────────────────────────────────────────────
    story += [p("After 2 Weeks — What Comes Next", H1)]
    hr(story)

    next_rows = [
        [p("Milestone", TC_HDR), p("When", TC_HDR), p("Why it matters", TC_HDR)],
        [p("Build the MCP server", TC_BOLD), p("Week 3", TC_MUT),
         p("Expose Jarvis via FastMCP so Claude Desktop auto-injects your memories on every "
           "message. Transforms the demo from 'open a web app' to 'open Claude and watch it "
           "already know you.'", TC)],
        [p("Recruit 3 design partners", TC_BOLD), p("Week 3–4", TC_MUT),
         p("Real professionals — not friends — who use Jarvis daily and give honest feedback. "
           "Offer white-glove onboarding: you seed their first 20 memories in a 30-minute call.", TC)],
        [p("Investor demo prep", TC_BOLD), p("Week 4–5", TC_MUT),
         p("The demo: open Claude Desktop cold, ask a real question, show it answering with "
           "context it couldn't have without Jarvis. Practice until under 5 minutes, unscripted.", TC)],
        [p("Embedding deduplication", TC_BOLD), p("Phase 2", TC_MUT),
         p("Pull nomic-embed-text, re-run Exp 10+11. Prevents memory bloat as the store grows "
           "past 100 entries. One command, 30 minutes to implement.", TC)],
    ]
    story.append(table(next_rows, [W * 0.26, W * 0.13, W * 0.61]))
    story.append(Spacer(1, 12))

    # ── Footer ────────────────────────────────────────────────────────────────
    hr(story, PURPLE, 1)
    story += [
        p("<b>The aha moment happens when Jarvis uses a specific memory from Week 1 to answer "
          "a new question in Week 2 — without you prompting it. That moment is the product. "
          "Everything else is infrastructure.</b>", CALLOUT),
        Spacer(1, 8),
        p("Project Jarvis — Confidential · Built locally on MacBook Pro M3 Pro · Zero cloud", CENTER),
    ]

    doc.build(story)
    print(f"PDF created: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
