# Annotation Guide v1 (RedFlag Lens)

Label **one sentence at a time**. The section name is shown as context. Don't open the PDF unless the sentence is truly unreadable.

Each sentence gets three labels.

## 1. `is_risk` (1 / 0)

**1 = the sentence states or warns about something that could hurt the company's finances, legal standing, compliance or operations.** This includes:
- A threat, uncertainty or exposure ("may", "could", "is exposed to")
- An adverse event that happened (default, penalty, loss, suit, fire, resignation with concerns)
- A pending dispute or contingent obligation (disputed tax demand, guarantee given, claim not acknowledged as debt)
- An auditor qualification, emphasis of matter, going-concern doubt or reported fraud

**0 = not a risk.** This includes:
- Neutral facts and descriptions ("The Company operates five plants")
- Positive results ("Revenue grew 12%")
- Accounting policy definitions ("Provisions are recognised when…")
- Governance procedure ("The Committee met four times")
- **CARO "nil" confirmations** ("The Company has not defaulted in repayment of loans", "no fraud was noticed")
- Statements that a control is adequate

**Rule of thumb:** would a credit analyst want this sentence on a "things to check" list? If yes → 1.

## 2. `category` (only if `is_risk` = 1; otherwise `None`)

| Category | Choose when the main source of the risk is… | Examples |
|---|---|---|
| **Financial** | Money and balance sheet: liquidity, debt, defaults, going concern, losses, impairments, provisions, unpaid statutory dues, FX/interest rate, credit risk, NPAs, guarantees, rating downgrades | "Lenders have recalled loans of ₹1,200 crore." |
| **Legal** | A **court or a private party** is the actor: suits, arbitration, claims, NCLT petitions by creditors, criminal cases | "A vendor has filed a suit claiming ₹8.5 crore." |
| **Regulatory** | A **regulator or government** is the actor, or rules change: SEBI/RBI/tax/GST/environment orders, penalties, show-cause notices, licence issues, new regulations, tax demands under dispute | "SEBI imposed a penalty of ₹10 lakh." |
| **Operational** | **Inside the company**: plants, supply chain, safety, IT/cyber, people, internal-control weaknesses, employee fraud, project delays | "Operations at the Pune plant were suspended for 41 days." |
| **Business** | **Outside the company, in the market**: competition, demand, customer concentration, prices, technology change, macro economy, geopolitics | "Intense competition may reduce margins." |

**Tie-breakers**
- Disputed **tax demands** (income tax, GST, excise) → Regulatory.
- **Contingent liabilities** from claims by private parties → Legal. Bank guarantees or corporate guarantees given → Financial.
- Going concern, default, inability to pay → Financial, even if lenders have gone to court (put Legal in `category_2`).
- SEBI order under appeal at SAT → Regulatory (secondary: Legal).
- Auditor qualification about accounts or records → Financial. About internal controls → Operational.
- Commodity or raw-material **price** risk → Business. Raw-material **supply** disruption → Operational.

## 3. `specificity` (only if `is_risk` = 1; otherwise `None`)

- **Specific:** names a concrete event, amount, counterparty, regulator action, date, or an actual auditor finding about **this** company. These are red flags.
  - "The Company has defaulted in repayment of ₹212 crore to banks."
  - "We draw attention to Note 52 regarding going concern."
  - "Disputed income tax demand of ₹45.2 crore is pending before CIT(A)."
- **Generic:** boilerplate that could appear in almost any report, often hedged ("may", "could", "is exposed to") with no concrete event.
  - "The Company is exposed to foreign exchange risk."
  - "Any slowdown in the economy may affect demand."

If a sentence has a real amount but describes a **routine** exposure (e.g. a table caption-like line, "Trade receivables of ₹X are subject to credit risk"), mark it **Generic**. Specific means *an event or finding*, not just a number.

## Process
1. Read the sentence, then decide `is_risk` first.
2. If unsure between two categories, pick one using the tie-breakers and write the other in `notes`.
3. If the sentence is broken or unreadable (bad extraction), set `is_risk = 0` and write `BAD_EXTRACTION` in `notes`.
4. Label alone: don't discuss sentences during the blind agreement round.

## Rules added after the pilot (v1.1)
Decided while labelling the 300-sentence pilot:
- **Auditor boilerplate → 0.** Standard auditing wording is not a risk about this company: "If we conclude that a material uncertainty exists, we are required to…", "future events may cause the Company to cease to continue as a going concern", "inherent limitations of internal financial controls".
- **Key Audit Matter descriptions → 0**, unless they state a concrete adverse fact. ("Revenue is susceptible to misstatement" = 0. "We were unable to obtain sufficient evidence for ₹3,253 crore of fair value changes" = 1.)
- **Emphasis of Matter, Qualified / Adverse / Disclaimer opinions, "except for the effects of…"** → 1, Specific.
- **Routine resignations** (retirement, new appointment elsewhere) → 0. Resignation citing concerns, or an auditor resigning → 1.
- **Litigation resolved in the company's favour** → 0. Pending appeals or cases → 1, Legal, Specific.
- **Table rows, page headers, fragments** → 0 with note `BAD_EXTRACTION`. They make up about 7% of the pilot and are the next cleaning target.

## Version history
- v1 (2026-10-28): first version, used for the 300-sentence pilot.
- v1.1 (2026-10-28): rules added from pilot decisions (above).
