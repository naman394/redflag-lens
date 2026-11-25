"""Post-classifier rule (improvement iteration): suppress nil confirmations.

Error analysis showed the classifiers flag CARO / auditor "nil" statements because they contain risk words
("The Company does not have any pending litigations", "has not defaulted in repayment"). Guide v1.1 says these
are not risks. This rule clears such predictions unless the sentence also contains a contrastive cue
("except", "however", "other than") that may introduce a real exception.
"""
import re

NIL_RE = re.compile(
    r"\b(does|do|did|has|have|had) not (have any|defaulted|granted any|noticed|come across)|"
    r"\bthere (were|was|are|is|has been|have been) no\b|"
    r"\bno (material |pending |such )?(litigations?|instances?|fraud|default|undisputed|dues|cases?|penalt\w*|"
    r"loans|frauds|whistle|complaints?|adverse|qualifications?|reservations?)\b|"
    r"\bnot (been )?(noticed|reported)\b|\bis not applicable\b",
    re.I)
EXCEPT_RE = re.compile(r"\bexcept|\bhowever\b|\bdespite\b|\bother than\b|\bsave for\b|\bbut\b", re.I)


def is_nil_statement(text: str) -> bool:
    return bool(NIL_RE.search(text)) and not EXCEPT_RE.search(text)


# ---------------------------------------------------------------------------------------------------------------
# Iteration 2 (after the report-level backtest): template filter + severity tier.
# Backtest showed healthy reports (TCS, Maruti) get as many flags as distressed ones. Two causes:
#  (1) questionnaire / template rows ("Number of complaints...", "Is there any case filed...") that every report has;
#  (2) every risk counts the same: a routine tax dispute weighs as much as a going-concern doubt.
# The severity lexicon below was written from RBI's early-warning-signal list for stressed accounts (RBI Master
# Directions on Frauds, EWS annex) and standard audit-opinion terms, BEFORE re-running the backtest.

TEMPLATE_RE = re.compile(
    r"^[\W\d]{0,4}(\(?[a-z0-9]{1,3}[.)]\s*)?(number of|no\. of|is there any|please indicate|what percentage|"
    r"whether the|does the company|details of)\b|\?\s*$|"
    r"\b(members|shareholders)\b.*\b(connecting|log ?in|e-?voting|video conferenc|vc ?/ ?oavm)\b|"
    r"\blook into the reasons for\b",
    re.I)

SEVERE_RE_V1 = re.compile(
    r"going concern|material uncertainty|adverse (remark|opinion)|qualified opinion|basis for qualified|"
    r"disclaimer of opinion|do not express an opinion|"
    r"default(ed|s)? (in|on) (the )?(re)?payment|delays? in (the )?(re)?payment|delays? in servicing|"
    r"wil+ful defaulter|non-performing|\bNPA\b|\bfraud|forensic (audit|review)|diversion of funds|siphon|"
    r"insolvency|\bNCLT\b|\bIBC\b|winding[- ]up|liquidat(or|ion) proceedings|"
    r"one[- ]time settlement|restructuring of (its )?(debt|loans|borrowings)|recall(ed)? (of )?(the )?loans?|"
    r"downgrad|invocation of (the )?pledge|pledged shares (were |have been )?invoked|"
    r"resignation of (the )?(statutory )?auditors?|auditors? (has |have )?resigned|"
    r"erosion of (its )?net ?worth|negative net ?worth|net worth (has been |is )?(fully )?eroded|"
    r"enforcement directorate|\bSFIO\b|\bCBI\b|search and seizure|"
    r"discrepanc\w+.{0,80}(bank|stock statement)|(bank|stock statement).{0,80}discrepanc|"
    r"unable to (pay|meet|service|repay)|grounded|suspension of (operations|flights|trading)|"
    r"salar(y|ies) (were |have been )?(delayed|unpaid)|overdue (to|for) (banks|lenders)",
    re.I)
SEVERE_RE = re.compile(
    r"going concern|material uncertainty|adverse (remark|opinion)|qualified opinion|basis for qualified|"
    r"disclaimer of opinion|do not express an opinion|"
    r"default(ed|s)? (in|on) (the )?(re)?payment|delays? in (the )?(re)?payment|delays? in servicing|"
    r"wil+ful defaulter|non-performing|\bNPA\b|\bfraud|forensic (audit|review)|diversion of funds|siphon|"
    r"insolvency|\bNCLT\b|National Company Law Tribunal|\bIBC\b|\bCIRP\b|winding[- ]up|liquidat(or|ion) proceedings|"
    r"one[- ]time (settlement|restructuring)|restructuring of (its )?(debt|loans|borrowings)|"
    r"inter[- ]creditor agreement|recall(ed)? (of )?(the )?loans?|"
    r"downgrad|"
    # v2 (RBI EWS): lenders invoking pledged promoter shares, in any word order
    r"invo(ked?|cation)\b.{0,60}pledg|pledg\w*.{0,60}\binvo(ked?|cation)\b|"
    # v2 (RBI EWS): covenant breach; CARO 3(vii)(a) statutory dues unpaid > 6 months
    r"breach(es)? of (the |its )?(debt |financial |loan )?covenants?|non[- ]?compliance with (the )?(debt |financial )?covenants?|"
    r"(statutory dues|undisputed amounts?).{0,200}outstanding.{0,100}more than six months|"
    r"resignation of (the )?(statutory )?auditors?|auditors? (has |have )?resigned|"
    r"erosion of (its )?net ?worth|negative net ?worth|net worth (has been |is )?(fully )?eroded|"
    r"enforcement directorate|\bSFIO\b|\bCBI\b|search and seizure|"
    r"discrepanc\w+.{0,80}(bank|stock statement)|(bank|stock statement).{0,80}discrepanc|"
    r"unable to (pay|meet|service|repay)|grounded|suspension of (operations|flights|trading)|"
    r"salar(y|ies) (were |have been )?(delayed|unpaid)|overdue (to|for) (banks|lenders)",
    re.I)

# v2: wording that contains a severe term but is routine. These spans are removed before SEVERE_RE is applied,
# so a sentence that ALSO has a real signal stays severe.
AUDIT_TEMPLATE_RE = re.compile(  # standard auditor / directors' responsibility paragraphs in every report
    r"(whether|if we conclude that) a material uncertainty exists[^.]*|"
    r"conclude on the appropriateness of management.{0,3}s use of the going concern basis[^.]*|"
    r"assess\w* the (company|group|bank|entity).{0,3}s? ability to continue as a going concern[^.]*|"
    r"disclosing, as applicable, matters related to going concern[^.]*|"
    r"using the going concern basis of accounting unless[^.]*|"
    r"(future )?events or conditions may cause the (company|group|bank|entity) to cease to continue as a going concern|"
    r"prepared the annual accounts[^.]{0,120}on a .{0,2}going concern.{0,2} basis|"
    r"no order has been passed by[^.]*|"                       # CARO nil clause
    r"downgrad\w*[^.]{0,60}\b(gdp|growth|estimates?|forecasts?|outlook)\b",  # macro forecasts, not credit ratings
    re.I)
TRANSFER_RE = re.compile(r"slump sale|business transfer|transfer\w* (of )?(the |its |this )?(business|undertaking)|"
                         r"\btransferred\b|\bundertaking\b|hive[- ]?off|demerg", re.I)
AS_GOING_CONCERN_RE = re.compile(r"as a going concern", re.I)
MERGER_RE = re.compile(r"amalgamation|merger|demerger|scheme of arrangement|sanction of (the )?scheme|"
                       r"\bthe scheme\b|transfer(ee|or) company|\bCAA\b|operations (of .{0,60})?(have been )?(transferred|shifted)", re.I)
# NB: merger schemes always name "creditors" (ss. 230-232), so creditors/lenders are NOT distress context here
DISTRESS_CTX_RE = re.compile(r"insolvency|\bIBC\b|\bCIRP\b|resolution professional|oppression|winding|"
                             r"liquidat|default|restructur", re.I)
# v2: "resolution plan" is severe only for the company's OWN debt (not a healthy acquirer buying an IBC asset)
RESOLUTION_RE = re.compile(r"resolution plan", re.I)
OWN_DEBT_RE = re.compile(r"\b(its|our|the company.?s|your company.?s) (lenders|bankers|borrowings|debt)|"
                         r"inter[- ]creditor|\bICA\b|submitted to the (consortium of )?(lenders|bankers)|"
                         r"debenture ?holders? resolution|debt resolution", re.I)
TRIBUNAL_RE = re.compile(r"\bNCLT\b|National Company Law Tribunal", re.I)


def _strip_benign(text: str) -> str:
    t = AUDIT_TEMPLATE_RE.sub(" ", text)
    if TRANSFER_RE.search(t):          # "transfer ... as a going concern on a slump sale basis" = a business sale
        t = AS_GOING_CONCERN_RE.sub(" ", t)
    if MERGER_RE.search(t) and not DISTRESS_CTX_RE.search(t):   # NCLT sanctioning a merger scheme
        t = TRIBUNAL_RE.sub(" ", t)
    return t


HYPOTHETICAL_RE = re.compile(r"\bif any\b|\bin case of\b|\bpolicy\b|\bframework\b|\bcommittee\b", re.I)


def is_template(text: str) -> bool:
    return bool(TEMPLATE_RE.search(text))


def is_severe(text: str) -> bool:
    """v2 severity: RBI EWS lexicon + context filters (audit template, business transfer, merger at NCLT)."""
    if HYPOTHETICAL_RE.search(text) or is_nil_statement(text):
        return False
    t = _strip_benign(text)
    return bool(SEVERE_RE.search(t)) or bool(RESOLUTION_RE.search(t) and OWN_DEBT_RE.search(t))


def is_severe_v1(text: str) -> bool:
    """Original lexicon, kept for before/after comparison."""
    return bool(SEVERE_RE_V1.search(text)) and not HYPOTHETICAL_RE.search(text) and not is_nil_statement(text)


def norm(text: str) -> str:
    """Normalise for boilerplate lookup: lower-case, digits -> #, punctuation dropped."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z# ]", " ", re.sub(r"\d+", "#", text.lower()))).strip()
