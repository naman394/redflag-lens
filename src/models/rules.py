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

SEVERE_RE = re.compile(
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
HYPOTHETICAL_RE = re.compile(r"\bif any\b|\bin case of\b|\bpolicy\b|\bframework\b|\bcommittee\b", re.I)


def is_template(text: str) -> bool:
    return bool(TEMPLATE_RE.search(text))


def is_severe(text: str) -> bool:
    return bool(SEVERE_RE.search(text)) and not HYPOTHETICAL_RE.search(text) and not is_nil_statement(text)


def norm(text: str) -> str:
    """Normalise for boilerplate lookup: lower-case, digits -> #, punctuation dropped."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z# ]", " ", re.sub(r"\d+", "#", text.lower()))).strip()
