# Viva prep: likely questions and short answers

**Why not just give the report to ChatGPT?**
We did not test ChatGPT or GPT itself. We tested an open LLM: on the 600-sentence test set, Qwen2.5-7B given the same annotation guide scored F1 0.56. Our fine-tuned FinBERT scored 0.77, and our best model (DeBERTa-v3-large) scored 0.86.
FinBERT is also about 180× faster per sentence (0.7 ms against 126 ms on the same GPU), runs on a laptop CPU, and keeps borrower documents inside the bank.
We still use an LLM for one job, the short summary, and a verifier drops any claim not supported by its cited sentences.

**Where did the labels come from? Isn't that circular?**
No human time was available, so the labels come from LLM annotators:
- **Train/val:** labelled by Gemini (`gemini-4-neon`) using our guide v1.2.
- **Test:** labelled separately by Claude and by a different Gemini model (`argon`). Where they disagree, a third model breaks the tie.

The training labeller never touched the test set. The two test annotators agree with κ = 0.90 on is_risk.
The limitation: this is LLM–LLM agreement, not human gold labels, so mistakes the models share would not show up. Two hours of human labelling is the first next step.

**How did you prevent leakage?**
- Splits are by company: no company appears in two splits. 28% of a company's sentences repeat word for word from the previous year, so a random split would inflate scores.
- Gensol is held out from everything.
- The test labellers are not the training labeller.

**Why recall over precision?**
A missed red flag can mean a bad loan; a false alarm costs the analyst a few seconds.
- On the 600-sentence test set, the best model reaches recall 0.92 at precision 0.81; the FinBERT in the app reaches 0.84 at 0.70.
- The review tier lowers the threshold further for analysts who want everything.

**Your test set is small. Can we trust the numbers?**
The first test set had only 16 risk sentences, so we doubled it overnight to 600 sentences (37 risk), from the same 5 unseen companies and with the same labelling protocol (Claude + Qwen-32B, Llama-70B breaking ties).
- **Best model, DeBERTa-v3-large + extra examples:** F1 0.86 (0.76–0.93), recall 0.92. Both targets met.
- **On the 300 new sentences alone,** which no model was ever chosen on, it scores F1 0.93.
- **Honest caveat:** the new labels' agreement is lower (κ 0.76) and Business has only 2 test examples.

**Did you try bigger or newer models?**
Yes: DeBERTa-v3-base and -large, 3 seeds each, with and without extra non-risk examples (real nil statements and template text). That's 36 GPU runs over two rounds, chosen on validation only.
- **On the 600-sentence test set:** DeBERTa-large + extra examples wins (F1 0.86 vs 0.77 for FinBERT).
- **The app still runs FinBERT by default,** because DeBERTa-large is about 7.6× slower on a laptop CPU (~16 min per report against ~2.5 min).
- **DeBERTa-large is saved and selectable** (`REDFLAG_MODEL=deberta`). It is the recommended setup on a bank GPU server, at about 5 s per report.
- **Its weak spot is the category:** macro-F1 0.29 against FinBERT's 0.70, so the next step is DeBERTa to detect, FinBERT to categorise.

**Why FinBERT and not a bigger model?**
It is pre-trained on financial text, small enough to run on a laptop CPU in about 2.5 minutes per report, and fine-tunes in 25 seconds on one GPU.
With only ~1,850 labelled sentences, a bigger model would mostly add cost.

**Does it actually tell a distressed company from a healthy one?**
Not by counting flags: on our first backtest, healthy TCS and Maruti had more flags than Gensol, because big companies disclose a lot of routine litigation (ROC-AUC 0.50).
- **Fix:** we added a severity tier from RBI's early-warning signals (going concern, default, audit qualification, insolvency, fraud…), written before re-running.
- **Result:** severe signs per 1,000 sentences separate 4 distressed from 4 healthy reports with ROC-AUC 0.94. The only miss is Gensol vs Titan, which disclosed real IL&FS exposure.
- **Caveat:** 8 reports is indicative, not proof.

**What did it find in Gensol?**
The auditor's adverse remark on subsidiary Scorpius Trackers (p.187) and a going-concern doubt (p.184), listed first. It also flagged the discrepancies in stock statements submitted to the bank (p.144), but not as severe: the report misspells the word, and we did not change our list after seeing results.

**What are the main failure modes?**
1. **Negation.** The model reacts to risk words even in "there were no instances of…"; 9 of 11 rewritten red flags were still flagged. A rule filter catches the common CARO nil statements.
2. **Text about risk processes** flagged as if it were an actual risk event.
3. **Red-flag amounts inside tables** are not read.

**How do you verify the summary?**
- Every LLM claim must cite finding IDs.
- A claim is dropped if an ID doesn't exist, or if a ₹ amount in the claim does not appear in the cited sentences.
- If the LLM fails, the summary is built directly from the findings.
- Honest caveat: the verifier has not run on live LLM output yet, because the API credit ran out.

**What happened to the "agentic" design in the proposal?**
Cut for time to one LLM summariser plus a rule-based verifier. The verifier is the part that matters for trust.

**How would this run at a bank?**
- One Docker image (API + dashboard), on the bank's own servers.
- A monitoring log per report, with alerts when the finding rate or the share of low-confidence scores drifts.
- Analyst 👍/👎 votes feed the next training round.

**Did you test it on reports it had never seen at all?**
Yes. On 25 Nov 2026 we ran two brand-new reports. SpiceJet FY2023 (distressed): 34 severe signs, 13.5 per 1,000 sentences, higher than any backtest report, including the going-concern material uncertainty, the qualified opinion, "unable to pay Rs 750 million" and lessors' IBC petitions (about 6 of the 34 are false alarms from the cargo business transfer "as a going concern"). Nestlé India FY2024 (healthy): 5 severe signs, 1.6 per 1,000, all false alarms (NCLT merger scheme, fraud KAM wording). See TECHNICAL_REPORT §8.1.
