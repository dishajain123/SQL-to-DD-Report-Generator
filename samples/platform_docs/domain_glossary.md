# Banking / NPA / ECL Domain Glossary

**NPA (Non-Performing Asset)**: A loan or advance where interest or principal
repayment is overdue for a specified period (commonly 90 days), per RBI/IRAC
prudential norms.

**DPD (Days Past Due)**: Number of days a payment obligation has been overdue.
Different DPD "streams" (e.g. DPD_Overdue, DPD_NoCredit, DPD_Overdrawn) track
different overdue conditions per account.

**SMA (Special Mention Account)**: An early-warning classification bucketed
by DPD range (e.g. SMA-0: 1-30 DPD, SMA-1: 31-60 DPD, SMA-2: 61-90 DPD),
used to flag accounts before they become NPA.

**Asset Classification**: The regulatory category assigned to a loan account
(Standard, Sub-Standard, Doubtful, Loss) based on DPD, restructuring history,
and other qualifying criteria.

**ECL (Expected Credit Loss)**: Forward-looking loss provisioning framework
(IFRS 9) based on Probability of Default (PD), Loss Given Default (LGD), and
Exposure at Default (EAD), often organized into stages (Stage 1/2/3).

**Restructuring**: A modification of loan terms (tenure, rate, moratorium)
granted to a borrower in financial difficulty; restructured accounts often
carry additional monitoring periods and default-classification rules.

**TIMEKEY**: An integer surrogate key representing a calendar date in the
platform's day-dimension table, used throughout batch procedures to gate
date-based / rule-versioning logic (e.g. `IF p_TIMEKEY > 26267 THEN ...`
marks a rule that changed on a specific date).

**Reference Period**: The DPD threshold that must be exceeded before an
account moves to the next asset classification bucket; varies by facility
type and scheme.
