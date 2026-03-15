# Data Validation: 2008/2009

> M5 validation of Torvik and results data added for 2008–2009 seasons.

---

## M5.1 Spot-Check: Torvik 2008/2009

**Source:** `data/torvik_2008.json`, `data/torvik_2009.json`  
**Reference:** BartTorvik / Torvik historical pages (pre-tournament ratings)

| Year | Team        | barthag | adj_o | adj_d | Notes                    |
|------|-------------|---------|-------|-------|--------------------------|
| 2008 | Kansas      | 0.9815  | 122.1 | 86.4  | #1 seed, champion        |
| 2008 | Memphis     | 0.9703  | 117.7 | 86.9  | #1 seed, runner-up       |
| 2008 | UCLA        | 0.9647  | 116.9 | 87.7  | #1 seed, Final Four      |
| 2008 | North Carolina | 0.9643 | 123.6 | 92.8 | #1 seed, Final Four   |
| 2009 | North Carolina | —    | —     | —     | Champion (verify in torvik_2009) |

Order by barthag (2008): Kansas > Memphis > UCLA > North Carolina. Matches expected pre-tournament rankings.

---

## M5.2 Cross-Check: Results vs NCAA Brackets

**Source:** `data/results_2008.json`, `data/results_2009.json`  
**Reference:** Wikipedia, NCAA.com

### 2008
- **Champion:** Kansas (def. Memphis 75–68) ✓
- **Final Four:** Kansas, Memphis, UCLA, North Carolina ✓
- **Semifinals:** Memphis def. UCLA; Kansas def. North Carolina ✓

### 2009
- **Champion:** North Carolina (def. Michigan State 89–72) ✓
- **Final Four:** North Carolina, Michigan State, UConn, Villanova ✓
- **Semifinals:** North Carolina def. Villanova; Michigan State def. UConn ✓

**Minor issue:** `results_2008.json` has one winner entry `"Davidson "` (trailing space). Consider normalizing.

---

## M5.3 Calibration Comparison: With vs Without 2008/2009

| Dataset | Games | Brier | Accuracy |
|---------|-------|-------|----------|
| Full (2008–2026) | 1,071 | 0.1646 | 73.9% |
| Exclude 2008, 2009 | 945 | 0.1674 | 73.3% |

**Delta:** Brier +0.0028, Accuracy −0.6% when excluding 2008/2009.

**Interpretation:** Small, expected variation. No large swings that would suggest data errors. The 2008/2009 games appear slightly more predictable by the model (or the model fits them slightly better). **Conclusion: 2008/2009 data appears valid.**

---

## M5.4 Summary

- **Torvik:** Structure and top-team order consistent with expectations.
- **Results:** Champions and Final Four match official records.
- **Calibration:** Excluding 2008/2009 yields modest Brier/accuracy changes; no anomalies.

**Recommendation:** Keep 2008/2009 data in the calibration set. Optional: fix `"Davidson "` trailing space in `results_2008.json`.
