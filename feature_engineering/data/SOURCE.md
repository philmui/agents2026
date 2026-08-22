# Dataset source and provenance

## UCI Online Retail

- Dataset: [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- DOI: [10.24432/C5BW33](https://doi.org/10.24432/C5BW33)
- Creator: Daqing Chen
- Repository: UCI Machine Learning Repository
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- Official CSV: <https://archive.ics.uci.edu/static/public/352/data.csv>
- UCI ID: 352

The dataset contains 541,909 line items from a UK-based non-store retailer. A line
item is one product on one invoice, so one order spans several rows. Its documented
period is 1 December 2010 through 9 December 2011. The company mainly sold gifts,
and many customers were wholesalers.

Suggested attribution:

> Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5BW33

## How this module uses the data

The notebook downloads the official CSV to the ignored `data/cache/` directory.
It does not redistribute the raw file, so every reader gets the copy UCI publishes.

The tutorial derives two event tables:

- completed orders, aggregated from positive, non-cancelled invoice lines;
- cancellations, aggregated from invoice numbers beginning with `C` or rows with
  negative quantity.

Rows without `CustomerID` cannot support customer-keyed history and are excluded
from the entity feature views after the notebook reports their count. The entity is
the thing whose history you summarize, here the customer, so such a row belongs to
nobody. Rows with nonpositive prices or zero quantities are also reported before
exclusion. Reporting the counts first is deliberate, because a silent drop hides
how much data you just discarded.

The source timestamps are naive and have minute resolution. Naive means they carry
no timezone, and minute resolution means two events in the same minute look
simultaneous. The pandas reference runner compares them consistently as recorded.
The optional Spark fixture uses explicit UTC timestamps, but it is a separate
boundary test rather than a claim about the source file's original timezone. A
production conversion should localize the retailer's source timezone with a reviewed
daylight-saving policy before converting to UTC. Skip that step and your cutoffs
shift by whole hours for part of the year, which shows up only near a boundary.

## Prediction task created for teaching

The UCI dataset does not ship with a repeat-purchase target. The notebook creates
one for a transparent teaching task:

- prediction row: one completed order;
- history cutoff: the order timestamp;
- positive label: a strictly later completed order occurs within 30 days;
- label maturity: rows in the last 30 days of the dataset are censored and are not
  used for supervised training.

Censored means the outcome window has not finished, so seeing nothing does not yet
mean nothing happened. An order placed on 1 December 2011 has only eight days of
future in a file ending 9 December 2011.

This target is an educational derivation, not a claim about the retailer's
production use case.

## Limitations that matter for feature-store design

- There is no ingestion or availability timestamp, so exact production-available
  replay is impossible from this file alone. Availability time is when a pipeline
  could first have seen a row, so any replay here only approximates what production
  knew.
- There is no event ID beyond invoice and product identifiers, and within-minute
  ordering cannot be recovered. Two events in the same minute cannot be ordered, so
  a tie at a cutoff is settled arbitrarily.
- Customer identifiers are pseudonymous but should still be treated as sensitive
  identifiers in a real access-control design. A number that follows one person
  across orders can still identify them once joined to other data.
- Currency is interpreted as pounds sterling from UCI's `UnitPrice` description, so
  treat £ totals as an interpretation, not a verified field.
- Invoice cancellation is a useful behavioral event, but it is not a verified
  fraud, return, or refund outcome. A model trained on it predicts cancellation
  behavior, not fraud.
- The observation window is a little over one year. Long-window and seasonal
  conclusions are therefore limited, because a yearly pattern appears only once,
  which is not enough to confirm it.
- UCI's page reports no missing values, while the landed CSV contains missing
  customer IDs and descriptions. The tutorial reports the actual landed quality.
  Trust what you measure over what the page claims.

These limitations are teaching material. A feature store cannot recreate clocks,
semantics, or identifiers that the source never preserved. No amount of engineering
downstream puts back a timestamp nobody wrote down. If you need availability times,
someone must record them as the events arrive, because by the time the data reaches
you it is gone.

© mui-group
