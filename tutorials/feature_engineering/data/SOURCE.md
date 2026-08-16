# Dataset source and provenance

## UCI Online Retail

- Dataset: [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- DOI: [10.24432/C5BW33](https://doi.org/10.24432/C5BW33)
- Creator: Daqing Chen
- Repository: UCI Machine Learning Repository
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- Official CSV: <https://archive.ics.uci.edu/static/public/352/data.csv>
- UCI ID: 352

The dataset contains 541,909 line items from a UK-based non-store retailer. Its
documented period is 1 December 2010 through 9 December 2011. The company mainly
sold gifts, and many customers were wholesalers.

Suggested attribution:

> Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5BW33

## How this module uses the data

The notebook downloads the official CSV to the ignored `data/cache/` directory.
It does not redistribute the raw file.

The tutorial derives two event tables:

- completed orders, aggregated from positive, non-cancelled invoice lines;
- cancellations, aggregated from invoice numbers beginning with `C` or rows with
  negative quantity.

Rows without `CustomerID` cannot support customer-keyed history and are excluded
from the entity feature views after the notebook reports their count. Rows with
nonpositive prices or zero quantities are also reported before exclusion.

The source timestamps are naive and have minute resolution. The pandas reference
runner compares them consistently as recorded. The optional Spark fixture uses
explicit UTC timestamps, but it is a separate boundary test rather than a claim
about the source file's original timezone. A production conversion should localize
the retailer's source timezone with a reviewed daylight-saving policy before
converting to UTC.

## Prediction task created for teaching

The UCI dataset does not ship with a repeat-purchase target. The notebook creates
one for a transparent teaching task:

- prediction row: one completed order;
- history cutoff: the order timestamp;
- positive label: a strictly later completed order occurs within 30 days;
- label maturity: rows in the last 30 days of the dataset are censored and are not
  used for supervised training.

This target is an educational derivation, not a claim about the retailer's
production use case.

## Limitations that matter for feature-store design

- There is no ingestion or availability timestamp, so exact production-available
  replay is impossible from this file alone.
- There is no event ID beyond invoice and product identifiers, and within-minute
  ordering cannot be recovered.
- Customer identifiers are pseudonymous but should still be treated as sensitive
  identifiers in a real access-control design.
- Currency is interpreted as pounds sterling from UCI's `UnitPrice` description.
- Invoice cancellation is a useful behavioral event, but it is not a verified
  fraud, return, or refund outcome.
- The observation window is a little over one year. Long-window and seasonal
  conclusions are therefore limited.
- UCI's page reports no missing values, while the landed CSV contains missing
  customer IDs and descriptions. The tutorial reports the actual landed quality.

These limitations are teaching material. A feature store cannot recreate clocks,
semantics, or identifiers that the source never preserved.

© mui-group
