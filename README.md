# heor-synth

A reproducible synthetic HEOR/RWE pipeline for type 2 diabetes using Synthea data.

The project ingests raw synthetic longitudinal records, normalizes them to Parquet, maps core entities into an OMOP-lite SQLite layer, applies deterministic cohort logic, and produces cohort attrition, baseline characteristics, treatment-pattern summaries, utilization/survival outputs, and poster-ready reports.