# LightRAG-Experimente

LightRAG wird später gegen dieselbe Textbasis unter `data/processed/` getestet.

Geplanter Vergleich:

- `lightrag_naive`: LightRAG mit `QueryParam(mode="naive")` als interne Ablation ohne Graph-Retrieval.
- `lightrag_mix`: LightRAG mit `QueryParam(mode="mix")` als graphgestütztes Hauptsystem.
- `haystack_naive_rag`: externe klassische Haystack-Baseline.

Für den finalen Vergleich sollen möglichst dieselben Modelle verwendet werden:

- Generator: `gpt-4o`
- Embedder: `text-embedding-3-small`

Zu messen bzw. manuell zu bewerten:

- Antwortqualität
- Quellenqualität
- Latenz
- Indexing-Zeit
- erkannte Entitäten und Relationen
- Graph-Fehler wie Duplikate oder falsche Relationen
- Unsicherheitsverhalten bei nicht eindeutig gefundenen Informationen
