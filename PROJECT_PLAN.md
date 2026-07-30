# MissionGraph

MissionGraph is an educational public-data application that connects
federal contract awards, government agencies, contractors and mission
capabilities in a traceable knowledge graph.

## Initial scope

The first version will use only the USAspending API.

A user will:

1. Enter a keyword.
2. Select a date range.
3. Search Department of the Army contract awards.
4. View matching results in a table.
5. View a graph connecting agencies, awards and recipients.
6. Inspect the evidence supporting each relationship.

## Technical stack

- Python
- Streamlit
- Requests
- Pandas
- NetworkX
- PyVis
- Pytest

## Important principles

- Do not invent information.
- Preserve the original USAspending record.
- Every graph relationship must point to supporting evidence.
- Keep API retrieval separate from data transformation.
- Keep data transformation separate from visualization.
- Add error handling and tests.
- Never expose secrets in source code.