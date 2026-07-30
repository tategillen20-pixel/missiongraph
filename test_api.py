from services.usaspending import search_contract_awards

results = search_contract_awards(
    keyword="autonomous systems",
    start_date="2023-01-01",
    end_date="2026-07-27",
    limit=10,
)

for result in results:
    print(result)