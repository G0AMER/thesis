import requests

url = "https://api.openalex.org/works"
params = {
    "search": "collaborative robot collision detection deep learning",
    "filter": "publication_year:>2023",
    "per-page": 8,
    "sort": "cited_by_count:desc",
    "select": "id,doi,title,publication_year,authorships"
}
try:
    response = requests.get(url, params=params)
    data = response.json()
    for i, work in enumerate(data.get("results", [])):
        authors = ", ".join([a["author"]["display_name"] for a in work.get("authorships", [])[:3]])
        if len(work.get("authorships", [])) > 3:
            authors += " et al."
        print(f"[{i+1}] {authors} ({work.get('publication_year')}). {work.get('title')}. DOI: {work.get('doi')}")
except Exception as e:
    print("Error:", e)
