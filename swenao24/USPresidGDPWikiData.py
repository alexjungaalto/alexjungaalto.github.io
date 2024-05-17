import requests
import datetime
import pandas as pd

def query_wikidata(sparql):
    url = "https://query.wikidata.org/sparql"
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'}
    data = requests.get(url, headers=headers, params={'format': 'json', 'query': sparql})
    return data.json()

def get_us_presidents(last_n_years=80):
    current_year = datetime.datetime.now().year
    start_year = current_year - last_n_years
    
    sparql = """
    SELECT ?president ?presidentLabel ?startTerm ?endTerm WHERE {
        ?president p:P39 ?statement.
        ?statement ps:P39 wd:Q11696; pq:P580 ?startTerm.
        OPTIONAL {?statement pq:P582 ?endTerm.}
        FILTER(YEAR(?startTerm) >= """ + str(start_year) + """)
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    ORDER BY ?startTerm
    """
    return query_wikidata(sparql)

def get_us_economic_data(year, property_code):
    # Common function to get economic data like GDP or unemployment rate
    sparql = f"""
    SELECT ?year ?value WHERE {{
        ?us wdt:{property_code} ?value;
            p:{property_code} [pq:P585 ?date].
        BIND(YEAR(?date) AS ?year)
        FILTER(?year = {year})
        FILTER(?value > 0)
    }}
    LIMIT 1
    """
    results = query_wikidata(sparql)
    data = results.get('results', {}).get('bindings', [])
    if data:
        return data[0]['value']['value']
    return None

def main():
    presidents_data = get_us_presidents()
    data_list = []

    for president in presidents_data['results']['bindings']:
        name = president['presidentLabel']['value']
        start_year = president['startTerm']['value'][:4]
        end_year = president['endTerm']['value'][:4] if 'endTerm' in president else datetime.datetime.now().year
        
        # GDP and unemployment rate queries
        start_gdp = get_us_economic_data(start_year, "P2131")  # GDP
        end_gdp = get_us_economic_data(end_year, "P2131")
        start_unemployment = get_us_economic_data(start_year, "P1198")  # Unemployment rate
        end_unemployment = get_us_economic_data(end_year, "P1198")

        data_list.append({
            "President": name,
            "Start Year": start_year,
            "Start GDP": start_gdp,
            "End Year": end_year,
            "End GDP": end_gdp,
            "Start Unemployment Rate": start_unemployment,
            "End Unemployment Rate": end_unemployment
        })

    df = pd.DataFrame(data_list)
    print(df)
    df.to_csv("us_presidents_economic_data.csv", index=False)

if __name__ == "__main__":
    main()

