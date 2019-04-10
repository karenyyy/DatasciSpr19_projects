import json
from SPARQLWrapper import SPARQLWrapper, JSON


with open('/home/karen/workspace/cse597/nmt_preprocessing/en-2012/onto_classes_system.json', encoding="utf8") as f:
    onto_classes_system = json.load(f)


def dbpedia_query(query_string, resource=False):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(query_string)
    sparql.setReturnFormat(JSON)

    if resource:
        try:
            results = sparql.query().convert()

            ontology_url = "http://dbpedia.org/ontology/"
            onto_classes = [x["x"]["value"][28:]
                                for x in results["results"]["bindings"]
                                if x["x"]["value"].startswith(ontology_url)]

            bestPrecision = 0
            mostPreciseType = "UNK"

            for type in onto_classes:
                if type in onto_classes_system:
                    if onto_classes_system[type][0] > bestPrecision:
                        mostPreciseType = type
                        bestPrecision = onto_classes_system[type][0]

            return onto_classes_system[mostPreciseType][1].upper()

        except Exception:
            return "UNK"

    else:
        try:
            results = sparql.query().convert()
            if len(results["results"]["bindings"]) == 0:
                return "*"
            else:
                return results["results"]["bindings"][0]["x"]["value"][28:].upper()

        except Exception:
            return "*"


def get_resource_type(resource):
    query_string = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?x
        WHERE { <http://dbpedia.org/resource/""" + resource + """> rdf:type ?x .}
    """

    return dbpedia_query(query_string, resource=True)


if __name__ == '__main__':
    print(get_resource_type('Alan_LaVern_Bean'))

