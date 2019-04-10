
def get_relex_content(nmt_result_path):
    with open(nmt_result_path, encoding="utf8") as nmt_result:
        results = nmt_result.readlines()
    return results


def get_relexed_dict(relex_path):
    relex_result = []
    
    with open(relex_path, encoding="utf8") as relex_file:
        lines = relex_file.readlines()
    
        for line in lines:
            relex_entities = line.split('\t')
    
            relexed_entity_dict = {}
    
            for (k, e) in enumerate(relex_entities):
                relexed_entity_dict['ENTITY_' + str(k + 1)] = e.replace('"', '').replace('\n', ' ')

            relex_result.append(relexed_entity_dict)
        return relex_result


def get_relexed_result(nmt_result_path, relex_path, output_path):
    final_results = []
    results = get_relex_content(nmt_result_path)
    relex_results = get_relexed_dict(relex_path)

    for (res, relex_res) in zip(results, relex_results):
        tmp = res
        for (id, lex) in relex_res.items():
            tmp = tmp.replace(id, lex)

        final_results.append(tmp)

    with open(output_path, 'a+', encoding="utf8") as output_file:
        output_file.write(''.join(final_results))
