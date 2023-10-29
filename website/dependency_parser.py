"""
Utility functions for processing .conllu formatted dependency trees
"""

from conllu import parse

def get_text(sentence):
    return " ".join([token["form"] for token in sentence])

def read_file(conll_file):
    ## Read the CoNLL-U file
    with open(conll_file, "r", encoding="utf-8") as file:
        ## Parse the file
        data = file.read()
        sentences = parse(data)
    return sentences

def get_children(target_token, sentence):
    target_token_id = target_token["id"]
    return [token for token in sentence if token["head"] == (target_token_id)]

def get_subtree(target_token, sentence):
    if not sentence:
        return []

    target_token_id = target_token["id"]

    ## Find the target token in the sentence
    target_token = next((token for token in sentence if token["id"] == target_token_id), None)

    if target_token is None:
        return []

    ## Initialize a list to store the subtree tokens
    subtree_tokens = [target_token]

    ## Define a function to recursively collect the dependent tokens
    def collect_dependents(token):
        nonlocal subtree_tokens
        for token2 in sentence:
            if token2["head"] == token["id"]:
                subtree_tokens.append(token2)
                collect_dependents(token2)

    ## Collect the dependent tokens to build the subtree
    collect_dependents(target_token)

    ## Sort the subtree tokens by their "id" attribute to preserve the order
    subtree_tokens.sort(key=lambda token: int(token["id"]))

    return subtree_tokens

def get_text_dep(sentence):
    return " ".join([str((token["form"], token["upostag"])) for token in sentence])

def get_subject(verb, sentence):
    subject = None

    ## Find the subject of the verb by looking for a nominal subject (nsubj)
    for token2 in sentence:
        if token2["head"] == verb["id"] and token2["deprel"] == "nsubj":
            subject = token2
            break
    
    return subject

def get_subject_phrase(token, sentence):
    if token is None:
        return []
    
    return get_subtree(token, sentence)

def get_verb_phase(token, sentence):
    children = get_children(token, sentence)
    
    phase = [token]

    for child in children:
        if child["upostag"] == "AUX":
            phase.append(child)
        elif child["upostag"] == "PART":
            phase.append(child)
        elif child["upostag"] == "VERB" and child["deprel"] == "xcomp":
            phase += get_verb_phase(child, sentence)
    
    return sorted(phase, key=lambda token: token["id"])


def extract_subject_verb_pairs(sentence):
    if not sentence:
        return []

    ## Initialize lists to store verbs and their corresponding subjects
    verbs_with_subjects = []

    for token in sentence:
        ## Check if the token is a verb based on its upostag
        if token["upostag"] == "VERB":
            subject = get_subject(token, sentence)

            verb_phrase = get_verb_phase(token, sentence)
            ## Append the verb and its subject to the list
            if subject is not None:
                verbs_with_subjects.append((subject["form"].lower(), get_text(verb_phrase).lower()))

    # TODO: only keep the longest VP
    return verbs_with_subjects


if __name__ == "__main__":
    conll_file = "../Data/UD_English-EWT/en_ewt-ud-test.conllu"
    
    sentences = read_file(conll_file)

    
    for sentence in sentences[20:30]:
        result = extract_subject_verb_pairs(sentence)
        print(get_text(sentence))
        print(result)
    
    