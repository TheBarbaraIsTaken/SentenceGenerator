"""
Utility functions for processing .conllu formatted dependency trees
"""

from conllu import parse

def get_text(sentence):
    return " ".join([token["form"] for token in sentence]).lower()

def get_text_dep(sentence):
    return " ".join([str((token["form"], token["deprel"])) for token in sentence])

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

def get_subject(verb, sentence):
    subject = None

    ## Find the subject of the verb by looking for a nominal subject (nsubj)
    for token in sentence:
        if token["head"] == verb["id"] and token["deprel"] == "nsubj":
            subject = token["form"].lower()
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

def get_object(verb, sentence):
    obj = None
    for token in sentence:
        if token['head'] == verb['id'] and token['deprel'] in {'obj', 'iobj'}:
            obj = token['form'].lower()
            break
    
    return obj

def get_object_phrase(token, sentence):
    if token is None:
        return []
    
    return get_subtree(token, sentence)

def extract_subject_verb_pairs(sentence):
    if not sentence:
        return []

    ## Initialize lists to store verbs and their corresponding subjects
    subjcet_verb_pairs = []

    for token in sentence:
        ## Check if the token is a verb based on its upostag
        if token["upostag"] == "VERB":
            subject = get_subject(token, sentence)
            verb_phrase = get_verb_phase(token, sentence)

            ## Append the verb and its subject to the list
            if subject is not None:
                subjcet_verb_pairs.append((subject, get_text(verb_phrase)))

    return subjcet_verb_pairs


def extract_verb_object_pairs(sentence):    
    ## Initialize an empty list to store the verb-object pairs
    verb_object_pairs = []

    for token in sentence:
        ## Check if the token is a verb based on its upostag
        if token['upos'] == 'VERB':
            ## Find the corresponding object (direct or indirect) of the verb
            verb_phrase = get_verb_phase(token, sentence)
            object = get_object(token, sentence)

            ## Append the verb and its object to the list
            ## Object might be None
            verb_object_pairs.append((get_text(verb_phrase), object))

    # Merge - longer VP with the object of shorter VP
    for i, (verb_i, obj_i) in enumerate(verb_object_pairs):
        for j, (verb_j, obj_j) in enumerate(verb_object_pairs):
            if i != j and verb_i in verb_j:
                if obj_i is not None and obj_j is None:
                    verb_object_pairs[j] = (verb_j, obj_i)
                
                verb_object_pairs.pop(i)

    return verb_object_pairs



if __name__ == "__main__":
    conll_file = "../Data/UD_English-EWT/en_ewt-ud-test.conllu"
    
    sentences = read_file(conll_file)

    
    for sentence in sentences[10:20]:
        result = extract_verb_object_pairs(sentence)
        print(get_text(sentence))
        print(result)
        print()
    
    