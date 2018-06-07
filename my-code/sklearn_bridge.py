from copy import deepcopy
from itertools import groupby
from operator import itemgetter
import pandas as pd
from collections import defaultdict
import spacy
import networkx as nx
from networkx import shortest_path
import spacy




def shift_idx(idx2shift, indices2delete):
    """Shifts the indices of the entities appropriately to keep consistency
    after deleting some elements from the token list.
    
    idx2shift: int
    indices2delete: list of integers
    """
    count = reduce(lambda count, i2del: count + (idx2shift > i2del), indices2delete, 0)
    return idx2shift - count



# #### shortest dep path
# def childIterator(root, level = 0, parent = "", debug=False, g=nx.Graph()):      
# #     global g
#     if debug:
#         print "\t" * level, "-", parent
# #     c+= 1
#     if parent:
#         g.add_edge(parent.norm_, root.norm_)
#     for child in root.children:
#         g = childIterator(child, level + 1, root, g=g)
#     return g

#### shortest dep path
def childIterator(root, level = 0, parent = "", debug=False, g=nx.Graph()):      
#     global g
    if debug:
        print "\t" * level, "-", parent
#     c+= 1
    if parent:
        g.add_edge(parent.i, root.i)
    for child in root.children:
        g = childIterator(child, level + 1, root, g=g)
    return g

def get_shortest_dep_path(sentence, T1, T2, nlp ):
    """Get only a subpart of sentence, connecting the words in two 
    using spacy and networkx"""
    
    #parse in spaCy
    doc = nlp(sentence)
#     print '\n',sentence
    if (doc[T1].norm_!=u'ENTITY1') or (doc[T2].norm_!=u'ENTITY2'):
        
#         print 'wrong indices. searching sentence for ENTITY1 and ENTITY2'
        T1,T2 = None,None
        for token in doc:
            if token.text == 'ENTITY1':
                T1 = token.i
            elif token.text == 'ENTITY2':
                T2 = token.i
        
        if (not T1) or (not T2):#if still None, return whole sentence
            return sentence
    
    
    root_idx = doc.get_lca_matrix()[T1,T2]
    # delete prev graph
    g = nx.Graph()
    g = childIterator(doc[root_idx], level=0, g=g)
#     nx.draw_networkx(g)

    # idk wtf is going on here, for some reason keys T1 and T2 do not exist sometimes.
#     print shortest_path(g)[T1][T2]
    try:
        short_sent = [doc[x].norm_ for x in  shortest_path(g)[T1][T2]]
    except:
        short_sent = sentence
    return ' '.join(short_sent)



def recreate_text_representation(candidate, entity_replacement = True, span_replacement = True, as_tokens=False , replace_conseq_entities = False, trim_text = False, window=0, lemmas = False, shortest_dep_path=False, nlp = None
                                ):
    """Re-generate text from snorkel dict format
    
    value: dict of a certain candidate (containing keys: words, entity_types, lemmas)
    span_replacement: Boolean: Whether to replace the whole spans of entity candidates with ENTITY1 and ENTITY2
    T_type_replacement: Boolean: Whether to replace all of the found entity types with their names (eg. CH2-> CHEMICAL)
    trim_text: Boolean: Get the text only between the two entities
    window: int: Window to trim text before/after entities (only valid if trim_text==True)
    
    shortest_dep_path: Boolean: If True, return only the shortest dependency path (required to also pass nlp = spacy.load('en') )
    nlp: the spaCy model for dependency parsing
    
    """
    value = deepcopy(candidate)
    if lemmas and not shortest_dep_path:
        tokens = deepcopy(value['lemmas'])
    else:
        tokens = deepcopy(value['words'])
#     print '\n','Initial indices:',value['gene_idx'],value['chem_idx']
    if entity_replacement:
        for i, t_type in enumerate(value['entity_types']):
            if t_type !=u'O':
                tokens[i] = str(t_type).upper()
    
    # if only_between, drop words before/after chem+gene
    if trim_text and not shortest_dep_path:
        start = min(value['gene_idx']+value['chem_idx'])
        start = max(0, start-window)
        
        stop = max(value['gene_idx']+value['chem_idx'])
        stop = stop + window
        
#         print 'sdas', start,stop
        
        keys_with_lists = ['entity_types', 'lemmas', 'pos_tags', 'words']
        
        #trim all lists containing lemmas etc.
        # TODO: fix issue that window also counts special characters.
        for key in keys_with_lists: 
            value[key] = value[key][start:stop+1]
        tokens = tokens[start:stop+1]

        #replace indices to keep consistency 
        value['chem_idx'][:] = [x - start for x in value['chem_idx']]
        value['gene_idx'][:] = [x - start for x in value['gene_idx']]
        
#         print 'new value indices:',value['chem_idx'],value['gene_idx']
        
#     print tokens
    #TODO: potentially replace with lemmas here maybe
    
    #replace span with ENTITY1, ENTITY2 --- destruction of list consistency after this point
    if span_replacement or shortest_dep_path:
        for idx in value['chem_idx']:
            tokens[idx] = "ENTITY1"
        for idx in value['gene_idx']:
            tokens[idx] = "ENTITY2"
            
        #convert whole span to 1 token
        idx_to_del = value['gene_idx'][1:]+value['chem_idx'][1:]
        # keep track of entity indices (merged & converted from list to int)
        gene_idx = shift_idx(value['gene_idx'][0], idx_to_del)
        chem_idx = shift_idx(value['chem_idx'][0], idx_to_del)
        for index in sorted(idx_to_del, reverse=True):
            del tokens[index]
            
    if shortest_dep_path:
        return get_shortest_dep_path(' '.join(tokens) , chem_idx, gene_idx, nlp)
        
        
    if replace_conseq_entities and not shortest_dep_path: 
        tokens = map(itemgetter(0), groupby(tokens))
        
#         #shit after that
#         if return_idcs:
#             chem_idx = deepcopy(value['chem_idx'])
#             gene_idx = deepcopy(value['gene_idx'])
            
#             if (len(chem_idx)>1) and (gene_idx[0]<chem_idx[0]):
#                 print ' '.join(tokens)
#                 print 'l1',chem_idx
#                 chem_idx = chem_idx[0] - len(gene_idx) - 1
#                 print 'l2',tokens[chem_idx], chem_idx
                
#             elif (len(gene_idx)>1) and (chem_idx[0]<gene_idx[0]):
#                 print ' '.join(tokens)
#                 gene_idx = gene_idx[0] - len(chem_idx) - 1
#                 print 'l3',gene_idx, tokens[gene_idx]
            
            
#     print tokens
    if as_tokens:
        return tokens
    elif shortest_dep_path:
        return (' '.join(tokens), chem_idx, gene_idx, 
                #tokens[chem_idx], tokens[gene_idx] #debug/test
               )
    else:
        return ' '.join(tokens)
    

def candidate_dict_to_df(candidate_dict, entity_replacement = True, span_replacement = True, as_tokens=False , replace_conseq_entities = False , trim_text = False, window=0, lemmas = False, shortest_dep_path = False):
    """Function to convert the snorkel candidate dict into a dataframe, 
    containing the text representation (with options to modify entity replacement etc)
    
    cand_dict: dictionary containing all required candidate elements from snorkel.
                ** requires inner split dictionary (eg cand_dict = candidate_dictionary[split_number])
    
    entity_replacement,
    span_replacement,
    as_tokens,
    replace_conseq_entities: arguments regarding recreating the string (check recreate_text_representation doc function)
    
    """
    if shortest_dep_path:
        nlp =  spacy.load('en')
    else:
        nlp=None
    df = pd.DataFrame(map(lambda x: recreate_text_representation(x,  entity_replacement = entity_replacement, span_replacement = span_replacement, as_tokens=as_tokens , replace_conseq_entities = replace_conseq_entities, trim_text = trim_text, window=window, lemmas = lemmas, shortest_dep_path = shortest_dep_path, nlp = nlp), 
                          candidate_dict.values()), 
                      columns = ['text'],
                      index = map(lambda x: x['cand_id'], candidate_dict.values())
                     )
    df['label'] = map(lambda x: x['label'], candidate_dict.values())
    df['doc_id'] = map(lambda x: x['doc_id'], candidate_dict.values())
    df['sent_id'] = map(lambda x: x['sent_id'], candidate_dict.values())
    return df



def export_snorkel_candidates(session, REGULATOR, split_nr, include_unlabelled):
    # TODO: write doc (args)
    """return a dictionary of docids -> cand_ids -> candidate features
    
    
    candidates = {doc_id1:
                 {cand_id312: candidate_object,
                 cand_id3124: candidate_object,
                 },
                  doc_id2:
                     {cand_id3124: candidate_object,
                     cand_id2312: candidate_object,
                     }
             }
    """
    candidates = defaultdict(lambda: defaultdict())
    for c in session.query(REGULATOR).filter(REGULATOR.split == split_nr):
        if (c.gold_labels) or include_unlabelled:

            
            features = {
                'label': c.gold_labels[0].value if (c.gold_labels) else None,
                'cand_id' : c.id,
                'sent_id' : c.get_parent().id,
                'doc_id' : c.get_parent().get_parent().name,
                'chem_idx' : [x for x in range(c.Chemical.get_word_start(),c.Chemical.get_word_end()+1)],
                'gene_idx' : [x for x in range(c.Gene.get_word_start(),c.Gene.get_word_end()+1)],
                'words' : c.get_parent().words,
                'entity_types' : c.get_parent().entity_types,
                'lemmas' : c.get_parent().lemmas,
                'pos_tags' : c.get_parent().pos_tags,
            }
            candidates[c.id] = features
            #
        
    return dict(candidates)


