import pandas as pd
from snorkel.models import StableLabel
from snorkel.db_helpers import reload_annotator_labels
import codecs, glob

def get_raw_document_txt(doc_id):
    """Get document from the txt which was used to import in snorkel db (for debugging)"""
    fpath = glob.glob("/home/antonis/data/biocreative6/goldset/*/%s.txt"%doc_id)
    if len(fpath)==1:
        with codecs.open(fpath[0]) as f:
            v = f.read()
        return v
    return None



def load_external_labels(session, candidate_class, tsv_path, annotator_name='gold', symmetric=False, reload = False):
    # FPATH = 'data/gold_labels.tsv'
    """
    Adapted from snorkel/tutorials/workshop/lib/load_external_annotations.py
    
    reload: Boolean:: Whether to reload annotations (perform mapping for splits 0,1,2)
    """
    gold_labels = pd.read_csv(tsv_path, sep="\t") # TODO: delete {DEBUG}
    for index, row in gold_labels.iterrows():

        # We check if the label already exists, in case this cell was already executed
        context_stable_ids = "~~".join([row['Chemical'], row['Gene']])
        query = session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_ids)
        query = query.filter(StableLabel.annotator_name == annotator_name)
        if query.count() == 0:
            session.add(StableLabel(
                context_stable_ids=context_stable_ids,
                annotator_name=annotator_name,
                value=row['label']))

    # If it's a symmetric relation, load both directions...
    if symmetric:
        for index, row in gold_labels.iterrows():    
            context_stable_ids = "~~".join([row['Gene'], row['Chemical']])
            query = session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_ids)
            query = query.filter(StableLabel.annotator_name == annotator_name)
            if query.count() == 0:
                session.add(StableLabel(
                    context_stable_ids=context_stable_ids,
                    annotator_name=annotator_name,
                    value=row['label']))

    # Commit session
    session.commit()

    # Reload annotator labels
    if reload:
        reload_annotator_labels(session, candidate_class, annotator_name, split=0, filter_label_split= filter_label_split, debug=debug)
        reload_annotator_labels(session, candidate_class, annotator_name, split=1, filter_label_split= filter_label_split, debug=debug)
        reload_annotator_labels(session, candidate_class, annotator_name, split=2, filter_label_split= filter_label_split, debug=debug)



#######################################################
### Load from pickle dictionary (on document level) ###
#######################################################
###  From snorkel/tutorials/cdr/load_external_annotations.py (v0.6.2)
# from six.moves.cPickle import load

# from snorkel.db_helpers import reload_annotator_labels
# from snorkel.models import StableLabel
# import bz2

# def load_external_labels(session, candidate_class, split, annotator='gold',
#     label_fname='data/cdr_relations_gold.pkl', id_fname='data/doc_ids.pkl'):
#     # Load document-level relation annotations
#     if label_fname.endswith('.bz2'):
#         with bz2.BZ2File(label_fname, 'rb') as f:
#             relations = load(f)
#     else:    
#         with open(label_fname, 'rb') as f:
#             relations = load(f)
#     # Get split candidates
#     candidates = session.query(candidate_class).filter(
#         candidate_class.split == split
#     ).all()
#     for c in candidates:
#         # Get the label by mapping document annotations to mentions
#         doc_relations = relations.get(c.get_parent().get_parent().name, set())
#         label = 2 * int(c.get_cids() in doc_relations) - 1        
#         # Get stable ids and check to see if label already exits
#         context_stable_ids = '~~'.join(x.get_stable_id() for x in c)
#         query = session.query(StableLabel).filter(
#             StableLabel.context_stable_ids == context_stable_ids
#         )
#         query = query.filter(StableLabel.annotator_name == annotator)
#         # If does not already exist, add label
#         if query.count() == 0:
#             session.add(StableLabel(
#                 context_stable_ids=context_stable_ids,
#                 annotator_name=annotator,
#                 value=label
#             ))

#     # Commit session
#     session.commit()

#     # Reload annotator labels
#     reload_annotator_labels(session, candidate_class, annotator, split=split, filter_label_split=False)