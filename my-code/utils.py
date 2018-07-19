import pandas as pd
from snorkel.models import StableLabel
from snorkel.db_helpers import reload_annotator_labels
import codecs, glob
import pickle
from scipy.sparse import csr_matrix
import numpy as np


def get_raw_document_txt(doc_id):
    """Get document from the txt which was used to import in snorkel db (for debugging)"""
    fpath = glob.glob("/home/antonis/data/biocreative6/goldset/*/%s.txt"%doc_id)
    if len(fpath)==1:
        with codecs.open(fpath[0]) as f:
            v = f.read()
        return v
    return None

#####################
##### Snorkel  ######
#####################

def load_external_labels(session, candidate_class, tsv_path, annotator_name='gold', symmetric=False, reload = False, filter_label_split = False, debug=True):
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



def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    
    # from https://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:,mask]
    else:
        return mat


##########################################
#####       Classifier selection    ######
##########################################


def merge_pickles_pred_dicts(pickle_list, f1_threshold=0):
    """Helper function to merge results from different pickle files
    
    file_list: list of pickles to read files from
    f1_threshold: float: classifiers with f1(+) below that threshold are not included    
    """
    results_merged = dict()
    for fname in pickle_list:
        name = fname.split('/')[-1].split('.pkl')[0].strip(',').replace('bin_,minFreq=3,_stopw=english','') # pipeline identifier
        print name
        with open(fname,'rb') as f:
            results_dict = pickle.load(f)
    #     diversity_heatmap(results_dict, title = name)

        for model in results_dict.keys():
            print model
    #         print name+'_'+model
            if (results_dict[model]['f1+']>f1_threshold):
    #             if (model.endswith('LogisticRegression') or model.endswith('SVC_linear') or model.endswith('SVC_rbf_C=500') ) :
#                 if (model.endswith('Regression') or 1) :
#                     if ('LSA100' in name) or ('LSA' not in name):
                results_merged[name+'_'+model] = results_dict[model]

    print 'Merged %i different pipeline variations'%len(results_merged)
    return results_merged


##########################################
#####           Minor stuff         ######
##########################################

def keep_cands(df,ids_to_keep):
    """Keep only specific ids out of a dataframe (based on index)"""
    ids_to_drop = set(df.index) - set(ids_to_keep)
    return df.drop(ids_to_drop)

def check_class_imbalance(labels):
    """Helper function to check actual/predicted class imbalance"""
    labels = pd.Series(labels)
    return labels.value_counts()/float(len(labels))