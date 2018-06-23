# Helper functions for the pipeline (creating many ML classifiers & their predictions)
from matplotlib import pyplot as plt
import seaborn, pickle, re
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import cohen_kappa_score, precision_score, accuracy_score, f1_score

from io import StringIO
from imblearn.under_sampling import RandomUnderSampler



def report_to_df(report):
    report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)        
    return(report_df)

def train_evaluate(model, to_latex=False):
    model.fit(X_train_dev,y_train_dev)
#     print "Dev score:", accuracy_score(model.predict(X_dev), y_dev)
#     print "Test score:", accuracy_score(model.predict(X_test), y_test)
#     print classification_report(list(lr.predict(X_dev))+list(lr.predict(X_test)), list(y_dev)+list(y_test))
    y_pred = model.predict(X_test)
    report = report_to_df(classification_report(y_test, y_pred))
    print "confusion_matrix:\n",confusion_matrix(y_test, y_pred)
    
    if to_latex:
#         print report
        print "Latex table:\n"
        print """\\begin{table}[H]\centering"""+report.to_latex()+"""\\caption{Table label}
\end{table}"""
    
    return report


def sort_list_on(lst, like):
    """"Takes a list (lst) and a string (like), 
    sorts items containing that string first"""
    found=[]
    not_found=[]
    for mn in lst:
        if like in mn:
            found.append(mn)
        else:
            not_found.append(mn)
    return found+not_found



##########################################
######   Explore model diversity    ######
##########################################
def diversity_matrix(results_dict, diagonal_key = False , metric = cohen_kappa_score, evaluate_on = 'label_val_prob+'):
    """Compute a diversity matrix based on a specific metric
    
    diagonal_key: str: replace diagonal values with precomputed metric out of dict eg. f1+ (would otherwise be ==1)
    metric: pairwise comparison metric (eg. cohens_kappa, accuracy)
    evaluate_on: dictionary key to get the results to evaluate on.
    """
    
    model_names = sorted(results_dict.keys())
#     if return_model_names:
#         cohens_df = pd.DataFrame(index=model_names, columns=model_names)
#     else:
#         model_names = ["Model %i"%i for i in range(len(model_names))]
#         print model_names
    cohens_df = pd.DataFrame(index=model_names, columns=model_names)
    
    for i,model1 in enumerate(cohens_df.index):
        for j,model2 in enumerate(cohens_df.columns):
            if i==j:
                if diagonal_key==False:
                    cohens_df.iloc[i,j] = 1
                elif not diagonal_key:
                    cohens_df.iloc[i,j] = None
                else:
                    cohens_df.iloc[i,j] = results_dict[model1][diagonal_key]
            else:
                #assuming that it will get dicts of probabilities 
                true_list, pred_list = dict_to_list(results_dict[model1][evaluate_on],
                                 results_dict[model2][evaluate_on])
                true_list = np.round(true_list)
                pred_list = np.round(pred_list)
                cohens_df.iloc[i,j] = metric(true_list, pred_list)
                
    return cohens_df

def diversity_heatmap(results_dict, title=None, figsize = (10, 10), evaluate_on= 'label_val_prob+',sort_on=None, metric = cohen_kappa_score, diagonal_key = False, plot_model_names = True):
    """
    
    sort_on: str: if substring of model name, then put those models higher in heatmap (heuristic for visualization)
    evaluate_on: str: on which key of the classifier to evaluate on
    
    """
    plt.figure(figsize=figsize)
    #construct DF with Cohens kappa inter-annot agreement between trained classifiers
    model_names = sorted(results_dict.keys())
    if sort_on:
        model_names = sort_list_on(model_names, sort_on)
        
#     diversity_df = pd.DataFrame(index=model_names, columns=model_names)
    diversity_df = diversity_matrix(results_dict, metric=metric, evaluate_on=evaluate_on, diagonal_key = diagonal_key )
    if not plot_model_names:
        diversity_df.columns = ["Model %i"%i for i in range(len(diversity_df.columns))]
        diversity_df.index = ["Model %i"%i for i in range(len(diversity_df.columns))]
    
    mask = np.ones_like(diversity_df, dtype=np.bool)
    mask[np.tril_indices_from(mask)] = False
    
#     mask
    #to delete: computed by diversity_matrix()
#     for i,model1 in enumerate(diversity_df.index):
#         for j,model2 in enumerate(diversity_df.columns):
#             if i==j:
#                 diversity_df.iloc[i,j] = results_dict[model1]['f1+']
#             else:
#                 #assuming that it will get dicts of probabilities 
#                 true_list, pred_list = dict_to_list(results_dict[model1][evaluate_on],
#                                  results_dict[model2][evaluate_on])
#                 true_list = np.round(true_list)
#                 pred_list = np.round(pred_list)
#                 diversity_df.iloc[i,j] = metric(true_list, pred_list)
    print diversity_df.columns
    diversity_df = diversity_df[diversity_df.columns].astype(float)
    if title:
        plt.title(title)
    return seaborn.heatmap(diversity_df,mask = mask,vmin=0,vmax=1,annot=True,cmap='Blues')



def reduce_results_dict(results_dict, your_keys):
    """Reduces results_dict to selected models (your_keys)"""
    
    return {your_key: results_dict[your_key] for your_key in your_keys }


def classif_report_from_dicts(true_dict, pred_dict):
    ids = true_dict.keys()
    true_list = map(lambda x: true_dict[x],ids)
    pred_list = map(lambda x: pred_dict[x],ids)
    return classification_report(true_list, pred_list)

def dict_to_list(true_dict, pred_dict):
    """"Returns the values of two lists based on common keys (ensuring consistency)"""
    ids = true_dict.keys()
    true_list = map(lambda x: true_dict[x],ids)
    pred_list = map(lambda x: pred_dict[x],ids)
    return true_list, pred_list
    
    
    
def merge_pickles_pred_dicts(pickle_list, f1_threshold=0, list_substr=[], best_model=True):
    """Helper function to merge results from different pickle files
    
    file_list: list of pickles to read files from
    f1_threshold: float: classifiers with f1(+) below that threshold are not included
    list_str: list of strings for clearing up name of classifier. those strings will
    """
    results_merged = dict()
    for fname in pickle_list:
        #fix name
        name = fname.split('/')[-1].split('.pkl')[0]
        for s in list_substr:
            name=name.replace(s,'')
        name=name.strip(',')
        with open(fname,'rb') as f:
            results_dict = pickle.load(f)

        if best_model: #only picks the best model over LogisticRegression, SVM etc
            model = max(results_dict, key=lambda k: results_dict[k]['f1+'])
            if (results_dict[model]['f1+']>f1_threshold):
                results_merged[name+'_'+model] = results_dict[model]
        else: #returns all the models
            for model in results_dict.keys():
                if (results_dict[model]['f1+']>f1_threshold):
                    results_merged[name+'_'+model] = results_dict[model]

    print 'Merged %i different model variations'%len(results_merged)
    return results_merged


def F1_positive_class(tp,fp,tn,fn):
    prec = (1.0*tp/(tp+fp))
    rec = (1.0*tp/ (tp+fn))
    f1 = (2*prec*rec)/float(prec+rec)
    print "Precision (+):\t %.2f \nRecall (+):\t %.2f \nF1 score (+):\t %.2f \n"%(prec,rec, f1)
    


def balance_candidates(cands, marginals, rs = 42):
    """Balance and shuffle candidates along with their (prob) labels.
    
    cands: list of candidate objects
    marginals: list of marginal (or not) labels
    rs: int: random_state (used in RandomUnderSampler and sklearn.utils.shuffle) 
    """
    rus = RandomUnderSampler(random_state=rs,return_indices=True)
    marginals_01 = np.round(marginals)

    _,_, indices = rus.fit_sample(pd.DataFrame(marginals), np.round(marginals))

    # shuffle indices
    indices = shuffle(indices, random_state = rs)

    # keep only selected items
    cands_us = [cands[i] for i in indices]
    marginals_us = [marginals[i] for i in indices]
    
    return cands_us, np.array(marginals_us)


#############################################################################
#######   Helpers to convert to/from Snorkels -1,1 class label lists  #######
#############################################################################


def get_positive_logit(logit_array, positive_logit_position=1):
    """Helper to get a list of logits/probabilities for the positive class 
    (by default assuming that order of classes is [negative, positive])
    eg: Input: [[1, 0],
           [0,1],
           [0,1]]
           Output: [0,1,1]
    """
    return list(map(lambda x: x[positive_logit_position], logit_array))

def logits_to_neg_labels(logit_array):
    """Input: [[0.93, 0.07],
       [0,1],
       [0,1]]
       Output: [-1,1,1]
    """
    return list(map(lambda x: 1 if x[1]>=0.5 else -1, logit_array))

def logits_to_bin_labels(logit_array):
    """Input: [[1, 0],
           [0,1],
           [0,1]]
           Output: [0,1,1]
    """
    return list(map(lambda x: 1 if x[1]>=0.5 else 0, logit_array))


def bin_to_neg_labels(label_list):
    return [-1 if x==0 else 1 for x in label_list]

def neg_to_bin_labels(label_list):
    return [0 if x==-1 else 1 for x in label_list]

def classif_report_from_dicts(true_dict, pred_dict):
    ids = true_dict.keys()
    true_list = map(lambda x: true_dict[x],ids)
    pred_list = map(lambda x: pred_dict[x],ids)
    return classification_report(true_list, pred_list)



#####################################################
####                  Diagnostics              ######       
#####################################################

def plot_marginals_histogram(pred_marginals, true_labels=None, bins=11, title=None):
    """Plots a histogram with red and green bars based on whether the marginals were correctly/incorrectly assigned.
    Make sure true_labels is {0,1}
    """
    if not true_labels:
        plt.hist(pred_marginals, bins=bins)
        if title:
            plt.title(title)
        plt.show()
        return
        
    if np.array(true_labels).max()==1 and np.array(true_labels).min()==0 :
        true_marginals = []
        false_marginals = []
        for i in range(len(pred_marginals)):
            if abs(pred_marginals[i]-true_labels[i]) > 0.5: # then its falsely assigned from the GM
                false_marginals.append(pred_marginals[i])
            else:
                true_marginals.append(pred_marginals[i])
        
        #Calculate F1 score to add on title
        preds = np.round(pred_marginals)
        f1 = f1_score(true_labels, preds)*100
        
        plt.hist([true_marginals,false_marginals], bins=bins, range=[0,1], color=['green', 'red'])
        plt.title("Correctly vs incorrectly assigned marginals: %i/%i, F1: %.2f%%"%(len(true_marginals),len(false_marginals), f1))
        plt.show()
        return
    else:
        print "Incorrect label mapping, ensure true_labels are within {0,1}"
        return