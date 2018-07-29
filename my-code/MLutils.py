# Helper functions for the pipeline (creating many ML classifiers & their predictions)
from matplotlib import pyplot as plt
import seaborn, pickle, re
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import cohen_kappa_score, precision_score, accuracy_score, f1_score, recall_score, mean_squared_error, precision_recall_fscore_support
from io import StringIO
from imblearn.under_sampling import RandomUnderSampler

from scipy.sparse import csr_matrix,vstack
from random import sample

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from statsmodels.stats.weightstats import DescrStatsW


def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
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
    
    
def sample_from_csr(L_unlab, percentage):
    
    indices = [i for i in range(L_unlab.shape[0])]
    percentage = 1.-percentage
    indices_to_del = sample(indices, int((L_unlab.shape[0])*percentage))
    L_unlab_rs = delete_from_csr(L_unlab,indices_to_del)
    
    return L_unlab_rs


    

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
######     Classifier selection     ######
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


# Helpers for "naming the models in visualization"

def get_trims(x):
    if 'trim=0' in x:
        return 'trim=0'
    elif 'trim=5' in x:
        return 'trim=5'
    elif 'ShortDepPath' in x:
        return 'ShortDepPath'
    else:
        return 'no_trim'
    
def get_model(x):
    if 'SVC_rbf' in x:
        return 'SVC_rbf'
    elif 'SVC_linear' in x:
        return 'SVC_linear'
    elif 'RandomForestClassifier' in x:
        return 'RandomForestClassifier'
    elif 'LogisticRegression' in x:
        return 'LogisticRegression'
    elif 'CNN' in x:
        return 'CNN'
    elif 'LSTM' in x:
        return 'LSTM'
    else:
        return 'other'

def get_representation(x):
    if 'CV' in x:
        return "CountVectorizer"
    elif 'TfIdf' in x:
        return 'TfIdf'
    else:
        return 'other'



def visualize_classifiers(X_tsne, cat_var1 = 'model', cat_var2 = 'trimming' , style = 'ggplot'):
    """2-D Visualization with options for two categorical values
    
    X: should contain columns 'x' and 'y'
    cat_var1
    """
    # enrich df with categorical vars
    X_tsne['trimming'] = map(lambda x: get_trims(x),X_tsne.index)
    X_tsne['model'] = map(lambda x: get_model(x),X_tsne.index)
    X_tsne['representation'] = map(lambda x: get_representation(x),X_tsne.index)
    
    cat1i = np.unique(X_tsne[cat_var1])
    cat2i = np.unique(X_tsne[cat_var2])
    
    plt.figure(figsize=(10,10))
    plt.style.use(style)
    markers = ['.','D','_','+','x','.','H', '<','p','*','h','D','d','1','v','']
    colors = plt.cm.tab20.colors
    color_patches, marker_patches = [], []

    for j, cat2 in enumerate(cat2i):
        marker_patches.append(mlines.Line2D([], [], color='black', marker=markers[j], linestyle='None',
                          markersize=4, label=cat2) )

    for i, cat1 in enumerate(cat1i):
        color_patches.append(mpatches.Patch(color=colors[i], label=cat1))

        for j, cat2 in enumerate(cat2i):
            for x,y in  X_tsne[(X_tsne[cat_var1]==cat1) & (X_tsne[cat_var2]==cat2)][['x','y']].values:
                plt.scatter(x, y, 
                            color=colors[i],marker=markers[j])


    # plt.legend(handles=color_patches, ncol=2, loc=2, bbox_to_anchor=(1.05, 1),)
    # plt.legend(handles=marker_patches, ncol=2, loc=2, bbox_to_anchor=(1.05, 1),)



    legend1 = plt.legend(handles=color_patches, ncol=1, loc=2, bbox_to_anchor=(1.05, 1),)
    plt.legend(handles=marker_patches, ncol=1, loc=3, bbox_to_anchor=(1.05, 0),)
    plt.gca().add_artist(legend1)
    plt.title("Visualization of different classifiers based on their predictions")
    return plt

    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,ncol=2);


from scipy.spatial import distance
def centroid_selection(X, clusterer):
    """X should contain x,y columns with the tSNE projections
    clusterer 
    """
    picked_models = []
    clusterer.fit_predict(X[['x','y']])
    
    for center in clusterer.cluster_centers_:
        print center
        closest_index = distance.cdist(center.reshape(-1,2), X[['x','y']]).argmin()
        picked_models.append(X.index[closest_index])
    return picked_models


def pick_centroid_models(X, clusterer):
    """X is feature matrix for clustering.
    clusterer is cluster instance implementing .fit_predict and .cluster_centers_
    """
    picked_models = []
    try:
        clusterer.fit_predict(X)
    except:
        clusterer.fit_transform(X)
    
    for center in clusterer.cluster_centers_:
        closest_index = distance.cdist(center.reshape(-1,len(X)), X).argmin()
        picked_models.append(X.index[closest_index])
    return picked_models


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
    


def balance_candidates(cands, marginals, rs = 42, shuffle_cands=True, undersample=True):
    """Balance and shuffle candidates along with their (prob) labels.
    
    cands: list of candidate objects
    marginals: list of marginal (or not) labels
    rs: int: random_state (used in RandomUnderSampler and sklearn.utils.shuffle) 
    """
    rus = RandomUnderSampler(random_state=rs,return_indices=True)
    marginals_01 = np.round(marginals)

    if undersample:
        _,_, indices = rus.fit_sample(pd.DataFrame(marginals), np.round(marginals))
    else:
        indices = [i for i in range(len(cands))]
        
    if shuffle:
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
    return np.array(map(lambda x: x[positive_logit_position], logit_array))

def logits_to_neg_labels(logit_array):
    """Input: [[0.93, 0.07],
       [0,1],
       [0,1]]
       Output: [-1,1,1]
    """
    return np.array(map(lambda x: 1 if x[1]>=0.5 else -1, logit_array))

def logits_to_bin_labels(logit_array):
    """Input: [[1, 0],
           [0,1],
           [0,1]]
           Output: [0,1,1]
    """
    return np.array(map(lambda x: 1 if x[1]>=0.5 else 0, logit_array))


def prob_to_bin_labels(label_list):
    return np.array([0 if x<0.5 else 1 for x in label_list])

def bin_to_neg_labels(label_list):
    return np.array([-1 if x==0 else x for x in label_list])

def neg_to_bin_labels(label_list):
#     return [0 if x==-1 else x for x in label_list]
    return np.array([0 if x==-1 else x for x in label_list])


def classif_report_from_dicts(true_dict, pred_dict):
    ids = true_dict.keys()
    true_list = map(lambda x: true_dict[x],ids)
    pred_list = map(lambda x: pred_dict[x],ids)
    return classification_report(true_list, pred_list)



#####################################################
####                  Diagnostics              ######       
#####################################################

def plot_marginals_histogram(pred_marginals, true_labels=None, bins=11, title=None, save_name = None):
    """Plots a histogram with red and green bars based on whether the marginals were correctly/incorrectly assigned.
    Make sure true_labels is {0,1}
    """    
    if true_labels is None:
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
            elif abs(pred_marginals[i]-true_labels[i]) < 0.5:
                true_marginals.append(pred_marginals[i]) 
            else: #randomly assign in 50-50 disaggreement
                if bool(random.getrandbits(1)):
                    true_marginals.append(pred_marginals[i]) 
                else:
                    false_marginals.append(pred_marginals[i])

        # add legends
        red_patch = mpatches.Patch(color='red', label='Misclassified')
        green_patch = mpatches.Patch(color='green', label='Correctly classified')
        plt.legend(handles=[green_patch, red_patch])

        #Calculate F1 score to add on title
        preds = np.round(pred_marginals)
#         f1 = f1_score(true_labels, preds)*100
        
        plt.hist([true_marginals,false_marginals], bins=bins, range=[0,1], color=['green', 'red'])
        plt.title("Correctly vs incorrectly assigned marginals: %i/%i"%(len(true_marginals),len(false_marginals) ))
        if save_name is not None:
            plt.savefig(save_name)
        return 
    else:
        print "Incorrect label mapping, ensure true_labels are within {0,1}"
        return
    

def error_analysis(L_dev, L_gold_dev, gen_model=None, majority_voting = False, average_voting = False, save_name = None):
    """Given a vote matrix and the ground truth labels, 
    perform an error analysis on the dev set. 
    gen_model is provided for combining vote labels (denoising). 
    If not provided, performs majority voting & error analysis on that """
    
    if gen_model is not None:
        dev_marginals = gen_model.marginals(L_dev)
    elif majority_voting:
        dev_marginals = majority_vote(L_dev)
    elif average_voting:
        dev_marginals = average_vote(L_dev)
    else:
        raise ValueError("Have to provide a way for denoising/combining labels")

    if isinstance(L_gold_dev,csr_matrix):
        dev_gold_lbls = neg_to_bin_labels(list(map(lambda x: int(x[0].item()),L_gold_dev.todense())))
    else:
        dev_gold_lbls = neg_to_bin_labels(L_gold_dev)

    return plot_marginals_histogram(dev_marginals,dev_gold_lbls ,
                             title = 'Histogram of marginals in val set' ,
                            bins = 11,
                            save_name = save_name
                                   )
    
import matplotlib.pyplot as plt



#####################################################
####                  Ensembles                ######       
#####################################################


import math
from scipy import special
import random
import copy as cp


class BrownBoost:
    def __init__(self, base_estimator, c=10, convergence_criterion=0.0001, max_iter=10000):
        """ Initiates BrownBoost classifier
        
        Parameters
        ----------
        base_estimator: classifier from scikit-learn
            The base leaner in ensemble
        c: int or float
            A positive real value
            default = 10
        convergence_criterion: float
            A small constant(>0) used to avoid degenerate cases.
            default = 0.0001
        """
        self.base_estimator = base_estimator
        self.c = c
        self.max_iter = max_iter
        self.max_iter_newton_raphson = max_iter / 100
        self.convergence_criterion = convergence_criterion
        self.alphas = []
        self.hs = []
        self.ss = []

        
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"base_estimator": self.base_estimator, 
                "c": self.c, "convergence_criterion": self.convergence_criterion,
                "max_iter": self.max_iter,
               }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    

    def fit(self, X, y):
        """ Trains the classifier
        Parameters
        ----------
        X: ndarray
            The training instances
        y: ndarray
            The target values for The training instances
            
        returns
        --------
            self
        """

        # Initiate parameters
        self.__init__(base_estimator=self.base_estimator,
                      c=self.c,
                      max_iter=self.max_iter,
                      convergence_criterion=self.convergence_criterion)

        s = self.c
        r = np.zeros(X.shape[0])
        k = 0
        while s >= 0 and k < self.max_iter :
#             print(f'iter is {k}\ts = {s}')
            self.ss.append(s)
            k += 1
            w = np.exp(-(r + s)**2 / self.c)

            h = cp.deepcopy(self.base_estimator)
            h.fit(X, y, sample_weight=w)
            pred = h.predict(X)
            
            error = np.multiply(pred, y)
            gamma = np.dot(w, error)

            alpha, t = self.newton_raphson(r, error, s, gamma)
#             theta = (0.1/self.c)**2
#             A = 32 * math.sqrt(self.c*math.log(2/theta))
#             if t < gamma**2/A:
#                 (new_t * w).sum()
#                 t = new_t + gamma**2/A

            r += alpha * error
            s -= t

            self.alphas.append(alpha)
            self.hs.append(h)

    def predict(self, X):
        """ Classify the samples
        Parameters
        ----------
        X: ndarray
            The test instances
            
        Returns
        -------
        y: ndarray
            The pred with BrownBoost for the test instances
        """

        y = np.zeros(X.shape[0])
        for i in range(0, len(self.hs)):
            y += self.alphas[i] * self.hs[i].predict(X)
        return np.sign(y)

    def newton_raphson(self, r, error, s, gamma):
        """ Computes alpha and t
        Parameters
        ----------
        r: array
            margins for the instances
        error: ndarray
            error vec between pred and true instances
        s: float
            'time remaining'
        gamma: float
            correlation
        y: ndarray
            the target values
            
        Retruns
        -------
        alpha: float
        t: float
        """

        # Theorem 3 & 5
        alpha = min([0.1, gamma])
        t = (alpha**2) / 3

        a = r + s
        change_amount = self.convergence_criterion + 1
        k = 0

        while change_amount > self.convergence_criterion and k < self.max_iter_newton_raphson:
            d = a + alpha * error - t
            w = np.exp(-d**2 / self.c)

            # Coefficients for jacobian
            W = w.sum()
            U = (w * d * error).sum()
            B = (w * error).sum()
#             if abs(B) < 0.001:
#                 break
            V = (w * d * error**2).sum()
            E = (special.erf(d / math.sqrt(self.c)) - special.erf(a / math.sqrt(self.c))).sum()

            sqrt_pi_c = math.sqrt(math.pi * self.c)
            denominator = 2*(V*W - U*B)
            alpha_step = (self.c*W*B + sqrt_pi_c*U*E)/denominator
            t_step = (self.c*B*B + sqrt_pi_c*V*E)/denominator

            alpha += alpha_step
            t += t_step
            change_amount = math.sqrt(alpha_step**2 + t_step**2)
#             print(f'\t newton_raphson iter is {k}, {change_amount}')
            k += 1
        
        return alpha, t
    
    
class RobustBoost:
    def __init__(self, base_estimator, epsilon=0.25, theta=1.0, sigma=0.1, max_iter=10000):
        """ Initiates BrownBoost classifier

        Parameters
        ----------
        base_estimator: classifier from scikit-learn
            The base leaner in ensemble
        c: int or float
            A positive real value
            default = 10
        convergence_criterion: float
            A small constant(>0) used to avoid degenerate cases.
            default = 0.0001
        """
        self.base_estimator = base_estimator
        self.epsilon = epsilon
        self.theta = theta
        self.sigma = sigma
        self.max_iter = max_iter
        self.alphas = []
        self.hs = []
        self.ss = []
        self.rho = 0.
        
        
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"base_estimator": self.base_estimator, 
                "epsilon": self.epsilon, "theta": self.theta,
               "sigma": self.sigma, "max_iter": self.max_iter,               
               }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    
    def fit(self, X, y):
        """ Trains the classifier
        Parameters
        ----------
        X: ndarray
            The training instances
        y: ndarray
            The target values for The training instances

        returns
        --------
            self
        """

        # Initiate parameters
        self.__init__(base_estimator=self.base_estimator,
                      epsilon=self.epsilon,
                      theta=self.theta,
                      sigma=self.sigma,
                      max_iter=self.max_iter)

        m_t = 0.
        m_t_old = 0.
        m_last_ds = 0
        m_last_dt = 0

        # equation 8
        self.rho = self._calculate_rho()

        while t < 1:
            m_weights_old = cp.deepcopy(m_weights)
            m_t_old = m_t

            h = cp.deepcopy(self.base_estimator)
            h.fit(X, y, sample_weight=w)
            pred = h.predict(X)

            if 1 - m_t < 0.001:
                continue

            # prepare for using NewtonRaphson
            foundSolution = false
            mask = np.where(pred == 1, True, False)
            ns = NewronRaphsonSolver(self.m_t, mask, value)

            # 1. go as far in the future as possible
            init_dt = 1 - m_t
            init_ds = math.sqrt(init_dt)
            initial_points = []
            initial_points.append([init_ds, init_dt])

            # 2. alpha in adaboost
            m_w = [0., 0.]
            EPS = 1e-7
            totalWeight = m_w[0] + m_w[1]
            if (totalWeight == 0.0 or math.abs(totalWeight) < EPS or math.abs(m_w[0] - m_w[1]) < EPS):
                init_ds = 0
            else:
                init_ds = 0.5 * math.log((m_w[1] + 0.5) / (m_w[0] + 0.5))

            init_dt = init_ds ** 2
            initial_points.append([init_ds, init_dt])

            # 3. most recently used
            init_ds = m_ds_last
            init_dt = m_dt_last
            initial_points.append([init_ds, init_dt])

    def _calculate_rho(self):
        """Calculate rho
        Returns
        -------
        rho: [float, float]
        """
        f1 = math.sqrt(np.exp(2.) * ((self.sigma**2 + 1.) - 1.))
        f2 = special.erfinv(1. - self.epsilon)
        numerator = f1*f2 + np.e * self.theta
        denominator = 2.*(np.e - 1.)
        return numerator/denominator
    
    def _calculate_weight(self, cost, m, t):
        mu = self._calculate_mu(self.rho, t)
        sigma_sq = self._calculate_sigma_square(t)
        if m > mu:
            return cost*np.exp(-(m - mu)**2 / sigma_sq)
        else:
            return 0.0
        
    def _calculate_sigma_square(self, t):
        if t > 1:
            return self.sigma**2
        else:
            return (self.sigma**2 + 1.) * np.exp(2. * (1. - t)) - 1.
    
    def _calculate_mu(self, t):
        if t > 1:
            return self.sigma
        else:
            return (self.theta - 2*self.rho) * np.exp(1. - t) + 2*self.rho


def average_vote(L):
    avg_vote_labels = np.array(list(map(lambda x:x[0].item(),L.mean(axis=1))))
    avg_vote_labels = (avg_vote_labels/2)+0.5 # convert to {0,1}
    avg_vote_labels = np.clip(avg_vote_labels,0,1) # rounding errors
    return avg_vote_labels

def majority_vote(L):
    '''Compute majority vote given a Label matrix L'''
    avg_vote_labels = average_vote(L)
    maj_vote_labels = np.array(list(map(lambda x: 1 if x>0.5 else 0 if x<0.5 else x , avg_vote_labels)))
    return maj_vote_labels
    
#     pred = L.sum(axis=1)
#     pred[(pred > 0).nonzero()[0]] = 1
#     pred[(pred < 0).nonzero()[0]] = 0
#     pred = np.array(list(map(lambda x: x[0].item(), pred)))
#     return pred

def majority_vote_score(L, gold_labels):
    """Computes the majority vote score of label matrix L.
    Resolves 50-50 conflicts as negative labels.
    
    Prints scores & returns (P,R,F) tuple"""
    y_pred = np.round(np.ravel(majority_vote(L)))
    y_true = neg_to_bin_labels(gold_labels.toarray().reshape(1,-1)[0])
    y_true = [1 if y_true[i] == 1 else 0 for i in range(y_true.shape[0])]
    
    ### TODO: change score to the same used by snorkel:
    # ========================================
    # Scores (Un-adjusted)
    # ========================================
    # Pos. class accuracy: 0.708
    # Neg. class accuracy: 0.866
    # Precision            0.597
    # Recall               0.708
    # F1                   0.648
    # ----------------------------------------
    # TP: 503 | FP: 339 | TN: 2193 | FN: 207
    # ========================================
    
    pos,neg = y_true.count(1),float(y_true.count(0))
#     print "pos/neg    {:d}:{:d} {:.1f}%/{:.1f}%".format(int(pos), int(neg), pos/(pos+neg)*100, neg/(pos+neg)*100)
#     print "precision  {:.2f}".format( 100 * precision_score(y_true, y_pred) )
#     print "recall     {:.2f}".format( 100 * recall_score(y_true, y_pred) )
#     print "f1         {:.2f}".format( 100 * f1_score(y_true, y_pred) )
    
    #print "accuracy  {:.2f}".format( 100 * accuracy_score(y_true, y_pred) 
#     return "Precision\t{:.2f}%\nRecall\t{:.2f}%\nF1\t{:.2f}%".format( 100 * precision_score(y_true, y_pred),  100 * recall_score(y_true, y_pred) ,  100 * f1_score(y_true, y_pred)  )
    return (precision_score(y_true, y_pred),  recall_score(y_true, y_pred) ,  f1_score(y_true, y_pred)  )



#####################################################
######            Learning curves              ######       
#####################################################


def get_lower_upper_CI(scores):
    if scores.shape[1]>1: #then 2-D
        lower_bound, upper_bound = DescrStatsW(scores.T).tconfint_mean() - scores.mean(axis=1)
    else: 
        lower_bound, upper_bound = 0, 0
    return abs(lower_bound)



def plot_learning_curve(train_scores, 
                        valid_scores,
                        train_sizes,
                        title = None,
                        ylim = None,
                       epochs = None,
                       xlabel = None, ylabel = None
                       ):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    if title:
        plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel("Training examples")
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    def scale_epochs(epochs):
        epochs1 = np.array(epochs)
        return np.interp(epochs1, (epochs1.min(), epochs1.max()), (10, 50))
    
    if epochs is not None:
        epochs_scaled = scale_epochs(epochs)
        
    else: 
        epochs_scaled = None
    #change this with t-test
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_CI = get_lower_upper_CI(train_scores)
    
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_CI = get_lower_upper_CI(valid_scores)
    
    plt.grid()
    plt.plot(train_sizes, train_scores_mean - 0,color="r") #draw line
    plt.fill_between(train_sizes, train_scores_mean - train_scores_CI,
                     train_scores_mean + train_scores_CI, alpha=0.2,
                     color="r")
    plt.plot(train_sizes, train_scores_mean, '-', color="r",
             label="Train score")
    plt.scatter(train_sizes, train_scores_mean, marker = 'o', color="r",
             s=epochs_scaled)
    
    plt.fill_between(train_sizes, valid_scores_mean - valid_scores_CI,
                     valid_scores_mean + valid_scores_CI, alpha=0.2, color="g")
    
    plt.plot(train_sizes, valid_scores_mean, '-', color="g",
             label="Test score")
    plt.scatter(train_sizes, valid_scores_mean, marker='o', color="g",
             s=epochs_scaled)

    legend0 = plt.legend(loc='lower right')
    plt.ylim(0,1)
    
    legend2 = plt.legend([mlines.Line2D([],[],color='g',alpha=3*0.2 )] ,
                         ["95% CI"] ,
                         loc='upper right')
    
    if epochs is not None:
        patches = []
        for i,legend_size in enumerate(np.sort(np.unique(epochs))):
            patches.append( plt.scatter([],[], s=legend_size, marker='o', color='#555555'))

        legend1 = plt.legend(patches,
               ['%i epochs'%epoch for epoch in np.unique(np.sort(epochs))],
               scatterpoints=1,
               loc='lower left',
               ncol=2,
               fontsize=8)
        
    plt.gca().add_artist(legend0)
    plt.gca().add_artist(legend2)

    return plt


from sklearn.model_selection import KFold, ShuffleSplit
from copy import deepcopy


def calculate_predictions(clf, X):
    """Shortcut for calculating predictions and their probabilities for sklearn & snorkel RNN"""
    try:
        pred = clf.predict(X)
        positive_position = clf.classes_.argmax()
        pred_marg = get_positive_logit(clf.predict_proba(X), positive_position)
    except:
        pred = clf.predictions(X, batch_size=1024)
        pred_marg = clf.marginals(X, batch_size=1024)
    return pred,pred_marg


def labels_to_array(L_gold):
    """Converts label lists & sparse matrices to np arrays """
    if isinstance(L_gold,csr_matrix):
        return np.array(list(map(lambda x: int(x[0].item()),L_gold.todense())))
    else:
        return np.array(L_gold)

def fix_lc_labels(lbls):
    return np.array(neg_to_bin_labels(labels_to_array(lbls))).astype(float) 

def custom_learning_curve(clf, X_increm, y_increm, X_val, y_val, 
                          X_init = None, y_init=None,
                          X_test=None, y_test=None, cv_splits=3, 
                          splt_sizes = np.array([ 0.1, 0.33, 0.55, 0.78, 1.]),
                          fit_params = {} ):
    """Custom function for returning learning curve scores of different classifiers. 
    Learning curve starts from initial training set (X_init) making sure it is always
    included in the training process and then samples incrementally over X_increm.
    Evaluates loss & F1 on X_val, y_val and (optionally) returns P,R,F1 on test set (X_test)
    
    clf: estimator,implementing 'fit/train' and 'predict/predictions' methods
    X_init: the training dataset on beginning of the learning curve (spot 0)
    X_increm: the additional examples for plotting incremental learning curves
    
    returns: train_sizes, train_mse, train_f1, valid_mse , valid_f1 , (P, R, F1) @ test set
    altered with new clf initialization every time
    """
    # X_splt: subset of the training dataset
    # X_merged: X_init + X_splt | full training dataset to perform training on that iteration

    
    #init
    train_mse = np.empty((0,1), float)
    train_f1 = np.empty((0,1), float)
    valid_mse = np.empty((0,cv_splits), float)
    valid_f1 = np.empty((0,cv_splits), float)
    
    #set probability estimates to true
    try:
        clf.probability = True
    except:
        print 'Cannot set probability estimates to True'
        pass
    
    # copy clf so I can re-initialize with random weights
    clf_init = deepcopy(clf)
    
    # TODO: split y_increm, y_increm_marginals ||| OR do smth with train test split
    # TODO: use shufflesplit instead???
    
    #shuffle 
    X_increm, y_increm = shuffle(X_increm, y_increm)
        
    #ensure labels are {0,1}    
    y_init = fix_lc_labels(y_init)
    y_val = fix_lc_labels(y_val)
    y_test = fix_lc_labels(y_test)
    
#     y_val, y_test = np.array(neg_to_bin_labels(y_val)).astype(float) , np.array(neg_to_bin_labels(y_test)).astype(float)
    print np.unique(np.concatenate(( y_init,y_increm, y_val, y_test)))
    
    kf = KFold(n_splits=3, random_state=42, shuffle=True) # KFold splitter for CV in valid set
    
    if X_init is not None:
        X_init, y_init = shuffle(X_init, y_init)
        if y_init is None:
            raise ValueError("Must also pass labels for the initial training set")
        splt_sizes[0] = 0  # force training to start from ONLY gold set
        train_sizes = len(y_init)+(splt_sizes)*len(y_increm)

    for splt_size in splt_sizes:
        
        clf = deepcopy(clf_init) #reload clf (re-initialize weights of lstm etc)
        
        #hack because sklearn complains if test size == 0
        if splt_size == 1:
            train_size = len(y_increm)-2
        elif splt_size ==0:
            train_size = 2
        else: 
            train_size = splt_size
            
        splitter = ShuffleSplit(n_splits=1, test_size=None, train_size=train_size, random_state=42)
        ind = list(splitter.split(X_increm))[0][0]
        print type(ind), type(X_increm)
        if isinstance(X_increm,list):
            X_splt = [X_increm[i] for i in ind]
        else:
            X_splt = X_increm[ind]
        y_splt = y_increm[ind]
        
        
        if X_init is not None: #augment X_init with X_splt
            if isinstance(X_init,list):
                X_merged = X_init+X_splt
            else:
                X_merged = vstack((X_init, X_splt))
            y_merged = np.concatenate((y_init,y_splt))
        else: #perform training only with X_merged 
            X_merged = deepcopy(X_splt), deepcopy(y_splt) 

        # and shuffle
        X_merged,y_merged = shuffle(X_merged,y_merged, random_state=42)
        y_merged_bin = np.array(prob_to_bin_labels(y_merged)).astype(int)
        
        # training
        try:
            
            clf.fit(X_merged,y_merged_bin)
        except:
            clf.train(X_merged,y_merged, **fit_params)
        
        # calc training errors
        pred,pred_marg = calculate_predictions(clf, X_merged)
        pred = np.array(neg_to_bin_labels(pred)).astype(int) # Don't think thats needed : TODO: clean-up
        
        train_mse = np.vstack([train_mse, -mean_squared_error(y_merged,pred_marg)])
        train_f1 = np.vstack([train_f1, f1_score(y_merged_bin,pred)])
        
        #compute CV sets on validation to plot variance too
        mse,f1 = [],[]
        for train_index, test_index in kf.split(X_test):
            if isinstance(X_test, list):
                X = [X_test[i] for i in test_index]
            else:
                X = X_test[test_index]
            y = y_test[test_index]
            
            pred, pred_marg = calculate_predictions(clf, X)
            pred = neg_to_bin_labels(pred)
            
            mse.append(-mean_squared_error(y,pred_marg))
            f1.append(f1_score(y,pred))
            print "MSE:", mse
            print "F1:", f1
        mse = np.array(mse)
        f1 = np.array(f1)
        valid_mse = np.vstack([valid_mse, mse])
        valid_f1 = np.vstack([valid_f1, f1])


        #compute PRF on test set
        if X_test!=None:
            pred, pred_marg = calculate_predictions(clf, X_test)
            pred = neg_to_bin_labels(pred)
            p,r,f1,_ = precision_recall_fscore_support(y_test, pred, average = 'binary')
            test_prf1 = (p,r,f1)
        else:
            test_prf1 = (None,None,None)
        
    train_sizes = np.array([int(i) for i in train_sizes])
    
    return train_sizes, train_mse, train_f1, valid_mse , valid_f1 , test_prf1