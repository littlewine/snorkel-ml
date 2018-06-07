# Helper functions for creating (many) ML classifiers
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from io import StringIO
import re
from sklearn.metrics import cohen_kappa_score, precision_score, accuracy_score
from matplotlib import pyplot as plt
import seaborn
import numpy as np
import pickle

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

def diversity_matrix(results_dict, diagonal = False , metric = cohen_kappa_score, evaluate_on = 'label_val_binary'):
    """Compute a diversity matrix based on a specific metric"""
    
    model_names = sorted(results_dict.keys())
        
    cohens_df = pd.DataFrame(index=model_names, columns=model_names)
    
    for i,model1 in enumerate(cohens_df.index):
        for j,model2 in enumerate(cohens_df.columns):
            if (i==j) and diagonal:
                cohens_df.iloc[i,j] = results_dict[model1]['f1+']
            else:
                cohens_df.iloc[i,j] = metric(results_dict[model1][evaluate_on],results_dict[model2][evaluate_on])
    
    return cohens_df

 

def diversity_heatmap(results_dict, title=None, figsize = (10, 10), evaluate_on= 'label_val_binary',sort_on=None, metric = cohen_kappa_score):
    """
    
    sort_on: str: if substring of model name, then put those models higher in heatmap (heuristic for visualization)
    evaluate_on: str: on which key of the classifier to evaluate on
    
    """
    plt.figure(figsize=figsize)
    #construct DF with Cohens kappa inter-annot agreement between trained classifiers
    model_names = sorted(results_dict.keys())
    if sort_on:
        model_names = sort_list_on(model_names, sort_on)
        
    cohens_df = pd.DataFrame(index=model_names, columns=model_names)
    mask = np.ones_like(cohens_df, dtype=np.bool)
    mask[np.tril_indices_from(mask)] = False
#     mask
    for i,model1 in enumerate(cohens_df.index):
        for j,model2 in enumerate(cohens_df.columns):
            if i==j:
                cohens_df.iloc[i,j] = results_dict[model1]['f1+']
            else:
                cohens_df.iloc[i,j] = metric(results_dict[model1][evaluate_on],results_dict[model2][evaluate_on])

    cohens_df = cohens_df[cohens_df.columns].astype(float)
    if title:
        plt.title(title)
    return seaborn.heatmap(cohens_df,mask = mask,vmin=0,vmax=1,annot=True,cmap='Blues')

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