from collections import defaultdict
import pandas as pd
import shutil
import time, codecs, re


def copyDirectory(src, dest):
    
    """
    Copies a whole directory to another path.
    Gives error if directory exists already.
    
    Usage: 
    copyDirectory('/home/antonis/data/biocreative6/corpus/',
             '/home/antonis/data/biocreative6/goldset/')
    
    """
    try:
        shutil.copytree(src, dest)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)
        


def BioC_corpus_parse(load_path, save_path ):
    '''Splits a tsv containing multiple docs into different txt files (one per doc).
    
    load_path: Location of tsv file
    save_path: Location to save multiple txt files
    
    Input data row format:
    doc_id    title    abstract
    
    Output data format:
    title    abstract
    '''
    
    save_path = save_path.rstrip("/")#trim trailing /
    
    # Splitting docid - abstract+title in same file
    with open(load_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            doc_id, text = line.split('\t',1)
            save_path_file = '%s/%s.txt'%(save_path,doc_id)
            with open(save_path_file, 'w') as f:
                f.write(text)
        print("Successfully wrote %i docs to txt (%s)"%(i,save_path))

        
def BioC_ent_parse(path_entities, mapping=None):
    '''Input: tsv BioCreative format
    mapping: dict for renaming entities to desired format (compatible with NER)
            eg: {'GENE-X': 'Gene',
                'CHEMICAL': 'Chemical'}
    '''
    
    entities = pd.read_table(path_entities,names=['doc_id','idx', 'type', 'start', 'stop', 'name'])
    #hack to write type, start stop without \t
    if mapping:
        entities['type'] = map(lambda x: mapping[x], entities.type) #rename 
    entities['type_start_stop'] = map(lambda x,y,z: "%s %i %i"%(x,y,z) , entities.type, entities.start, entities.stop)
    return entities

def BioC_rel_parse(path_relationships):
    '''Input: tsv BioCreative format'''
    
    relationships = pd.read_table(path_relationships, names = ['doc_id', 'group', 'eval', 'CPR', 'arg1', 'arg2'])
    relationships['regulator'] = map(lambda x: 'REGULATOR' if x in ['CPR:3', 'CPR:4'] else 'NONREGULATOR'  ,relationships.group)
    print 'Total number of relationships:'
    print relationships['regulator'].value_counts()
    
    # encode in same column
    relationships['rel'] = map(lambda x,y,z: "%s %s %s"%(x,y,z) , relationships.regulator, relationships.arg1, relationships.arg2)
    #enrich with index (eg. R28)
    relationships['idx'] = map(lambda x: "R%i"%x ,relationships.index)
    
    return relationships

def write_ann_file(entities, relationships=pd.DataFrame(), save_path=None, debug=False):
    """Jointly write entities and relationships in .ann files 
    given their dataframes
    
    entities cols:        ['doc_id,'idx','type_start_stop','name']
    relationships cols:   ['doc_id,'idx','rel']
    """
    for doc_id in entities.doc_id.unique():
        
        save_path = save_path.rstrip("/")#trim trailing /
        save_path_file = '%s/%s.ann'%(save_path,doc_id)

        #write out entities
        entities[entities.doc_id==doc_id][['idx','type_start_stop','name']].to_csv(save_path_file,sep='\t',header=None,index=False)
        #write out relationships
        if len(relationships)>0:
            relationships[relationships.doc_id==doc_id][['idx', 'rel']].to_csv(save_path_file,sep='\t',header=None,mode='a',index=False)

        if debug:
            print save_path_file.split('/')[-1], ': ',len(entities[entities.doc_id==doc_id]), 'entities', len(relationships[relationships.doc_id==doc_id]),'relations'
    print "Successfully wrote %i ann files in: %s" %( len(entities.doc_id.unique()), 
                                                 save_path)

def process_ann_file(filepath, regex, groups = None):
    with codecs.open(filepath, 'r' ,'utf-8') as f:    
        matches = re.findall(regex, f.read(), re.MULTILINE)
    if groups:
        return pd.DataFrame(matches,names=groups)
    else:
        return pd.DataFrame(matches,)
    
def is_empty_str(identifier):
    """Checks if variable is pd.isnull, empty string (unicode or not)"""
    if (pd.isnull(identifier) or identifier=='' or identifier==u""):
        return True
    else:
        return False

def standoff_to_entitydf(file_list, filter_types = None, entity_type_rename = None):
    '''Takes as input a list of standoff files (.ann)
    and creates a dataframe of entities found.
    Identifier line (Notes) should be directly after entity lines, otherwise a counter identifier is initialized (eg. Chemical:3)
    Identifiers have a 1:1 mapping to entityText (case-sensitive)
    
    Input:
        file_list: list of paths of .ann files (assume list of: doc_id.ann)
        filter_types: list of entity type names to keep (eg. ['Chemical', 'Gene'])
        entity_type_rename: dictionary for renaming NER entity names - applied after filtering types (eg: {'CHEMICAL': 'Chemical' })
        
    Returns:
        entities: pd.DataFrame ready to plug in to entities_to_dict()
        
    Usage:
        standoff_to_df(ann_files, entity_type_mapping = {'CHEMICAL': 'Chemical'})
        
    '''
    
    print('Found %i standoff files'%len(file_list))
    
    start_time = time.time()
    
    
    ### Specify regex for parsing along with the groups expected back
    
    #matches only T lines:
    T_RE = ur'^(T\d+)\t(.*) (\d+) (\d+)\t(.*)$' 
    #matches only N lines
    N_RE = ur'^N\d+\tReference (T\d+) (.*)\t(.*)$' 
    # matches only entities with identifiers (notes in line below)
    TN_RE = ur'^(T\d+)\t(.*) (\d+) (\d+)\t(.*)[\n]N\d+\tReference \1 (.*)\t\5+$'
    # matches entities with optional identifiers
    TNopt_RE = ur'^(T\d+)\t(.*) (\d+) (\d+)\t(.*)[\n]?(N\d+\tReference \1 (.*)\t\5)?$' 
    # specify groups returned by regex
    groups = ['T','EntityType', 'start', 'stop', 'EntityText', '_' ,'identifier']
    entities = pd.DataFrame()
    
    for i, filepath in enumerate(file_list):
        if i%500==0:
            print 'Processing file %i of %i (%.1f min)'%(i,len(file_list), (time.time()-start_time)/60)
        doc_id = filepath.split('/')[-1].rstrip('.ann')
        new_df = pd.DataFrame(process_ann_file(filepath,TNopt_RE))
#         new_df.columns = groups
        new_df['doc_id'] = filepath.split('/')[-1].rstrip('.ann')
        entities = entities.append(new_df,ignore_index=True)
        
    print "Parsing took %.2f sec"%(time.time()-start_time)
    
    #rename columns
    entities.columns = groups+['doc_id']
    
    #select specific entities to keep
    if filter_types:
        entities = entities[map(lambda x: x in filter_types, entities['EntityType'])]
        
    if entity_type_rename:
        entities['EntityType'] = map(lambda x: entity_type_rename[x] , entities.EntityType)
    
    #fix missing identifiers:
    #fill empty identifiers with dummy indexing list (eg Chemical:329)
    entities['identifier'] = map(lambda x,y,z: unicode("%s:%i"%(z,y)) if is_empty_str(x) else x ,entities.identifier,entities.index,entities.EntityType) #fill empty with dummy identif
    
    #create mappings dict overwriting dummy counter ids
    mapping_counter = entities[map(lambda x: is_empty_str(x), entities['_'])].drop_duplicates(subset=['EntityText']).set_index('EntityText').to_dict()['identifier'] 
    mapping_notes = entities[map(lambda x: not is_empty_str(x), entities['_'])].drop_duplicates(subset=['EntityText']).set_index('EntityText').to_dict()['identifier'] 
    mapping_counter.update(mapping_notes) #override values from dummy counter
    entities['identifier'] = map(lambda txt: mapping_counter[txt] ,entities.EntityText )

#     entities.drop('_',axis=1,inplace=True)
    print 'Extracted',len(entities),entities.EntityType.unique(),'entities'
    
    # TODO: throw away useless columns [_, T(?)]
    return entities


#######################################################
    ### Convert to formats ingestible by Snorkel ###
#######################################################

def entitydf_to_meshdict(entities, entity_types = None):
    '''From a dataframe of entities to a mesh dictionary
    
    Input:
    entities = pd.DataFrame() 
            columns: ['EntityText', 'identifier'] & unique index to ensure unique mapping
            
    entity_types = list of entity types to write (eg ['Chemical', 'Genes'])
    '''
    mesh_dict = dict()
    if entity_types is None:
        entity_types = entities.EntityType.unique()
    for ent_type in entity_types:
        #filter out irrelevant entities & keep only needed cols
        temp = entities[entities.EntityType==ent_type][['EntityText','identifier']]

        temp['identifier'] = map(lambda x,y: unicode("%s:%i"%(ent_type,y)) if pd.isnull(x) else x ,temp.identifier,temp.index)
        mesh_dict[ent_type] = pd.Series(temp.identifier.values,index=temp.EntityText).to_dict()

    return mesh_dict

def entitydf_to_tagdict(entities, doc_col= 'doc_id', ident_col = 'identifier', type_col = 'EntityType', start_col = 'start', stop_col = 'stop'):
    '''From a dataframe to unary tags dictionary (compatible
    to Snorkels predifined tagger format)

    input: pd.DataFrame containing doc_id, entity type, entity name, identifier, start and stop index
    '''
    entities['tup'] = map(lambda x,y,start,end: ("|".join([x,y]),int(start),int(end)) , entities[type_col], entities[ident_col], entities[start_col],entities[stop_col])
    unary_tags = entities[[doc_col,'tup']].set_index(doc_col).stack().groupby(doc_col).apply(set).to_dict()
    return defaultdict(set,unary_tags)

from itertools import product

def gold_relations_to_tsv(input_path, output_path, T_offsets, T_types, true_rels=['CPR:3', 'CPR:4'], correct_last_offset = True, create_none_rels=True):
    '''
    Parse relationship goldsets (BioCreative input format), 
    into TSV file, format ingestible by snorkels' load_external_labels
    function (from tutorials/workshop utils) .
    
    input:
        input_path: str or list of strings containing gold relationships in tsv (BioCreative input format)
        output_path: path for saving tsv file
        T_offsets: dict of entity indexing mapping of each document (eg: {"doc12382": {"T4": [32,88] } } )
        T_types: dict of entity types per doc (eg: {"doc12382": {"Chemical": ["T4","T8"] } } )
        true_rels: list, containing relationship names to be parsed (considered relevant)
        correct_last_offset: Substract -1 from the last offset due for snorkel compatibility
        create_none_rels: Boolean: Wheather to creative exhaustive list of negative relationships - representing none relationships between chemicals - genes.
        
    '''

    relationships = pd.DataFrame()
    if isinstance(input_path,str):
        input_path = [input_path]
    for path in input_path:
        relationships = relationships.append(pd.read_table(path, names = ['doc_id', 'CPR', 'arg1', 'arg2']) , ignore_index=True)

    # convert doc_id to string
    relationships['doc_id'] = map(lambda x: str(x) , relationships['doc_id'])
    
    # create all possible combinations (incl not existing, append them to relationships df and drop duplicates by keeping first value)
    df_columns =  [u'doc_id', u'CPR', u'arg1', u'arg2']
    neg_rels = pd.DataFrame(columns=df_columns)
    for doc_id in relationships.doc_id.unique():
        neg_rels1 = pd.DataFrame(map(lambda (T1,T2): (doc_id,-1,T1,T2) , product(T_types[doc_id]['Chemical'],T_types[doc_id]['Gene'])) , columns = df_columns)
        neg_rels = neg_rels.append(neg_rels1,ignore_index=True,)

    neg_rels['arg1'] = map( lambda arg: "Arg1:"+arg , neg_rels['arg1'])
    neg_rels['arg2'] = map( lambda arg: "Arg2:"+arg , neg_rels['arg2'])

    #add them to the goldset rels and drop the duplicates (keep first, which are the GS)
    relationships = relationships.append(neg_rels,ignore_index=True).drop_duplicates(subset=['doc_id','arg1','arg2'])
    
    #continue building the snorkel input format
    relationships['offsetsT1'] = map(lambda doc_id, arg1: T_offsets[str(doc_id)][arg1.split(':')[-1]], relationships.doc_id , relationships.arg1)
    relationships['offsetsT2'] = map(lambda doc_id, arg2: T_offsets[str(doc_id)][arg2.split(':')[-1]], relationships.doc_id , relationships.arg2)
    relationships['label'] = map(lambda x: 1 if x in true_rels else -1, relationships.CPR)
    
    
    if correct_last_offset:
        ## !! correction for last character (substract by 1)
        offsets_format = lambda docid,offsets: "%s::span:%i:%i"%(docid,offsets[0],offsets[1]-1)
    else:
        offsets_format = lambda docid,offsets: "%s::span:%i:%i"%(docid,offsets[0],offsets[1])
        
    # prepare output
    relationships['Chemical'] = map(offsets_format,
                                  relationships.doc_id,relationships.offsetsT1
                                 )
    relationships['Gene'] = map(offsets_format,
                                  relationships.doc_id,relationships.offsetsT2
                                 )
    
    relationships = relationships[['Chemical', 'Gene', 'label']]
    relationships.to_csv(output_path,sep='\t',index=False)
    
    print relationships.label.value_counts()

    return 


# def gold_relations_to_tsv(input_path, output_path, T_offsets, true_rels=['CPR:3', 'CPR:4'], correct_last_offset = True):
#     '''
#     Parse relationship goldsets (BioCreative input file), 
#     into TSV file, format ingestible by snorkels' load_external_labels
#     function (from tutorials/workshop utils) .
    
#     input:
#         input_path: str or list of strings containing gold relationships in tsv (BioCreative input format)
#         output_path: path for saving tsv file
#         T_offsets: dict of entity indexing mapping of each document (eg: {"doc12382": {"T4": [32,88] } } )
#         true_rels: list, containing relationship names to be parsed (considered relevant)
#         correct_last_offset: Substract -1 from the last offset due for snorkel compatibility
        
#     '''

#     relationships = pd.DataFrame()
#     if isinstance(input_path,str):
#         input_path = [input_path]
#     for path in input_path:
#         relationships = relationships.append(pd.read_table(path, names = ['doc_id', 'CPR', 'arg1', 'arg2']) , ignore_index=True)


    
#     relationships['doc_id'] = map(lambda x: str(x) , relationships['doc_id'])
#     relationships['offsetsT1'] = map(lambda doc_id, arg1: T_offsets[str(doc_id)][arg1.split(':')[-1]], relationships.doc_id , relationships.arg1)
#     relationships['offsetsT2'] = map(lambda doc_id, arg2: T_offsets[str(doc_id)][arg2.split(':')[-1]], relationships.doc_id , relationships.arg2)
#     relationships['label'] = map(lambda x: 1 if x in true_rels else -1, relationships.CPR)
    
#     print relationships.label.value_counts()
    
#     if correct_last_offset:
#         ## !! correction for last character (substract by 1)
#         offsets_format = lambda docid,offsets: "%s::span:%i:%i"%(docid,offsets[0],offsets[1]-1)
#     else:
#         offsets_format = lambda docid,offsets: "%s::span:%i:%i"%(docid,offsets[0],offsets[1])
        
#     # prepare output
#     relationships['Chemical'] = map(offsets_format,
#                                   relationships.doc_id,relationships.offsetsT1
#                                  )
#     relationships['Gene'] = map(offsets_format,
#                                   relationships.doc_id,relationships.offsetsT2
#                                  )
    
#     relationships = relationships[['Chemical', 'Gene', 'label']]
#     relationships.to_csv(output_path,sep='\t',index=False)
#     return 

# def gold_relations_to_tsv(input_path, output_path, T_offsets, true_rels=['CPR:3', 'CPR:4']):
#     '''
#     Parse relationship goldsets (BioCreative input file), 
#     into TSV file, format ingestible by snorkels' load_external_labels
#     function (from tutorials/workshop utils) .
    
#     input:
#         input_path: str or list of strings containing gold relationships in tsv (BioCreative input format)
#         output_path: path for saving tsv file
#         T_offsets: dict of entity indexing mapping of each document (eg: {"doc12382": {"T4": [32,88] } } )
#         true_rels: list, containing relationship names to be parsed (considered relevant)
        
#     '''

#     relationships = pd.DataFrame()
#     if isinstance(input_path,str):
#         input_path = [input_path]
#     for path in input_path:
#         relationships = relationships.append(pd.read_table(path, names = ['doc_id', 'CPR', 'arg1', 'arg2']) , ignore_index=True)


#     relationships['doc_id'] = map(lambda x: str(x) , relationships['doc_id'])
#     relationships['offsetsT1'] = map(lambda doc_id, arg1: T_offsets[str(doc_id)][arg1.split(':')[-1]], relationships.doc_id , relationships.arg1)
#     relationships['offsetsT2'] = map(lambda doc_id, arg2: T_offsets[str(doc_id)][arg2.split(':')[-1]], relationships.doc_id , relationships.arg2)
#     relationships['label'] = map(lambda x: 1 if x in true_rels else -1, relationships.CPR)
    
#     print relationships.label.value_counts()
    
#     offsets_format = lambda docid,offsets: "%s::span:%i:%i"%(docid,offsets[0],offsets[1])
#     # prepare output
#     relationships['Chemical'] = map(offsets_format,
#                                   relationships.doc_id,relationships.offsetsT1
#                                  )
#     relationships['Gene'] = map(offsets_format,
#                                   relationships.doc_id,relationships.offsetsT2
#                                  )
    
#     relationships = relationships[['Chemical', 'Gene', 'label']]
#     relationships.to_csv(output_path,sep='\t',index=False)
#     return 


import os
from collections import defaultdict
def gold_relations_to_dict(fpath, T_mapping, true_rels=['CPR:3', 'CPR:4']):
    '''
    ONLY document-level annotations supported
    Parse relationship goldsets (BioCreative input file), 
    into snorkel defaultdict of gold relations 
    (consists of ONLY the TRUE/relevant rels).


    input:
        fpath: str or list of strings containing gold relationships in tsv (BioCreative input format)
        T_mapping: dict of entity indexing mapping of each document (eg: {"doc12382": {"T4": "MESH:2382" } } )
        true_rels: list, containing relationship names to be parsed (considered relevant)
        
    output:
        snorkel compatible dict (only for document_level annotations, to be parsed with CPR tutorial util)
    '''

    relationships = pd.DataFrame()
    if isinstance(fpath,str):
        fpath = [fpath]
    for path in fpath:
#         relationships = pd.read_table(path, names = ['doc_id', 'CPR', 'arg1', 'arg2'])
        relationships = relationships.append(pd.read_table(path, names = ['doc_id', 'CPR', 'arg1', 'arg2']) , ignore_index=True)

    print 'Total number of relationships:'
    print len(relationships)

    print 'Total number of TRUE (desired) relationships:'
    relationships = relationships[map(lambda x: x in true_rels, relationships.CPR)]
    print len(relationships)
    relationships['doc_id'] = map(lambda x: str(x) , relationships['doc_id'])
    relationships['tup'] = map(lambda doc_id, arg1,arg2: (T_mapping[str(doc_id)][arg1.split(':')[-1]],T_mapping[str(doc_id)][arg2.split(':')[-1]]), relationships.doc_id , relationships.arg1, relationships.arg2)
    return defaultdict(set,relationships.groupby('doc_id')['tup'].apply(lambda g: set(g.values.tolist())))



# ~~~~~~~~ DEPRECATED ~~~~~~~~~~
# def standoff_to_entitydf(file_list, filter_types = None, entity_type_rename = None):
#     '''Takes as input a list of standoff files (.ann)
#     and creates a dataframe of entities found.
    
#     Input:
#         file_list: list of paths of .ann files
#         entities: pd.DataFrame to append into (for concatenating entities from different NER)
#         filter_types: list of entity type names to keep (eg. ['Chemical', 'Gene'])
#         entity_type_rename: dictionary for renaming NER entity names - applied after filtering types (eg: {'CHEMICAL': 'Chemical' })
        
#     Returns:
#         entities: pd.DataFrame ready to plug in to entities_to_dict()
        
#     Usage:
#         standoff_to_df(ann_files, entity_type_mapping = {'CHEMICAL': 'Chemical'})
        
#     '''
    
#     print('Found %i standoff files'%len(file_list))
    
#     entities = pd.DataFrame(columns = ['T','doc_id','EntityType','start', 'stop','EntityText', 'identifier'])

#     start_time = time.time()

#     for filepath in file_list:
#         doc_id = filepath.split('/')[-1].rstrip('.ann')
#         with codecs.open(filepath, 'r' ,'utf-8') as f:
#             content = f.read()
            
#         for line in content.splitlines():
#             # TODO: replace with regex for faster processing
#             # ^(T\d+)\t(.*) (\d+) (\d+)\t([^\n\r]*)\n$
            
#             if line.startswith("T"): #entity
#                 T, type_start_stop, name = line.split('\t')
#                 ent_type, start, stop = type_start_stop.split(' ')

#                 entities = entities.append({'doc_id': doc_id,
#                                              'EntityType': ent_type,
#                                              'start': int(start),
#                                              'stop': int(stop),
#                                              'EntityText': name,
#                                             'T': T
#                                             },
#                                             ignore_index=True,)
#             if line.startswith("N"): #note - contains mapping
# #                 N1	Reference T1 NCBIGENE:18391	sigma-1 receptor

#                 ref, txt = line.split('\t')[1:3]
#                 T, identif  = ref.split(' ')[1:3]
# #                 print T, identif 
#                 entities.loc[(entities['T'] == T) & (entities['doc_id'] == doc_id), 'identifier'] = identif
    
#     print "Parsing took %.2f sec"%(time.time()-start_time)

#     #select specific entities to keep
#     if filter_types:
#         entities = entities[map(lambda x: x in filter_types, entities['EntityType'])]
        
#     if entity_type_rename:
#         entities['EntityType'] = map(lambda x: entity_type_rename[x] , entities.EntityType)
    
#     #fix missing identifiers
#     entities['identifier'] = map(lambda x,y,z: unicode("%s:%i"%(z,y)) if pd.isnull(x) else x ,entities.identifier,entities.index,entities.EntityType)
    
#     print 'Extracted',len(entities),entities.EntityType.unique(),'entities'
#     return entities


# def standoff_to_df(file_list, filter_types = None, entity_type_mapping = None):
#     '''Takes as input a list of standoff files (.ann)
#     and creates a dataframe of entities found.
    
#     Input:
#         file_list: list of paths of .ann files
#         entities: pd.DataFrame to append into (for concatenating entities from different NER)
#         filter_types: list of entity type names to keep (eg. ['Chemical', 'Gene'])
#         entity_type_mapping: dictionary for renaming NER entity names - applied after filtering types (eg: {'CHEMICAL': 'Chemical' })
        
#     Returns:
#         entities: pd.DataFrame ready to plug in to entities_to_dict()
        
#     Usage:
#         standoff_to_df(ann_files, entity_type_mapping = {'CHEMICAL': 'Chemical'})
        
#     '''
    
#     print('Found %i standoff files'%len(file_list))
    
#     entities = pd.DataFrame(columns = ['doc_id','EntityType','start', 'stop','EntityText'])

#     start_time = time.time()

#     for filepath in file_list:
#         doc_id = filepath.split('/')[-1].rstrip('.ann')
#         with codecs.open(filepath, 'r' ,'utf-8') as f:
#             content = f.read()
            
#         for line in content.splitlines():
#             # TODO: replace with regex for faster processing
#             # ^(T\d+)\t(.*) (\d+) (\d+)\t([^\n\r]*)\n$
            
#             if line.startswith("T"): #entity
#                 _, type_start_stop, name = line.split('\t')
#                 ent_type, start, stop = type_start_stop.split(' ')

#                 entities = entities.append({'doc_id': doc_id,
#                                              'EntityType': ent_type,
#                                              'start': start,
#                                              'stop': stop,
#                                              'EntityText': name
#                                             },
#                                             ignore_index=True,)

#             #     content.split
#     print "Conversion took %.2f sec"%(time.time()-start_time)

#     #select specific entities to keep
#     if filter_types:
#         entities = entities[map(lambda x: x in filter_types, entities['EntityType'])]
        
#     if entity_type_mapping:
#         entities['EntityType'] = map(lambda x: entity_type_mapping[x] , entities.EntityType)
        
#     print 'Extracted',len(entities),entities.EntityType.unique(),'entities'
#     return entities


# def entities_to_dict(entities, doc_col= 'doc_id', type_col = 'EntityType', name_col = 'EntityText', start_col = 'start', stop_col = 'stop'):
#     '''From a dataframe to unary tags dictionary (compatible 
# 	to Snorkels predifined tagger format)
	
# 	input: pd.DataFrame containing doc_id, entity type, entity name, start and stop index
# 	'''
#     entities["tup"] = map(lambda x,y,start,end: ("|".join([x,y]),start,end) , entities[type_col], entities[name_col], entities[start_col],entities[stop_col])
#     unary_tags = entities[[doc_col,'tup']].set_index(doc_col).stack().groupby(doc_col).apply(set).to_dict()
#     return defaultdict(set,unary_tags)


# def standoff_to_entitydf(file_list, filter_types = None, entity_type_rename = None):
#     '''Takes as input a list of standoff files (.ann)
#     and creates a dataframe of entities found.
#     Identifier line (Notes) should be directly after entity lines, otherwise a counter identifier is initialized (eg. Chemical:3)
#     Identifiers have a 1:1 mapping to entityText (case-sensitive)
    
#     Input:
#         file_list: list of paths of .ann files (assume list of: doc_id.ann)
#         filter_types: list of entity type names to keep (eg. ['Chemical', 'Gene'])
#         entity_type_rename: dictionary for renaming NER entity names - applied after filtering types (eg: {'CHEMICAL': 'Chemical' })
        
#     Returns:
#         entities: pd.DataFrame ready to plug in to entities_to_dict()
        
#     Usage:
#         standoff_to_df(ann_files, entity_type_mapping = {'CHEMICAL': 'Chemical'})
        
#     '''
    
#     print('Found %i standoff files'%len(file_list))
    
#     start_time = time.time()
    
    
#     ### Specify regex for parsing along with the groups expected back
    
#     #matches only T lines:
#     T_RE = ur'^(T\d+)\t(.*) (\d+) (\d+)\t(.*)$' 
#     #matches only N lines
#     N_RE = ur'^N\d+\tReference (T\d+) (.*)\t(.*)$' 
#     # matches only entities with identifiers (notes in line below)
#     TN_RE = ur'^(T\d+)\t(.*) (\d+) (\d+)\t(.*)[\n]N\d+\tReference \1 (.*)\t\5+$'
#     # matches entities with optional identifiers
#     TNopt_RE = ur'^(T\d+)\t(.*) (\d+) (\d+)\t(.*)[\n]?(N\d+\tReference \1 (.*)\t\5)?$' 
#     # specify groups returned by regex
#     groups = ['T','EntityType', 'start', 'stop', 'EntityText', '_' ,'identifier']
#     entities = pd.DataFrame()
    
#     for i, filepath in enumerate(file_list):
#         if i%500==0:
#             print 'Processing file %i of %i (%.1f min)'%(i,len(file_list), (time.time()-start_time)/60)
#         doc_id = filepath.split('/')[-1].rstrip('.ann')
#         new_df = pd.DataFrame(process_ann_file(filepath,regex))
# #         new_df.columns = groups
#         new_df['doc_id'] = filepath.split('/')[-1].rstrip('.ann')
#         entities = entities.append(new_df,ignore_index=True)
        
#     print "Parsing took %.2f sec"%(time.time()-start_time)
    
#     #rename columns
#     entities.columns = groups+['doc_id']
    
#     #select specific entities to keep
#     if filter_types:
#         entities = entities[map(lambda x: x in filter_types, entities['EntityType'])]
        
#     if entity_type_rename:
#         entities['EntityType'] = map(lambda x: entity_type_rename[x] , entities.EntityType)
    
#     #fix missing identifiers:
#     #fill empty identifiers with dummy indexing list (eg Chemical:329)
#     entities['identifier'] = map(lambda x,y,z: unicode("%s:%i"%(z,y)) if is_empty_str(x) else x ,entities.identifier,entities.index,entities.EntityType) #fill empty with dummy identif
    
#     #create mappings dict overwriting dummy counter ids
#     mapping_counter = entities[map(lambda x: is_empty_str(x), entities['_'])].drop_duplicates(subset=['EntityText']).set_index('EntityText').to_dict()['identifier'] 
#     mapping_notes = entities[map(lambda x: not is_empty_str(x), entities['_'])].drop_duplicates(subset=['EntityText']).set_index('EntityText').to_dict()['identifier'] 
#     mapping_counter.update(mapping_notes) #override values from dummy counter
#     entities['identifier'] = map(lambda txt: mapping_counter[txt] ,entities.EntityText )

# #     entities.drop('_',axis=1,inplace=True)
#     print 'Extracted',len(entities),entities.EntityType.unique(),'entities'
    
#     # TODO: throw away useless columns [_, T(?)]
#     return entities