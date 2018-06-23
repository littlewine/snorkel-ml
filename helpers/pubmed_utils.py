##################################################################
#######   Utilities for parsing articles from pubmed   ###########
##################################################################

from Bio import Entrez
import time

def get_pubmed_citations(ids, outgoing = True, api_output_loc='temp/citations_list.txt', chunks=199):
    """Get the references/citations of a list of pubmed ids.
    
    ids: list of articles to get citations/refs from
    outgoing: If True, return the documents that the given article(s) cite (outgoing citations),
            otherwise return documents that have cited that article (incoming citations).
    chunks: how many articles to request at once
    api_output_loc: location to save temp txt file
    
    """
    
    # TODO: probably need to put timer because of new pubmed API requests limit
    
    citations = []
    for i in xrange(0,len(ids), n):
        print 'Download citations from articles %i to %i  . . . .'%(i+1,i+n+1)
        pmids = ','.join(ids[i:i+n])
        if outgoing:
            url_Submit = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&linkname=pubmed_pubmed_refs&id=%s&tool=my_tool&email=my_email@example.com"%(pmids)
        else:
            url_Submit = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&linkname=pubmed_pubmed_citedin&id=%s&tool=my_tool&email=my_email@example.com"%(pmids)
        urllib.urlretrieve(url_Submit, api_output_loc)
    #     print url_Submit
        tree = ET.parse(api_output_loc)
        root = tree.getroot()
        citations.extend(map(lambda x: x.text ,root.findall("./LinkSet/LinkSetDb/Link/Id")))
        
    print "Found %s citations (%s unique) from %s articles"%(len(citations),len(set(citations)), len(ids))
    return citations




def get_pubmed_similar_docs(ids, top_n = 50):
    """Given a list of pubmed ids, get the top_n most similar pubmed ids"""
    
    Entrez.email = "amkrasakis@gmail.com"

    similar_ids = []

    for i,primary_id in enumerate(ids):
        handle = Entrez.elink(db="pubmed", id=primary_id, cmd="neighbor_score", rettype="xml")
        records = Entrez.read(handle)

        scores = sorted(records[0]['LinkSetDb'][0]['Link'], key=lambda k: int(k['Score']), reverse=True)
        similar_ids.extend(map(lambda x: x['Id'],scores[:top_n]))
        
        time.sleep(.34) # do not send more than 3 requests/sec
        if i%100==0:
            print 'Getting %i most similar docs to doc %i...'%(top_n, i)
            
    return list(set(similar_ids))
