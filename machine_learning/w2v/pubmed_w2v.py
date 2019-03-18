#!/usr/bin/env python
# coding: utf-8

# In[142]:


import pandas as pd
import numpy as np
import datetime
import warnings
import pickle
import codecs
import string
import psycopg2
import re
import multiprocessing
import timeit
import os
import spacy
from time import time
from gensim.models import Word2Vec
from lxml import etree
from sqlalchemy import create_engine
warnings.filterwarnings('ignore')
from configparser import ConfigParser


# # DB Connection

# In[143]:


def config(filename='database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)
 
    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
 
    return db


# In[144]:


def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        conn.autocommit = True
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        return conn
conn = connect()


# In[145]:


def run_query(sql, data=()):
    try:
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, data)
        # get the number of updated rows
        results = cur.fetchall()
        # Commit the changes to the database
        conn.commit()
        # Close communication with the PostgreSQL database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return results

def run_update(sql, data=(), is_insert=False):
    try:
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, data)
        if is_insert:
            updated_rows = cur.fetchone()[0]
        else:
            # get the number of updated rows
            updated_rows = cur.rowcount
        # Commit the changes to the database
        conn.commit()
        # Close communication with the PostgreSQL database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return updated_rows


# In[146]:


def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if "nxml" in name: 
                r.append(os.path.join(root, name))
    return r 


# ## update database

# In[147]:


def getText(root, path):
    node = root.xpath(path)
    if len(node) == 0:
        return None
    else:
        return node[0].text


# In[148]:


def update_article(article_id, pmc, doi, filename, fauthor, authors):
    sql = """
        update temp_article
        set pmc=%s
        , doi=%s
        , filename=%s
        , fauthor=%s
        , authors=%s
        where id=%s
    """
    run_update(sql, (pmc,doi,filename,fauthor,", ".join(authors),article_id))


# In[149]:


def update_reference(pmid, pmc, doi, raw, source, title, fauthor, authors, year, reftype):
    ref_id = pmid
    id_col = "pmid"
    if ref_id is None:
        ref_id = pmc
        id_col = "pmc"
        if pmc is None:
            ref_id = doi
            id_col = "doi"
            if doi is None:
                ref_id = raw
                id_col = "raw"
                if raw is None:
                    ref_id = title
                    id_col = "articletitle"
                    if title is None:
                        ref_id = source
                        id_col = "articlesource"
                        if source is None:
                            return None

    # let's see if this is an internal article
    articles = run_query("select id from temp_article where id=%s",(ref_id,))
    article_id = None
    if len(articles) > 0:
        article_id = articles[0][0]
    # check if this reference already exists
    ref = run_query("select id from reference where " + id_col + "=%s", (ref_id,))
    if len(ref) > 0:
        return ref[0][0]
    # new reference, let's save it then
    return run_update("""
        insert into reference(pmid
        ,pmc
        ,doi
        ,raw
        ,articletitle
        ,articlesource
        ,fauthor
        ,authors
        ,articleyear
        ,article_id
        ,referencetype)
        values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        returning id;
    """ , (pmid, pmc, doi, raw, title, source, fauthor, ", ".join(authors), re.sub("[^0-9]", "", year), article_id, reftype)
        , True)


# In[150]:


def update_citation(article_id, citation_id, reference_id):
    return run_update("""
        insert into citation_ref(articleid, citationid, reference_id)
        values(%s, %s, %s)
        returning reference_id;
    """ , (article_id, citation_id, reference_id)
        , True)


# ## parse data

# In[151]:


file_list = list_files("./data/citations/datasets")
processed_files = run_query("select distinct filename from temp_article where filename is not null")
processed_files = [f[0] for f in processed_files]


# In[152]:


def parse_file(f, n):
    root = etree.parse(f)
    ref_list = root.xpath("//back/ref-list/ref")
    article_pmid = getText(root, "//front/article-meta/article-id[@pub-id-type='pmid']")
    article_pmc = getText(root, "//front/article-meta/article-id[@pub-id-type='pmc']")
    if article_pmc is not None and "PMC" not in article_pmc:
        article_pmc = "PMC" + article_pmc
        
    # make sure we have an article id to update.. otherwise break
    if article_pmid is None:
        if article_pmc is not None:
            article_pmid = article_pmc
        else:
            return
    article_doi = getText(root, "//front/article-meta/article-id[@pub-id-type='doi']")
    article_filename = f.split("/")[-1]
    
    article_authors = []
    authors = root.xpath("//front/article-meta/contrib-group/contrib[@contrib-type='author']/name")
    for author in authors:
        fname = getText(author, ".//given-names")
        surname = getText(author, ".//surname")
        if fname is not None:
            fname = re.sub(r"[^A-Za-z]+", '', fname)
        else:
            fname = ""
        article_authors.append((fname + " " + surname).strip())

    article_fauthor = None
    if len(article_authors) > 0:
        article_fauthor = article_authors[0]
    
    # update article table
    update_article(article_pmid, article_pmc, article_doi, article_filename, article_fauthor, article_authors)
    
    # now update our citation references
    for ref in ref_list:
        elem = ref.xpath(".//element-citation")
        reftype = "mixed"
        if len(elem) > 0:
            reftype = elem[0].get("publication-type")
        title = getText(ref, ".//element-citation/article-title")
        year = getText(ref, ".//element-citation/year")
        source = getText(ref, ".//element-citation/source")
        if title is None:
            title = source
            if title is None:
                continue
        if year is None:
            year = ""
            
        # skip papers we've already processed
        title_year = title+"//"+year
        
        # collect missing meta data
        pmid = getText(ref, ".//pub-id[@pub-id-type='pmid']")
        pmc = getText(ref, ".//pub-id[@pub-id-type='pmc']")
        doi = getText(ref, ".//pub-id[@pub-id-type='doi']")
        raw = getText(ref, ".//mixed-citation")
        source = getText(ref, ".//source")
        title = getText(ref, ".//article-title")
        year = getText(ref, ".//year")
        # get authors
        authors = root.xpath("//front/article-meta/contrib-group/contrib[@contrib-type='author']/name")
        ref_authors = []
        for author in authors:
            fname = getText(author, ".//given-names")
            surname = getText(author, ".//surname")
            if fname is not None:
                fname = re.sub(r"[^A-Za-z]+", '', fname)
            else:
                fname = ""
            ref_authors.append((fname + " " + surname).strip())
        # save first author for easy access
        fauthor = None
        if len(ref_authors) > 0:
            fauthor = ref_authors[0]
        
        # update table with new meta data
        ref_id = update_reference(pmid, pmc, doi, raw, source, title, fauthor, ref_authors, year, reftype)
        update_citation(article_pmid, ref.get("id"), ref_id)
    
    n.value += 1
    if n.value % 10000 == 0:
        print(n.value)


# In[153]:


# start_time = timeit.default_timer()
# pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1 or 1)
# manager = multiprocessing.Manager()
# n = manager.Value('i',0)
# for f in file_list:
#     pool.apply_async(parse_file, args=(f, n))
# pool.close()
# pool.join()
# elapsed = timeit.default_timer() - start_time
# print(elapsed)


# # Create word2vec model

# In[154]:


nlp = spacy.load('en', disable=['ner', 'parser'])
EMBEDDINGS_WORD2VEC_MODEL_FILE = "word2vec.model"


def text_to_tokens(text: str) -> [str]:
    text = re.sub(r'\W+', ' ', text)
    doc = nlp(text)
    return [token.lemma_ for token in doc             if token.lemma_ not in nlp.Defaults.stop_words             and len(token.lemma_) > 2             and token.lemma_.isalpha()            ]


def tokens_generator():
    for qa in qa_generator():
        yield text_to_tokens(qa[2])


def qa_generator():
    """Generator which returns Q/A tuples.
    Each tuple has an user, a group, a timestamp, the tokens and raw text.
    Consecutive messages from the same user in the same group are aggregated.
    """
    rows = run_query("select id, articletitle, papertext from article")
    for row in rows:
        yield row


class SentencesIterator():
    def __init__(self, generator_function):
        self.generator_function = generator_function
        self.generator = self.generator_function()

    def __iter__(self):
        # reset the generator
        self.generator = self.generator_function()
        return self

    def __next__(self):
        result = next(self.generator)
        if result is None:
            raise StopIteration
        else:
            return result
        
        
def get_model():
    model = Word2Vec.load(EMBEDDINGS_WORD2VEC_MODEL_FILE)
    return model
        

def build_model():
    print('Connecting to the database...')
    sentences = SentencesIterator(tokens_generator)
    print('Calculating the embeddings...')
    model = Word2Vec(sentences, min_count=10)
    print('Saving the model...')
    model.save(EMBEDDINGS_WORD2VEC_MODEL_FILE)
    print('Word2Vec Model saved.')
    return model


# In[155]:


t = time()
w2v = build_model()
print('Time to build model: {} s'.format(round((time() - t), 2)))


# In[156]:


# w2v.most_similar("result")


# In[ ]:




