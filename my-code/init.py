# Shortcuts to connect to database, initialize candidate subclass and return snorkel session
import os
# os.environ['SNORKELDB'] = 'postgres:///snorkel'
os.environ['SNORKELDB'] = 'postgres:///snorkel50similar'

from snorkel import SnorkelSession
session = SnorkelSession()
from snorkel.models import  Document, Sentence
import matplotlib.pyplot as plt
from snorkel.annotations import save_marginals
from snorkel.models import Candidate, candidate_subclass
REGULATOR = candidate_subclass('REGULATOR', ['Chemical', 'Gene'])

print "Created snorkel session from ",os.environ['SNORKELDB']