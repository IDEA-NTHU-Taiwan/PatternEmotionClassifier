import os

import numpy as np
import pandas as pd

from functools import reduce

from .PatternVectorizer import PatternVectorizer

class SimpleClassifier:
    
  def __init__(self, scoreMatrix, classes=None):
    # scoreMatrix each represent the vector of one pattern
    self.scoreMatrix = scoreMatrix
    
    # Default set classes as integer
    if np.all(classes is None):
      classes = list(range(scoreMatrix.shape[1]))
    if len(classes) < scoreMatrix.shape[1]:
      raise ValueError('Classes dimentions does not fit with the score matrix')
    
    self.classes = classes  
          
  def get_top_classes(self, documentPatternVectors, n=1, ascending=True):
    d = 1 if ascending else -1
    return [[self.classes[c] for c in vect[::d][:n]] for vect in self.get_emotion_score(documentPatternVectors).argsort()]
    
  def get_max_score_class(self, documentPatternVectors):
    return [self.classes[c] for c in self.get_emotion_score(documentPatternVectors).argmax(axis=1)]
    
  def get_min_score_class(self, documentPatternVectors):
    return [self.classes[c] for c in self.get_emotion_score(documentPatternVectors).argmin(axis=1)]
  
  def get_emotion_score(self, documentPatternVectors):
    return documentPatternVectors.dot(self.scoreMatrix)

  def get_emotion_prob(self, documentPatternVectors, ascending=True):
    emo_score = self.get_emotion_score(documentPatternVectors)
    emo_sum = emo_score.sum(axis=1)
    emo_sum[emo_sum == 0] = 1
    if ascending:
      emo_prob = self.softmax(1 - (emo_score / emo_sum[:, np.newaxis]))
    else:
      emo_prob = self.softmax(emo_score / emo_sum[:, np.newaxis])
    return emo_prob
  
  def softmax(self, z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div
   
  @classmethod
  def load_from_folder(cls, patternFolderPath, rank='average', ascending=False, emo_file_suffix='tficf_'):
    
    def load_file(file, patternFolderPath=patternFolderPath, rank=rank, ascending=ascending, emo_file_suffix=emo_file_suffix):
        emotion = file.replace(emo_file_suffix,'')
        res = pd.read_csv(
            patternFolderPath + file, 
            names=['pattern', emotion],
            sep='\t',
            dtype={'pattern': str, emotion:np.float32})
        return res
        
    df_list = [load_file(f) for f in os.listdir(patternFolderPath)]
    patterns_df = reduce( (lambda right, left: pd.merge(right, left, on='pattern')), df_list )
    emotions = patterns_df.columns.tolist()[1:]
    
    patterns_df[emotions] = patterns_df[emotions].rank(method=rank, ascending=ascending, pct=True)
    
    pv = PatternVectorizer(list(patterns_df.pattern))
    new_cls = cls(patterns_df[emotions].values, classes=emotions)
    return (pv, new_cls)
