from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
 
class AprioriModel:
    def __init__(self,min_support:float,min_confidence:float):
        self.min_support=min_support
        self.min_confidence=min_confidence
        self.rules=pd.DataFrame()
    def fit(self,transactions:list):
        te=TransactionEncoder()
        te_array=te.fit(transactions).transform(transactions)
        df_trans=pd.DataFrame(te_array,columns=te.columns_)
        frequent_items=apriori(df_trans,min_support=self.min_support,use_colnames=True)
        self.rules=association_rules(frequent_items,metric="confidence",min_threshold=self.min_confidence)
        return self.rules
    def get_rules(self):
        return self.rules
