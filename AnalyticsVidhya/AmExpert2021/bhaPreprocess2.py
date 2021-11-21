# Trial 2
# Cleaning Train Data

import numpy as np
import pandas as pd

def extractor(data,prefix=None):
    l = []
    for i in data:
        l.append(i.replace('[',"").replace(']',"").replace('\'',"").replace(" ","").split(','))

    bha = {}
    for key in uniqueProduct:
        if prefix:
            key = prefix+'_'+key
        bha[key]=[]
    
    for i in l:
        for key in uniqueProduct:
            if prefix:
                newkey = prefix+'_'+key
            if key in i:
                bha[newkey].append(1)
            else:
                bha[newkey].append(0)
    return(pd.DataFrame(bha))

if __name__ == '__main__':
    df = pd.read_csv('train_go05W65.csv')
    print(df.head())
    
    uniqueProduct = []
    for i in df['Product_Holding_B1']:
        for j in i.replace('[',"").replace(']',"").replace('\'',"").replace(" ","").split(','):
            uniqueProduct.append(j)
    uniqueProduct = sorted(list(set(uniqueProduct)))
    print('Unique Products :',uniqueProduct)

    df1 = df.drop(['Gender','City_Category','Customer_Category','Customer_ID', 'Product_Holding_B1', 'Product_Holding_B2'], axis=1)
    df1 = pd.concat([df1,pd.get_dummies(df[['Gender','City_Category','Customer_Category']],drop_first=True), extractor(df['Product_Holding_B1'],prefix='Info_B1'),extractor(df['Product_Holding_B2'],prefix='Target_B2')], axis=1).astype(np.float64)
    print(df1.head())
    print(df1.info())

    print(df1.tail())
    print(df1[['Age','Vintage']])

    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()

    
    df1[['Age','Vintage']]= sc.fit_transform(df1[['Age','Vintage']])

    df1.to_csv('trainData.csv',index=None)

    print(df1.head())