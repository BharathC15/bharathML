# Python program for flames

import numpy as np


if __name__ == '__main__':

    flamesDictionary = {
        'F' : 'Friends',
        'L' : 'Love',
        'A' : 'Affection',
        'M' : 'Marriage',
        'E' : 'Enemy',
        'S' : 'Sister',
    }

    flames=np.array(list("flames".upper()))
    name1=np.array(list(input("Enter First Name : ").replace(' ','').upper()))
    name2=np.array(list(input("Enter Second Name : ").replace(' ','').upper()))

    print(name1,name2)
    common_count=0
    for i in name1:
        if i in name2:
            common_count+=1
            continue

    total_count=len(name1)+len(name2)-2*common_count
    #print("Total Count:",total_count)
    ind=0
    while len(flames)>1:
        ind=(total_count%len(flames))+ind
        ind=ind%len(flames)-1
        if ind==-1: ind+=len(flames)
        flames=np.delete(flames,ind)
        #print(flames)
    print("\nFinal Result->",flamesDictionary[str(flames[0])])