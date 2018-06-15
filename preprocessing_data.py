

import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')

import sklearn
import numpy as np
import csv
import sys

limit = 10e7
monomer_id_key = {}
monomer_id = 0
monomer_counts = np.zeros(int(10e5))
count = 0
target_dict = {}

db_name1 = 'purchase_target_10000.tsv'
db_name2 = 'BindingDB_All.tsv'


csv.field_size_limit(100000000)

with open(db_name2) as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t",quoting=csv.QUOTE_NONE)
    
    for line in tsvreader:
        if count>limit:
            break
        if count!=0:
            if len(line)>=9:
                target_name = line[6]
                current_monomer_id = line[4]
                monomer_smiles=line[1]
                monomer_names=line[5]
                Ki_value = line[8]
                #if count%10000==0:
                    #print(target_name)
                if Ki_value != '':
                    Ki_value = Ki_value.strip('<')
                    Ki_value = Ki_value.strip('>')
                    
                    if current_monomer_id not in monomer_id_key:
                        monomer_id_key[current_monomer_id] = monomer_id
                        monomer_id += 1
                        monomer_id_new = monomer_id
                        
                    else:
                        monomer_id_new = monomer_id_key[current_monomer_id]
                    monomer_counts[monomer_id_new] += 1
                    if target_name in target_dict:
                        target_dict[target_name].append([monomer_id_new,float(Ki_value),monomer_names,monomer_smiles,target_name ])
                    else:
                        target_dict[target_name] = []
                        target_dict[target_name].append([monomer_id_new,float(Ki_value),monomer_names,monomer_smiles,target_name ])
        count += 1
		
	##In the above code,compound and target dictionary is created by first1)Extracting values column by column from tsv file.
#2)Values obtained are appended to the dictionary object created 	
		
		#Further these dictionaries are accessed to form the data matrix
        #From the data matrix,the protein that needs to be considered as the binding receptor is removed from the training data and forms the ylabels
        #The remaining training data now consists of target profiles of the compounds whose binding affinity to the target receptor needs to be predicted.
## We don't want too sparse a matrix, so we only use monomers that have 
## greater than n_data points
n_data = 6
n_removals = 0
ii = 0
monomer_mapping = np.arange(monomer_id+1)
monomer_counts_copy = np.copy(monomer_counts[0:monomer_id+1])
while ii<len(monomer_mapping):
    if monomer_counts_copy[ii]<n_data:
       #monomer_mapping[ii:-1] = monomer_mapping[ii:-1]-1
       monomer_mapping = np.delete(monomer_mapping, ii)
       monomer_counts_copy = np.delete(monomer_counts_copy, ii)
      
       n_removals += 1
    else:
       ii += 1
        


##Forming the data matrix

#getting the number of targets
n_monomers_per_target = 30
n_targets = 0
for target in target_dict:
    length = len(target_dict[target])
    if length > n_monomers_per_target:
        n_targets += 1


       
data_matrix = np.zeros((n_targets, len(monomer_mapping)))
target_id = 0
target_reverse_mapping = {}
target_mapping = {}
result_array = np.array([])
res_comp=np.array([])
f=open("guru99.csv", "a+")
for target in target_dict:
    length = len(target_dict[target])
    if length > n_monomers_per_target:
        if target not in target_reverse_mapping:
            target_reverse_mapping[target] = target_id
            current_target_id = target_id
            target_mapping[target_id] = target
            target_id += 1
        else:
            current_target_id = target_reverse_mapping[target]     
        for ii in range(length):
            current_monomer_id = target_dict[target][ii][0]
            monomer_names=target_dict[target][ii][2]
            monomer_smiles1=target_dict[target][ii][3]
            affinityvalue=target_dict[target][ii][1]
            if monomer_counts[current_monomer_id]>n_data:
                new_monomer_id = np.where(monomer_mapping ==current_monomer_id)[0][0]
                Ki_value = target_dict[target][ii][1]
                data_matrix[current_target_id, new_monomer_id] = Ki_value
                f.write(str(current_target_id)+","+str(new_monomer_id)+","+str(Ki_value)+"\n")
                #resultant=str(current_target_id)
                #res_compounds=str(new_monomer_id)+","+str(monomer_names)
                #res_comp=np.append(res_comp,np.array(res_compounds))
                #result_array=np.append(result_array,np.array(resultant))
				



#import pandas as pd    
#df=pd.DataFrame(result_array)

#df.to_csv('filenamehellodb1.csv',sep=',')


#df1=pd.DataFrame(res_comp)

#df3=pd.DataFrame(data_matrix)
#df3.to_csv("Datadb1.csv",sep=',')
#df1.to_csv('filenamehello2compoundsdb1.csv',sep=',')



x=data_matrix[695,:]
#print((target_mapping))
#print((monomer_mapping))
np.array(monomer_mapping)
#np.savetxt("compound.csv",np.array(monomer_mapping))
#np.savetxt("proteins.csv",np.frombuffer(target_mapping))
#target_map=np.reshape(target_mapping,-1)
#mono_map=np.reshape(monomer_mapping,-1)

#np.savetxt('targetmapping.csv',target_map,delimiter=',')
#np.savetxt('momoner_mapping.csv',mono_map,delmiter=',')

##Extracting positively interacting compounds

pos=np.where(x!=0)				
data_matrix_new = np.delete(data_matrix, 695, 0)

"Split into X_train, X_dev, X_test"

X_train = data_matrix_new[:,0:5000]
X_dev = data_matrix_new[:,5001:6071]
X_test = data_matrix_new[:,6072:7139]

#The trick is here:MAke the estrogen receptor row as your y-labels to repurpose compounds for targetting estrogen

Y_train = data_matrix[695,0:5000]
Y_dev = data_matrix[695,5001:6071]
Y_test = data_matrix[695,6072:7139]
				

np.savetxt('X_trainbd.csv',X_train, delimiter=',')
np.savetxt('X_devbd.csv',X_dev, delimiter=',')
np.savetxt('X_testbd.csv',X_test, delimiter=',')
np.savetxt('Y_trainbd.csv',Y_train, delimiter=',')
np.savetxt('Y_devbd.csv',Y_dev, delimiter=',')
np.savetxt('Y_testbd.csv',Y_test, delimiter=',')
