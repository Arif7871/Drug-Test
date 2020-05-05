import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('drugdata.csv')

#dataset['rate'].fillna(0, inplace=True)
dataset['family live'].fillna(0, inplace=True)  #null value fill up

dataset['gender'].fillna(0, inplace=True)  #null value fill up

dataset['age'].fillna(dataset['age'].mean(), inplace=True)  #bqz age number

dataset['address'].fillna(0, inplace=True)  #null value fill up

dataset['profession'].fillna(0, inplace=True)  #null value fill up

dataset['distance'].fillna(0, inplace=True)  #null value fill up

dataset['efficiency'].fillna(0, inplace=True)  #null value fill up

dataset['stress'].fillna(0, inplace=True)  #null value fill up

dataset['economic'].fillna(0, inplace=True)  #null value fill up

dataset['addictedhome'].fillna(0, inplace=True)  #null value fill up

dataset['trauma'].fillna(0, inplace=True)  #null value fill up

dataset['relation'].fillna(0, inplace=True)  #null value fill up

dataset['alone'].fillna(0, inplace=True)  #null value fill up

dataset['care'].fillna(dataset['care'].mean(), inplace=True)

dataset['lostjob'].fillna(0, inplace=True)  #null value fill up

dataset['sexual'].fillna(0, inplace=True)  #null value fill up

dataset['interest'].fillna(0, inplace=True)  #null value fill up

dataset['sleep'].fillna(0, inplace=True)  #null value fill up

dataset['outsidenight'].fillna(0, inplace=True)  #null value fill up

dataset['weight'].fillna(0, inplace=True)  #null value fill up

dataset['solution'].fillna(0, inplace=True)  #null value fill up

dataset['addictedfriend'].fillna(0, inplace=True)  #null value fill up

dataset['reason'].fillna(0, inplace=True)  #null value fill up

dataset['addicted'].fillna(0, inplace=True)  #null value fill up

X = dataset.iloc[:, :23]


def convert_to_int(word):
    word_dict = {'yes':1, 'no':0, 'prefer not to say':3, 'student':0, 'business':1, 'govt job':2,'private job':3, 
                 'doctor':4, 'freelancer':5,'driver':7, 'engineer':6, 'excellent':0, 'good':1, 
                 'satisfactory':2, 'unsatisfactory':3, 'poor':4, 'not sure':5,
                 'bad':1, 'average':2, 'upper':0, 'upper middle':1,
                 'lower middle':2, 'lower':3, 'confused':2, 'friend':1, 'none':3, 
                 'mind':0, 'business and having money':4, 'family problem':2,
                 'dhaka':0,'sylhet':1,'jessore':2,'chottogram':3,'sirajgonj':4,'b-baria':5,
                 'kushtia':6,'khulna':7,'gazipur':8,'barisal':9,'bogura':10,'nauga':11,
                 'rajbari':13,'cumilla':14,'rajshahi':15,'narayanganj':16,'comilla':17,'dhanmondi':18,
                 'rangpur':19,'mirpur':20,'tangail':21,'thakurgaon':22,'jamalpur':23,'barishal':24,
                 'sherpur':25,'uttara':26,'moulvibazar':27,'demra':28,'faridpur':29,'manikganj':30,
                 'male':1,'female':0}
    return word_dict[word]

X['family live'] = X['family live'].apply(lambda x : convert_to_int(x))

X['gender'] = X['gender'].apply(lambda x : convert_to_int(x))

X['address'] = X['address'].apply(lambda x : convert_to_int(x))

X['profession'] = X['profession'].apply(lambda x : convert_to_int(x))

X['distance'] = X['distance'].apply(lambda x : convert_to_int(x))

X['efficiency'] = X['efficiency'].apply(lambda x : convert_to_int(x))

X['stress'] = X['stress'].apply(lambda x : convert_to_int(x))

X['economic'] = X['economic'].apply(lambda x : convert_to_int(x))

X['addictedhome'] = X['addictedhome'].apply(lambda x : convert_to_int(x))

X['trauma'] = X['trauma'].apply(lambda x : convert_to_int(x))

X['relation'] = X['relation'].apply(lambda x : convert_to_int(x))

X['alone'] = X['alone'].apply(lambda x : convert_to_int(x))

X['lostjob'] = X['lostjob'].apply(lambda x : convert_to_int(x))

X['sexual'] = X['sexual'].apply(lambda x : convert_to_int(x))

X['interest'] = X['interest'].apply(lambda x : convert_to_int(x))

X['sleep'] = X['sleep'].apply(lambda x : convert_to_int(x))

X['outsidenight'] = X['outsidenight'].apply(lambda x : convert_to_int(x))

X['weight'] = X['weight'].apply(lambda x : convert_to_int(x))

X['solution'] = X['solution'].apply(lambda x : convert_to_int(x))

X['addictedfriend'] = X['addictedfriend'].apply(lambda x : convert_to_int(x))

X['reason'] = X['reason'].apply(lambda x : convert_to_int(x))

#X['addicted'] = X['addicted'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

def output_int(word):
    word_dict = {'yes':1, 'no':0}
    return word_dict[word]

y = y.apply(lambda x : convert_to_int(x))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=164)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)



pickle.dump(knn, open('drugmodel.pkl','wb'))

model = pickle.load(open('drugmodel.pkl','rb'))
print(model.predict([[1,0,21,0,1,0,1,0,2,0,0,0,1,10,1,0,0,3,0,1,0,0,0]]))

ans = model.predict([[1,0,21,0,1,0,1,0,2,0,0,0,1,10,1,0,0,3,0,1,0,0,0]])

if ans == 1:
    print("Possibility: Yes")
else:
    print("Possibility: No")




