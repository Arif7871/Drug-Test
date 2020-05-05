import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'family live':1, 'gender':1, 'age':23, 'address':0, 
                            'profession':2, 'distance':1, 'efficiency':2, 'stress':1, 
                            'economic':0, 'addictedhome':0, 'trauma':1, 'relation':1, 
                            'alone':0, 'care':7, 'lostjob':0, 'sexual':0, 'interest':1, 
                            'sleep':1, 'outsidenight':1, 'weight':1, 'solution':1, 
                            'addictedfriend':1, 'reason':1})

print(r.json())


