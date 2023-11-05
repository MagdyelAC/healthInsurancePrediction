import requests

body = {
    'age' : 19,
    'sex' : 1,
    'bmi' : 27.9,
    'children' : 0,
    'smoker' : 1,
    'region' : 4
}
response = requests.post(url = 'http://127.0.0.1:8000/prediction',
json = body)
print("Approx: 16884.924")
print (response.json())

body = {
    'age' : 50,
    'sex' : 0,
    'bmi' : 30.97,
    'children' : 3,
    'smoker' : 0,
    'region' : 2
    }
response = requests.post(url = 'http://127.0.0.1:8000/prediction',
json = body)
print("Approx: 10600.5483")
print (response.json())

body = {
    'age' : 61,
    'sex' : 1,
    'bmi' : 29.07,
    'children' : 0,
    'smoker' : 1,
    'region' : 2
    }
response = requests.post(url = 'http://127.0.0.1:8000/prediction',
json = body)
print("Approx: 29141.3603")
print (response.json())
# output: {'score': 0.866490130600765}