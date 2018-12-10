import requests
import sys
import os

#CURRENT_DIR = os.path.dirname(__file__)
#file_path = os.path.join(CURRENT_DIR, 'test.txt')
#f = open(file_path,"w")
location = ""
for i in range(1,len(sys.argv)):
	# f.write(sys.argv[i])
	if(i!=1 and i!= len(sys.argv)):
		location = location +",+"+ sys.argv[i]
	else:
		location = location + sys.argv[i]
#f.write(location)
#f.close()
address = 'https://maps.googleapis.com/maps/api/geocode/json?address='+location+'&key={yourkey}'
response = requests.get(address)
resp_json_payload = response.json()

print(resp_json_payload['results'][0]['geometry']['location']['lat'])
print(resp_json_payload['results'][0]['geometry']['location']['lng'])