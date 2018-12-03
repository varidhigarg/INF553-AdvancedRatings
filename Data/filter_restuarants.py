import json

restaurant = []
reviews = []
businesses = set()

		
with open("yelp_academic_dataset_business.json", encoding ="utf-8") as f:
	for x in f.readlines():
		a = json.loads(x)
		id = a["business_id"]
		if a["attributes"]:
			for y in a["attributes"]:
				attr = y.lower()
				if "restaurant" in attr:
					businesses.add(id)
					restaurant.append(json.dumps(a))
					break
		
		
print("Now writing business")		
with open('restaurant_data_2.json', 'w') as outfile:
    outfile.write("\n".join(restaurant))

print("Done writing business")
with open("yelp_academic_dataset_review.json", encoding ="utf-8") as f:
	for x in f.readlines():
		a = json.loads(x)
		id = a["business_id"]
		if id in businesses:
			reviews.append(json.dumps(a))
		
		
		
print("Now Writing")
with open('restaurant_review_2.json', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(reviews))

print("Done")
print(len(reviews),len(businesses))