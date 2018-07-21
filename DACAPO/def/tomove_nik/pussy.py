import numpy as np
import pandas as pd


config_predictions = {
    "left_outs": [5],
    "top_from_predictions": [10]
}

# utility function to load parameters from file
def load_config(filename="input.dat"):
    f = open(filename)
    s = f.readlines()
    result = {}
    for p in s:
        params = p.replace("\n","").split(" ")
        result[params[0]]=params[1]
	
    return result


def train_generator_softmax(ratings, users_pool, n_movies, left_out=1):
	
	while(True):
		user = np.random.choice(users_pool)
		d = ratings[ ratings['userID']==user]
		d.sort_values(['time'])
		user_movies_x = d.iloc[ : (d.shape[0] - left_out), 1 ]
		user_movies_y = d.iloc[ d.shape[0] - left_out : , 1 ]

		
		X_train = np.zeros((1, d.shape[0] - left_out, n_movies))
		y_train = np.zeros((1, n_movies))
		
		for _ in range((d.shape[0] - left_out)):
			X_train[ 0, _, user_movies_x.iloc[_] ] = 1
		
		for _ in range(left_out):
			y_train[ 0, user_movies_y.iloc[_] ] = 1/left_out

		
		yield np.array(X_train), np.array(y_train)
		

#
# train generator for the sps, return just the last element of the seq, not the softmax of the last ones
#


def train_generator_sps(ratings, users_pool, n_movies, left_out=1):
	
	while(True):
		user = np.random.choice(users_pool)
		d = ratings[ ratings['userID']==user]
		d.sort_values(['time'])

		user_movies_x = d.iloc[ : (d.shape[0] - left_out), 1 ]
		user_movies_y = d.iloc[ d.shape[0] - left_out , 1 ]
		
		X_train = np.zeros((1, d.shape[0] - 1, n_movies))
		y_train = np.zeros((1, n_movies))
		
		
		for _ in range((d.shape[0] - left_out)):
			X_train[ 0, _, user_movies_x.iloc[_] ] = 1
		
		y_train[ 0, user_movies_y ] = 1

		
		yield np.array(X_train), np.array(y_train)
    

#
# this function returns "left_out" movies actually seen by the user and the "top_from" predicted by 
# the model using the sequence up to "left_out" movies
#
def predict_user(user, in_model, train_gen, left_out, top_from, ratings, n_movies):

	x, y = next(train_gen(ratings, np.array([user]), n_movies, left_out=left_out))
	real_movies = y[0].argsort()[ -left_out : ]
	# print("y[0, real_movies] {}".format(y[0, real_movies]))
	prediction = in_model.predict(x)
	predicted_movies = prediction[0].argsort()[ -top_from : ]

	return real_movies, predicted_movies


def make_predictions(model, train_gen, users, left_out, top_from, ratings, n_movies):
    result = []
    for user in users:
        real, pred = predict_user(user, model, train_gen, left_out, top_from, ratings, n_movies)
        result.append({"user":user, "real":real, "pred":pred})
    
    return result



def evaluate_predictions(predictions, method):
    mean = 0
    m2 = 0
    n= len(predictions)
    if(method == "precision"):
        for p in predictions:
            s = np.sum(np.in1d(p["real"], p["pred"]))/ p["pred"].shape[0]
            mean += s/n
            m2 += (s**2)/(n-1)
    elif(method == "recall"):
        for p in predictions:
            s = np.sum(np.in1d(p["real"], p["pred"]))/ p["real"].shape[0]
            mean += s/n
            m2 += (s**2)/(n-1)
    elif(method == "sps"):
        for p in predictions:
            s = np.sum(np.in1d(p["real"], p["pred"]))
            mean += s/n
            m2 += (s**2)/(n-1)

    sd = np.sqrt(m2 - (n/(n-1))*mean**2)
    return {"score": mean, "sd": sd}    


def evaluate_model(model, users, ratings, n_movies):
    result = {"left_out":[], "top_from":[], "method": [], "score":[], "sd":[]}

    def appendToResult(left_out, top_from, evaluation, method):
        result["left_out"].append(left_out)
        result["top_from"].append(top_from)
        result["method"].append(method)
        result["score"].append(evaluation["score"])
        result["sd"].append(evaluation["sd"])

    for left_out in config_predictions["left_outs"]:
        for top_from in config_predictions["top_from_predictions"]:
            print("loop {} {}".format(left_out, top_from))
            predictions = make_predictions(model, train_generator_softmax, users, left_out, top_from, ratings, n_movies)
            print("ev")
            evaluation = evaluate_predictions(predictions, "precision")
            appendToResult(left_out, top_from, evaluation, "precision")
            print("p")
            evaluation = evaluate_predictions(predictions, "recall")
            appendToResult(left_out, top_from, evaluation, "recall")
            print("r")
            predictions = make_predictions(model, train_generator_sps, users, left_out, top_from, ratings, n_movies)
            print("ev")
            evaluation = evaluate_predictions(predictions, "sps")
            appendToResult(left_out, top_from, evaluation, "sps")
            print("s")
        
    return pd.DataFrame.from_dict(result)













def make_vectors(users, ratings):

	seqs = {}

	for user in users:
		d = ratings[ ratings['userID']==user]
		d.sort_values(['time'])
		seqs[user] = d.iloc[:, 1].values

	return seqs







def train_generator(seqs, users_pool, n_movies, max_left_out, kept):
	
	while(True):
		if users_pool.shape == ():
			user = int(users_pool)
			# print(user)
		else:
			user = np.random.choice(users_pool)

		s_len = len(seqs[user])
		

		left_out = np.random.choice(np.arange(max_left_out)+1, size=1)[0]
		
		user_movies_x = seqs[ user ][ : s_len - left_out ]
		user_movies_y = seqs[ user ][ s_len - left_out : ]


		
		X_train = np.zeros((1, s_len - left_out, n_movies))
		y_train = np.zeros((1, n_movies))
		
		for _ in range((s_len - left_out)):
			X_train[ 0, _, user_movies_x[_] ] = 1
			
		kept = min(left_out, kept)
		
		for _ in range(kept):
			y_train[ 0, user_movies_y[_] ] = 1/kept


		yield np.array(X_train), np.array(y_train)






def predict_users_all(users, vec, model, left_out, top_from, n_movies):
        
    evals = np.zeros((3, users.shape[0]))
    
        
    for u in range(users.shape[0]):
        
        seq = vec[users[u]]
        real = seq[ -left_out : ]
        x, y = next( train_generator(vec, np.array(users[u]), n_movies, left_out, 1 ) )
        
        prediction = model.predict(x)
        predicted = prediction[0].argsort()[ -top_from : ]
        
        #print("pred.shape {} = top {}".format(predicted.shape[0], top_from))
        #print("real.shape {} = top {}".format(real.shape[0], left_out))
        
        
        
        
        # precision
        evals[0, u] = np.sum(np.in1d(real, predicted)) / predicted.shape[0]
            
        # recall
        evals[1, u] = np.sum(np.in1d(real, predicted)) / real.shape[0]
        
        # sps
        evals[2, u] = np.sum(np.in1d(real[0], predicted))
            
    return evals    # the vector of measures for each user
    



# d = {"left_out" : [], "top_from" : [], "measure" : [], "mean" : [], "sd" : [] }

def append_to_dict(d, stat, left_out, top_from, model):
    
    mm = np.mean(stat, axis=1)
    ss = np.std(stat, axis=1)
    
    d["model_name"].append(model)
    d["left_out"].append(left_out)
    d["top_from"].append(top_from)
    d["measure"].append("precision")
    d["mean"].append(mm[0])
    d["sd"].append(ss[0])
    
    d["model_name"].append(model)
    d["left_out"].append(left_out)
    d["top_from"].append(top_from)
    d["measure"].append("recall")
    d["mean"].append(mm[1])
    d["sd"].append(ss[1])

    d["model_name"].append(model)
    d["left_out"].append(left_out)
    d["top_from"].append(top_from)
    d["measure"].append("sps")
    d["mean"].append(mm[2])
    d["sd"].append(ss[2])
    
    return d   
