config_predictions = {
    left_outs: [1,2,3,4,5],
    top_from_predictions: [1,2,3,4,5,6,7,8,9,10]
}

# utility function to load parameters from file
def load_config(filename="input.dat"):
    f = open(filename)
    s = f.readlines()
    result = {}
    for p in s:
        params = p.split(" ")
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
		print(d)
		user_movies_x = d.iloc[ : (d.shape[0] - left_out), 1 ]
		user_movies_y = d.iloc[ d.shape[0] - left_out , 1 ]
		print(user_movies_y)
		
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
def predict_user(user, in_model, train_gen, left_out=1, top_from=10):

	x, y = next(train_gen(np.array([user]), left_out=left_out))
	real_movies = y[0].argsort()[ -left_out : ]
	# print("y[0, real_movies] {}".format(y[0, real_movies]))
	prediction = in_model.predict(x)
	predicted_movies = prediction[0].argsort()[ -top_from : ]

	return real_movies, predicted_movies


def make_predictions(model, train_gen, users, left_out=1,top_from=5)
    for user in users:
        real, pred = predict_user(user, model, train_gen, left_out, top_from)



def evaluate_predictions(predictions)
    
