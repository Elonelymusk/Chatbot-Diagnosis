from nltk_lib import prep_dataset, build_model, response

# Create the model
prep = prep_dataset()

words = prep[0]
classes = prep[1]
train_x = prep[2]
train_y = prep[3]

build_model(train_x, train_y)

# Query a response
response(words=words, classes=classes, sentence='Hello!')
response(words=words, classes=classes, sentence='What do you sell?')
response(words=words, classes=classes, sentence='Bye!')
