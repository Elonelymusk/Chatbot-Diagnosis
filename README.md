# chatbot-diagnosis
A chat-bot designed to provide a self-diagnosis for the user based on NLP.

## Files:
 * diag_chat_basic   ->		This is a basic implementation of the django asynchronous chat web-app. I'm keeping this here as a backup, just in case I mess up future versions.

 * diag_chat_basic_v2 	->	Still a basic implementation, but has random room-key generation and a slighly more comprehensive layout. Kept as a backup for future versions.

 * DEMO_CHATBOT_WITH_REDDIT_DATASET.py -> This is just a demo of a chatbot model using SQL Database with about 51 million reddit comments to train the model. Keeping this here temporarily for future reference.

 * NLTK.CHATBOT.ipynb ->	This is a fully trained chatbot using NLTK using `intents.json` file for training. However the 'intents' file is specified for moped rental services. This could be changed for specific things in our case diagnosis.

 * nltk_lib.py	->		Basically the same as the `.ipynb` file, but cleaned up slightly and converted into a library of individually callable functions.

 * chatbot_controller.py ->	A demo file of how to use the `nltk_lib.py` file to make repeated response queries.

## ToDo:
 * Model:
 * [ ] Look at the symptoms-disease dataset and try to derive a suitable `intents.json` file from it.
 * [ ] Optimize the model.
 * Website:
 * [ ] Implement a nicer UI - vue.js? angular? react?
 * *After everything else is done:*
 * [ ] Consider buying a cheap domain name.
 * [ ] Host the site somehow and make it public?
 * After that, everything should be finished.
