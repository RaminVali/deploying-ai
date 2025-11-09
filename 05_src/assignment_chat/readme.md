# Jarvis the Research Assistant

In this project we make Jarvis a chatbot research assistant that can perform several services:

### Service 1 
API calls to arxiv (can expand it later if I have time as semantic scholar is free). This is the first service. 


### Service 2 
Perform semantic searches based on user query n its local database. 
TO construct the local database, Idownloaded a large csv file from [Kallgle](https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset?resource=download)
The file is larger than 40 MB so I trimmed it down (kept the first 30,000 rows)
To construct the embedding I did the fllowing in a notbook:




Note: I trided doing this in a .py file but the same path that worked in the same directory for the ipynb file failed to work for the .py file (probably ipton kernel is in a different directory that the execution for the .py file)


Listed the required columnes, and checks if the required columsn are missing (title, author publication date etc.)

Then loaded them in a pandas df as backend,

I then initialised a chromadb client inthe same direcgroy, I used the Persistet Client 

I made a list of the document ids and summaries and then I kept the metadata as ["title", "authors", "published_date", "category"] and put them in a dictionary

I then fed the rows of my csv file in batches of 100 to be vectorised intot he chromadb client I had made and I used OpenAIEmbeddings to create the emneddings using text-embedding-3-small