# category_classifier_ml_model_api

## Abstract

This project is a API to classify products informations into 6 classes with machine learning model Random Forests. These classes are "Decoração", "Papel e Cia", "Outros", "Bebê", "Lembrancinhas" and "Bijuterias e Jóias". This API was developed in Python language with Flask web framework and It is containerized so that it is runned using docker-compose up command. This application is documented by swagger so it is possible for the user to manipulate the api using the swagger UI. 

---

## Swagger UI

![image](https://user-images.githubusercontent.com/40969977/182303397-0b75c291-d3b9-4891-a5dd-cb2aaf556f8c.png)
---

## Technologies

- Flask
- Docker
- Machine Learning 
- Jupyter Notebook

---

## Command to Execute all aplication

```docker-compose up command```

## Endpoint to predict classes (POST)

```http://0.0.0.0:5000/predict```

This route receive post requests with products data and its response is a json with class predictions and probabilities  


```
Example of json to predict scores

{
  "query": [      
     "espirito santo","cartao de visita","expositor de esmaltes","medidas lencol para berco americano"
  ],
  "search_page": [
    2,2,1,1	
  ],
  "position": [
    6,0,38,6
  ],
  "title": [
    "Mandala Espirito Santo","Cartao de Visita","Organizador expositor p 70 esmaltes","Jogo de Lençol Berço Estampado"	
  ],
  "concatenated_tags": [
    "mandala mdf","cartao visita panfletos tag adesivos copos lon...","expositor","t jogo lencol menino lencol berco"
  ],
  "price": [
    171.89,77.67,73.92,118.77
  ],
  "weight": [
    1200,8,2709,0
  ],
  "express_delivery": [
    1,1,1,1
  ],
  "minimum_quantity": [
    4,5,1,1
  ],
  "view_counts": [
    244,124,59,180
  ]
}
```

```
Example of response

{
  "class_pred": [
    "Decoração",
    "Papel e Cia",
    "Outros",
    "Bebê"
  ],
  "class_proba": [
    0.48,
    0.75,
    0.67,
    0.74
  ]
}

```

```
Example of response if the model isn't trained 

{
  "model status": "Model isn't trained yet, request train endpoint"
}
```
---

## Endpoint to train the machine learning model (GET)

```http://0.0.0.0:5000/train```

This route receive **get request** and train the machine learning model


```
Example of response

{

  "model status": "Model is trained",
  "f1_score_macro": 0.53,
  "precision_macro": 0.65,
  "recall_macro": 0.50
}
```



---
# Tests

The python file called **test_api.py** in the root directory has 2 aplication tests (one for each route). These tests are executed by the **library pytest** using the command **py.test**. You have to run **py.test** in the root directory. These tests verify if the 2 endpoints return code 200 (status okay).


---
# Model Development

Results of the model (Random Forests)

![image](https://user-images.githubusercontent.com/40969977/182295562-74171382-36ef-4bf1-839a-4a92fa905542.png)


---

## Directory *training*

This directory has the Jupyter notebook called *model_development_pipeline.ipynb* which was used to develop the machine learning model 


---

## Directory *Data* 

This directory has the data used to develop the machine learning model 


---

## Module *toolkit_model*

This module has all functions used in the preprocessing of the model 

---

## Dependencies 

requirements.txt has all the dependencies of this aplication