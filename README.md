# category_classifier_ml_model_api

## Abstract

This project is a API to classify products informations into 6 classes with machine learning model Random Forests. These classes are "Decoração", "Papel e Cia", "Outros", "Bebê", "Lembrancinhas" and "Bijuterias e Jóias". This API was developed in Python language with Flask web framework and It is containerized so that it is runned using docker-compose up command. This application is documented by swagger so it is possible for the user to manipulate the api using the swagger UI. 

---

## Swagger UI

---

## Technologies

- Flask
- MySQL
- Docker

---

## Command to Execute all aplication

```docker-compose up command```

## Endpoint to insert values into project_name.table_name (POST)

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
---
