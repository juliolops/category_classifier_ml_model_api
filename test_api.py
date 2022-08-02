import requests

def test_get_train():

     response = requests.get("http://0.0.0.0:5000/train")
    
     assert response.status_code == 200



def test_post_predict():

     data = {
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
                        
     response = requests.post("http://0.0.0.0:5000/predict", json=data)

     assert response.status_code == 200