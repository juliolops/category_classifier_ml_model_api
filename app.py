from flask import Flask, request, jsonify
from flask_restx import Resource, Api, fields
import pandas as pd
from toolkit_model.preprocessing import *
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

app = Flask(__name__)
api = Api(app)



global category_model


category_model = None


insert_user_data = api.model(
    "Insert_user_data",
    {
        "query": fields.List(cls_or_instance=fields.String(), required=True),
        "search_page": fields.List(cls_or_instance=fields.Integer(), required=True),
        "position": fields.List(cls_or_instance=fields.Integer(), required=True),
        "title": fields.List(cls_or_instance=fields.String(), required=True),
        "concatenated_tags": fields.List(cls_or_instance=fields.String(), required=True),
        "price": fields.List(cls_or_instance=fields.Float(), required=True),
        "weight": fields.List(cls_or_instance=fields.Float(), required=True),
        "express_delivery": fields.List(cls_or_instance=fields.Integer(), required=True),
        "minimum_quantity": fields.List(cls_or_instance=fields.Integer(), required=True),
        "view_counts": fields.List(cls_or_instance=fields.Integer(), required=True)
    },
) 



words_processing_query = ['alianca','tapete','saida','roupa','quadro','pulseira','planner','pet','personalizada','parede',
                        'papel','moeda','ouro','anel','maternidade','lembrancinhas','lembrancinha','kit','caixa','caderno',
                        'cachorro','bolsa','bebe','tecido']


words_processing_title = ['par','tapete','tag','silicone','sabonete','quadro','prata',
                            'pet','personalizado','personalizada','parede','aliancas','papel',
                            'anel','ouro','maternidade','lembrancinha','croche','caixa','caderno',
                            'bolsa','berco','bebe','tecido']


words_processing_title = ['alianca','maternidade','quarto','quadros','pulseiras','prata','pet',
                            'parede','pais','moeda','lacos','lembrancinhas','lembrancinha','aliancas',
                            'fitas','decoracao','cha','casa','bebe','baby','tags']


list_dummies = ['query_alianca', 'query_anel',
                'query_bebe', 'query_bolsa', 'query_caderno', 'query_caixa',
                'query_kit', 'query_lembrancinha', 'query_lembrancinhas_lembrancinha',
                'query_maternidade', 'query_ouro', 'query_papel', 'query_personalizada',
                'query_pet', 'query_planner', 'query_pulseira', 'query_quadro',
                'query_roupa', 'query_saida', 'query_tapete', 'query_tecido',
                'concatenated_tags_alianca', 'concatenated_tags_alianca_aliancas',
                'concatenated_tags_baby', 'concatenated_tags_bebe',
                'concatenated_tags_casa', 'concatenated_tags_cha',
                'concatenated_tags_decoracao', 'concatenated_tags_fitas',
                'concatenated_tags_lacos', 'concatenated_tags_lembrancinha',
                'concatenated_tags_lembrancinhas_lembrancinha',
                'concatenated_tags_maternidade', 'concatenated_tags_pais',
                'concatenated_tags_pet', 'concatenated_tags_prata',
                'concatenated_tags_pulseiras', 'concatenated_tags_quadros',
                'concatenated_tags_quarto', 'concatenated_tags_tags', 'title_aliancas',
                'title_anel', 'title_bebe', 'title_berco', 'title_bolsa',
                'title_caderno', 'title_caixa', 'title_lembrancinha', 'title_papel',
                'title_par', 'title_pet', 'title_quadro', 'title_sabonete', 'title_tag',
                'title_tapete', 'title_tecido']




@api.route("/predict")
class database(Resource):
    @api.expect(insert_user_data)
    def post(self):
        
        global category_model

        #Parsing json
        input_json = request.get_json(force=True) 

        
        #Convert into DataFrame
        df_input = pd.DataFrame(input_json)


        #Preprossecing Data
        df_input["query_cleaned"] = df_input["query"].apply(lambda x: text_cleaner(x))
        df_input["concatenated_tags_cleaned"] = df_input["concatenated_tags"].apply(lambda x: text_cleaner(x))
        df_input["title_cleaned"] = df_input["title"].apply(lambda x: text_cleaner(x))


        df_input["query_cleaned_modified"] = df_input["query_cleaned"].apply(lambda x: text_categorizer(text=x,labels=words_processing_query))
        df_input["concatenated_tags_cleaned_modified"] = df_input["concatenated_tags_cleaned"].apply(lambda x: text_categorizer(text=x,labels=words_processing_query))
        df_input["title_cleaned_modified"] = df_input["title_cleaned"].apply(lambda x: text_categorizer(text=x,labels=words_processing_query))

        df_input = create_dummies(df_input,list_dummies)
        df_input.drop(['query_cleaned_modified',
                        'concatenated_tags_cleaned_modified',
                        'title_cleaned_modified',
                        'query_cleaned',
                        'concatenated_tags_cleaned',
                        'title_cleaned',"title","concatenated_tags","query"],inplace=True,axis=1)


        if category_model:

            predict = category_model.predict(df_input)

            predict_proba = category_model.predict_proba(df_input)
            
            predict_proba = [max(i) for i in predict_proba.tolist()]

            return jsonify({"class_pred":predict.tolist(),"class_proba":predict_proba})

        else:

            return jsonify({"model status":"Model isn't trained yet, request train endpoint"})


@api.route("/train")
class database(Resource):

    def get(self):

        #load data
        df_input = pd.read_csv("data/category_dataset.csv")
        df_input = df_input[["query","search_page","position","title","concatenated_tags","price","weight","express_delivery","minimum_quantity","view_counts","category"]] 
        df_input = df_input.dropna()

        #Preprossecing Data
        df_input["query_cleaned"] = df_input["query"].apply(lambda x: text_cleaner(x))
        df_input["concatenated_tags_cleaned"] = df_input["concatenated_tags"].apply(lambda x: text_cleaner(x))
        df_input["title_cleaned"] = df_input["title"].apply(lambda x: text_cleaner(x))


        df_input["query_cleaned_modified"] = df_input["query_cleaned"].apply(lambda x: text_categorizer(text=x,labels=words_processing_query))
        df_input["concatenated_tags_cleaned_modified"] = df_input["concatenated_tags_cleaned"].apply(lambda x: text_categorizer(text=x,labels=words_processing_query))
        df_input["title_cleaned_modified"] = df_input["title_cleaned"].apply(lambda x: text_categorizer(text=x,labels=words_processing_query))

        df_input = create_dummies(df_input,list_dummies)
        df_input.drop(['query_cleaned_modified',
                        'concatenated_tags_cleaned_modified',
                        'title_cleaned_modified',
                        'query_cleaned',
                        'concatenated_tags_cleaned',
                        'title_cleaned',"title","concatenated_tags","query"],inplace=True,axis=1)


        X = df_input.drop(["category"],axis=1)
        y = df_input["category"].values



        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf_rf = RandomForestClassifier().fit(X_train, y_train)

        y_pred =  clf_rf.predict(X_test)

        global category_model

        category_model = clf_rf

        return jsonify({"model status":"Model is trained",
                        "f1_score_macro":f1_score(y_test, y_pred, average='macro'),
                        "precision_macro":precision_score(y_test, y_pred, average='macro'),
                        "recall_macro":recall_score(y_test, y_pred, average='macro')})


if __name__ == '__main__':

    app.run(host='0.0.0.0')
