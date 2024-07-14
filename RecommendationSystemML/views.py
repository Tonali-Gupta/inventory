from django.http import JsonResponse
from sklearn.externals import joblib
import traceback
import string
import pandas as pd
import os

def home(request):
    return JsonResponse({'code' : 'success', 'response': 'Welcome!', 'message': 'Recommended for LDP Recommendations!'})

def preprocess(text):
    #text = text.replace(" ","")
    text = text.lower()
    return text

def sort_recommendation(query,recommendation):
    df1 = recommendation.loc[(recommendation['category_detail_code']==query[9]) & (recommendation['location_code']==query[5]) & (recommendation['selling_price']>=(query[1]-query[1]*0.15)) & (recommendation['selling_price']<= (query[1]+query[1]*0.15)) & (recommendation['body_type_code']==query[6])]
    if query[3] != -1 :
        df2 = df1.loc[df1['model_code']==query[3]]
    else :
        df2 = df1
    if query[6] != -1 and query[2] !=1 :
        df3 = df1.loc[(df1['body_type_code']==query[6]) & (df1['make_code']==query[2])]
    else :
        df3 = df1
    if query[6] != -1 :
        df4 = df1.loc[df1['body_type_code']==query[6]]
    else :
        df4 = df1
    return  pd.concat([df2,df3,df4,df1]).drop_duplicates(keep='first').reset_index(drop=True)

def predict(request):
    module_dir = os.path.dirname(__file__)
    model_file_path = os.path.join(module_dir, 'recommendation_model.pkl')
    csv_file_path = os.path.join(module_dir, 'Recommendation_cluster_data.csv')
    model = joblib.load(model_file_path)

    if model:
        try:
            data  = request.GET;
            query = {}
            codes = []
            input_list = ['year', 'selling_price', 'make', 'model', 'trim', 'location', 'body_type', 'fuel_type', 'condition', 'category_detail'];
            for key in input_list:
                if key not in data :
                    query[key] = "-1";
                else :
                    key_value = preprocess(data[key])
                    query[key] = key_value;
            codes_var = ['make', 'model', 'trim', 'location', 'body_type', 'fuel_type', 'condition', 'category_detail'];
            for item in codes_var:
                df1 = pd.read_csv(os.path.join(module_dir, 'codes/' +item + '_codes.csv'))
                result = list(df1.index[df1[item] == query[item]])
                if(len(result) == 0):
                    codes.append(-1)
                else :
                    codes.append(df1.loc[result[0]][item + '_code'])

            list1 = [int(query['year']),int(query['selling_price'])]
            list1 = list1 + codes
            list1 = [list1, list1]
            final_cluster = model.predict(list1)[0]
            recommendation = pd.read_csv(os.path.join(module_dir, 'clusters/cluster_data_'+str(final_cluster)+'.csv'))
            recommendation  = sort_recommendation(list1[0],recommendation)
            lids = list(recommendation['lid'])
            if len(lids) > 20 :
                lids = lids[0:20]
            response = {'code' : 'success', 'data': lids }
            resp = JsonResponse(response)
            resp.status_code = 200
            return resp

        except:
            return JsonResponse({'code' : 'failed', 'error': traceback.format_exc()})
    else:
        print ('Train the model first')
        return JsonResponse({'code' : 'failed', 'error': 'No model found'})
