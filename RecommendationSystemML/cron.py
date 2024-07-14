from django.conf import settings
from sklearn.externals import joblib
import traceback
import pandas as pd
import os
import pymongo
from kmodes.kmodes import KModes
from datetime import datetime
import pickle

def codes():
    client = pymongo.MongoClient(host=settings.MONGO_HOST,username=settings.MONGO_USER,password=settings.MONGO_PASSWORD,readPreference='secondaryPreferred')
    db = client["droom"]
    collection = db['cmp_listings']
    data = collection.find({'status':'active', 'condition':'used', 'quantity_available':{"$gt":0}, 'quick_sell':0, 'category_detail.bucket':{"$in":['car','bike','scooter','bicycle']},  'category_detail.bucket':{"$exists":'true', '$ne':''},'make':{"$exists":'true', '$ne':''},'model':{"$exists":'true', '$ne':''},'trim':{"$exists":'true', '$ne':''},'location':{"$exists":'true', '$ne':''},'body_type':{"$exists":'true', '$ne':''},'fuel_type':{"$exists":'true', '$ne':''},'condition':{"$exists":'true', '$ne':''},'year':{"$exists":'true', '$ne':0},'selling_price':{"$exists":'true', '$ne':0},'lid':{"$exists":'true', '$ne':''}},{'category_detail.bucket':1,'make':1,'model':1,'trim':1,'location':1,'body_type':1,'fuel_type':1,'condition':1,'year':1,'selling_price':1,'lid':1})
    df = pd.DataFrame(list(data))
    df = df.apply(lambda x: x.astype(str).str.lower())
    df['lid'] = pd.to_numeric(df['lid'])
    df['year'] = pd.to_numeric(df['year'])
    df['selling_price'] = pd.to_numeric(df['selling_price'])
    df['category_detail'] = df['category_detail'].apply(lambda x: x[12:-2])
    
    module_dir = os.path.dirname(__file__)
    try:
        os.mkdir(os.path.join(module_dir, 'codes'))
        print("Directory codes Created")
    except FileExistsError:
        print("Directory codes already exists")

    try:
        os.mkdir(os.path.join(module_dir, 'clusters'))
        print("Directory clusters Created")
    except FileExistsError:
        print("Directory clusters already exists")

    # df1 temporary dataframe
    df.make = pd.Categorical(df.make)
    df['make_code'] = df.make.cat.codes
    df1 = df[['make', 'make_code']]
    df1.drop_duplicates(subset ="make", keep = 'first', inplace = True)
    df1.to_csv(os.path.join(module_dir, 'codes/make_codes.csv'),index = False)

    df.model = pd.Categorical(df.model)
    df['model_code'] = df.model.cat.codes
    df1 = df[['model', 'model_code']]
    df1.drop_duplicates(subset ="model", keep = 'first', inplace = True)
    df1.to_csv(os.path.join(module_dir, 'codes/model_codes.csv'),index = False)

    df.condition = pd.Categorical(df.condition)
    df['condition_code'] = df.condition.cat.codes
    df1 = df[['condition', 'condition_code']]
    df1.drop_duplicates(subset ="condition", keep = 'first', inplace = True)
    df1.to_csv(os.path.join(module_dir, 'codes/condition_codes.csv'),index = False)

    df.location = pd.Categorical(df.location)
    df['location_code'] = df.location.cat.codes
    df1 = df[['location', 'location_code']]
    df1.drop_duplicates(subset ="location", keep = 'first', inplace = True)
    df1.to_csv(os.path.join(module_dir, 'codes/location_codes.csv'),index = False)

    df.body_type = pd.Categorical(df.body_type)
    df['body_type_code'] = df.body_type.cat.codes
    df1 = df[['body_type', 'body_type_code']]
    df1.drop_duplicates(subset ="body_type", keep = 'first', inplace = True)
    df1.to_csv(os.path.join(module_dir, 'codes/body_type_codes.csv'),index = False)

    df.fuel_type = pd.Categorical(df.fuel_type)
    df['fuel_type_code'] = df.fuel_type.cat.codes
    df1 = df[['fuel_type', 'fuel_type_code']]
    df1.drop_duplicates(subset ="fuel_type", keep = 'first', inplace = True)
    df1.to_csv(os.path.join(module_dir, 'codes/fuel_type_codes.csv'),index = False)

    df.trim = pd.Categorical(df.trim)
    df['trim_code'] = df.trim.cat.codes
    df1 = df[['trim', 'trim_code']]
    df1.drop_duplicates(subset ="trim", keep = 'first', inplace = True)
    df1.to_csv(os.path.join(module_dir, 'codes/trim_codes.csv'),index = False)

    df.category_detail = pd.Categorical(df.category_detail)
    df['category_detail_code'] = df.category_detail.cat.codes
    df1 = df[['category_detail', 'category_detail_code']]
    df1.drop_duplicates(subset ="category_detail", keep = 'first', inplace = True)
    df1.to_csv(os.path.join(module_dir, 'codes/category_detail_codes.csv'),index = False)
    # df2 dataframe to train model
    df2 = df[['year','selling_price','make_code','model_code','trim_code','location_code','body_type_code','fuel_type_code','condition_code','category_detail_code']] 
    model = KModes(n_clusters=30, init='Huang', n_init=10, verbose=1,max_iter = 500)
    model.fit(df2)
    df2['cluster'] = model.predict(df2)
    df2['lid'] = df['lid']
    for i in range(30):
        df4 = df2[df2['cluster'] == i]
        df4.to_csv(os.path.join(module_dir, 'clusters/cluster_data_'+str(i)+'.csv'),index = False)
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    model_file_path = os.path.join(module_dir, 'recommendation_model.pkl')
    joblib.dump(model,model_file_path)
    print('')
    print('Model Updated!')
