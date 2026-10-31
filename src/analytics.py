import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

REQUIRED = ["city","locality","property_type","bhk","area_sqft","price_inr"]

def clean_properties(data):
    df=data.copy()
    df.columns=[str(c).strip().lower().replace(" ","_") for c in df.columns]
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: "+", ".join(missing))
    for col in ["bhk","area_sqft","price_inr","monthly_rent_inr","property_age_years","amenities_count"]:
        if col in df:
            df[col]=pd.to_numeric(df[col],errors="coerce")
    df=df.dropna(subset=["city","locality","property_type","bhk","area_sqft","price_inr"])
    df=df[(df.area_sqft>0)&(df.price_inr>0)&(df.bhk>0)]
    df=df.drop_duplicates()
    if "price_per_sqft" not in df:
        df["price_per_sqft"]=df.price_inr/df.area_sqft
    else:
        df["price_per_sqft"]=pd.to_numeric(df.price_per_sqft,errors="coerce")
        df["price_per_sqft"]=df.price_per_sqft.fillna(df.price_inr/df.area_sqft)
    if "monthly_rent_inr" not in df:
        df["monthly_rent_inr"]=np.nan
    if "property_age_years" not in df:
        df["property_age_years"]=np.nan
    if "amenities_count" not in df:
        df["amenities_count"]=np.nan
    if "listing_date" in df:
        df["listing_date"]=pd.to_datetime(df.listing_date,errors="coerce")
    for col in ["city","locality","property_type"]:
        df[col]=df[col].astype(str).str.strip()
    return df.reset_index(drop=True)

def build_kpis(df):
    return {
        "count":len(df),
        "median_price":float(df.price_inr.median()),
        "avg_ppsf":float(df.price_per_sqft.mean()),
        "median_area":float(df.area_sqft.median()),
        "median_rent":float(df.monthly_rent_inr.median()) if df.monthly_rent_inr.notna().any() else 0
    }

def location_summary(df):
    out=df.groupby(["city","locality"],as_index=False).agg(
        listings=("price_inr","size"),
        median_price=("price_inr","median"),
        avg_price_per_sqft=("price_per_sqft","mean"),
        median_area=("area_sqft","median")
    )
    out["locality_label"]=out["locality"]+" · "+out["city"]
    return out.sort_values("median_price",ascending=False)

def train_price_model(df):
    features=["city","locality","property_type","bhk","area_sqft","property_age_years"]
    usable=df.copy()
    for f in features:
        if f not in usable:
            usable[f]=np.nan
    usable=usable.dropna(subset=["price_inr","city","locality","property_type","bhk","area_sqft"])
    X=usable[features]
    y=usable["price_inr"]
    cat=["city","locality","property_type"]
    num=["bhk","area_sqft","property_age_years"]
    pre=ColumnTransformer([
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat),
        ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num)
    ])
    model=Pipeline([("preprocess",pre),("regressor",RandomForestRegressor(n_estimators=160,min_samples_leaf=2,random_state=42,n_jobs=-1))])
    Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.2,random_state=42)
    model.fit(Xtrain,ytrain)
    pred=model.predict(Xtest)
    return model,float(r2_score(ytest,pred)),float(mean_absolute_error(ytest,pred)),features

def predict_price(model,city,locality,bhk,area,property_type,age):
    row=pd.DataFrame([{"city":city,"locality":locality,"property_type":property_type,
                       "bhk":bhk,"area_sqft":area,"property_age_years":age}])
    return float(model.predict(row)[0])
