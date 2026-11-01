import pandas as pd

df = pd.read_csv("data/for_assessment.csv")
print(df.shape)
dates = pd.to_datetime(df["InitialDateReported"], format="%m/%d/%Y")
print(dates.min(), dates.max())
names_per_cas = df.groupby("CasNumber")["ChemicalName"].nunique()
multi = names_per_cas[names_per_cas > 1]
print("CAS numbers with more than one name:", len(multi))
print(multi)
print(df[df["CasNumber"]=="79-81-2" ]["ChemicalName"].unique())
print(df.isna().sum())
no_cas = df[ df["CasNumber"].isna()] 
print(no_cas["ChemicalName"].value_counts())
companies_per_brand = df.groupby("BrandName")["CompanyName"].nunique()
multi_brand = companies_per_brand[companies_per_brand > 1]
print("Brands under more than one company:", len(multi_brand))
print(multi_brand.sort_values(ascending=False).head(10))