#Müşterilerin alışveriş davranışlarına göre gruplanması ve aykırı verilerin keşfi

#order_details,customers,orders

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

user = "postgres"
password = "12345"
host = "localhost"
port = "5432"
database = "northwind"

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

query = """
select 
c.customer_id,
count(o.order_id) as total_orders,
sum(od.unit_price*od.quantity) as total_spent,
avg(od.unit_price*od.quantity) as avg_order_value
from customers c inner join orders o
on c.customer_id =o.customer_id
inner join order_details od
on o.order_id = od.order_id
group by c.customer_id
having count(o.order_id)>0
"""

df = pd.read_sql_query(query,engine)
print(df.head())

X = df[["total_orders","total_spent","avg_order_value"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def find_optimal_eps(X_scaled,min_samples=3):
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances,_  = neighbors.kneighbors(X_scaled)

    distances = np.sort(distances[:,min_samples-1])

    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    optimal_eps = distances[kneedle.elbow]

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.axvline(x=kneedle.elbow, color='r', linestyle='--', label=f'Optimal eps: {optimal_eps:.2f}')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{min_samples}-th nearest neighbor distance')
    plt.title('Elbow Method for Optimal eps')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_eps

optimal_eps = find_optimal_eps(X_scaled)
dbscan = DBSCAN(eps=optimal_eps,min_samples=3)

df["cluster"] = dbscan.fit_predict(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(df['total_orders'], df['total_spent'], c=df['cluster'], cmap='plasma', s=60)
plt.xlabel("Toplam Sipariş Sayısı")
plt.ylabel("Toplam Harcama")
plt.title("Müşteri Segmentasyonu (DBSCAN)")
plt.grid(True)
plt.colorbar(label='Küme No')
plt.show()

outliers = df[df["cluster"]==-1]
print("Aykırı veri sayısı : ", len(outliers))
print(outliers[["customer_id","total_orders","total_spent"]])