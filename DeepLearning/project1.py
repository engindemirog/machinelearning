import pandas as pd
import numpy as np
import psycopg2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#1 connect to the database
connection = psycopg2.connect(host="localhost", dbname="northwind", user="postgres", password="12345", port=5432)

query = """
with last_order_date as
(
  select max(order_date) as max_date from orders
),
customer_order_stats as (
select 
c.customer_id,
count(o.order_id) as total_orders,
sum(od.unit_price*od.quantity) as total_spent,
avg(od.unit_price*od.quantity) as avg_order_value
from orders o
inner join customers c
on o.customer_id = c.customer_id
inner join order_details od
on od.order_id = o.order_id
group by c.customer_id),
label_data as(
 select c.customer_id,
	case when exists(
	 select 1 from orders o2,last_order_date lod
		where o2.customer_id = c.customer_id
		and o2.order_date>(lod.max_date-Interval '6 months')
	)
	then 1 else 0
	end as will_order_again
	from customers c
)
select 
s.customer_id,
s.total_orders,
s.total_spent,
s.avg_order_value,
l.will_order_again
from customer_order_stats s join label_data l
on s.customer_id = l.customer_id
"""

df = pd.read_sql(query,connection)
connection.close()

X = df[["total_orders","total_spent","avg_order_value"]]
y = df["will_order_again"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(8,activation="relu",input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(4,activation="relu"),
        tf.keras.layers.Dense(1,activation="sigmoid")
    ]
)

model.compile(optimizer="adam",loss="mean_squared_error",metrics=["accuracy"])

model.fit(X_train,y_train,epochs=50,validation_data=(X_test,y_test),verbose=1)

loss,acc = model.evaluate(X_test,y_test)
print(acc)