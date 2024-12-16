import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#importing vanna library

from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore


#Initializing class of Vanna

class MyVanna(ChromaDB_VectorStore,Ollama):
    def __init__(self,config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

vn = MyVanna(config={'model':'llama3.1'})

vn.connect_to_sqlite('olist.sqlite')

#Training the LLM (Ollama) with the schema of the database

df_ddl = vn.run_sql("select type, sql from sqlite_master where sql is not null")
for ddl in df_ddl['sql'].to_list(): #This is going to get the schema of the tables present in the database and we are going to train on that
    vn.train(ddl=ddl)

# Providing some documentation of the data
vn.train(documentation = "This is a comprehensive e-commerce dataset provided by Olist, a Brazilian e-commerce platform. The dataset contains information about orders, customers, products, and sellers, offering a rich source of data for analysis and insights generation.")

vn.train(sql="SELECT customer_state, \
       COUNT(*) AS total_customers, \
       SUM(CASE WHEN order_status = 'delivered' THEN 0 ELSE 1 END) AS churned_customers FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY customer_state")

training_data = vn.get_training_data()

#Asking section
from vanna.flask import VannaFlaskApp
app = VannaFlaskApp(vn, allow_llm_to_see_data=True, title="Homelander.AI", subtitle="I'm not just like ordinary LLMs, I'm smarter, I'm stronger, I'm better!", followup_questions=False, suggested_questions=False, index_html_path="/Users/h347285/Documents/LLM_HACKATHON/assets/index.html", assets_folder="/Users/h347285/Documents/LLM_HACKATHON/assets")
app.run()