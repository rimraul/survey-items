import pyreadstat
import numpy as np
import pandas as pd
from pyvis import network as net

file_name="ESS10_labels.csv"
df = pd.read_csv(file_name)
df.columns=["name","label"]

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

embeddings = model.encode(df['label'].to_list(), convert_to_tensor=True)

from sentence_transformers import util

cosine_scores = util.cos_sim(embeddings, embeddings)


pairs = []
for i in range(len(cosine_scores)-1):
    for j in range(i+1, len(cosine_scores)):
        pairs.append({'index': [i, j], 'score': cosine_scores[i] 
                                                             [j]})

weights_list = [abs(w['score'].item()) for w in pairs]

edge_threshold = np.percentile(np.asarray(weights_list),95)
edge_threshold

G_pyvis=net.Network('800px', '1000px') #(notebook=True) #('8000px', '8000px')

# value_range = [-0.1,0.4]
# value_range = [-edge_threshold,edge_threshold]
for edge in pairs:
    f,t,value=edge['index'][0],edge['index'][1],float(edge['score'].item())
    # if value not in value_range:
    if abs(value) > edge_threshold:
        # print(f,t,value)
        if value > 0:
            color='#9aceeb'
        else:
            color='#f2ad85'
        # print(color)
        # print(a['data'].item())
        # print(f,t,a)

        G_pyvis.add_node(f,label=df['label'].loc[f])
        G_pyvis.add_node(t,label=df['label'].loc[t])
        # G_pyvis.add_edge(f,t, weight=value)
        G_pyvis.add_edge(f,t,value=value,title =round(value,2),color=color)

from datetime import datetime
now = datetime.now()
date_time_stamp = now.strftime('%Y-%m-%d_%H%M') 

network_name='Semantic'
G_pyvis.toggle_physics(True)
G_pyvis.show_buttons(filter_=True) #(filter_=['physics'])
output_name=f'{network_name}_pyvis_{date_time_stamp}.html'
G_pyvis.show(output_name)


st.title('Semantic similarity Network of Questionnaire Items')

# https://towardsdatascience.com/how-to-deploy-interactive-pyvis-network-graphs-on-streamlit-6c401d4c99db
# Save and read graph as HTML file (on Streamlit Sharing)
try:
   path = '/tmp'
   G_pyvis.save_graph(f'{path}/pyvis_graph.html')
   HtmlFile = open(f'{path}/pyvis_graph.html','r',encoding='utf-8')
# Save and read graph as HTML file (locally)
except:
    path = '/html_files'
    G_pyvis.save_graph(f'{path}/pyvis_graph.html')
    HtmlFile = open(f'{path}/pyvis_graph.html','r',encoding='utf-8')
    
    
# Load HTML into HTML component for display on Streamlit
components.html(HtmlFile.read())


