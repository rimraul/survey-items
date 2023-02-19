import pyreadstat
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pyvis import network as net
import faiss
import streamlit as st
import streamlit.components.v1 as components

# 0. Generate pyvis graphs dynamically from github (unable to acomplish so far)
# 1. (Alternative) Generate pyvis graphs locally and upload to hithub, e.g.:

# file_name="ESS10_labels.csv"
# df = pd.read_csv(file_name)
# df.columns=["name","label"]

# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# embeddings = model.encode(df['label'].to_list(), convert_to_tensor=True)

# from sentence_transformers import util

# cosine_scores = util.cos_sim(embeddings, embeddings)


# pairs = []
# for i in range(len(cosine_scores)-1):
#     for j in range(i+1, len(cosine_scores)):
#         pairs.append({'index': [i, j], 'score': cosine_scores[i] 
#                                                              [j]})

# weights_list = [abs(w['score'].item()) for w in pairs]

# edge_threshold = np.percentile(np.asarray(weights_list),95)
# edge_threshold

# G_pyvis=net.Network('800px', '1000px') #(notebook=True) #('8000px', '8000px')

# # value_range = [-0.1,0.4]
# # value_range = [-edge_threshold,edge_threshold]
# for edge in pairs:
#     f,t,value=edge['index'][0],edge['index'][1],float(edge['score'].item())
#     # if value not in value_range:
#     if abs(value) > edge_threshold:
#         # print(f,t,value)
#         if value > 0:
#             color='#9aceeb'
#         else:
#             color='#f2ad85'
#         # print(color)
#         # print(a['data'].item())
#         # print(f,t,a)

#         G_pyvis.add_node(f,label=df['label'].loc[f])
#         G_pyvis.add_node(t,label=df['label'].loc[t])
#         # G_pyvis.add_edge(f,t, weight=value)
#         G_pyvis.add_edge(f,t,value=value,title =round(value,2),color=color)

# from datetime import datetime
# now = datetime.now()
# date_time_stamp = now.strftime('%Y-%m-%d_%H%M') 

# network_name='Semantic'
# G_pyvis.toggle_physics(False)
# G_pyvis.show_buttons(filter_=True) #(filter_=['physics'])

# # # output_name=f'{network_name}_pyvis_{date_time_stamp}.html'
# # output_name=f'{network_name}_pyvis.html'
# # # G_pyvis.show(output_name)
# # html_code = G_pyvis.save_html(file=output_name)


# # # Commit and push the HTML code to GitHub
# # !git add network.html
# # !git commit -m "Add Semantic_pyvis.html"
# # !git push


# 2. Show the graph in Streamlit

st.title('Semantic Similarity Network of Questionnaire Items')

# st.markdown(G_pyvis, unsafe_allow_html=True)

# # # https://towardsdatascience.com/how-to-deploy-interactive-pyvis-network-graphs-on-streamlit-6c401d4c99db
# # # Save and read graph as HTML file (on Streamlit Sharing)
# # try:
# #     path = '/tmp'
# # #     G_pyvis.save_graph(f'{path}/pyvis_graph.html')
# # #     HtmlFile = open(f'{path}/pyvis_graph.html','r',encoding='utf-8')
# #     HtmlFile = open(f'{path}/Semantic_pyvis_2022-12-27_0016.html','r',encoding='utf-8')
# # # Save and read graph as HTML file (locally)
# # except:
# #     path = '/html_files'
# # #     G_pyvis.save_graph(f'{path}/pyvis_graph.html')
# # #     HtmlFile = open(f'{path}/pyvis_graph.html','r',encoding='utf-8')
# #     HtmlFile = open(f'{path}/Semantic_pyvis_2022-12-27_0016.html','r',encoding='utf-8')

    
import streamlit as st


# if "visibility" not in st.session_state:
#     st.session_state.visibility = "visible"
#     st.session_state.disabled = False
    

option = st.selectbox(
    'Select one of the surveys:',
    ('V-Dem', 'ESS9', 'ESS10'))

files=['V-Dem_pyvis_2023-02-19_1845.html','ESS9_pyvis_2023-02-19_1845.html','ESS10_pyvis_2023-02-19_1845.html']

selected_file = [f for f in files if option in f][0]

# st.write('You selected:', option)    
    
HtmlFile = open(f'tmp/{selected_file}','r',encoding='utf-8')
    
# Load HTML into HTML component for display on Streamlit
components.html(HtmlFile.read(), height=600)


### Add semantic search text box

file_name='Survey items 2023-02-19_1357.csv'
df=pd.read_csv(file_name)
df.columns=['no','survey','item']

model =SentenceTransformer('msmarco-MiniLM-L-12-v3')
items=df["item"].tolist()
items_embds=model.encode(items)

index = faiss.read_index('faiss_index')

def search(query,k=10):
    query_vector = model.encode([query])
    top_k = index.search(query_vector, k)
    results=[items[_id] for _id in top_k[1].tolist()[0]]
    scores=[round(s,2) for s in top_k[0].tolist()[0]]
    results=pd.DataFrame({'score':top_k[0][0],'item':[items[_id] for _id in top_k[1].tolist()[0]]})
    unique_results = results.drop_duplicates(subset=['item'])
    enriched_results=pd.DataFrame(columns=["survey","item",'score'])
    for i,c in unique_results.iterrows():
        multiple=df.loc[df['item'] == c[1]]
        for ind,row in multiple.iterrows():
            enriched_results.loc[len(enriched_results)]= [row[1],row[2],c[0]]
    return enriched_results



query = st.text_input(
    "Enter some text for semantic search in the variable labels 👇")
# ,
#     label_visibility=st.session_state.visibility,
#     disabled=st.session_state.disabled,
#     placeholder=st.session_state.placeholder,
# )

limits = st.text_input(
    "Enter the number of hits to be returned 👇")

if limits:
    limits=int(limits)
else:
    limits=20



if query:
    results=search(query,k=limits)
    results.loc[len(results)]=['Query',query,0.01]
    st.write("The closests items (lowest scores) are: ", results)
    
    
        






