import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
import altair as alt

sns.set(style="ticks", context="talk")
plt.style.use('default')


# Chargement du dataset et preparation
def log(func):
    def wrapper(*args,**kwargs):
        with open("log.txt","a") as f:
            debut = time.time()
            value = func(*args,**kwargs)
            fin = time.time()
            f.write("\nCalled function "+ func.__name__ + " in "+ str(fin - debut)+"\n")
            return value
    return wrapper

@st.cache(allow_output_mutation=True)
def load_previous_years():
    df_2016 = pd.read_csv(r'./full_2016.csv')
    df_2017 = pd.read_csv(r'./full_2017.csv')
    df_2018 = pd.read_csv(r'./full_2018.csv')
    df_2019 = pd.read_csv(r'./full_2019.csv')
    return df_2016, df_2017, df_2018, df_2019

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv(r'./full_2020.csv')

    df['adresse_code_voie'] = df['adresse_code_voie'].astype(str)
    df['code_commune'] = df['code_commune'].astype(str)
    df['code_departement'] = df['code_departement'].astype(str)
    df['numero_volume'] = df['numero_volume'].astype(str)
    df.drop(columns = ["lot1_numero", "lot2_numero", "lot3_numero", "lot4_numero", "lot5_numero", "lot1_surface_carrez", "lot2_surface_carrez", "lot3_surface_carrez", "lot4_surface_carrez", "lot5_surface_carrez", "adresse_suffixe", "ancien_nom_commune", "ancien_code_commune", "ancien_id_parcelle", "code_nature_culture", "nature_culture_speciale", "code_nature_culture_speciale", "numero_volume"],  inplace=True)
    # Add new column and compute price per m2 
    df['price_per_m2'] = df['valeur_fonciere']/df['surface_reelle_bati']
    return df

# Fonctions 
@log
def to_plot_bar(to_plot):
    # Affichage dans fenetre à part
    fig = px.bar(to_plot)
    fig.show()

@log
def highest_dept_transactions(df, top = 100):
    """
    get the departements with the highest number of real estate
    input : df : dataframe
    return : list of "highest" departements
    """
    # Groupby to find total number of transactiosn 
    mutation_by_dept = df.groupby(['code_departement']).count()['id_mutation']
    # Then sort descending 
    mutation_by_dept = mutation_by_dept.sort_values(ascending=False).copy()
    return mutation_by_dept.iloc[:top]

@log
def get_most_expensive_com_in_dept(df, dept, top = 100):
    """
    Given a dept number, get the highest communes with the highest and lowest prices per m2
    input : df : dataframe, dept number, highest
    return : list of "highest" communes, list of the lowest communes
    """
    
    # Select mutations for dept
    df_com = df[(df['code_departement']==dept) & (df['nature_mutation']=='Vente')].copy(deep=True)
    # Some price are missing, we drop them 
    df_com = df_com.dropna(subset=['price_per_m2'])
    # find the highest 
    
    most_expensive_communes = df_com.groupby(['code_commune']).mean()['price_per_m2']
    most_expensive_communes = most_expensive_communes.sort_values(ascending=False)
    most_expensive_communes = most_expensive_communes[:top]

    # find the lowest 
    less_expensive_communes = df_com.groupby(['code_commune']).mean()['price_per_m2']
    less_expensive_communes = less_expensive_communes.sort_values()
    less_expensive_communes = less_expensive_communes[:top]
    return most_expensive_communes, less_expensive_communes

@log
def plot_most_expensive_communes(most_expensive_communes, top = 100):
    plt.figure(figsize=(16,6))
    sns.lineplot(x=most_expensive_communes.index, y=most_expensive_communes.values)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig=plt)

@log
def plot_less_expensive_communes(less_expensive_communes, top = 100):
    plt.figure(figsize=(16,6))
    sns.lineplot(x=less_expensive_communes.index, y=less_expensive_communes.values)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig=plt)

@log
def get_mutation_repartition_by_types_in_dept(df, dept):
    """
    Given a dept number, get the  prices per m2
    input : df : dataframe, dept number, highest
    return : list of "highest" communes, list of the lowest communes
    """
    # Select mutations for dept
    df_types = df[df['code_departement']==dept].copy(deep=True)
    # group by 'type_local'
    df_types = df_types.groupby(['type_local']).count()['id_mutation']
    # sort ascending
    df_types = df_types.sort_values(ascending=False)
    
    return df_types

@log
def plot_mut_repartition_by_type(df_types):

    data = df_types.values
    labels = df_types.index

    #define Seaborn color palette to use
    colors = sns.color_palette('pastel')[0:len(data)]

    #create pie chart
    plt.figure(figsize=(16,6))
    plt.pie(data, labels = labels, colors = colors, autopct='%.1f%%')
    plt.show()
    st.pyplot(fig=plt)


    # Use `hole` to create a donut-like pie chart
    #fig = go.Figure(data=[go.Pie(labels=labels, values=data, hole=.3)])
    #fig.show()

@log
def get_price_trends(df, dept, top = 100):
    
    # select only communes for dept
    df_trend = df[df['code_departement']==dept].copy(deep=True)
    
    # Convert date_mutation to datetime type 
    df_trend['date_mutation'] = pd.to_datetime(df_trend['date_mutation'])

    # create multindex
    df_trend = df_trend.set_index([pd.to_datetime(df_trend['date_mutation']), 'code_commune' ], drop=True)
    
    # drop unused colum
    df_trend = df_trend.drop(['date_mutation'], axis=1)
    
    # group by multindex
    df_trend = df_trend.groupby(['date_mutation', 'code_commune']).mean()

    # unstack the data
    data_flat = df_trend.unstack().resample('M').mean()['price_per_m2']
    data_flat = data_flat.stack().reset_index()
    data_flat.columns=['date_mutation','code_commune','price_per_m2']
    
    return data_flat

@log
def plot_price_trends(dept, data_flat, top = 100):
    # get the most expensive and less expensive communes regarding price per m2
    most_expensive_communes, less_expensive_communes = get_most_expensive_com_in_dept(df, dept)

    # get price trend for this dept
    data_flat = get_price_trends(df, dept)
    data_flat = data_flat[data_flat['code_commune'].isin(most_expensive_communes.index.to_list())]

    

    #create pie chart
    plt.figure(figsize=(16,16))
    sns.relplot(data=data_flat, x='date_mutation', y='price_per_m2', hue='code_commune', kind="line")
    plt.show()
    st.pyplot(fig=plt)

@log
def plot_mutation(highest_dept_by_mutation, df_2020, top = 100):
    plt.figure(figsize=(16,6))
    sns.barplot(x=highest_dept_by_mutation.index, y=highest_dept_by_mutation.values)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig=plt)


# Map : Répartition géographique des biens 
def display_map(df):
    #delete the NaN values for longitude and latitude
    df.dropna(subset = ["latitude"], inplace=True)
    df.dropna(subset = ["longitude"], inplace=True)
    min_surface_to_filter, max_surface_to_filter = st.slider('Choisissez la surface min et max du bien : ', 1, 300, (1, 300), 1)
    type_to_filter = st.selectbox("Choisissez un type de bien :", ["Maison", "Dépendance", "Appartement", "Local industriel. commercial ou assimilé" ])
    num_dept = st.selectbox("Choisissez un département :", ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11','12', '13', '14', '15', '16', '17', '18', '19', '21', '22', '23',
        '24', '25', '26', '27', '28', '29', '2A', '2B', '30', '31', '32','33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43',
        '44', '49', '45', '46', '47', '48', '50', '51', '52', '53', '54','55', '56', '58', '59', '60', '61', '62', '63', '64', '65', '66',
        '69', '70', '71', '72', '73', '74', '76', '77', '78', '79', '80','81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91',
        '92', '93', '94', '95', '971', '972', '973', '974', '75'])
    filtered_data = df[(df["surface_reelle_bati"] >= min_surface_to_filter) & (df["surface_reelle_bati"] <= max_surface_to_filter) &
     (df["type_local"] == type_to_filter) & (df['code_departement'] ==  num_dept)]
    #st.subheader('Situation géographique des %ss de %s mètre carré' % (type_to_filter, surface_to_filter))
    st.map(filtered_data)







def main():
    
    # Affichage général :
    st.sidebar.markdown(' ### OUKACI Mathilde')
    st.sidebar.title("Demandes de valeurs foncières : étude des biens par départements")
    df_2020 = load_data()
    df_2016, df_2017, df_2018, df_2019 = load_previous_years()
    top = 97

    
    select = st.sidebar.radio("Sommaire :",('Accueil', 'Mutations par département', 'Étude des prix par mètre carré', 'Répartition par catégories de biens', 'Répartition géographique des biens'))

    if select == 'Accueil':
        # Affichage des metrics :
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("2016", len(df_2016.index))
        col2.metric("2017", len(df_2017.index), "{:.2f}".format(-(100-(len(df_2017.index)*100/len(df_2016.index)))))
        col3.metric("2018", len(df_2018.index), "{:.2f}".format(-(100-(len(df_2018.index)*100/len(df_2017.index)))))
        col4.metric("2019", len(df_2019.index), "{:.2f}".format(-(100-(len(df_2019.index)*100/len(df_2018.index)))))
        col5.metric("2020", len(df_2020.index), "{:.2f}".format(-(100-(len(df_2020.index)*100/len(df_2019.index)))))

        if st.checkbox("Aperçu du dataframe"): 
            df_2020_sample = df_2020.sample(n=100000, random_state=1)
            st.dataframe(df_2020_sample)

    if select == 'Mutations par département': 
        mutation_by_dept = highest_dept_transactions(df_2020, top)
        nbr_dept = st.slider("Nombre de départements à afficher", 1, 97, 10)
        
        # get department with the highest number of mutations
        highest_dept_by_mutation =  highest_dept_transactions(df_2020, nbr_dept) 
        plot_mutation(highest_dept_by_mutation, df_2020)
        if st.button("Nombre de mutations par département complet"): 
            to_plot_bar(mutation_by_dept)


    if select == 'Étude des prix par mètre carré':
        top = st.slider("Nombre de communes à afficher :", 1, 100, 10)
        dept = st.selectbox("Département : ", 
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11','12', '13', '14', '15', '16', '17', '18', '19', '21', '22', '23',
        '24', '25', '26', '27', '28', '29', '2A', '2B', '30', '31', '32','33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43',
        '44', '49', '45', '46', '47', '48', '50', '51', '52', '53', '54','55', '56', '58', '59', '60', '61', '62', '63', '64', '65', '66',
        '69', '70', '71', '72', '73', '74', '76', '77', '78', '79', '80','81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91',
        '92', '93', '94', '95', '971', '972', '973', '974', '75']) 

        # get the most expensive and less expensive communes regarding price per m2
        most_expensive_communes, less_expensive_communes = get_most_expensive_com_in_dept(df_2020, dept, top)
        st.markdown('## Communes les plus chères :')
        plot_most_expensive_communes(most_expensive_communes)
        st.markdown('## Communes les moins chères :')
        plot_most_expensive_communes(less_expensive_communes)

    
    if select == 'Répartition par catégories de biens': 

        dept = st.selectbox("Département : ", 
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11','12', '13', '14', '15', '16', '17', '18', '19', '21', '22', '23',
        '24', '25', '26', '27', '28', '29', '2A', '2B', '30', '31', '32','33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43',
        '44', '49', '45', '46', '47', '48', '50', '51', '52', '53', '54','55', '56', '58', '59', '60', '61', '62', '63', '64', '65', '66',
        '69', '70', '71', '72', '73', '74', '76', '77', '78', '79', '80','81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91',
        '92', '93', '94', '95', '971', '972', '973', '974', '75']) 
        # Given a department, what is the repartition of mutations by category
        df_types = get_mutation_repartition_by_types_in_dept(df_2020, dept)
        plot_mut_repartition_by_type(df_types)

        
    

    if select == 'Répartition géographique des biens': 
        display_map(df_2020)


    #get price trend for communes withe highest price per m2
    # data_flat = get_price_trends(df_2020, dept)
    # plot_price_trends(top, dept, data_flat)

    



if __name__=="__main__":
    main()