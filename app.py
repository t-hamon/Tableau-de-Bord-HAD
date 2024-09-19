import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Étape 1: Charger les fichiers Lib_GP et Lib_sGP une seule fois
@st.cache_data
def load_lib_gp_sgp():
    lib_gp_path = r"C:\Users\hamon\OneDrive\Bureau\AH\prospero91\Données à coder machine learning\fichiers csv\GP.csv"
    lib_sgp_path = r"C:\Users\hamon\OneDrive\Bureau\AH\prospero91\Données à coder machine learning\fichiers csv\sGP.csv"
    
    lib_gp = pd.read_excel(lib_gp_path)
    lib_sgp = pd.read_excel(lib_sgp_path)
    
    return lib_gp, lib_sgp

lib_gp, lib_sgp = load_lib_gp_sgp()

# Étape 2: Charger le fichier gpsl que l'utilisateur va importer
st.title("Analyse des données avec les groupages HAD")

uploaded_file = st.file_uploader("Téléchargez votre fichier CSV contenant la colonne : gpsl", type="csv")

if uploaded_file is not None:
    df_gpsl = pd.read_csv(uploaded_file, sep=';', encoding='ISO-8859-1', on_bad_lines='skip')
    
    # Étape 3: Extraire le préfixe des colonnes
    df_gpsl['gpsl_prefix'] = df_gpsl['gpsl'].str[:4]
    
    lib_gp['ID_Groupe_prefix'] = lib_gp['ID_Groupe'].str.extract(r'_([A-Za-z0-9]{4})_')
    lib_sgp['superGP_prefix'] = lib_sgp['superGP'].str.extract(r'_([A-Za-z0-9]{4})_')
    
    # Fusionner avec Lib_GP et Lib_sGP
    df_with_libelle = df_gpsl.merge(lib_gp[['ID_Groupe_prefix', 'Lib_GP']], left_on='gpsl_prefix', right_on='ID_Groupe_prefix', how='left')
    df_with_libelle = df_with_libelle.merge(lib_sgp[['superGP_prefix', 'Lib_sGP']], left_on='gpsl_prefix', right_on='superGP_prefix', how='left')
    
    # Créer une colonne de traduction
    df_with_libelle['translation'] = df_with_libelle['Lib_GP'].fillna('') + " " + df_with_libelle['Lib_sGP'].fillna('')
    df_with_libelle['translation'] = df_with_libelle['translation'].str.strip()
    
    # Affichage des premières lignes du dataframe fusionné
    st.write("Les données fusionnées sont les suivantes :")
    st.dataframe(df_with_libelle.head())
    
    # Extraire les 6 items les plus fréquents de Lib_GP
    top_items = df_with_libelle['Lib_GP'].value_counts().nlargest(6).index.tolist()
    
    # Filtrer les données pour ne conserver que ces items
    filtered_df = df_with_libelle[df_with_libelle['Lib_GP'].isin(top_items)]
    
    # Extraire les 5e et 6e caractères de gpsl
    filtered_df['gpsl_suffix'] = filtered_df['gpsl'].str[4:6]

    # Vérification des données
    st.write("Les valeurs de gpsl_suffix sont les suivantes :")
    st.write(filtered_df['gpsl_suffix'].unique())
    
    # Grouper les données par Lib_GP et gpsl_suffix pour obtenir les occurrences
    grouped_df = filtered_df.groupby(['Lib_GP', 'gpsl_suffix']).size().unstack(fill_value=0)
    
    # Afficher un aperçu des données regroupées
    st.write("Les données regroupées sont :")
    st.write(grouped_df)
    
    # Graphique 1 : Distribution des codes gpsl_prefix traduits par Lib_sGP
    st.write("Distribution des codes gpsl_prefix traduits par Lib_sGP :")
    sorted_sgp = df_with_libelle['Lib_sGP'].value_counts().sort_values(ascending=False)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sorted_sgp.plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_title("Distribution des codes gpsl_prefix traduits par Lib_sGP")
    ax1.set_xlabel("Lib_sGP")
    ax1.set_ylabel("Nombre d'occurrences")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    st.pyplot(fig1)
    
    # Graphique 2 : Distribution des codes gpsl_prefix traduits par Lib_GP
    st.write("Distribution des codes gpsl_prefix traduits par Lib_GP :")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    df_with_libelle['Lib_GP'].value_counts().plot(kind='bar', ax=ax2)
    ax2.set_title("Distribution des codes gpsl_prefix traduits par Lib_GP")
    ax2.set_xlabel("Lib_GP")
    ax2.set_ylabel("Nombre d'occurrences")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig2)
    
    # Graphique 3 : Répartition des niveaux de lourdeur et de sévérité par item de Lib_GP
    st.write("Répartition des niveaux de lourdeur et de sévérité par item de Lib_GP :")
    if not grouped_df.empty:
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        grouped_df.plot(kind='bar', stacked=True, color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6'], ax=ax3)
        ax3.set_title("Répartition des niveaux de lourdeur et de sévérité par item de Lib_GP")
        ax3.set_xlabel("Lib_GP")
        ax3.set_ylabel("Nombre d'occurrences")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.legend(title="Niveau de Lourdeur/Sévérité", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig3)
    else:
        st.warning("Le DataFrame regroupé est vide ou ne contient pas de données numériques à tracer.")
    
    # Graphique 4 : Camembert pour la distribution des niveaux de lourdeur et de sévérité
    st.write("Camembert pour la distribution des niveaux de lourdeur et de sévérité :")
    if not filtered_df['gpsl_suffix'].empty:
        fig4, ax4 = plt.subplots(figsize=(8, 8))
        filtered_df['gpsl_suffix'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors, ax=ax4)
        ax4.set_title("Niveau de lourdeur et de sévérité")
        ax4.set_ylabel("")  # On masque le label de l'axe y pour un affichage plus propre
        ax4.legend(title="gpsl_suffix")
        st.pyplot(fig4)
    else:
        st.warning("Les données de gpsl_suffix sont vides ou ne peuvent pas être tracées en camembert.")
    
else:
    st.info("Veuillez télécharger un fichier CSV pour commencer l'analyse.")
