# ============================================================
# APPLICATION DE PREDICTION - VEHICULE
# Version Streamlit
# ============================================================

import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Prédiction de l'état d'un véhicule",
    page_icon="🚗",
    layout="centered"
)


# ============================================================
# CHARGEMENT DES MODELES
# ============================================================

@st.cache_resource
def charger_modeles():

    encoders = jb.load("encoders (1).joblib")
    uniques = jb.load("uniques (1).joblib")
    scaler = jb.load("scaler (1).joblib")
    xgb = jb.load("xgb_model (1).joblib")

    return encoders, uniques, scaler, xgb


encoders, uniques, scaler, xgb = charger_modeles()


# ============================================================
# NOMS DES CLASSES
# ============================================================

classnames = uniques[3]


# ============================================================
# FONCTION DE PREDICTION
# ============================================================

def Pred_func(annee, marque, transmission, quartier, prix):

    # --------------------------------------------------------
    # Encodage de la marque
    # --------------------------------------------------------

    marque_encodee = encoders[0].transform([marque])[0]

    # --------------------------------------------------------
    # Encodage de la transmission
    # --------------------------------------------------------

    transmission_encodee = encoders[1].transform([transmission])[0]

    # --------------------------------------------------------
    # Encodage du quartier
    # --------------------------------------------------------

    quartier_encode = encoders[2].transform([quartier])[0]

    # --------------------------------------------------------
    # Création du vecteur
    #
    # Ordre des variables :
    # annee
    # marque
    # transmission
    # quartier
    # prix
    # --------------------------------------------------------

    x_new = np.array([
        annee,
        marque_encodee,
        transmission_encodee,
        quartier_encode,
        prix
    ], dtype=float)

    # Transformation en 2D
    x_new = x_new.reshape(1, -1)

    # --------------------------------------------------------
    # Normalisation
    # --------------------------------------------------------

    x_new = scaler.transform(x_new)

    # --------------------------------------------------------
    # Prédiction
    # --------------------------------------------------------

    y_pred = xgb.predict(x_new)

    return classnames[y_pred[0]]


# ============================================================
# TITRE
# ============================================================

st.title("🚗 Prédiction de l'état d'un véhicule")

st.write(
    """
    Cette application permet de prédire l'état d'un véhicule
    à partir de son année, sa marque, sa transmission,
    son quartier et son prix.
    """
)


# ============================================================
# MENU
# ============================================================

option = st.sidebar.radio(
    "Choisissez une fonctionnalité",
    [
        "Prédiction simple",
        "Prédiction multiple"
    ]
)


# ============================================================
# PREDICTION SIMPLE
# ============================================================

if option == "Prédiction simple":

    st.header("🔮 Prédiction simple")

    # --------------------------------------------------------
    # Année
    # --------------------------------------------------------

    annee = st.number_input(
        "Année",
        min_value=1900,
        max_value=2100,
        value=2020,
        step=1
    )

    # --------------------------------------------------------
    # Marque
    # --------------------------------------------------------

    marque = st.selectbox(
        "Marque",
        options=uniques[0]
    )

    # --------------------------------------------------------
    # Transmission
    # --------------------------------------------------------

    transmission = st.selectbox(
        "Transmission",
        options=uniques[1]
    )

    # --------------------------------------------------------
    # Quartier
    # --------------------------------------------------------

    quartier = st.selectbox(
        "Quartier",
        options=uniques[2]
    )

    # --------------------------------------------------------
    # Prix
    # --------------------------------------------------------

    prix = st.number_input(
        "Prix",
        min_value=0.0,
        value=1000000.0,
        step=10000.0
    )

    st.divider()

    # --------------------------------------------------------
    # Bouton
    # --------------------------------------------------------

    if st.button(
        "🔮 Prédire",
        type="primary"
    ):

        try:

            prediction = Pred_func(
                annee,
                marque,
                transmission,
                quartier,
                prix
            )

            st.success(
                f"### État prédit : {prediction}"
            )

        except Exception as e:

            st.error(
                f"Erreur lors de la prédiction : {e}"
            )


# ============================================================
# PREDICTION MULTIPLE
# ============================================================

elif option == "Prédiction multiple":

    st.header("📂 Prédiction multiple avec un fichier CSV")

    st.write(
        """
        Importez un fichier CSV contenant les colonnes suivantes :
        
        `Année`, `Marque`, `Transmission`, `Prix`, `Quartier`
        """
    )

    # --------------------------------------------------------
    # Upload du fichier
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Importer un fichier CSV",
        type=["csv"]
    )

    # --------------------------------------------------------
    # Fichier chargé
    # --------------------------------------------------------

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.subheader("📋 Données importées")

            st.dataframe(
                df,
                use_container_width=True
            )

            # ------------------------------------------------
            # Colonnes attendues
            # ------------------------------------------------

            required_columns = [
                "Année",
                "Marque",
                "Transmission",
                "Prix",
                "Quartier"
            ]

            # ------------------------------------------------
            # Vérification
            # ------------------------------------------------

            missing_columns = [
                col
                for col in required_columns
                if col not in df.columns
            ]

            if missing_columns:

                st.error(
                    "Colonnes manquantes : "
                    + ", ".join(missing_columns)
                )

            else:

                # --------------------------------------------
                # Bouton prédiction
                # --------------------------------------------

                if st.button(
                    "🔮 Lancer les prédictions",
                    type="primary"
                ):

                    predictions = []

                    progress = st.progress(0)

                    total = len(df)

                    # ----------------------------------------
                    # Parcours des lignes
                    # ----------------------------------------

                    for i, row in df.iterrows():

                        try:

                            prediction = Pred_func(
                                row["Année"],
                                row["Marque"],
                                row["Transmission"],
                                row["Quartier"],
                                row["Prix"]
                            )

                            predictions.append(prediction)

                        except Exception as e:

                            predictions.append(
                                f"Erreur : {e}"
                            )

                        # Progression
                        progress.progress(
                            (i + 1) / total
                        )

                    # ----------------------------------------
                    # Ajouter les prédictions
                    # ----------------------------------------

                    df["etat"] = predictions

                    st.success(
                        "✅ Les prédictions sont terminées !"
                    )

                    # ----------------------------------------
                    # Résultats
                    # ----------------------------------------

                    st.subheader("📊 Résultats")

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

                    # ----------------------------------------
                    # Conversion CSV
                    # ----------------------------------------

                    csv = df.to_csv(
                        index=False
                    ).encode("utf-8")

                    # ----------------------------------------
                    # Téléchargement
                    # ----------------------------------------

                    st.download_button(
                        label="⬇️ Télécharger les prédictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:

            st.error(
                f"Impossible de lire le fichier CSV : {e}"
            )


# ============================================================
# INFORMATIONS
# ============================================================

st.sidebar.divider()

st.sidebar.info(
    """
    ### 🤖 Modèle
    
    XGBoost
    
    ### Variables
    
    • Année  
    • Marque  
    • Transmission  
    • Quartier  
    • Prix
    """
)
