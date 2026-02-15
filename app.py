import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Human Voice Analysis",
    page_icon="🎙️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Sidebar background */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #dbe9f4, #c8d9e6);
    padding-top: 30px;
}

/* Sidebar title */
.sidebar-title {
    font-size: 28px;
    font-weight: 700;
    color: #1f3c5b;
    margin-bottom: 20px;
}

/* Radio button text */
div[role="radiogroup"] > label {
    font-size: 18px !important;
    padding: 8px 0px;
}

/* Highlight selected radio */
div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
    background-color: #1f77b4 !important;
    border-color: #1f77b4 !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    """
    <h1 style='text-align: center; font-size: 48px; font-weight: bold;'>
    🎤 Human Voice Classification & Clustering
    </h1>
    """,
    unsafe_allow_html=True
)

with st.sidebar:

    st.markdown(
        "<div class='sidebar-title'>📌 Navigation</div>",
        unsafe_allow_html=True
    )

    page = st.radio(
        "Go to:",
        ["🏠 Home", "📄 View Table", "📊 EDA", "🤖 Prediction", "📈 Clustering","Introduction"],
    )



if page == "🏠 Home":

    st.subheader("Project Overview")

    st.write(""" 
             
    A machine learning system that classifies and clusters human voice profiles using pre-extracted numerical audio features. The workflow includes preprocessing the dataset, performing clustering to identify natural groupings, training classification models to predict predefined voice categories, and evaluating model performance. The final application will provide a Streamlit interface where users can manually enter numerical feature values and receive both cluster assignment and classification predictions in real time.
         
    
    This project analyzes human voice features to:
    - Classify gender using Machine Learning
    - Perform clustering (KMeans, DBSCAN)
    - Apply PCA for dimensionality reduction
    - Visualize cluster separation
             

    Business Use Cases:
             
    - Speaker Identification:
    - Identify individuals based on their voice features.
    - Gender Classification:
        - Classify voices as male or female for various applications like call center analytics.
    - Speech Analytics:
        - Extract insights from audio data for industries such as media, security, and customer service.
    - Assistive Technologies:
        - Improve accessibility solutions by analyzing voice patterns.

    """)


elif page == "📊 EDA":

    st.subheader("📊 Exploratory Data Analysis")

    df = pd.read_csv("vocal_gender_features_cleaned.csv")
    df["Gender"] = df["label"].map({0: "Female", 1: "Male"})

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Overview",
    "🎵 Pitch",
    "🔊 Energy",
    "🔥 Correlation",
    "🎼 Spectral Features"
    
])


    with tab1:

        st.markdown("### 📊 Dataset Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Samples", df.shape[0])

        with col2:
            st.metric("Female", len(df[df["Gender"]=="Female"]))

        with col3:
            st.metric("Male", len(df[df["Gender"]=="Male"]))

        st.bar_chart(df["Gender"].value_counts())

        st.markdown("---")

        with st.expander("📖 View Audio Feature Description"):

            st.markdown("""
            The dataset consists of extracted audio features from human voice recordings.

            ### 🔊 Spectral Features
            - **mean_spectral_centroid** -Represents brightness (center of mass of spectrum).
            - **std_spectral_centroid** -Variability in brightness.
            - **mean_spectral_bandwidth**- Spread of frequencies in the signal.
            - **std_spectral_bandwidth**- Variability in frequency spread.
            - **mean_spectral_contrast**-  Tonal contrast between peaks and valleys.
            - **mean_spectral_flatness**- Measures noisiness of the signal.
            - **mean_spectral_rolloff**- Frequency below which most spectral energy resides.
            - **spectral_skew**- Asymmetry of spectral distribution.
            - **spectral_kurtosis**- Peak sharpness of the spectrum.

            ### 🎵 Pitch & Energy Features
            - **mean_pitch** - Average pitch frequency.
            - **min_pitch** - Minimum pitch.
            - **max_pitch** - Maximum pitch.
            - **std_pitch** -Pitch variability.
            - **rms_energy**- Loudness of the signal.
            - **log_energy**- Log-compressed energy representation.
            - **energy_entropy**- Randomness of signal energy.
            - **zero_crossing_rate**- Measures noisiness or percussiveness.

            ### 🎧 MFCC Features
            - **mfcc_1_mean to mfcc_13_mean** : Capture timbral characteristics.
            - **mfcc_1_std to mfcc_13_std**: Variability in timbral features.

            ### 🎯 Target Variable
            - **label**  Gender classification  
            - Female = 0  
            - Male = 1
            """)

        st.info("These extracted acoustic features are used for gender classification and clustering analysis.")
        


    with tab2:

        st.markdown("### Mean Pitch Analysis")

        female_pitch = df[df["Gender"]=="Female"]["mean_pitch"].mean()
        male_pitch = df[df["Gender"]=="Male"]["mean_pitch"].mean()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Female Mean Pitch", f"{female_pitch:.2f} Hz")
            st.metric("Male Mean Pitch", f"{male_pitch:.2f} Hz")

        with col2:
            fig1, ax1 = plt.subplots(figsize=(6,4))
            sns.kdeplot(
                data=df,
                x="mean_pitch",
                hue="Gender",
                fill=True,
                alpha=0.4,
                ax=ax1
            )
            ax1.set_title("Pitch Distribution")
            st.pyplot(fig1)

      
            st.info("""
            The distribution shows that female voices generally have higher pitch values compared to male voices. 
            Although some overlap exists, pitch remains one of the strongest acoustic indicators for gender classification.
            """)
        
    with tab3:

        st.markdown("### RMS Energy Analysis")

        col1, col2 = st.columns(2)

        with col1:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.kdeplot(
                data=df,
                x="rms_energy",
                hue="Gender",
                fill=True,
                alpha=0.4,
                ax=ax2
            )
            ax2.set_title("RMS Energy Distribution")
            st.pyplot(fig2)

        with col2:
            fig3, ax3 = plt.subplots(figsize=(6,4))
            sns.barplot(
                data=df,
                x="Gender",
                y="rms_energy",
                estimator=np.mean,
                ax=ax3
            )
            ax3.set_title("Average RMS Energy")
            st.pyplot(fig3)

        st.info("""
            RMS energy reflects vocal loudness. While females show slightly higher average RMS values,
            the overlap between distributions indicates that loudness alone cannot fully distinguish gender.
            It serves as a complementary feature in classification.
            """)

    with tab4:

        st.markdown("### Feature Correlation Analysis")

        fig4, ax4 = plt.subplots(figsize=(10,6))
        sns.heatmap(
            df.drop(columns=["Gender"]).corr(),
            cmap="coolwarm",
            annot=False,
            ax=ax4
        )
        ax4.set_title("Correlation Heatmap")
        st.pyplot(fig4)

        st.markdown("### Correlation with Gender")

        fig5, ax5 = plt.subplots(figsize=(4,8))
        sns.heatmap(
            df.drop(columns=["Gender"]).corr()[["label"]],
            cmap="coolwarm",
            annot=True,
            ax=ax5
        )
        ax5.set_title("Feature vs Gender")
        st.pyplot(fig5)

        st.warning(
            "Pitch and spectral features show stronger correlation with gender, supporting their importance in classification."
        )


    with tab5:   

        st.markdown("### 🎼 Spectral Feature Analysis")

        col1, col2 = st.columns(2)

        # -------- Spectral Centroid --------
        with col1:
            fig1, ax1 = plt.subplots(figsize=(6,4))
            sns.kdeplot(
                data=df,
                x="mean_spectral_centroid",
                hue="Gender",
                fill=True,
                alpha=0.4,
                ax=ax1
            )
            ax1.set_title("Mean Spectral Centroid Distribution")
            st.pyplot(fig1)

            st.markdown("""
            
            - Spectral centroid represents the brightness of the sound.
            - Higher centroid values indicate brighter or sharper sounds.
            - Female voices typically show slightly higher centroid values,
            suggesting brighter vocal characteristics.
            - Overlap indicates it is useful but not fully separative.
            """)

        # -------- Spectral Rolloff --------
        with col2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.kdeplot(
                data=df,
                x="mean_spectral_rolloff",
                hue="Gender",
                fill=True,
                alpha=0.4,
                ax=ax2
            )
            ax2.set_title("Mean Spectral Rolloff Distribution")
            st.pyplot(fig2)

            st.markdown("""
            
            - Spectral rolloff indicates the frequency below which most energy resides.
            - Higher values suggest stronger high-frequency components.
            - Female voices often show slightly higher rolloff,
            reflecting sharper frequency distribution.
            - Some overlap exists, meaning rolloff supports classification
            but is not independently decisive.
            """)


elif page == "📄 View Table":

    st.subheader("📄 Dataset Viewer")

    df = pd.read_csv("vocal_gender_features_cleaned.csv")

    df["Gender"] = df["label"].map({0: "Female", 1: "Male"})

    st.markdown("### Filter Based on Male or Female")

    gender_filter = st.selectbox(
        "Select Gender",
        ["All", "Female", "Male"]
    )

    # filter
    if gender_filter == "Female":
        filtered_df = df[df["Gender"] == "Female"]
    elif gender_filter == "Male":
        filtered_df = df[df["Gender"] == "Male"]
    else:
        filtered_df = df

    # Show metrics
    st.metric("Total Records", len(filtered_df))

    # Display table
    st.dataframe(filtered_df.drop(columns=["label"]), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Females", len(df[df["Gender"] == "Female"]))

    with col2:
        st.metric("Males", len(df[df["Gender"] == "Male"]))


elif page == "🤖 Prediction":


    st.subheader("🤖 Voice Gender Prediction")

    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans


   
    # Load saved model and scaler
    model = joblib.load("models/svm_20.pkl")
    scaler = joblib.load("models/scaler_20.pkl")
    kmeans = joblib.load("models/kmeans.pkl")
    pca = joblib.load("models/pca.pkl")
    df = pd.read_csv("vocal_gender_features_cleaned.csv")

    top_features = [
        "mfcc_5_mean",
        "mean_spectral_contrast",
        "mfcc_3_std",
        "mfcc_2_mean",
        "std_spectral_bandwidth",
        "mfcc_12_mean",
        "mfcc_1_mean",
        "mfcc_2_std",
        "mfcc_10_mean",
        "rms_energy",
        "mfcc_10_std",
        "mfcc_6_mean",
        "mfcc_8_mean",
        "mfcc_4_mean",
        "mfcc_13_mean",
        "mfcc_7_mean",
        "mfcc_3_mean",
        "mfcc_8_std",
        "mfcc_11_mean",
        "mfcc_5_std"
    ]


    df_scaled = scaler.transform(df[top_features])
    df_pca = pca.transform(df_scaled)
    df["PC1"] = df_pca[:, 0]
    df["PC2"] = df_pca[:, 1]
    df["cluster"] = kmeans.predict(df_pca)
    cluster_mapping = (
        df.groupby("cluster")["label"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )



    
  
    st.markdown("### 🎛 Adjust Acoustic Feature Values")
    input_data = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(top_features):

        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())

        if i % 2 == 0:
            input_data[feature] = col1.slider(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )
        else:
            input_data[feature] = col2.slider(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )


    st.markdown("---")

    st.markdown("---")

    action = st.radio(
    "Choose Action",
    ["🎯 Predict Gender", 
     "📊 Full Feature Clustering", 
     ],
    horizontal=True
    )



    if st.button("🚀 Run"):
    
        input_df = pd.DataFrame([input_data])

        # Always scale first
        input_scaled = scaler.transform(input_df)

        # ==================================
        # 🎯 GENDER PREDICTION
        # ==================================
        if action == "🎯 Predict Gender":

            prediction = model.predict(input_scaled)[0]

            label_map = {0: "Female 👩", 1: "Male 👨"}
            st.success(f"Predicted Gender: {label_map[prediction]}")

        # CLUSTER VIEW
        else:
            
            # PREPARE DATASET PCA
            df_scaled = scaler.transform(df[top_features])
            df_pca = pca.transform(df_scaled)

            df["PC1"] = df_pca[:, 0]
            df["PC2"] = df_pca[:, 1]
            # Create 3-cluster model
            kmeans_3 = KMeans(n_clusters=3, random_state=42)

            # Fit on scaled data
            df["cluster_3"] = kmeans_3.fit_predict(df_scaled)


            # Transform user input
            input_scaled = scaler.transform(input_df)
            input_pca = pca.transform(input_scaled)

            # 🔵 2 CLUSTERS
            import seaborn as sns
            from sklearn.cluster import KMeans

            sns.set_style("white")



            # Transform user input
            input_scaled = scaler.transform(input_df)
            input_pca = pca.transform(input_scaled)

            # ===============================
            # 2-CLUSTER MODEL
            # ===============================
            kmeans_2 = KMeans(n_clusters=2, random_state=42)
            df["cluster_2"] = kmeans_2.fit_predict(df_scaled)

            # Predict cluster for user input
            user_cluster = kmeans_2.predict(input_scaled)[0]

            # ===============================
            # PLOT
            # ===============================
            st.subheader("📊 K-Means Clusters (2 Groups)")

            fig, ax = plt.subplots(figsize=(7,6))

            sns.scatterplot(
                x=df["PC1"],
                y=df["PC2"],
                hue=df["cluster_2"],
                palette="tab10",      # Blue & Orange
                s=50,
                edgecolor="white",
                linewidth=0.5,
                ax=ax
            )

            # Highlight user input
            ax.scatter(
                input_pca[0][0],
                input_pca[0][1],
                s=200,
                marker="X",
                color="black",
                edgecolor="white",
                linewidth=1,
                label="Your Input"
            )

            ax.set_title("K-Means Clusters")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

            ax.legend(title="Cluster ID")

            st.pyplot(fig)

            # ===============================
            # SHOW USER CLUSTER RESULT
            # ===============================
            st.success(f"Your input belongs to Cluster {user_cluster}")



            # 🟢 3 CLUSTERS
            st.subheader("📊 K-Means Clustering (3 Clusters)")

            fig, ax = plt.subplots(figsize=(7,6))

            # Define clean distinct colors
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue, orange, green

            for cluster_id in range(3):
                cluster_points = df[df["cluster_3"] == cluster_id]

                ax.scatter(
                    cluster_points["PC1"],
                    cluster_points["PC2"],
                    s=50,
                    color=colors[cluster_id],
                    edgecolor="white",
                    linewidth=0.5,
                    label=f"Cluster {cluster_id}"
                )

            # Highlight user point
            ax.scatter(
                input_pca[0][0],
                input_pca[0][1],
                s=200,
                marker="X",
                color="black",
                edgecolor="white",
                linewidth=1,
                label="Your Input"
            )

            ax.set_title("K-Means Clusters (3 Groups)")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

            ax.legend(title="Cluster ID")
            st.pyplot(fig)


elif page == "📈 Clustering":

    st.title("📊 K-Means Clustering (Unsupervised Learning)")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_csv("vocal_gender_features_cleaned.csv")

    X = df.drop(columns=["label"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot
    fig, ax = plt.subplots(figsize=(7,6))

    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=clusters,
        palette="tab10",
        s=50,
        ax=ax
    )

    ax.set_title("K-Means Clusters (2 Groups)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Cluster ID")

    st.pyplot(fig)

    st.markdown("""
    This visualization shows how voice samples group naturally 
    into 2 clusters using all acoustic features.
    PCA is used to reduce dimensionality for visualization.
    """)


elif page=="Introduction":

    

    st.write("**Name:** Avanthi U C")
    st.write("**Course:** Data Science")
    
    
