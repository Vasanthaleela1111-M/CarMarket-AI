import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_price_model():
    return joblib.load("car_price_model.joblib")


@st.cache_data
def load_car_data():
    return pd.read_csv("Cleaned_cars.csv")

@st.cache_resource
def load_recommendation_artifacts():
    return joblib.load("recommendation_artifacts.joblib")


price_model = load_price_model()
df = load_car_data()

recommendation_artifacts = load_recommendation_artifacts()

encoder = recommendation_artifacts["encoder"]
scaler = recommendation_artifacts["scaler"]
recommendation_matrix = recommendation_artifacts["recommendation_matrix"]
rec = recommendation_artifacts["rec"]
final_kmeans = recommendation_artifacts["final_kmeans"]
categorical_features = recommendation_artifacts["categorical_features"]
numerical_features = recommendation_artifacts["numerical_features"]


# ============================================================
# CHATBOT HELPERS
# ============================================================

def extract_budget(text):

    text = text.lower()

    # Examples:
    # 8 lakh
    # 8 lakhs
    # 8 l
    # 8.5 lakh
    # 800000

    lakh_match = re.search(
        r'(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|l)',
        text
    )

    if lakh_match:
        return float(lakh_match.group(1))

    rupee_match = re.search(
        r'₹?\s*(\d{5,7})',
        text
    )

    if rupee_match:
        amount = float(rupee_match.group(1))
        return amount / 100000

    return None


def extract_brand(text):

    text_lower = text.lower()

    brands = df["brand"].dropna().unique()

    for brand in brands:

        if brand.lower() in text_lower:
            return brand

    return None


def extract_fuel(text):

    text_lower = text.lower()

    if "petrol" in text_lower:
        return "PETROL"

    if "diesel" in text_lower:
        return "DIESEL"

    if "electric" in text_lower or "ev" in text_lower:
        return "ELECTRIC"

    if "cng" in text_lower:
        return "CNG"

    if "lpg" in text_lower:
        return "LPG"

    return None


def extract_body_type(text):

    text_lower = text.lower()

    body_types = df["body_type"].dropna().unique()

    for body in body_types:

        if body.lower() in text_lower:
            return body

    # Common shortcuts

    if "suv" in text_lower:
        return "SUV"

    if "sedan" in text_lower:
        return "Sedan"

    if "hatchback" in text_lower:
        return "Hatchback"

    return None


def extract_transmission(text):

    text_lower = text.lower()

    if "automatic" in text_lower or "auto" in text_lower:
        return "Automatic"

    if "manual" in text_lower:
        return "Manual"

    return None


def extract_city(text):

    text_lower = text.lower()

    cities = df["city"].dropna().unique()

    for city in cities:

        if city.lower() in text_lower:
            return city

    return None


def chatbot_recommend_cars(
    budget=None,
    brand=None,
    fuel=None,
    body_type=None,
    transmission=None,
    city=None,
    top_n=5
):

    candidates = rec.copy()

    # --------------------------------------------------------
    # BUDGET FILTER
    # --------------------------------------------------------

    if budget is not None:

        candidates = candidates[
            candidates["price_lakh"] <= budget
        ]

    # --------------------------------------------------------
    # FUEL FILTER
    # --------------------------------------------------------

    if fuel is not None:

        candidates = candidates[
            candidates["fuel_type"].str.upper() == fuel.upper()
        ]

    if candidates.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------

    user_preferences = pd.DataFrame([{

        "brand": brand if brand else "Unknown",

        "model": "Unknown",

        "fuel_type": fuel if fuel else "PETROL",

        "body_type": body_type if body_type else "SUV",

        "transmission": transmission if transmission else "Automatic",

        "city": city if city else "chennai",

        "km_driven": rec["km_driven"].median(),

        "model_year": rec["model_year"].median(),

        "mileage": rec["mileage"].median(),

        "engine": rec["engine"].median(),

        "max_power": rec["max_power"].median(),

        "price_lakh": budget
            if budget is not None
            else rec["price_lakh"].median()

    }])

    user_encoded = encoder.transform(
        user_preferences[
            [
                "brand",
                "model",
                "fuel_type",
                "body_type",
                "transmission",
                "city"
            ]
        ]
    )

    user_scaled = scaler.transform(
        user_preferences[
            [
                "km_driven",
                "model_year",
                "mileage",
                "engine",
                "max_power",
                "price_lakh"
            ]
        ]
    )

    user_vector = np.hstack([
        user_encoded,
        user_scaled
    ])

    similarity_scores = cosine_similarity(
        user_vector,
        recommendation_matrix
    )[0]

    candidates["similarity"] = similarity_scores[
        candidates.index
    ]

    # --------------------------------------------------------
    # PREFERENCE MATCH
    # --------------------------------------------------------

    match_score = pd.Series(
        0.0,
        index=candidates.index
    )

    match_count = 0

    if brand:

        match_score += (
            candidates["brand"].str.upper()
            == brand.upper()
        ).astype(int)

        match_count += 1

    if body_type:

        match_score += (
            candidates["body_type"].str.upper()
            == body_type.upper()
        ).astype(int)

        match_count += 1

    if transmission:

        match_score += (
            candidates["transmission"].str.upper()
            == transmission.upper()
        ).astype(int)

        match_count += 1

    if city:

        match_score += (
            candidates["city"].str.lower()
            == city.lower()
        ).astype(int)

        match_count += 1

    if match_count > 0:

        candidates["preference_match"] = (
            match_score / match_count
        )

    else:

        candidates["preference_match"] = 0.0

    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    user_cluster = final_kmeans.predict(
        user_vector
    )[0]

    candidates["cluster_match"] = (
        candidates["cluster"] == user_cluster
    ).astype(int)

    # --------------------------------------------------------
    # HYBRID SCORE
    # --------------------------------------------------------

    candidates["hybrid_score"] = (

        0.70 * candidates["preference_match"]

        + 0.25 * candidates["similarity"]

        + 0.05 * candidates["cluster_match"]

    )

    candidates = candidates.sort_values(
        "hybrid_score",
        ascending=False
    )

    # --------------------------------------------------------
    # GET ORIGINAL CAR DETAILS
    # --------------------------------------------------------

    result = df.loc[
        candidates.head(top_n).index
    ].copy()

    result["hybrid_score"] = candidates.loc[
        result.index,
        "hybrid_score"
    ]

    result["similarity"] = candidates.loc[
        result.index,
        "similarity"
    ]

    return result
# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CarResale AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🚗 CarResale AI")

    st.caption("AI-Powered Used Car Intelligence")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📘 Project Introduction",
            "💰 Price Prediction",
            "⭐ Car Recommendations",
            "🤖 AI Car Chatbot",
            "👩‍💻 Creator Info"
        ]
    )

    st.divider()

    st.caption("Machine Learning • Recommendation • AI")


# ============================================================
# PROJECT INTRODUCTION
# ============================================================

if page == "📘 Project Introduction":

    st.title("🚗 CarResale AI")

    st.caption(
        "AI-Powered Car Resale Price Prediction & Smart Recommendations"
    )

    st.write(
        """
        CarResale AI is an intelligent used-car analysis platform that
        combines Machine Learning, recommendation techniques and
        conversational AI to help users understand car prices and
        discover suitable vehicles.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # QUICK STATS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🚗 Cars",
            "8,239"
        )

    with col2:
        st.metric(
            "🧠 ML Model",
            "HistGradientBoosting"
        )

    with col3:
        st.metric(
            "⭐ Recommendation",
            "Hybrid"
        )

    with col4:
        st.metric(
            "📊 R² Score",
            "94.46%"
        )

    st.write("")

    # --------------------------------------------------------
    # ABOUT PROJECT
    # --------------------------------------------------------

    with st.container(border=True):

        st.subheader("📖 About CarResale AI")

        st.write(
            """
            The system predicts the resale price of used cars using
            historical vehicle information such as brand, model,
            fuel type, body type, transmission, mileage, engine,
            power, model year and location.

            It also provides personalized car recommendations using
            a hybrid recommendation approach combining:

            • Explicit user preference matching

            • Cosine similarity

            • K-Means clustering

            • Hybrid ranking
            """
        )

    st.write("")

    # --------------------------------------------------------
    # PROJECT HIGHLIGHTS
    # --------------------------------------------------------

    st.subheader("🚀 Project Highlights")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.success(
            """
            ### 💰 Price Prediction

            • Machine Learning

            • Regression

            • HistGradientBoosting

            • MLflow Tracking

            • Price estimation
            """
        )

    with col2:

        st.info(
            """
            ### ⭐ Recommendations

            • Cosine Similarity

            • K-Means Clustering

            • Preference Matching

            • Hybrid Ranking

            • Personalized results
            """
        )

    with col3:

        st.warning(
            """
            ### ☁️ Deployment

            • Streamlit

            • AWS

            • S3

            • RDS

            • MLflow

            • Docker
            """
        )

    st.write("")

    # --------------------------------------------------------
    # CORE FEATURES
    # --------------------------------------------------------

    st.subheader("✨ Core Features")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
            ### 💰 Price Prediction

            Enter vehicle details and
            estimate its resale price.
            """
        )

    with col2:

        st.info(
            """
            ### ⭐ Smart Recommendations

            Find cars that match your
            preferences and budget.
            """
        )

    with col3:

        st.info(
            """
            ### 🤖 AI Chatbot

            Interact with the system
            using natural language.
            """
        )

    st.write("")

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    with st.expander("🔄 How CarResale AI Works"):

        st.markdown(
            """
            **Step 1:** User provides vehicle information

            ⬇️

            **Step 2:** Data preprocessing

            ⬇️

            **Step 3:** Machine Learning price prediction

            ⬇️

            **Step 4:** Recommendation engine

            ⬇️

            **Step 5:** Hybrid ranking

            ⬇️

            **Step 6:** Personalized car results

            ⬇️

            **Step 7:** AI chatbot interaction
            """
        )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    with st.expander("📂 Dataset Information"):

        st.markdown(
            """
            **Dataset:** Cleaned CarDekho Dataset

            **Records:** 8,239

            **Features:** 22

            **Target:** Car resale price

            **Target Unit:** Lakhs (₹)

            **Cities:**

            - Bangalore
            - Chennai
            - Delhi
            - Hyderabad
            - Jaipur
            - Kolkata
            """
        )


# ============================================================
# PRICE PREDICTION
# ============================================================

elif page == "💰 Price Prediction":

    st.title("💰 Car Resale Price Prediction")

    st.caption(
        "Enter the vehicle details to estimate its resale value."
    )

    st.divider()

    # ========================================================
    # VEHICLE INFORMATION
    # ========================================================

    st.subheader("🚗 Vehicle Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        brand = st.selectbox(
            "Brand",
            sorted(df["brand"].dropna().unique())
        )

        model = st.selectbox(
            "Model",
            sorted(
                df[df["brand"] == brand]["model"]
                .dropna()
                .unique()
            )
        )

        variant = st.selectbox(
            "Variant",
            sorted(
                df[
                    (df["brand"] == brand) &
                    (df["model"] == model)
                ]["variant"]
                .dropna()
                .unique()
            )
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            sorted(df["fuel_type"].dropna().unique())
        )

        body_type = st.selectbox(
            "Body Type",
            sorted(df["body_type"].dropna().unique())
        )

    with col2:

        transmission = st.selectbox(
            "Transmission",
            sorted(df["transmission"].dropna().unique())
        )

        city = st.selectbox(
            "City",
            sorted(df["city"].dropna().unique())
        )

        insurance_validity = st.selectbox(
            "Insurance Validity",
            sorted(df["insurance_validity"].dropna().unique())
        )

        owner_number = st.number_input(
            "Owner Number",
            min_value=1,
            max_value=5,
            value=1,
            step=1
        )

        seats = st.number_input(
            "Seats",
            min_value=2,
            max_value=10,
            value=5,
            step=1
        )

    with col3:

        km_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=1_000_000,
            value=50_000,
            step=1_000
        )

        model_year = st.number_input(
            "Model Year",
            min_value=int(df["model_year"].min()),
            max_value=int(df["model_year"].max()),
            value=int(df["model_year"].median()),
            step=1
        )

        registration_year = st.number_input(
            "Registration Year",
            min_value=int(df["registration_year"].min()),
            max_value=int(df["registration_year"].max()),
            value=int(df["registration_year"].median()),
            step=1
        )

        mileage = st.number_input(
            "Mileage",
            min_value=float(df["mileage"].min()),
            max_value=float(df["mileage"].max()),
            value=float(df["mileage"].median()),
            step=0.1
        )

    st.write("")

    # ========================================================
    # ENGINE INFORMATION
    # ========================================================

    st.subheader("⚙️ Engine & Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        engine = st.number_input(
            "Engine (cc)",
            min_value=float(df["engine"].min()),
            max_value=float(df["engine"].max()),
            value=float(df["engine"].median()),
            step=50.0
        )

    with col2:

        max_power = st.number_input(
            "Max Power (bhp)",
            min_value=float(df["max_power"].min()),
            max_value=float(df["max_power"].max()),
            value=float(df["max_power"].median()),
            step=1.0
        )

    with col3:

        torque = st.number_input(
            "Torque (Nm)",
            min_value=float(df["torque"].min()),
            max_value=float(df["torque"].max()),
            value=float(df["torque"].median()),
            step=5.0
        )

    st.write("")

    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_button = st.button(
        "💰 Predict Resale Price",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        input_data = pd.DataFrame([{
            "fuel_type": fuel_type,
            "body_type": body_type,
            "km_driven": km_driven,
            "transmission": transmission,
            "owner_number": owner_number,
            "brand": brand,
            "model": model,
            "model_year": model_year,
            "variant": variant,
            "registration_year": registration_year,
            "insurance_validity": insurance_validity,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "torque": torque,
            "seats": seats,
            "city": city
        }])

        try:

            prediction = price_model.predict(input_data)[0]

            st.divider()

            st.subheader("🎯 Estimated Resale Price")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Estimated Price",
                    f"₹{prediction:.2f} Lakh"
                )

            with col2:

                st.metric(
                    "Estimated Price",
                    f"₹{prediction * 100000:,.0f}"
                )

            with col3:

                st.metric(
                    "Model",
                    "HistGradientBoosting"
                )

            st.success(
                f"🚗 Estimated resale value: ₹{prediction:.2f} lakh"
            )

            with st.expander("📋 View Input Details"):

                st.dataframe(
                    input_data,
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# CAR RECOMMENDATIONS
# ============================================================

# ============================================================
# CAR RECOMMENDATIONS
# ============================================================

elif page == "⭐ Car Recommendations":

    st.title("⭐ Smart Car Recommendations")

    st.caption(
        "Find used cars using your budget, preferences and AI-powered similarity."
    )

    st.divider()

    # ========================================================
    # USER PREFERENCES
    # ========================================================

    st.subheader("🎯 Tell us what you are looking for")

    col1, col2, col3 = st.columns(3)

    with col1:

        budget = st.number_input(
            "💰 Maximum Budget (₹ Lakh)",
            min_value=1.0,
            max_value=float(df["price_lakh"].max()),
            value=10.0,
            step=0.5
        )

        fuel_options = sorted(
            df["fuel_type"]
            .dropna()
            .unique()
        )

        fuel = st.selectbox(
            "⛽ Fuel Type",
            fuel_options
        )

    with col2:

        brand_options = sorted(
            df["brand"]
            .dropna()
            .unique()
        )

        brand = st.selectbox(
            "🏷️ Preferred Brand",
            brand_options
        )

        body_type_options = sorted(
            df["body_type"]
            .dropna()
            .unique()
        )

        body_type = st.selectbox(
            "🚙 Body Type",
            body_type_options
        )

    with col3:

        transmission_options = sorted(
            df["transmission"]
            .dropna()
            .unique()
        )

        transmission = st.selectbox(
            "⚙️ Transmission",
            transmission_options
        )

        city_options = sorted(
            df["city"]
            .dropna()
            .unique()
        )

        city = st.selectbox(
            "📍 City",
            city_options
        )

    st.write("")

    # ========================================================
    # SEARCH BUTTON
    # ========================================================

    recommend_button = st.button(
        "🔎 Find Best Cars",
        type="primary",
        use_container_width=True
    )

    # ========================================================
    # RECOMMENDATION ENGINE
    # ========================================================

    if recommend_button:

        with st.spinner("🤖 Finding the best cars for you..."):

            # ------------------------------------------------
            # 1. HARD FILTER
            # ------------------------------------------------

            candidates = rec[
                (rec["price_lakh"] <= budget) &
                (
                    rec["fuel_type"].str.upper()
                    == fuel.upper()
                )
            ].copy()

            if candidates.empty:

                st.warning(
                    "😔 No cars found within your budget and fuel preference."
                )

            else:

                # --------------------------------------------
                # 2. CREATE USER PREFERENCE VECTOR
                # --------------------------------------------

                user_preferences = pd.DataFrame([{
                    "brand": brand,
                    "model": "Unknown",
                    "fuel_type": fuel,
                    "body_type": body_type,
                    "transmission": transmission,
                    "city": city,

                    "km_driven": rec["km_driven"].median(),
                    "model_year": rec["model_year"].median(),
                    "mileage": rec["mileage"].median(),
                    "engine": rec["engine"].median(),
                    "max_power": rec["max_power"].median(),
                    "price_lakh": budget
                }])

                # --------------------------------------------
                # 3. ENCODE CATEGORICAL FEATURES
                # --------------------------------------------

                user_encoded = encoder.transform(
                    user_preferences[
                        categorical_features
                    ]
                )

                # --------------------------------------------
                # 4. SCALE NUMERICAL FEATURES
                # --------------------------------------------

                user_scaled = scaler.transform(
                    user_preferences[
                        numerical_features
                    ]
                )

                # --------------------------------------------
                # 5. COMBINE USER FEATURES
                # --------------------------------------------

                user_vector = np.hstack([
                    user_encoded,
                    user_scaled
                ])

                # --------------------------------------------
                # 6. COSINE SIMILARITY
                # --------------------------------------------

                user_similarity = cosine_similarity(
                    user_vector,
                    recommendation_matrix
                )[0]

                candidates["similarity"] = (
                    user_similarity[candidates.index]
                )

                # --------------------------------------------
                # 7. EXPLICIT PREFERENCE MATCHING
                # --------------------------------------------

                candidates["preference_match"] = (

                    (
                        candidates["brand"].str.upper()
                        == brand.upper()
                    ).astype(int)

                    +

                    (
                        candidates["body_type"].str.upper()
                        == body_type.upper()
                    ).astype(int)

                    +

                    (
                        candidates["transmission"].str.upper()
                        == transmission.upper()
                    ).astype(int)

                    +

                    (
                        candidates["city"].str.lower()
                        == city.lower()
                    ).astype(int)

                ) / 4

                # --------------------------------------------
                # 8. K-MEANS USER CLUSTER
                # --------------------------------------------

                user_cluster = final_kmeans.predict(
                    user_vector
                )[0]

                candidates["cluster_match"] = (
                    candidates["cluster"]
                    == user_cluster
                ).astype(int)

                # --------------------------------------------
                # 9. HYBRID SCORE
                # --------------------------------------------

                candidates["hybrid_score"] = (

                    0.70 * candidates["preference_match"]

                    +

                    0.25 * candidates["similarity"]

                    +

                    0.05 * candidates["cluster_match"]

                )

                # --------------------------------------------
                # 10. SORT BY HYBRID SCORE
                # --------------------------------------------

                candidates = candidates.sort_values(
                    by="hybrid_score",
                    ascending=False
                )

                # --------------------------------------------
                # 11. GET TOP 10 RESULTS
                # --------------------------------------------

                top_indices = candidates.head(10).index

                result = df.loc[
                    top_indices
                ].copy()

                result["preference_match"] = (
                    candidates.loc[
                        top_indices,
                        "preference_match"
                    ]
                )

                result["similarity"] = (
                    candidates.loc[
                        top_indices,
                        "similarity"
                    ]
                )

                result["cluster_match"] = (
                    candidates.loc[
                        top_indices,
                        "cluster_match"
                    ]
                )

                result["hybrid_score"] = (
                    candidates.loc[
                        top_indices,
                        "hybrid_score"
                    ]
                )

                # ====================================================
                # RESULTS
                # ====================================================

                st.divider()

                st.subheader(
                    "🏆 Recommended Cars"
                )

                st.success(
                    f"Found {len(candidates):,} cars matching "
                    f"your budget and fuel preference."
                )

                # ------------------------------------------------
                # USER SUMMARY
                # ------------------------------------------------

                with st.container(border=True):

                    st.markdown("### 🎯 Your Preferences")

                    pref1, pref2, pref3, pref4, pref5, pref6 = st.columns(6)

                    with pref1:
                        st.metric(
                            "Budget",
                            f"₹{budget:.1f} L"
                        )

                    with pref2:
                        st.metric(
                            "Fuel",
                            fuel
                        )

                    with pref3:
                        st.metric(
                            "Brand",
                            brand
                        )

                    with pref4:
                        st.metric(
                            "Body",
                            body_type
                        )

                    with pref5:
                        st.metric(
                            "Transmission",
                            transmission
                        )

                    with pref6:
                        st.metric(
                            "City",
                            city.title()
                        )

                st.write("")

                # ------------------------------------------------
                # DISPLAY RECOMMENDATIONS
                # ------------------------------------------------

                for rank, (index, car) in enumerate(
                    result.iterrows(),
                    start=1
                ):

                    with st.container(border=True):

                        col1, col2, col3 = st.columns(
                            [1, 4, 2]
                        )

                        with col1:

                            st.markdown(
                                f"## #{rank}"
                            )

                            st.markdown(
                                "⭐ Recommended"
                            )

                        with col2:

                            st.subheader(
                                f"{car['brand']} "
                                f"{car['model']}"
                            )

                            st.write(
                                f"**Variant:** "
                                f"{car['variant']}"
                            )

                            st.write(
                                f"⛽ {car['fuel_type']}  |  "
                                f"⚙️ {car['transmission']}  |  "
                                f"🚙 {car['body_type']}  |  "
                                f"📍 {car['city'].title()}"
                            )

                            st.write(
                                f"📅 Model Year: "
                                f"{int(car['model_year'])}  |  "
                                f"🛣️ KM Driven: "
                                f"{int(car['km_driven']):,}"
                            )

                        with col3:

                            st.metric(
                                "💰 Price",
                                f"₹{car['price_lakh']:.2f} L"
                            )

                            st.metric(
                                "🎯 Match Score",
                                f"{car['hybrid_score'] * 100:.1f}%"
                            )

                # ------------------------------------------------
                # DETAILED TABLE
                # ------------------------------------------------

                st.write("")

                with st.expander(
                    "📊 View Recommendation Details"
                ):

                    display_result = result[
                        [
                            "brand",
                            "model",
                            "variant",
                            "fuel_type",
                            "body_type",
                            "transmission",
                            "city",
                            "km_driven",
                            "model_year",
                            "price_lakh",
                            "preference_match",
                            "similarity",
                            "cluster_match",
                            "hybrid_score"
                        ]
                    ].copy()

                    display_result["preference_match"] = (
                        display_result["preference_match"] * 100
                    ).round(1)

                    display_result["similarity"] = (
                        display_result["similarity"] * 100
                    ).round(1)

                    display_result["hybrid_score"] = (
                        display_result["hybrid_score"] * 100
                    ).round(1)

                    st.dataframe(
                        display_result,
                        use_container_width=True,
                        hide_index=True
                    )

                # ------------------------------------------------
                # ALGORITHM EXPLANATION
                # ------------------------------------------------

                with st.expander(
                    "🧠 How were these cars recommended?"
                ):

                    st.markdown(
                        """
                        The recommendation engine uses a **Hybrid
                        Recommendation System**.

                        ### 1️⃣ Preference Matching — 70%

                        Checks whether the car matches:

                        - Brand
                        - Body type
                        - Transmission
                        - City

                        ### 2️⃣ Cosine Similarity — 25%

                        Measures how similar the car is to your
                        overall preference profile using:

                        - Brand
                        - Model
                        - Fuel type
                        - Body type
                        - Transmission
                        - City
                        - Mileage
                        - Engine
                        - Power
                        - Price
                        - Kilometers driven
                        - Model year

                        ### 3️⃣ K-Means Cluster — 5%

                        Identifies whether the car belongs to the
                        same vehicle segment as your preference profile.

                        ### 🏆 Final Hybrid Score

                        **70% Preference Match**

                        +

                        **25% Cosine Similarity**

                        +

                        **5% Cluster Match**

                        =

                        **Final Recommendation Score**
                        """
                    )


# ============================================================
# AI CHATBOT
# ============================================================

# ============================================================
# AI CHATBOT
# ============================================================

elif page == "🤖 AI Car Chatbot":

    st.title("🤖 AI Car Assistant")

    st.write(
        "Ask me about used cars, prices and recommendations."
    )
    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    if "chat_history" not in st.session_state:

        st.session_state.chat_history = []

    # Display previous messages

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.write(message["content"])

            if (
                message["role"] == "assistant"
                and "data" in message
                and not message["data"].empty
            ):

                st.dataframe(
                    message["data"][
                        [
                            "brand",
                            "model",
                            "variant",
                            "fuel_type",
                            "body_type",
                            "transmission",
                            "city",
                            "price_lakh",
                            "km_driven",
                            "model_year"
                        ]
                    ],
                    use_container_width=True
                )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    user_question = st.chat_input(
        "Example: I have ₹8 lakh budget. Suggest an Audi..."
    )

    if user_question:

        # ----------------------------------------------------
        # SAVE USER MESSAGE
        # ----------------------------------------------------

        st.session_state.chat_history.append({

            "role": "user",

            "content": user_question

        })

        with st.chat_message("user"):

            st.write(user_question)

        # ----------------------------------------------------
        # EXTRACT USER REQUIREMENTS
        # ----------------------------------------------------

        budget = extract_budget(user_question)

        brand = extract_brand(user_question)

        fuel = extract_fuel(user_question)

        body_type = extract_body_type(user_question)

        transmission = extract_transmission(user_question)

        city = extract_city(user_question)

        # ----------------------------------------------------
        # FIND RECOMMENDATIONS
        # ----------------------------------------------------

        recommendations = chatbot_recommend_cars(

            budget=budget,

            brand=brand,

            fuel=fuel,

            body_type=body_type,

            transmission=transmission,

            city=city,

            top_n=5

        )

        # ----------------------------------------------------
        # ASSISTANT RESPONSE
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            if recommendations.empty:

                response = (
                    "😔 I couldn't find cars matching "
                    "your requirements. Try increasing "
                    "your budget or changing the preferences."
                )

                st.write(response)

                st.session_state.chat_history.append({

                    "role": "assistant",

                    "content": response

                })

            else:

                response_parts = []

                if budget:

                    response_parts.append(
                        f"budget of ₹{budget:.1f} lakh"
                    )

                if brand:

                    response_parts.append(
                        f"brand **{brand}**"
                    )

                if fuel:

                    response_parts.append(
                        f"**{fuel}** fuel"
                    )

                if body_type:

                    response_parts.append(
                        f"**{body_type}**"
                    )

                if transmission:

                    response_parts.append(
                        f"**{transmission}** transmission"
                    )

                if city:

                    response_parts.append(
                        f"in **{city.title()}**"
                    )

                if response_parts:

                    response = (
                        "🚗 Based on your "
                        + ", ".join(response_parts)
                        + ", here are my top recommendations:"
                    )

                else:

                    response = (
                        "🚗 Here are some recommended cars "
                        "from the available dataset:"
                    )

                st.markdown(response)

                st.dataframe(

                    recommendations[
                        [
                            "brand",
                            "model",
                            "variant",
                            "fuel_type",
                            "body_type",
                            "transmission",
                            "city",
                            "price_lakh",
                            "km_driven",
                            "model_year"
                        ]
                    ],

                    use_container_width=True

                )

                st.success(
                    f"Found {len(recommendations)} suitable "
                    "cars for you."
                )

                st.session_state.chat_history.append({

                    "role": "assistant",

                    "content": response,

                    "data": recommendations

                })


# ============================================================
# CREATOR INFO
# ============================================================

elif page == "👩‍💻 Creator Info":

    st.markdown("# 👩‍💻 Creator")

    with st.container(border=True):

        st.markdown(
            """
            ### Vasantha Leela M

            🎓 B.E Computer Science and Engineering

            🏫 Karpagam Academy of Higher Education

            ### 🚀 Skills

            - Python
            - Machine Learning
            - Data Science
            - SQL
            - Pandas
            - NumPy
            - Scikit-learn
            - Streamlit
            - MLflow
            - AWS

            ### 💡 Project Focus

            - Car Resale Price Prediction
            - Recommendation Systems
            - Machine Learning
            - AI Chatbot
            - Data Analytics
            - Cloud Deployment

            ### 🎯 Career Objective

            Passionate about building intelligent,
            scalable and user-friendly AI-powered
            applications for real-world problems.
            """
        )

    st.write("")

    st.success(
        "Building AI-powered applications using Machine Learning, "
        "Data Science and Generative AI technologies."
    )