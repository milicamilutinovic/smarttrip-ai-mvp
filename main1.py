import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from backend.api.schemas import GroupRequest
from backend.services.data_loader import load_destinations
from backend.services.intent_extraction import extract_intent
from backend.services.aggregation import aggregate_group_preferences
from backend.services.recommender import compute_hybrid_text_score
from backend.services.scoring import apply_scoring
from backend.services.diversification import (
    calculate_destination_similarity_matrix,
    diversify_recommendations
)
from backend.services.geocoding import get_coordinates
from backend.services.constants import TEMP_MAP, DURATION_DAYS

# Podešavanje stranice
st.set_page_config(page_title="SmartTrip App", page_icon="🌍", layout="wide")

st.markdown("""
<style>

.match-badge {
    display: inline-block;
    background: linear-gradient(135deg, #00c853, #64dd17);
    color: white;
    font-weight: 700;
    padding: 6px 10px;
    border-radius: 10px;
    text-align: center;
    font-size: 16px;
    line-height: 1.2;
    box-shadow: 0 3px 8px rgba(0,0,0,0.12);
}

.match-label {
    font-size: 10px;
    opacity: 0.9;
}

</style>
""", unsafe_allow_html=True)


st.title("🌍 Welcome To SmartTrip")
st.markdown("### *Your AI-powered group travel planner*")
st.markdown("Tell us how many of you are traveling and what are your preferences — we'll find the perfect destination for everyone.")
st.markdown("---")

# --- SIDEBAR: ZAJEDNIČKI PARAMETRI ---
st.sidebar.header("📍 Trip Basics")
st.sidebar.markdown("**Step 1:** Fill in your trip basics below, then add each traveler's preferences on the right.")
start_city = st.sidebar.text_input("Departure city", placeholder="e.g. Belgrade")
month_options = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
month_display = st.sidebar.selectbox("Month of travel", list(month_options.keys()), index=3)
month = month_options[month_display]

# front handling za regione
region_options = {
    "Europe": "europe",
    "Asia": "asia",
    "North America": "north_america",
    "South America": "south_america",
    "Middle East": "middle_east",
    "Africa": "africa",
    "Oceania": "oceania"
}

region_display = st.sidebar.selectbox("Region", list(region_options.keys()))
region = region_options[region_display]

# --- GLAVNI DEO: GRUPA PUTNIKA ---
st.subheader("👥 Travel group")
num_people = st.number_input("How many people are traveling?", min_value=1, max_value=10, value=2)

group_data = []

# Kreiramo kolone za putnike da bi izgledalo kao "grid"
cols = st.columns(2) 

for i in range(num_people):
    with cols[i % 2]: # Naizmenično punimo dve kolone
        with st.expander(f"Traveler {i+1} - Preferences", expanded=True):
            budget = st.number_input(f"Maximum budget (€)", min_value=100, value=500, key=f"bud_{i}")
            duration = st.selectbox(f"Trip duration", list(DURATION_DAYS.keys()), index=2, key=f"dur_{i}")
            temp = st.radio(f"Preferred temperature", ["cold", "neutral", "warm"], index=1, horizontal=True, key=f"temp_{i}")
            description = st.text_area(f"What would you like to do during the trip?", placeholder="e.g. I want to visit museums and try local food", key=f"desc_{i}")
            
            group_data.append({
                "max_budget_eur": float(budget),
                "max_days": duration,
                "temperature": temp,
                "description": description
            })

st.markdown("---")

st.markdown("#### Ready to find your perfect destination? 🗺️")
# --- LOGIKA PREPORUKE ---
if st.button("✨ Find the best destinations", use_container_width=True):
    with st.spinner("Analyzing travel data and matching group preferences..."):
        # 1. Geokodiranje polazišta
        coords = get_coordinates(start_city)
        
        if coords is None:
            st.error("Unfortunately, we couldn't find the departure city. Please try again.")
        else:
            # Sklapanje zahteva u formatu koji tvoj backend očekuje
            try:
                validated = GroupRequest(
                    start_city=start_city,
                    month=month,
                    region=region,
                    group=group_data
                )
                group_input = validated.dict()
            except Exception as e:
                st.error(f"Input validation error: {e}")
                st.stop()

            # 2. Load destinations
            df, vectorizer, tfidf_matrix = load_destinations(
                start_city_coords=coords,
                month=month,
                region=region
            )

            if df is None:
                st.error("No destinations available for the selected region.")
                st.stop()

            if not description.strip():
                st.warning("Please add a description for all travelers.")
                st.stop()

            for person in group_input["group"]:
                intent_result = extract_intent(person["description"])
                person["activities"] = intent_result.get("activities", [])

            # 3. Agregacija preferencija grupe
            group_profile = aggregate_group_preferences(group_input, strategy="average_with_fairness")

            # 4. Hybrid Text Scoring (Intent + TF-IDF)
            df = compute_hybrid_text_score(df, group_profile["group_description"], vectorizer, tfidf_matrix = tfidf_matrix[:len(df)], beta=0.6, alpha=0.7)

            # 5. Final Scoring (Fairness, Budget, Temp...)
            df = apply_scoring(df, group_profile, group_input=group_input, use_fairness=True)

            # 6. Sortiranje i Diversifikacija
            sorted_df = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
            
            similarity_matrix = calculate_destination_similarity_matrix(tfidf_matrix[df.index])
            top_results = diversify_recommendations(sorted_df, similarity_matrix, top_k=10, diversity_level="balanced")

            # --- PRIKAZ REZULTATA ---
            st.balloons()
            st.success("Here are the top 10 destinations for your group!")

            top_results = top_results.reset_index(drop=True)
            
            for idx, row in top_results.iterrows():
                with st.container():
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        # Generisanje ocene kao "Bedž"
                        #st.metric("Score", f"{row['final_score']:.2f}")
                        match_pct = int(row['final_score'] * 100)
                        badge_html = f"""
                            <div class="match-badge">
                            {match_pct}%<br>
                            <span class="match-label">Match</span>
                            </div>
                            """
                        st.markdown(badge_html, unsafe_allow_html=True)

                    with c2:
                        st.markdown(f"### {idx+1}. {row['city'].title()}, {row['country']}")
                        lat, lon = get_coordinates(row["city"])
                        map_html = f"""
                        <iframe
                        width="100%"
                        height="300"
                        style="border:0; border-radius: 10px;"
                        loading="lazy"
                        allowfullscreen
                        src="https://maps.google.com/maps?q={lat},{lon}&z=10&output=embed">
                        </iframe>"""
                        components.html(map_html, height=320)
                        #image_url = f"https://loremflickr.com/800/400/{row['city']},travel/all" # city slika
                        #st.image(image_url, use_container_width=True, caption=f"Explore {row['city']}")
                        st.caption(
                             f"💰 Estimated cost per person: **€{row['estimated_cost']:.0f}** | "
                             f"🌡️ Average temperature: **{row['avg_temp']:.1f}°C**"
                        )
                        st.write(row['short_description'])
                        
                        # Dodatna vizuelna informacija o AI logici
                        exp = st.expander("Why was this destination recommended?")
                        with exp:
                            exp.write(
                                f"✅ Fits your group budget of €{group_profile['group_budget']}, estimated cost per person: €{row['estimated_cost']:.0f}\n\n"
                                f"✅ Average temperature in selected month: {row['avg_temp']:.1f}°C\n\n"
                            )
                            
                            # Score breakdown
                            exp.write("**🧠 AI Score Breakdown**")
                            
                            scores = {
                                "🎯 Interest match": row.get("hybrid_text_score", 0),
                                "💰 Budget fit":     row.get("budget_score", 0),
                                "🌡️ Temperature fit": row.get("temp_score", 0),
                                "📅 Duration fit":   row.get("duration_score", 0),
                            }
                            
                            if "fairness_score" in row:
                                scores["⚖️ Group fairness"] = row.get("fairness_score", 0)

                            for label, value in scores.items():
                                col_label, col_bar = st.columns([1.5, 3])
                                with col_label:
                                    st.markdown(f"<small>{label}</small>", unsafe_allow_html=True)
                                with col_bar:
                                    st.progress(float(min(max(value, 0), 1)))
                    st.markdown("---")