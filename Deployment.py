import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout="wide")

tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])

with tab1:
    st.header("This is Tab 1")
    st.write("Any Documentation")

with tab2:
    st.header("This is Tab 2")
    st.write("Data Visualization")

with tab3:
    input_keys = [
        "FRUITS_VEGGIES", "DAILY_STRESS", "PLACES_VISITED", "CORE_CIRCLE", "SUPPORTING_OTHERS",
        "SOCIAL_NETWORK", "ACHIEVEMENT", "DONATION", "BMI_RANGE", "TODO_COMPLETED", "FLOW",
        "DAILY_STEPS", "LIVE_VISION", "SLEEP_HOURS", "LOST_VACATION", "DAILY_SHOUTING",
        "SUFFICIENT_INCOME", "PERSONAL_AWARDS", "TIME_FOR_PASSION", "WEEKLY_MEDITATION",
        "AGE", "GENDER"
    ]

    default_values = {
        "FRUITS_VEGGIES": 1, "DAILY_STRESS": 1, "PLACES_VISITED": 1, "CORE_CIRCLE": 1,
        "SUPPORTING_OTHERS": 1, "SOCIAL_NETWORK": 1, "ACHIEVEMENT": 1, "DONATION": 1,
        "BMI_RANGE": 1, "TODO_COMPLETED": 1, "FLOW": 1, "DAILY_STEPS": 1, "LIVE_VISION": 1,
        "SLEEP_HOURS": 1, "LOST_VACATION": 1, "DAILY_SHOUTING": 1, "SUFFICIENT_INCOME": 1,
        "PERSONAL_AWARDS": 1, "TIME_FOR_PASSION": 1, "WEEKLY_MEDITATION": 1,
        "AGE": "36 to 50", "GENDER": "Female"
    }

    for key, val in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = val

    top_col1, top_col2, top_col3 = st.columns([5, 1, 1])
    with top_col1:
        st.header("Prediction of ⚖️ Work Life Balance Score")

    with top_col2:
        if st.button("Reset"):
            for key in input_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    with top_col3:
        if st.button("Predict Score"):
            model = joblib.load("xgb_pipeline_model.joblib")
            input_dict = {key:st.session_state[key] for key in input_keys}
            input_df = pd.DataFrame([input_dict])

            try:
                prediction = model.predict(input_df)[0]
                st.success(f"🎯 Predicted Work-Life Balance Score: **{prediction:.2f}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    col1, space, col2, space, col3 = st.columns([1, 0.1, 1, 0.1, 1])

    with col1:
        sub1, _ = st.columns([1, 1])
        with sub1:
            st.number_input("FRUITS_VEGGIES", min_value=0, step=1, max_value=5, key="FRUITS_VEGGIES")
            st.number_input("DAILY_STRESS", min_value=0, step=1, max_value=5, key="DAILY_STRESS")
            st.number_input("PLACES_VISITED", min_value=0, step=1, max_value=10, key="PLACES_VISITED")
            st.number_input("CORE_CIRCLE", min_value=0, step=1, max_value=10, key="CORE_CIRCLE")
            st.number_input("SUPPORTING_OTHERS", min_value=0, step=1, max_value=10, key="SUPPORTING_OTHERS")
            st.number_input("SOCIAL_NETWORK", min_value=0, step=1, max_value=10, key="SOCIAL_NETWORK")
            st.number_input("ACHIEVEMENT", min_value=0, step=1, max_value=10, key="ACHIEVEMENT")

    with col2:
        sub1, _ = st.columns([1, 1])
        with sub1:
            st.number_input("DONATION", min_value=0, step=1, max_value=5, key="DONATION")
            st.number_input("BMI_RANGE", min_value=1, step=1, max_value=2, key="BMI_RANGE")
            st.number_input("TODO_COMPLETED", min_value=0, step=1, max_value=10, key="TODO_COMPLETED")
            st.number_input("FLOW", min_value=0, step=1, max_value=10, key="FLOW")
            st.number_input("DAILY_STEPS", min_value=1, step=1, max_value=10, key="DAILY_STEPS")
            st.number_input("LIVE_VISION", min_value=0, step=1, max_value=10, key="LIVE_VISION")
            st.number_input("SLEEP_HOURS", min_value=0, step=1, max_value=10, key="SLEEP_HOURS")

    with col3:
        sub1, _ = st.columns([1, 1])
        with sub1:
            st.number_input("LOST_VACATION", min_value=0, step=1, max_value=10, key="LOST_VACATION")
            st.number_input("DAILY_SHOUTING", min_value=0, step=1, max_value=10, key="DAILY_SHOUTING")
            st.number_input("SUFFICIENT_INCOME", min_value=1, step=1, max_value=2, key="SUFFICIENT_INCOME")
            st.number_input("PERSONAL_AWARDS", min_value=0, step=1, max_value=10, key="PERSONAL_AWARDS")
            st.number_input("TIME_FOR_PASSION", min_value=0, step=1, max_value=10, key="TIME_FOR_PASSION")
            st.number_input("WEEKLY_MEDITATION", min_value=0, step=1, max_value=10, key="WEEKLY_MEDITATION")
            st.selectbox("AGE", ['36 to 50', '51 or more', '21 to 35', 'Less than 20'], key="AGE")
            st.selectbox("GENDER", ['Female', 'Male'], key="GENDER")