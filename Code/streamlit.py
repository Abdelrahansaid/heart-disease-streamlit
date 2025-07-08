import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# Load model and feature columns
model = joblib.load("heart_disease_model_only.pkl")
columns = joblib.load("columns.pkl")

# Session init
if "user_input" not in st.session_state:
    st.session_state["user_input"] = None

# Language selection
lang = st.sidebar.selectbox("Choose Language / اختر اللغة", ["English", "العربية"])

text = {
    "English": {
        "title": "💓 Heart Disease Prediction App",
        "intro": "Enter the patient's health parameters below to predict risk of heart disease.",
        "age": "Age", "sex": "Sex", "cp": "Chest Pain Type", "trestbps": "Resting Blood Pressure (mm Hg)",
        "chol": "Cholesterol (mg/dl)", "fbs": "Fasting Blood Sugar > 120 mg/dl", "restecg": "Resting ECG",
        "thalach": "Max Heart Rate Achieved", "exang": "Exercise Induced Angina", "oldpeak": "ST Depression",
        "slope": "Slope of Peak Exercise ST", "ca": "Number of Major Vessels Colored (0–3)",
        "thal": "Thalassemia Type", "predict": "Predict", "result": "🩺 Prediction Result:",
        "high": "⚠️ High Risk of Heart Disease", "low": "✅ Low Risk of Heart Disease",
        "page": ["🏠 Prediction", "📊 Visualizations"]
    },
    "العربية": {
        "title": "💓 تطبيق توقع مرض القلب",
        "intro": "ادخل بيانات المريض الصحية لتوقع احتمال الإصابة بمرض القلب.",
        "age": "العمر", "sex": "النوع", "cp": "نوع ألم الصدر", "trestbps": "ضغط الدم أثناء الراحة",
        "chol": "الكوليسترول", "fbs": "سكر صائم > 120", "restecg": "رسم القلب أثناء الراحة",
        "thalach": "أقصى معدل ضربات القلب", "exang": "ذبحة صدرية بالمجهود", "oldpeak": "انخفاض ST",
        "slope": "انحدار ST أثناء المجهود", "ca": "عدد الأوعية الملونة", "thal": "نوع الثلاسيميا",
        "predict": "توقع", "result": "🩺 النتيجة:", "high": "⚠️ خطر مرتفع", "low": "✅ خطر منخفض",
        "page": ["🏠 التوقع", "📊 التصورات"]
    }
}

selected_page = st.sidebar.radio("📄 Navigation", text[lang]["page"])

# =========================
# Page 1: Prediction
# =========================
if selected_page == text[lang]["page"][0]:
    st.title(text[lang]["title"])
    st.markdown(text[lang]["intro"])

    # Inputs
    age = st.slider(text[lang]["age"], 20, 90, 45)
    sex = st.selectbox(text[lang]["sex"], ["Male", "Female"] if lang == "English" else ["ذكر", "أنثى"])
    cp = st.selectbox(text[lang]["cp"],
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"] if lang == "English"
        else ["الذبحة الصدرية النموذجية", "الذبحة غير النمطية", "ألم غير قلبي", "بدون أعراض"]
    )
    trestbps = st.number_input(text[lang]["trestbps"], 80, 200, 120)
    chol = st.number_input(text[lang]["chol"], 100, 600, 200)
    fbs = st.selectbox(text[lang]["fbs"], ["Yes", "No"] if lang == "English" else ["نعم", "لا"])
    restecg = st.selectbox(text[lang]["restecg"],
        ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"] if lang == "English"
        else ["طبيعي", "شذوذ ST-T", "تضخم البطين الأيسر"]
    )
    thalach = st.number_input(text[lang]["thalach"], 60, 220, 150)
    exang = st.selectbox(text[lang]["exang"], ["Yes", "No"] if lang == "English" else ["نعم", "لا"])
    oldpeak = st.number_input(text[lang]["oldpeak"], 0.0, 6.0, 1.0)
    slope = st.selectbox(text[lang]["slope"], ["Upsloping", "Flat", "Downsloping"] if lang == "English" else ["صاعد", "مستوٍ", "هابط"])
    ca = st.selectbox(text[lang]["ca"], [0, 1, 2, 3])
    thal = st.selectbox(text[lang]["thal"],
        ["Normal", "Fixed Defect", "Reversible Defect"] if lang == "English"
        else ["طبيعي", "خلل ثابت", "خلل قابل للعكس"]
    )

    # Data preprocessing
    def build_feature_df():
        input_dict = {col: 0 for col in columns}
        input_dict["age"] = age
        input_dict["trestbps"] = trestbps
        input_dict["chol"] = chol
        input_dict["thalach"] = thalach
        input_dict["oldpeak"] = oldpeak
        input_dict["ca"] = ca

        if sex in ["Male", "ذكر"]:
            input_dict["sex_1.0"] = 1

        cp_map = {
            "Typical Angina": 1, "الذبحة الصدرية النموذجية": 1,
            "Atypical Angina": 2, "الذبحة غير النمطية": 2,
            "Non-anginal Pain": 3, "ألم غير قلبي": 3,
            "Asymptomatic": 4, "بدون أعراض": 4
        }
        cp_val = cp_map[cp]
        if f"cp_{cp_val}.0" in input_dict:
            input_dict[f"cp_{cp_val}.0"] = 1
        elif f"cp_{cp_val}" in input_dict:
            input_dict[f"cp_{cp_val}"] = 1

        if fbs in ["Yes", "نعم"]:
            input_dict["fbs_1"] = 1

        restecg_map = {
            "Normal": 0, "طبيعي": 0,
            "ST-T Abnormality": 1, "شذوذ ST-T": 1,
            "Left Ventricular Hypertrophy": 2, "تضخم البطين الأيسر": 2
        }
        r_val = restecg_map[restecg]
        if f"restecg_{r_val}" in input_dict:
            input_dict[f"restecg_{r_val}"] = 1

        if exang in ["Yes", "نعم"]:
            if "exang_1" in input_dict:
                input_dict["exang_1"] = 1
            elif "exang_1.0" in input_dict:
                input_dict["exang_1.0"] = 1

        slope_map = {
            "Upsloping": 1, "صاعد": 1,
            "Flat": 2, "مستوٍ": 2,
            "Downsloping": 3, "هابط": 3
        }
        slope_val = slope_map[slope]
        if f"slope_{slope_val}.0" in input_dict:
            input_dict[f"slope_{slope_val}.0"] = 1
        elif f"slope_{slope_val}" in input_dict:
            input_dict[f"slope_{slope_val}"] = 1

        thal_map = {
            "Normal": 3, "طبيعي": 3,
            "Fixed Defect": 6, "خلل ثابت": 6,
            "Reversible Defect": 7, "خلل قابل للعكس": 7
        }
        thal_val = thal_map[thal]
        if f"thal_{thal_val}.0" in input_dict:
            input_dict[f"thal_{thal_val}.0"] = 1
        elif f"thal_{thal_val}" in input_dict:
            input_dict[f"thal_{thal_val}"] = 1

        return pd.DataFrame([input_dict])

    if st.button(text[lang]["predict"]):
        input_df = build_feature_df()
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0][1]

        st.session_state["user_input"] = input_df.copy()
        st.session_state["user_input"]["target"] = prediction[0]

        st.subheader(text[lang]["result"])
        if prediction[0] == 1:
            st.error(f"{text[lang]['high']} ({proba*100:.2f}%)")
        else:
            st.success(f"{text[lang]['low']} ({(1 - proba)*100:.2f}%)")

# =========================
# Page 2: Visualizations
# =========================
else:
    st.title("📊 " + ("التصورات التفاعلية" if lang == "العربية" else "Interactive Visualizations"))

    # ملاحظة توضيحية للمستخدم عن النقطة الحمراء
    note = "🔴 Your red point represents your data input — compare yourself with other cases." if lang == "English" \
        else "🔴 النقطة الحمراء تمثل بياناتك التي أدخلتها – يمكنك مقارنة حالتك مع باقي الحالات."
    st.markdown("### 💡 عرض حسب حالة القلب")
    st.info(note)

    try:
        data = pd.read_csv("data1.csv")

        # اجعل القيم الداخلية ثابتة (target: 0/1, source: general/user)
        data["source"] = "general"
        if st.session_state["user_input"] is not None:
            user_row = st.session_state["user_input"].copy()
            user_row["source"] = "user"
            user_row["target"] = int(user_row["target"].values[0])
            data = pd.concat([data, user_row], ignore_index=True)

        # للعرض فقط: ترجمات حسب اللغة
        def get_target_display(x):
            if lang == "العربية":
                return "💓 مريض" if x == 1 else "✅ سليم"
            else:
                return "Disease" if x == 1 else "No Disease"

        def get_source_display(x):
            return "🧍‍♂️ أنت" if x == "user" else "📊 عامة"

        data["target_display"] = data["target"].apply(get_target_display)
        data["source_display"] = data["source"].apply(get_source_display)

        numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

        # أسماء المميزات الظاهرة للمستخدم باللغتين
        feature_labels = {
            "English": {
                "age": "Age",
                "trestbps": "Resting Blood Pressure",
                "chol": "Cholesterol",
                "thalach": "Max Heart Rate",
                "oldpeak": "ST Depression",
                "ca": "Number of Vessels Colored"
            },
            "العربية": {
                "age": "العمر",
                "trestbps": "ضغط الدم أثناء الراحة",
                "chol": "الكوليسترول",
                "thalach": "أقصى معدل ضربات القلب",
                "oldpeak": "انخفاض ST",
                "ca": "عدد الأوعية الدموية المصبوغة"
            }
        }

        chart_type = st.selectbox("📈 اختر نوع الرسم / Select Chart Type", ["📊 Boxplot", "📉 Histogram", "🧮 Scatter"])

        # قائمة تظهر للمستخدم بالتسمية المفهومة
        display_features = [feature_labels[lang][f] for f in numeric_features]
        selected_display = st.selectbox("🧬 اختر الميزة / Select Feature", display_features)
        selected_feature = numeric_features[display_features.index(selected_display)]

        color_map = {
            "📊 عامة": "blue",
            "🧍‍♂️ أنت": "red"
        }

        if chart_type == "📊 Boxplot":
            fig = px.box(
                data,
                x="target_display",
                y=selected_feature,
                color="source_display",
                points="all",
                color_discrete_map=color_map,
                labels={selected_feature: feature_labels[lang][selected_feature],
                        "target_display": get_target_display(1)}
            )
        elif chart_type == "📉 Histogram":
            fig = px.histogram(
                data,
                x=selected_feature,
                color="source_display",
                barmode="overlay",
                opacity=0.7,
                color_discrete_map=color_map,
                labels={selected_feature: feature_labels[lang][selected_feature]}
            )
        elif chart_type == "🧮 Scatter":
            available_second = [f for f in numeric_features if f != selected_feature]
            display_second = [feature_labels[lang][f] for f in available_second]
            second_display = st.selectbox("➕ اختر ميزة إضافية / Extra Feature", display_second)
            second_feature = available_second[display_second.index(second_display)]
            fig = px.scatter(
                data,
                x=selected_feature,
                y=second_feature,
                color="source_display",
                symbol="target_display",
                hover_name="source_display",
                color_discrete_map=color_map,
                labels={
                    selected_feature: feature_labels[lang][selected_feature],
                    second_feature: feature_labels[lang][second_feature]
                }
            )

        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.warning("🔍 تأكد من وجود ملف 'data1.csv' لعرض الرسومات.")
