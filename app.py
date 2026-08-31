import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sentiment AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# Replace this with your actual Hugging Face model ID.
MODEL_ID = "saimagesh/distilbert-sentiment-final"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99, 102, 241, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(6, 182, 212, 0.10),
                transparent 30%
            ),
            #050816;
    }


    /* Main container */

    .main .block-container {
        max-width: 1400px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }


    /* Hide default Streamlit elements */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* Text */

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    p {
        color: #94a3b8;
    }


    /* Text area */

    textarea {
        background-color: rgba(255,255,255,0.04) !important;
        color: #f8fafc !important;

        border: 1px solid rgba(255,255,255,0.10) !important;

        border-radius: 15px !important;
    }


    /* Buttons */

    .stButton > button {

        border-radius: 12px !important;

        min-height: 42px;

        font-weight: 700 !important;

        background: rgba(255,255,255,0.04) !important;

        border: 1px solid rgba(255,255,255,0.10) !important;

        color: #e2e8f0 !important;

        transition: 0.2s ease;
    }


    .stButton > button:hover {

        border-color: rgba(129,140,248,0.55) !important;

        background: rgba(99,102,241,0.12) !important;

        transform: translateY(-2px);
    }


    /* Primary button */

    button[kind="primary"] {

        background:
            linear-gradient(
                90deg,
                #4f46e5,
                #7c3aed
            ) !important;

        border: none !important;

        color: white !important;
    }


    /* Metrics */

    [data-testid="stMetric"] {

        background: rgba(255,255,255,0.035);

        border: 1px solid rgba(255,255,255,0.08);

        border-radius: 16px;

        padding: 15px;
    }


    [data-testid="stMetricLabel"] {

        color: #94a3b8 !important;
    }


    [data-testid="stMetricValue"] {

        color: #ffffff !important;
    }


    /* Divider */

    hr {

        border-color: rgba(255,255,255,0.08) !important;
    }


    /* Footer */

    .custom-footer {

        text-align: center;

        color: #475569;

        margin-top: 50px;

        padding-top: 20px;

        border-top: 1px solid rgba(255,255,255,0.07);

        font-size: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


if "review" not in st.session_state:
    st.session_state.review = ""


if "result" not in st.session_state:
    st.session_state.result = None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID
    )

    model.eval()

    return tokenizer, model


try:

    tokenizer, model = load_model()

    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error(
        "Unable to load the Hugging Face model."
    )

    st.code(str(e))

    st.info(
        "Check MODEL_ID at the top of app.py."
    )

    st.stop()


# ============================================================
# PREDICTION
# ============================================================

def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )


    with torch.no_grad():

        outputs = model(**inputs)


    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )


    predicted_class = torch.argmax(
        probabilities,
        dim=-1
    ).item()


    confidence = probabilities[
        0,
        predicted_class
    ].item()


    negative_probability = probabilities[
        0,
        0
    ].item()


    positive_probability = probabilities[
        0,
        1
    ].item()


    # Use model labels when available

    if hasattr(model.config, "id2label"):

        sentiment = model.config.id2label.get(
            predicted_class,
            "POSITIVE" if predicted_class == 1 else "NEGATIVE"
        )

    else:

        sentiment = (
            "POSITIVE"
            if predicted_class == 1
            else "NEGATIVE"
        )


    sentiment = sentiment.upper()


    return (
        sentiment,
        confidence,
        negative_probability,
        positive_probability
    )


# ============================================================
# 3D AI CORE
# ============================================================

def render_3d_ai_core(sentiment):

    if sentiment == "POSITIVE":

        primary_color = "#22c55e"

    elif sentiment == "NEGATIVE":

        primary_color = "#ef4444"

    else:

        primary_color = "#6366f1"


    html = f"""
    <!DOCTYPE html>

    <html>

    <head>

        <script
            src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js">
        </script>

        <style>

            html, body {{

                margin: 0;
                padding: 0;

                overflow: hidden;

                background: transparent;

            }}


            #scene {{

                width: 100%;
                height: 440px;

                position: relative;

            }}


            canvas {{

                display: block;

            }}


            .center-label {{

                position: absolute;

                left: 50%;
                top: 50%;

                transform:
                    translate(-50%, -50%);

                text-align: center;

                pointer-events: none;

                font-family:
                    Arial,
                    sans-serif;

                color: white;

            }}


            .brain {{

                font-size: 45px;

            }}


            .title {{

                font-size: 20px;

                font-weight: 800;

                letter-spacing: 3px;

                margin-top: 5px;

            }}


            .model {{

                color: #94a3b8;

                font-size: 10px;

                letter-spacing: 2px;

                margin-top: 5px;

            }}

        </style>

    </head>


    <body>

        <div id="scene">

            <div class="center-label">

                <div class="brain">
                    🧠
                </div>

                <div class="title">
                    AI CORE
                </div>

                <div class="model">
                    DISTILBERT
                </div>

            </div>

        </div>


        <script>

            const container =
                document.getElementById(
                    "scene"
                );


            const scene =
                new THREE.Scene();


            const camera =
                new THREE.PerspectiveCamera(
                    55,
                    container.clientWidth /
                    container.clientHeight,
                    0.1,
                    1000
                );


            camera.position.z = 7;


            const renderer =
                new THREE.WebGLRenderer({{
                    alpha: true,
                    antialias: true
                }});


            renderer.setPixelRatio(
                Math.min(
                    window.devicePixelRatio,
                    2
                )
            );


            renderer.setSize(
                container.clientWidth,
                container.clientHeight
            );


            container.appendChild(
                renderer.domElement
            );


            const group =
                new THREE.Group();


            scene.add(group);


            const nodeMaterial =
                new THREE.MeshBasicMaterial({{
                    color: "{primary_color}"
                }});


            const lineMaterial =
                new THREE.LineBasicMaterial({{
                    color: "{primary_color}",
                    transparent: true,
                    opacity: 0.20
                }});


            const nodes = [];


            // ============================================
            // NEURAL NETWORK NODES
            // ============================================

            for (
                let i = 0;
                i < 42;
                i++
            ) {{

                const geometry =
                    new THREE.SphereGeometry(
                        0.045 +
                        Math.random() * 0.035,
                        12,
                        12
                    );


                const node =
                    new THREE.Mesh(
                        geometry,
                        nodeMaterial
                    );


                const radius =
                    1.8 +
                    Math.random() * 1.6;


                const angle =
                    Math.random() *
                    Math.PI * 2;


                const z =
                    (
                        Math.random() -
                        0.5
                    ) * 3;


                node.position.set(

                    Math.cos(angle) *
                    radius,

                    Math.sin(angle) *
                    radius,

                    z

                );


                group.add(node);

                nodes.push(node);

            }}


            // ============================================
            // CONNECTIONS
            // ============================================

            for (
                let i = 0;
                i < nodes.length;
                i++
            ) {{

                for (
                    let j = i + 1;
                    j < nodes.length;
                    j++
                ) {{

                    const distance =
                        nodes[i]
                        .position
                        .distanceTo(
                            nodes[j].position
                        );


                    if (distance < 1.25) {{

                        const geometry =
                            new THREE.BufferGeometry()
                            .setFromPoints([

                                nodes[i].position,

                                nodes[j].position

                            ]);


                        const line =
                            new THREE.Line(
                                geometry,
                                lineMaterial
                            );


                        group.add(line);

                    }}

                }}

            }}


            // ============================================
            // CENTRAL AI CORE
            // ============================================

            const coreGeometry =
                new THREE.IcosahedronGeometry(
                    1.05,
                    3
                );


            const coreMaterial =
                new THREE.MeshBasicMaterial({{

                    color:
                        "{primary_color}",

                    wireframe: true,

                    transparent: true,

                    opacity: 0.55

                }});


            const core =
                new THREE.Mesh(
                    coreGeometry,
                    coreMaterial
                );


            group.add(core);


            // ============================================
            // INNER CORE
            // ============================================

            const innerGeometry =
                new THREE.SphereGeometry(
                    0.68,
                    32,
                    32
                );


            const innerMaterial =
                new THREE.MeshBasicMaterial({{

                    color:
                        "{primary_color}",

                    transparent: true,

                    opacity: 0.10

                }});


            const inner =
                new THREE.Mesh(
                    innerGeometry,
                    innerMaterial
                );


            group.add(inner);


            // ============================================
            // ORBIT RINGS
            // ============================================

            const rings = [];


            for (
                let i = 0;
                i < 3;
                i++
            ) {{

                const geometry =
                    new THREE.TorusGeometry(
                        1.55 +
                        i * 0.30,

                        0.008,

                        8,

                        100
                    );


                const material =
                    new THREE.MeshBasicMaterial({{

                        color:
                            "{primary_color}",

                        transparent: true,

                        opacity: 0.28

                    }});


                const ring =
                    new THREE.Mesh(
                        geometry,
                        material
                    );


                ring.rotation.x =
                    Math.random() *
                    Math.PI;


                ring.rotation.y =
                    Math.random() *
                    Math.PI;


                group.add(ring);

                rings.push(ring);

            }}


            // ============================================
            // MOUSE INTERACTION
            // ============================================

            let mouseX = 0;

            let mouseY = 0;


            container.addEventListener(
                "mousemove",
                function(event) {{

                    const rect =
                        container
                        .getBoundingClientRect();


                    mouseX =
                        (
                            event.clientX -
                            rect.left
                        ) /
                        rect.width -
                        0.5;


                    mouseY =
                        (
                            event.clientY -
                            rect.top
                        ) /
                        rect.height -
                        0.5;

                }}
            );


            // ============================================
            // ANIMATION
            // ============================================

            const clock =
                new THREE.Clock();


            function animate() {{

                requestAnimationFrame(
                    animate
                );


                const time =
                    clock.getElapsedTime();


                group.rotation.y +=
                    0.0025;


                group.rotation.x =
                    Math.sin(
                        time * 0.4
                    ) * 0.06;


                group.rotation.y +=
                    mouseX * 0.001;


                group.rotation.x +=
                    mouseY * 0.001;


                core.rotation.x +=
                    0.004;


                core.rotation.y +=
                    0.006;


                inner.scale.setScalar(

                    1 +

                    Math.sin(
                        time * 2
                    ) * 0.05

                );


                rings.forEach(
                    function(ring, index) {{

                        ring.rotation.z +=
                            0.002 *
                            (index + 1);

                    }}
                );


                renderer.render(
                    scene,
                    camera
                );

            }}


            animate();


            // ============================================
            // RESIZE
            // ============================================

            window.addEventListener(
                "resize",
                function() {{

                    camera.aspect =
                        container.clientWidth /
                        container.clientHeight;


                    camera.updateProjectionMatrix();


                    renderer.setSize(
                        container.clientWidth,
                        container.clientHeight
                    );

                }}
            );

        </script>

    </body>

    </html>
    """


    components.html(
        html,
        height=440,
        scrolling=False
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 Sentiment AI")

    st.caption(
        "Transformer-powered sentiment analysis"
    )

    st.divider()


    if model_loaded:

        st.success(
            "🟢 Model Online"
        )

    else:

        st.error(
            "🔴 Model Offline"
        )


    st.divider()


    st.subheader("⚙️ Model")

    st.write(
        "**Architecture:** DistilBERT"
    )

    st.write(
        "**Task:** Sentiment Classification"
    )

    st.write(
        "**Dataset:** IMDb"
    )

    st.write(
        "**Max Tokens:** 256"
    )


    st.divider()


    st.subheader("📊 Performance")


    st.metric(
        "Accuracy",
        "91.41%"
    )


    st.metric(
        "F1 Score",
        "91.44%"
    )


    st.divider()


    st.subheader("🧪 Session")


    st.metric(
        "Predictions",
        len(st.session_state.history)
    )


    if st.button(
        "🗑️ Clear History",
        use_container_width=True
    ):

        st.session_state.history = []

        st.session_state.result = None

        st.rerun()


# ============================================================
# HERO
# ============================================================

st.markdown(
    "### ✨ TRANSFORMER INTELLIGENCE"
)


st.title(
    "🧠 SENTIMENT AI"
)


st.subheader(
    "Understand the emotion behind every review"
)


st.caption(
    "Fine-tuned DistilBERT • IMDb Movie Reviews"
)


st.success(
    "🟢 MODEL ONLINE  •  DISTILBERT"
)


# ============================================================
# 3D CORE
# ============================================================

st.divider()


st.header(
    "🌐 Neural Intelligence Core"
)


st.caption(
    "Interactive 3D visualization of the sentiment engine"
)


current_sentiment = "NEUTRAL"


if st.session_state.result:

    current_sentiment = (
        st.session_state.result["sentiment"]
    )


render_3d_ai_core(
    current_sentiment
)


# ============================================================
# INPUT
# ============================================================

st.divider()


st.header(
    "📝 Analyze a Review"
)


st.caption(
    "Enter a movie review and let DistilBERT predict its sentiment."
)


# ============================================================
# QUICK EXAMPLES
# ============================================================

st.write("⚡ **Quick Examples**")


example1, example2, example3 = st.columns(3)


with example1:

    if st.button(
        "😍 Positive Example",
        use_container_width=True
    ):

        st.session_state.review = (
            "This movie was absolutely fantastic! "
            "The acting was brilliant and the story "
            "kept me entertained from beginning to end."
        )

        st.rerun()


with example2:

    if st.button(
        "😡 Negative Example",
        use_container_width=True
    ):

        st.session_state.review = (
            "This movie was terrible. "
            "The story was boring and the acting "
            "was absolutely awful."
        )

        st.rerun()


with example3:

    if st.button(
        "🎬 Mixed Example",
        use_container_width=True
    ):

        st.session_state.review = (
            "The movie had some interesting ideas "
            "and a few good performances, although "
            "some parts felt slow."
        )

        st.rerun()


# ============================================================
# TEXT INPUT
# ============================================================

review = st.text_area(
    "Movie Review",
    value=st.session_state.review,
    placeholder=(
        "Write your movie review here..."
    ),
    height=170,
    max_chars=500
)


st.caption(
    f"✏️ {len(review)} / 500 characters"
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

left, center, right = st.columns(
    [1, 2, 1]
)


with center:

    analyze = st.button(
        "✨ ANALYZE WITH DISTILBERT",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    if not review.strip():

        st.warning(
            "⚠️ Please enter a review first."
        )

    else:

        with st.spinner(
            "🧠 DistilBERT is analyzing..."
        ):

            (
                sentiment,
                confidence,
                negative_probability,
                positive_probability
            ) = predict_sentiment(
                review
            )


        result = {

            "sentiment":
                sentiment,

            "confidence":
                confidence,

            "negative":
                negative_probability,

            "positive":
                positive_probability,

            "review":
                review

        }


        st.session_state.result = result


        st.session_state.history.insert(
            0,
            result
        )


        st.session_state.history = (
            st.session_state.history[:8]
        )


        st.rerun()


# ============================================================
# RESULT
# ============================================================

if st.session_state.result:

    result = st.session_state.result


    sentiment = result["sentiment"]

    confidence = result["confidence"]

    negative_probability = result["negative"]

    positive_probability = result["positive"]


    st.divider()


    st.header(
        "🎯 AI Prediction"
    )


    # --------------------------------------------------------
    # RESULT MESSAGE
    # --------------------------------------------------------

    if sentiment == "POSITIVE":

        st.success(
            f"😊 POSITIVE — {confidence:.1%} confidence"
        )

    else:

        st.error(
            f"😞 NEGATIVE — {confidence:.1%} confidence"
        )


    # --------------------------------------------------------
    # RESULT METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "🎯 Confidence",
            f"{confidence:.2%}"
        )


    with col2:

        st.metric(
            "😞 Negative",
            f"{negative_probability:.2%}"
        )


    with col3:

        st.metric(
            "😊 Positive",
            f"{positive_probability:.2%}"
        )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    st.subheader(
        "Confidence"
    )


    st.progress(
        confidence
    )


    if confidence >= 0.90:

        st.success(
            "High-confidence prediction"
        )

    elif confidence >= 0.70:

        st.info(
            "Moderate-confidence prediction"
        )

    else:

        st.warning(
            "Low-confidence prediction"
        )


    # --------------------------------------------------------
    # PROBABILITY CHART
    # --------------------------------------------------------

    st.subheader(
        "📊 Sentiment Probability"
    )


    chart = go.Figure()


    chart.add_trace(
        go.Bar(

            x=[
                negative_probability,
                positive_probability
            ],

            y=[
                "Negative",
                "Positive"
            ],

            orientation="h",

            text=[
                f"{negative_probability:.1%}",
                f"{positive_probability:.1%}"
            ],

            textposition="outside"

        )
    )


    chart.update_layout(

        height=300,

        margin=dict(
            l=20,
            r=50,
            t=20,
            b=20
        ),

        xaxis=dict(
            range=[0, 1],
            tickformat=".0%"
        ),

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="#cbd5e1"
        ),

        showlegend=False

    )


    st.plotly_chart(
        chart,
        use_container_width=True
    )


    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.subheader(
        "🔎 Interpretation"
    )


    if sentiment == "POSITIVE":

        st.info(
            f"The fine-tuned DistilBERT model "
            f"predicts **positive sentiment** with "
            f"**{confidence:.1%} confidence**."
        )

    else:

        st.info(
            f"The fine-tuned DistilBERT model "
            f"predicts **negative sentiment** with "
            f"**{confidence:.1%} confidence**."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()


st.header(
    "📈 Model Intelligence"
)


st.caption(
    "Evaluation metrics from the unseen IMDb test dataset."
)


metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "🧠 Architecture",
        "DistilBERT"
    )


with metric2:

    st.metric(
        "🎬 Dataset",
        "IMDb"
    )


with metric3:

    st.metric(
        "🎯 Accuracy",
        "91.41%"
    )


with metric4:

    st.metric(
        "📊 F1 Score",
        "91.44%"
    )


# ============================================================
# HISTORY
# ============================================================

if st.session_state.history:

    st.divider()


    st.header(
        "🕘 Recent Predictions"
    )


    for index, item in enumerate(
        st.session_state.history
    ):

        if item["sentiment"] == "POSITIVE":

            icon = "😊"

        else:

            icon = "😞"


        preview = item["review"]


        if len(preview) > 140:

            preview = (
                preview[:140] +
                "..."
            )


        st.write(
            f"{icon} **{item['sentiment']}** "
            f"• {item['confidence']:.1%}"
        )


        st.caption(
            preview
        )


        if index < len(
            st.session_state.history
        ) - 1:

            st.divider()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="custom-footer">

        Built with Python • PyTorch • Hugging Face
        • DistilBERT • Streamlit • Three.js

        <br><br>

        Fine-tuned Transformer Sentiment Analysis

    </div>
    """,
    unsafe_allow_html=True
)