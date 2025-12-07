import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import time
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches

# Configure the page with radiology theme
st.set_page_config(
    page_title="Chest X-Ray Disease Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for radiology theme
st.markdown("""
    <style>
    /* Main theme colors - medical/radiology inspired */
    .main {
        background-color: #0a1929;
    }
    
    /* Headers */
    h1 {
        color: #00d4ff !important;
        text-align: center;
        font-family: 'Arial', sans-serif;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    h2, h3 {
        color: #00d4ff !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0d1b2a;
        border-right: 2px solid #00d4ff;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #00d4ff;
        color: #0a1929;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #00a8cc;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.6);
        transform: scale(1.05);
    }
    
    /* Info boxes */
    .stAlert {
        background-color: rgba(0, 212, 255, 0.1);
        border-left: 4px solid #00d4ff;
        border-radius: 5px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 2rem !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #00d4ff;
        border-radius: 10px;
        padding: 20px;
        background-color: rgba(0, 212, 255, 0.05);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: #00d4ff;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(0, 212, 255, 0.1);
        border-radius: 5px;
    }
    
    /* Result box styling */
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 2px solid;
    }
    
    .healthy-box {
        background-color: rgba(0, 255, 136, 0.1);
        border-color: #00ff88;
    }
    
    .disease-box {
        background-color: rgba(255, 68, 68, 0.1);
        border-color: #ff4444;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    """Load TensorFlow model from h5 file"""
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def get_model_input_shape(model):
    """Extract the expected input shape from the model"""
    try:
        input_shape = model.input_shape
        if len(input_shape) >= 4:
            return (input_shape[1], input_shape[2], input_shape[3])
        elif len(input_shape) == 3:
            return input_shape
    except:
        pass
    return None

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess the uploaded image for model prediction"""
    image = image.resize(target_size)
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    
    return img_array

def create_gauge_chart(confidence, prediction):
    """Create a gauge chart using matplotlib"""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0a1929')
    ax.set_facecolor('#0a1929')
    
    # Determine color based on prediction
    if prediction == "Healthy":
        color = '#00ff88'
    else:
        color = '#ff4444'
    
    # Create gauge
    theta = confidence * 180
    
    # Background arc
    arc1 = Wedge((0.5, 0), 0.4, 0, 180, width=0.1, facecolor='#2a3f5f', edgecolor='white', linewidth=2)
    ax.add_patch(arc1)
    
    # Confidence arc
    arc2 = Wedge((0.5, 0), 0.4, 0, theta, width=0.1, facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(arc2)
    
    # Add needle
    angle_rad = np.radians(theta)
    needle_x = 0.5 + 0.35 * np.cos(angle_rad)
    needle_y = 0.35 * np.sin(angle_rad)
    ax.plot([0.5, needle_x], [0, needle_y], color='white', linewidth=3)
    ax.plot(0.5, 0, 'o', color='white', markersize=10)
    
    # Add text
    ax.text(0.5, -0.15, f'{confidence*100:.1f}%', 
            ha='center', va='top', fontsize=28, fontweight='bold', color=color)
    ax.text(0.5, -0.25, prediction, 
            ha='center', va='top', fontsize=20, fontweight='bold', color=color)
    
    # Add scale labels
    ax.text(0.05, 0.05, '0%', ha='left', va='bottom', fontsize=10, color='white')
    ax.text(0.5, 0.45, '50%', ha='center', va='bottom', fontsize=10, color='white')
    ax.text(0.95, 0.05, '100%', ha='right', va='bottom', fontsize=10, color='white')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 0.5)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_probability_bars(healthy_prob, disease_prob):
    """Create probability bars using matplotlib"""
    fig, ax = plt.subplots(figsize=(8, 2), facecolor='#0a1929')
    ax.set_facecolor('#0a1929')
    
    categories = ['Healthy', 'Disease']
    values = [healthy_prob * 100, disease_prob * 100]
    colors = ['#00ff88', '#ff4444']
    
    bars = ax.barh(categories, values, color=colors, height=0.6)
    
    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val/2, i, f'{val:.1f}%', 
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability (%)', fontsize=12, color='white')
    ax.tick_params(colors='white', labelsize=11)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, color='white', linestyle='--')
    
    plt.tight_layout()
    return fig

def get_disease_info(prediction):
    """Get information about the prediction"""
    if prediction == "Disease":
        return {
            "title": "⚠️ Disease Detected",
            "description": "The model has detected potential abnormalities in the chest X-ray that may indicate disease.",
            "recommendations": [
                "Consult with a radiologist for professional evaluation",
                "Additional imaging or tests may be recommended",
                "Do not rely solely on AI diagnosis - seek medical attention",
                "Keep a record of this screening for medical consultation"
            ],
            "css_class": "disease-box"
        }
    else:
        return {
            "title": "✅ Healthy Indication",
            "description": "The model indicates that the chest X-ray appears to be within normal parameters.",
            "recommendations": [
                "Continue regular health check-ups",
                "Maintain a healthy lifestyle",
                "This is a screening tool - not a replacement for medical advice",
                "Consult a doctor if you have any symptoms or concerns"
            ],
            "css_class": "healthy-box"
        }

def main():
    # Header with medical cross icon
    st.markdown("<h1>🫁 Chest X-Ray Disease Classification System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #00d4ff; font-size: 1.2rem;'>Advanced AI-Powered Radiology Analysis</p>", unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        
        # Model selection
        st.markdown("### 🧠 Model Selection")
        
        # Find available models
        search_paths = [
            "./*.h5",
            "./models/*.h5",
            "./saved_models/*.h5",
            "../*.h5",
        ]
        
        available_models = []
        for pattern in search_paths:
            available_models.extend(glob.glob(pattern, recursive=False))
        
        available_models = sorted(list(set(available_models)))
        
        if available_models:
            model_display = {}
            for path in available_models:
                filename = os.path.basename(path)
                model_display[filename] = path
            
            selected_model_name = st.selectbox(
                "Choose Model",
                options=list(model_display.keys()),
                help="Select the AI model for X-ray analysis"
            )
            model_path = model_display[selected_model_name]
            
            # Highlight recommended model
            if "unet" in selected_model_name.lower():
                st.success("🏆 **Recommended**: U-Net model shows superior real-world performance")
            
            st.caption(f"📂 {model_path}")
        else:
            st.warning("⚠️ No models found")
            model_path = None
        
        st.markdown("---")
        
        # About section
        with st.expander("ℹ️ About This System"):
            st.markdown("""
            This AI-powered system analyzes chest X-ray images to detect potential abnormalities.
            
            **Key Features:**
            - Binary classification (Healthy vs Disease)
            - Real-time analysis
            - Confidence scoring
            - Interactive visualizations
            
            **⚕️ Medical Disclaimer:**
            This tool is for screening purposes only and should not replace professional medical diagnosis.
            """)
        
        # Statistics
        with st.expander("📊 Model Performance"):
            st.markdown("""
            **U-Net Model:**
            - Accuracy: 63%
            - ROC-AUC: 0.6692
            - Sensitivity: 57.30%
            - Specificity: 68.40%
            
            **Custom CNN:**
            - Accuracy: 67%
            - ROC-AUC: 0.7183
            - Sensitivity: 58.15%
            - Specificity: 75.12%
            """)
    
    # Load model
    if model_path:
        with st.spinner("🔄 Loading AI model..."):
            model = load_model(model_path)
        
        if model:
            # Get input shape
            detected_shape = get_model_input_shape(model)
            if detected_shape and detected_shape[0] is not None:
                img_height, img_width = detected_shape[0], detected_shape[1]
            else:
                img_height, img_width = 128, 128
            
            # Main content area
            col1, col2 = st.columns([1, 1], gap="large")
            
            with col1:
                st.markdown("### 📤 Upload X-Ray Image")
                uploaded_file = st.file_uploader(
                    "Drop your chest X-ray here or click to browse",
                    type=["jpg", "jpeg", "png", "bmp"],
                    help="Supported formats: JPG, PNG, BMP"
                )
                
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)
                    
                    # Image info
                    st.info(f"📏 Original size: {image.size[0]}×{image.size[1]} pixels")
                    st.info(f"🔄 Processing size: {img_width}×{img_height} pixels")
            
            with col2:
                if uploaded_file is not None:
                    st.markdown("### 🔬 Analysis Results")
                    
                    # Analyze button
                    if st.button("🚀 Analyze X-Ray", use_container_width=True):
                        # Progress animation
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("⚙️ Preprocessing image...")
                        progress_bar.progress(25)
                        time.sleep(0.3)
                        
                        # Preprocess
                        processed_image = preprocess_image(image, target_size=(img_width, img_height))
                        
                        status_text.text("🧠 Running AI analysis...")
                        progress_bar.progress(50)
                        time.sleep(0.3)
                        
                        # Predict
                        prediction = model.predict(processed_image, verbose=0)
                        
                        status_text.text("📊 Generating results...")
                        progress_bar.progress(75)
                        time.sleep(0.3)
                        
                        # Calculate results
                        disease_prob = float(prediction[0][0])
                        healthy_prob = 1 - disease_prob
                        
                        if disease_prob > 0.5:
                            result = "Disease"
                            confidence = disease_prob
                        else:
                            result = "Healthy"
                            confidence = healthy_prob
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")
                        time.sleep(0.5)
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display results
                        st.markdown("---")
                        
                        # Create two columns for metrics
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("Prediction", result, delta=None)
                        with metric_col2:
                            st.metric("Confidence", f"{confidence*100:.1f}%", delta=None)
                        
                        # Gauge chart
                        st.markdown("#### Confidence Score")
                        fig_gauge = create_gauge_chart(confidence, result)
                        st.pyplot(fig_gauge)
                        plt.close()
                        
                        # Probability bars
                        st.markdown("#### Class Probabilities")
                        fig_bars = create_probability_bars(healthy_prob, disease_prob)
                        st.pyplot(fig_bars)
                        plt.close()
                        
                        # Disease information
                        info = get_disease_info(result)
                        
                        st.markdown(f"""
                        <div class='result-box {info['css_class']}'>
                            <h3>{info['title']}</h3>
                            <p>{info['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### 📋 Recommendations:")
                        for rec in info['recommendations']:
                            st.markdown(f"- {rec}")
                        
                        # Raw prediction data
                        with st.expander("🔍 View Raw Prediction Data"):
                            st.json({
                                "model_used": selected_model_name,
                                "prediction": result,
                                "confidence_score": f"{confidence:.4f}",
                                "disease_probability": f"{disease_prob:.4f}",
                                "healthy_probability": f"{healthy_prob:.4f}",
                                "input_shape": f"{img_width}×{img_height}",
                                "threshold": 0.5
                            })
                else:
                    st.info("👈 Upload a chest X-ray image to begin analysis")
                    
                    st.markdown("### 💡 Quick Start Guide")
                    st.markdown("""
                    1. **Upload** a chest X-ray image (JPG, PNG, or BMP)
                    2. **Click** the "Analyze X-Ray" button
                    3. **View** the AI-powered analysis results
                    4. **Review** the confidence score and recommendations
                    """)
        else:
            st.error("❌ Failed to load model. Please check the model path and try again.")
    else:
        st.warning("⚠️ No model selected. Please add model files to the directory.")
        st.info("💡 Place your .h5 model files in the current directory, `models/`, or `saved_models/` folder")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #00d4ff; padding: 20px;'>
            <p>🏥 <b>Medical AI Research Project</b> | Chest X-Ray Disease Classification</p>
            <p style='font-size: 0.9rem; color: #888;'>⚠️ For research and educational purposes only - Not for clinical diagnosis</p>
            <p style='font-size: 0.8rem;'>Developed by Mohamad Konsouh | <a href='https://github.com/MKonsouh' style='color: #00d4ff;'>GitHub</a></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
