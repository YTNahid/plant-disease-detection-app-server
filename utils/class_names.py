from typing import Dict, List, TypedDict

class DiseaseInfo(TypedDict):
    description: str
    treatment: str
    prevention: str

# ── The 38 exact class labels from the New Plant Diseases Dataset ─────────────
CLASS_NAMES: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# ── Description, Treatment, and Prevention Metadata ───────────────────────────
DISEASE_INFO: Dict[str, DiseaseInfo] = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple scab",
        "description": "A fungal disease that causes olive-green to black lesions on leaves and fruit.",
        "treatment": "Apply fungicides containing captan, myclobutanil, or sulfur.",
        "prevention": "Rake and destroy fallen leaves; prune trees for better air circulation."
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black rot",
        "description": "A disease causing frog-eye leaf spots, fruit rot, and branch cankers.",
        "treatment": "Prune out infected branches and apply appropriate copper-based fungicides.",
        "prevention": "Remove dead wood and mummified fruits; ensure proper tree spacing."
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar apple rust",
        "description": "A fungal disease requiring a juniper host, causing yellow-orange spots on apple leaves.",
        "treatment": "Apply preventative fungicides (like myclobutanil) early in the season.",
        "prevention": "Remove nearby juniper/cedar hosts or plant rust-resistant apple varieties."
    },
    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "description": "The apple plant is healthy with no visible signs of disease.",
        "treatment": "None required.",
        "prevention": "Maintain standard watering, pruning, and fertilizing schedules."
    },

    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "description": "The blueberry plant is healthy with no visible signs of disease.",
        "treatment": "None required.",
        "prevention": "Maintain acidic soil (pH 4.5-5.5) and ensure adequate drainage."
    },

    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery mildew",
        "description": "Fungal infection creating a white, powdery fungal growth on leaves and stems.",
        "treatment": "Apply sulfur or potassium bicarbonate-based fungicides.",
        "prevention": "Ensure good airflow through pruning and avoid overhead watering."
    },
    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "description": "The cherry plant is healthy.",
        "treatment": "None required.",
        "prevention": "Monitor for pests and maintain proper pruning."
    },

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn (maize)",
        "disease": "Cercospora leaf spot / Gray leaf spot",
        "description": "Fungal disease causing rectangular, pale brown-to-gray spots on corn leaves.",
        "treatment": "Foliar fungicides applied at the tasseling stage can reduce severity.",
        "prevention": "Rotate crops and use resistant corn hybrids; manage crop residue."
    },
    "Corn_(maize)___Common_rust_": {
        "plant": "Corn (maize)",
        "disease": "Common rust",
        "description": "Causes rust-colored pustules on both upper and lower leaf surfaces.",
        "treatment": "Fungicide applications are effective if applied early upon detection.",
        "prevention": "Plant rust-resistant varieties and avoid planting late in the season."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn (maize)",
        "disease": "Northern Leaf Blight",
        "description": "Fungal disease characterized by large, cigar-shaped grayish-green lesions.",
        "treatment": "Apply targeted fungicides before lesions spread significantly.",
        "prevention": "Use resistant hybrids and practice one-to-two-year crop rotation."
    },
    "Corn_(maize)___healthy": {
        "plant": "Corn (maize)",
        "disease": "Healthy",
        "description": "The corn plant is healthy.",
        "treatment": "None required.",
        "prevention": "Provide adequate nitrogen and maintain soil health."
    },

    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black rot",
        "description": "Fungal disease resulting in brown lesions on leaves and shriveled, hard black berries.",
        "treatment": "Apply mancozeb or myclobutanil fungicides from early bloom until berries size up.",
        "prevention": "Remove mummified berries from vines and clear debris from the ground."
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca (Black Measles)",
        "description": "A complex fungal disease causing \"tiger-stripe\" leaf patterns and dark spots on fruit.",
        "treatment": "No chemical cure exists for infected vines. Severely infected vines must be removed.",
        "prevention": "Protect pruning wounds with sealants and avoid pruning during wet weather."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf blight (Isariopsis Leaf Spot)",
        "description": "Causes dark red to brown spots on leaves, which can lead to premature defoliation.",
        "treatment": "Copper-based fungicides can help manage the spread.",
        "prevention": "Improve air circulation and destroy fallen diseased leaves."
    },
    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "description": "The grape vine is healthy.",
        "treatment": "None required.",
        "prevention": "Regular pruning and proper trellising."
    },

    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing (Citrus greening)",
        "description": "A fatal bacterial disease spread by psyllids, causing yellow shoots and bitter, misshapen fruit.",
        "treatment": "No cure is available. Infected trees must be removed and destroyed.",
        "prevention": "Control the Asian citrus psyllid vector using appropriate insecticides."
    },

    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial spot",
        "description": "Bacterial infection causing angular dark spots on leaves and pitted blemishes on fruit.",
        "treatment": "Copper sprays or antibiotic sprays (like oxytetracycline) can suppress the disease.",
        "prevention": "Plant resistant cultivars and avoid excessive nitrogen fertilization."
    },
    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "description": "The peach tree is healthy.",
        "treatment": "None required.",
        "prevention": "Maintain standard orchard hygiene."
    },

    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "disease": "Bacterial spot",
        "description": "Causes small, water-soaked spots on leaves that turn brown, leading to leaf drop.",
        "treatment": "Copper-based bactericides can help slow the spread if applied early.",
        "prevention": "Use disease-free seeds and avoid overhead irrigation."
    },
    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper",
        "disease": "Healthy",
        "description": "The bell pepper plant is healthy.",
        "treatment": "None required.",
        "prevention": "Provide well-draining soil and consistent watering."
    },

    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early blight",
        "description": "Fungal disease causing dark spots with concentric rings (target spots) on older leaves.",
        "treatment": "Apply fungicides like chlorothalonil or mancozeb as soon as symptoms appear.",
        "prevention": "Practice crop rotation and avoid wetting the foliage when watering."
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late blight",
        "description": "Severe disease causing water-soaked spots on leaves, rapid browning, and tuber rot.",
        "treatment": "Requires aggressive fungicide programs; infected plants should be destroyed quickly.",
        "prevention": "Plant certified disease-free seed potatoes and destroy volunteer potato plants."
    },
    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "description": "The potato plant is healthy.",
        "treatment": "None required.",
        "prevention": "Hill potatoes properly and ensure loose soil."
    },

    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "description": "The raspberry plant is healthy.",
        "treatment": "None required.",
        "prevention": "Prune old canes to ground level after harvesting to ensure good airflow."
    },

    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "description": "The soybean plant is healthy.",
        "treatment": "None required.",
        "prevention": "Use proper row spacing and monitor for aphids."
    },

    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery mildew",
        "description": "Fungus causing white, powdery spots on both surfaces of squash leaves.",
        "treatment": "Fungicides, neem oil, or potassium bicarbonate solutions.",
        "prevention": "Plant resistant varieties and allow ample spacing for air circulation."
    },

    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf scorch",
        "description": "Fungal disease causing irregular, dark purple to brown spots on leaves.",
        "treatment": "Apply protective fungicides during wet weather periods.",
        "prevention": "Remove and destroy infected foliage; replace plants every 3-4 years."
    },
    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "description": "The strawberry plant is healthy.",
        "treatment": "None required.",
        "prevention": "Keep fruit off the soil using mulch or straw."
    },

    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial spot",
        "description": "Bacterial disease causing small, dark, greasy spots on leaves and scabs on fruit.",
        "treatment": "Copper fungicides combined with mancozeb can reduce spread.",
        "prevention": "Water at the base to keep leaves dry; practice 3-year crop rotation."
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early blight",
        "description": "Causes target-like lesions on lower leaves, leading to yellowing and leaf drop.",
        "treatment": "Use fungicides containing chlorothalonil or copper soap.",
        "prevention": "Mulch around the base of the plant to prevent soil splashing onto leaves."
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late blight",
        "description": "Causes large, dark, water-soaked patches on leaves and brown lesions on stems and fruit.",
        "treatment": "Apply strong protective fungicides. Pull and destroy heavily infected plants.",
        "prevention": "Ensure plenty of air circulation and avoid overhead watering."
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf mold",
        "description": "Fungal disease common in greenhouses, causing pale spots on upper leaves and mold underneath.",
        "treatment": "Fungicides can be used, though improved ventilation is most effective.",
        "prevention": "Reduce humidity, prune lower leaves, and improve air circulation."
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria leaf spot",
        "description": "Causes numerous small, circular spots with dark borders and gray centers on lower leaves.",
        "treatment": "Apply fungicides; remove heavily infected leaves immediately.",
        "prevention": "Mulch soil and stake plants to keep foliage off the ground."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider mites (Two-spotted spider mite)",
        "description": "Tiny pests causing stippled, yellowing leaves and fine webbing on the plant.",
        "treatment": "Use insecticidal soap, neem oil, or introduce predatory mites.",
        "prevention": "Keep plants well-watered, as mites thrive in hot, dry, and dusty conditions."
    },
    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target spot",
        "description": "Fungal disease causing target-like lesions with pale centers on leaves and fruit.",
        "treatment": "Apply appropriate broad-spectrum fungicides.",
        "prevention": "Ensure good airflow and remove infected plant debris at the end of the season."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "description": "Viral disease transmitted by whiteflies; causes stunted growth and cupped, yellowing leaves.",
        "treatment": "No cure for the virus. Infected plants must be removed and destroyed.",
        "prevention": "Control whitefly populations using reflective mulches and insecticidal soaps."
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato mosaic virus",
        "description": "Viral infection causing mottled light and dark green leaves and stunted growth.",
        "treatment": "No chemical cure. Remove and destroy infected plants immediately.",
        "prevention": "Wash hands and tools thoroughly, as the virus spreads mechanically (e.g., via touch)."
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "description": "The tomato plant is healthy.",
        "treatment": "None required.",
        "prevention": "Maintain consistent watering, fertilization, and staking."
    }
}


def get_class_name(index: int) -> str:
    if index < 0 or index >= len(CLASS_NAMES):
        return f"unknown_class_{index}"
    return CLASS_NAMES[index]


def get_all_classes() -> Dict[int, str]:
    return {i: name for i, name in enumerate(CLASS_NAMES)}


def get_disease_info(index_or_name) -> DiseaseInfo:
    if isinstance(index_or_name, int):
        name = get_class_name(index_or_name)
    else:
        name = index_or_name
        
    return DISEASE_INFO.get(name, {
        "plant": "Information not found.",
        "disease": "Information not found.",
        "description": "Information not found.",
        "treatment": "Information not found.",
        "prevention": "Information not found."
    })