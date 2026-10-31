CarMarket AI – Intelligent Used-Car Price Prediction & Recommendations
An AI-powered platform that helps users estimate the resale value of a used car and discover vehicles that better match their preferences. CarMarket AI combines a machine-learning price prediction model, a hybrid recommendation engine, and an interactive chatbot in a Streamlit application.
Project Overview
Buying or selling a pre-owned car can be challenging. Sellers may struggle to set a competitive asking price, while buyers often need to compare many listings against a specific budget and set of preferences.

CarMarket AI brings these tasks together in one application. Users can explore an estimated resale price, receive relevant vehicle recommendations, and interact with a chatbot that interprets common car-search preferences.
Key Capabilities
1. Used-Car Resale Price Estimation
- Predicts an estimated resale price from the vehicle information provided by the user.
- Uses a trained machine-learning regression model based on HistGradientBoosting.
- Loads the saved model at application startup so predictions can be made without retraining it each time.
2. Hybrid Vehicle Recommendations
- Suggests cars based on user preferences.
- Combines preference-based matching with cosine similarity to identify relevant vehicles.
- Uses K-Means clustering as part of the recommendation workflow.
- Loads the prepared recommendation components from a saved artifact bundle.
3. Conversational Car Search
- Provides an interactive chatbot inside the Streamlit application.
- Extracts useful search preferences from natural-language queries, including budget, brand, fuel type, body type, transmission, and city.
- Uses the interpreted preferences to help retrieve relevant car recommendations.
Solution Workflow
1. Prepare the data – Use the cleaned vehicle dataset as the common source for the application and recommendation workflow.
2. Train the price model – Train the resale-price regression model using the machine-learning notebook.
3. Build recommendation components – Prepare the encoders, scaler, similarity data, and clustering components used by the recommendation workflow.
4. Save model artifacts – Store the trained price model and recommendation components so the application can load them.
5. Run the application – Launch the Streamlit interface to make price predictions and explore recommendations through the chatbot.
Machine-Learning Approach
Resale-Price Prediction
The project uses a HistGradientBoosting regression model to estimate used-car resale prices. The trained model is serialized with Joblib and loaded by the Streamlit application.
Recommendation Engine
The recommendation workflow combines multiple techniques rather than relying on a single similarity measure:
Technique	Role in the project
Preference matching	Filters or prioritizes vehicles using the user's stated requirements.
Cosine similarity	Measures similarity between prepared vehicle feature representations.
K-Means clustering	Groups vehicles with similar characteristics to support recommendation.


The recommendation components are packaged in recommendation_artifacts.joblib for use by the application.
Dataset
The application uses a prepared used-car dataset stored as Cleaned_cars.csv.

The dataset provides the vehicle records used by the application and recommendation workflow. Dataset provenance, the original source URL, and the full list of training features should be documented here once they are confirmed and ready to publish.
Model Evaluation
The price-prediction model should be assessed using regression metrics such as:
- Mean Absolute Error (MAE): average absolute difference between predicted and actual prices.
- Root Mean Squared Error (RMSE): gives greater weight to larger prediction errors.
- R² score: indicates how much of the variation in the target values is explained by the model.
Add the measured scores from the final evaluation run before publishing performance claims. No numerical accuracy or error value is stated here because a verified result was not provided for this README.
Technology Stack
Area	Tools and libraries
Programming language	Python
Machine learning	Scikit-learn
Data handling	Pandas, NumPy
Similarity search	Scikit-learn cosine similarity
Model persistence	Joblib
Application interface	Streamlit
Development and experimentation	Jupyter Notebook


Repository Structure
CarMarket-AI/
│
├── car.py
├── machine_learning.ipynb
├── Cleaned_cars.csv
├── car_price_model.joblib
├── recommendation_artifacts.joblib
├── requirements.txt
└── README.md
Important files
- car.py – Streamlit application containing the user interface, price prediction flow, recommendation functionality, and chatbot.
- machine_learning.ipynb – Notebook used for machine-learning experimentation and model training.
- Cleaned_cars.csv – Prepared vehicle dataset used by the application.
- car_price_model.joblib – Saved resale-price prediction model.
- recommendation_artifacts.joblib – Saved components required by the recommendation workflow.
- requirements.txt – Python dependencies required to run the project.
Keep the dataset and model artifacts available at the paths expected by car.py. The application currently loads these files by filename.
Getting Started
Prerequisites
- Python installed on your system.
- pip package manager.
- The project dataset and saved model artifacts listed above.
1. Clone the repository
git clone https://github.com/Vasanthaleela1111-M/CarMarket-AI.git
cd CarMarket-AI
2. Create and activate a virtual environment
Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
** macOS / Linux:**
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
When requirements.txt is available:
pip install -r requirements.txt
Alternatively, install the core packages used by the application:
pip install streamlit pandas numpy scikit-learn joblib
4. Launch the application
streamlit run car.py
Streamlit will print a local URL in the terminal. Open that URL in your browser to use the application.
Using CarMarket AI
1. Open the Streamlit application.
2. Enter the available vehicle details to request a resale-price estimate.
3. Explore recommended vehicles based on the preferences you provide.
4. Use the chatbot to describe a car you're looking for. Include details such as your budget, preferred brand, fuel type, body type, transmission, or city to help narrow the recommendations.
The results are model-generated estimates and recommendations. They should be treated as decision-support information, not as a guaranteed sale price or a substitute for inspecting a vehicle.
Future Improvements
- Expand the dataset and regularly refresh vehicle-market information.
- Add model monitoring and a repeatable model-training pipeline.
- Improve chatbot understanding of varied and conversational search queries.
- Add filters for mileage, manufacturing year, ownership history, and other relevant vehicle attributes where the data supports them.
- Provide clearer explanations for price estimates and recommendation rankings.
- Deploy the application with a managed hosting service and a documented model/data update process.
Author
Vasantha Leela M
Bachelor of Engineering – Computer Science and Engineering
Karpagam Academy of Higher Education
