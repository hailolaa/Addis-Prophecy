# AddisProphecy 🏠🇪🇹

AddisProphecy is a professional-grade real estate valuation tool designed specifically for the Addis Ababa market. It uses machine learning to provide accurate property price estimates based on local factors.

## ✨ Key Features
- **Addis Ababa Scoped**: Covers major neighborhoods including Bole, Kazanchis, CMC, Ayat, and more.
- **Realistic Pricing Hierarchy**: Accurately distinguishes between **Condominiums**, **Apartments**, and **Villas**.
- **Detailed Features**: Factors in area size, property age, room counts, and distance to the city center.
- **Premium UI**: Modern dark-themed dashboard with glassmorphism and smooth animations.
- **Optimized for Vercel**: Ultra-lightweight JSON-based model (under 50MB) with minimal dependencies.

## 📁 Project Structure
- `/api`: FastAPI backend with JSON-based ML model.
- `/frontend`: Responsive dashboard UI.
- `vercel.json`: Deployment configuration.
- `requirements.txt`: Minimal Python dependencies (FastAPI, Uvicorn, Pydantic only).
- `.vercelignore`: Excludes training scripts and data from deployment.

## 🚀 Getting Started

### Local Development
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Generate the model: `python train_addis.py` (requires pandas, scikit-learn for training only).
4. Run the backend: `python api/index.py`.
5. Open `frontend/index.html` in your browser.

### Deployment to Vercel
1. Create a new repository on GitHub and push your code.
2. Link the repository in the [Vercel Dashboard](https://vercel.com).
3. Vercel will automatically detect the settings and deploy your site.

**Note**: The production deployment uses a lightweight JSON model and doesn't require scikit-learn or pandas, keeping the bundle size minimal.

## 🛠️ Built With
- **FastAPI**: Modern, high-performance web framework for the API.
- **Scikit-Learn**: Machine learning library (training only, not in production).
- **Vanilla CSS & HTML**: Premium responsive UI with custom aesthetics.

## 📊 Model Architecture
The project uses a **Linear Regression** model trained on 1,500 synthetic but realistic Addis Ababa property records. For production deployment, the model weights are exported to JSON format, allowing the backend to perform predictions without heavy ML dependencies.
