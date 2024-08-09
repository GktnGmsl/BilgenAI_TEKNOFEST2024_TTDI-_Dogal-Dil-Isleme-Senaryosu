#main.py
import uvicorn
import logging
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp import sentiment_analysis, ner
from train import load_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data and fit vectorizer
df = load_data()
vectorizer = TfidfVectorizer(max_features=10000)
vectorizer.fit(df['description'])

# Pydantic model for input validation
class Item(BaseModel):
    text: str = Field(..., example=df['description'].iloc[0])

# FastAPI app
app = FastAPI()

# Timeout duration in seconds
TIMEOUT_DURATION = 5

@app.post("/predict", response_model=dict)
async def predict(item: Item):
    async def run_prediction():
        text = item.text

        # NER processing
        try:
            entity_list, lstm_preds = ner(text, vectorizer)
        except Exception as ner_error:
            logger.error(f"NER processing error: {ner_error}")
            entity_list, lstm_preds = [], []

        sentiment_list = []
        for entity in entity_list:
            try:
                sentiment = sentiment_analysis(text)
                sentiment_label = 'olumsuz' if sentiment == 0 else 'nötr' if sentiment == 1 else 'olumlu'
                sentiment_list.append({'entity': entity, 'sentiment': sentiment_label})
            except Exception as sentiment_error:
                logger.error(f"Sentiment analysis error: {sentiment_error}")
                sentiment_list.append({'entity': entity, 'sentiment': 'unknown'})

        return {'entity_list': entity_list, 'results': sentiment_list}

    try:
        result = await asyncio.wait_for(run_prediction(), timeout=TIMEOUT_DURATION)
    except asyncio.TimeoutError:
        logger.error("Prediction timed out")
        raise HTTPException(status_code=408, detail="Prediction request timed out")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        result = {'entity_list': [], 'results': []}

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9992)
