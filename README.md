## Architecture
<img width="596" height="214" alt="image" src="https://github.com/user-attachments/assets/a02dc317-a8e0-4194-b706-59962cfd8b2e" />

## Data Collection
A unified pipeline for generating social media content using text, image, and audio data. Supports Retrieval-Augmented Generation (RAG), multimodal processing, and real-time evaluation. Designed to streamline the creation of blogs, tweets, captions, and video narratives across platforms.
Curated datasets across modalities:
- Text: TED Talks, arXiv, Twitter, Sentiment140, Wikipedia
- Image: Flickr8k
- Audio: Common Voice
Datasets selected to align with content types (e.g., blogs, tweets, captions, podcasts, short videos) and unified into `.jsonl` format to support RAG-enabled retrieval.

## Preprocessing & Cleaning
- Text: Noise removal, normalization, metadata extraction
- Image: Caption cleaning, lexicon creation
- Audio: Metadata filtering, irrelevant attribute removal
- All data categorized and cleaned for downstream modeling

## Transformation & Embedding
- Text chunked and embedded using Sentence Transformers
- FAISS vector store used for fast retrieval
- Dataset split: 70% training, 15% validation, 15% testing
- Total corpus size: 2.75M+ RAG-ready records

## Machine Learning Modeling

### Modal-Specific Models
- Text Generation: DeepSeek, Gemini, LLaMA, Mistral
- Image Generation: Stable Diffusion
- Audio Generation: Deepgram (TTS)
- Image Captioning: BLIP, GIT, LLaVA
- Speech-to-Text: Whisper, Deepgram
- PDF Parsing: PyMuPDF, pypdf-llama-parser

### Enhancements
- Fine-tuned models for domain-specific accuracy
- Modular dispatch across modalities
- RAG-enabled inference via FAISS + Vertex AI
- Hot-swappable LLM wrapper for BASE and RAG variants

### Performance Comparison
- BASE models: Better grammar and latency
- RAG models: More creative/contextual but slightly slower

## Evaluation Metrics : LLM-as-a-Judge
- Correctness: Evaluates grammar, syntax, and clarity of generated text; ensures outputs are fluent and structurally sound.

<img width="374" height="181" alt="image" src="https://github.com/user-attachments/assets/794668ac-97a1-44fb-9f89-6d59939025a5" />

- Faithfulness: Measures how accurately the output reflects the original input or retrieved context, without hallucinations.

<img width="312" height="150" alt="image" src="https://github.com/user-attachments/assets/4c43354c-342f-4bd1-8a09-f935b0946b6e" />

- Relevance: Assesses how well the output aligns with the user’s prompt or intended topic.

<img width="325" height="152" alt="image" src="https://github.com/user-attachments/assets/5ca5f0e3-0d00-419a-885f-ee7c5826a620" />

- Helpfulness: Gauges whether the output is informative, actionable, or useful in accomplishing the user's goal.

<img width="316" height="143" alt="image" src="https://github.com/user-attachments/assets/4009a3df-692f-4773-9f46-bee71c7cbcb9" />

- Truthfulness: Checks for factual accuracy, ensuring generated content aligns with known or retrieved knowledge.

<img width="314" height="141" alt="image" src="https://github.com/user-attachments/assets/4247d4ae-9745-4426-8e44-be6a795c6c39" />

- Virality: Scores the potential of content to be engaging, creative, and shareable—ideal for social media use cases.

<img width="317" height="170" alt="image" src="https://github.com/user-attachments/assets/a1076599-ecdf-4a7a-9c26-1e25d066883c" />

### Observations

- Correctness & Faithfulness: BASE models scored 5.0; RAG versions dropped due to loosely aligned context.
- Relevance: Minor decline in RAG; overall alignment with prompts remained strong.
- Truthfulness: Slight drop in RAG (4.6–4.9), but still reliable for fact-based content.
- Virality: RAG improved creative engagement, especially for social media tasks.
- Latency: RAG added latency (DeepSeek: ~38ms)

## Latency Distribution Across BASE and RAG LLMs

<img width="548" height="134" alt="image" src="https://github.com/user-attachments/assets/a21c2950-320a-43dc-86e8-63af84956859" />

- Box plot visualizes latency variation across different LLMs (Base vs. RAG variants), highlighting performance trade-offs.
- DeepSeek and DeepSeek-RAG showed the highest latency, with wider variability due to longer context and retrieval steps.
- Gemini, Mistral, and LLaMA (including RAG variants) maintained low and consistent latency, making them ideal for real-time generation use cases.

## Web Portal System

- System demo: [YouTube](https://www.youtube.com/watch?v=U8Tjm2GVrkM)

<img width="305" height="205" alt="image" src="https://github.com/user-attachments/assets/15df3124-99c9-4ef2-b8b3-1b466f589815" />

- Designed to streamline social media content creation, the system generates platform-optimized blogs, tweets, posts, and videos tailored for real-time engagement.
- Modular & Scalable Architecture : Built using Streamlit, allowing flexible development and easy extension across new modalities.
- Users can select from LLaMA, Gemini, Mistral, or DeepSeek to compare model behavior in real-time.
- Centralized state management to persist inputs across sessions and modules.
- Asynchronous processing for faster response times and handling concurrent user interactions.
- Easily deployable on cloud (e.g., Streamlit Sharing, GCP, or Vertex AI) for production use.

## Testing

- The MCG system was tested across all modules Blog, Twitter/X, Instagram, Podcast, Short Video using a shared LLM backend with configurable model selection. Each output was evaluated on multiple quality metrics and latency to ensure robustness and efficiency.
- Models Evaluated: DeepSeek, Gemini, LLaMA, Mistral (BASE and RAG variants)
- Metrics Tracked: Relevance, Correctness, Faithfulness, Virality, Latency

## Visualization 

- Top Chart: Relevance Score all models maintain high mean relevance (>4.5), with minor drop in RAG variants.
- Bottom Chart: Latency BASE models are generally faster; DeepSeek RAG has the highest variance (~35–38ms), while others remain under 10ms.

<img width="163" height="225" alt="image" src="https://github.com/user-attachments/assets/e3ac012a-cf5a-4851-b7e5-f0f3df082c81" />



