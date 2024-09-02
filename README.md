# Conversational_AI_model

The model with functionality as: Personalized Inetarction Engine, Legal Insight Enhanceemnt model, 
Advanced Image Classification feature.

## Personalized Inetarction Engine:
RAG based Interaction Engine.
Workflow:
Text extraction -> Converted into Vectors using Word2Vec model -> Converted into chunks of information -> stored in vector database -> According to user query the most relevant documents are retrieved -> passed into llm to make the emaningful out of it -> Ouptut given.

Files associated are- rag_imp1

## Legal Insight Enhancement Model:
Fine tuned llm model
Workflow: 
model is loaded after quantization for memory effiency -> Tokenizer is loaded -> with LoRA approach the weigth are freezed -> new weights are intodueced -> Text from PDf is extracted -> Text converted into suitable format -> Text is Tokenized ->  Trained on dataset -> Prediction is Made.  

Files associated are- Fine_Tune_llama2/File_tune_corp.LW.ipynb, corporate_laws.pdf, requiremets.txt (containing all the libraries, modules required)

## Advanced Image Classification feature
Fine tuned CNN Model
Workflow: 
pre-traind CNN model is loaded -> weights are freezed -> new weights are introduced -> Image is Augmented -> Trained -> made Predictions

Files associated are- fine_tune_CNN, 
