from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

load_dotenv()

# Get environment variables
AURA_INSTANCENAME = os.getenv("AURA_INSTANCENAME")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

api_key = os.getenv("api_key")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

chat = ChatOpenAI(api_key=api_key)

# Initialize Neo4jGraph
kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

# Create vector index if it doesn't exist
kg.query(
    """
    CREATE VECTOR INDEX health_providers_embeddings IF NOT EXISTS
    FOR (hp:HealthcareProvider) ON (hp.comprehensiveEmbedding)
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
      }
    }
    """
)

# Test to see if the index was created
res = kg.query(
    """
    SHOW VECTOR INDEXES
    """
)
print(res)

# Encode and set vector properties
kg.query(
    """
    MATCH (hp:HealthcareProvider)-[:TREATS]->(p:Patient)
    WHERE hp.bio IS NOT NULL
    WITH hp, genai.vector.encode(
        hp.bio,
        "OpenAI",
        {
          token: $openAiApiKey,
          endpoint: $openAiEndpoint
        }) AS vector
    WITH hp, vector
    WHERE vector IS NOT NULL
    CALL db.create.setNodeVectorProperty(hp, "comprehensiveEmbedding", vector)
    """,
    params={
        "openAiApiKey": api_key,
        "openAiEndpoint": OPENAI_ENDPOINT,
    },
)

# Query the graph for healthcare providers
result = kg.query(
    """
    MATCH (hp:HealthcareProvider)
    WHERE hp.bio IS NOT NULL
    RETURN hp.bio, hp.name, hp.comprehensiveEmbedding
    LIMIT 5
    """
)

# Loop through the results
for record in result:
    print(f"bio: {record['hp.bio']}, name: {record['hp.name']}")

# Querying the graph for a healthcare provider based on a question
question = "give me a list of healthcare providers in the area of dermatology"

# Execute the query
result = kg.query(
    """
    WITH genai.vector.encode(
        $question,
        "OpenAI",
        {
          token: $openAiApiKey,
          endpoint: $openAiEndpoint
        }) AS question_embedding
    CALL db.index.vector.queryNodes(
        'health_providers_embeddings',
        $top_k,
        question_embedding
        ) YIELD node AS healthcare_provider, score
    RETURN healthcare_provider.name, healthcare_provider.bio, score
    """,
    params={
        "openAiApiKey": api_key,
        "openAiEndpoint": OPENAI_ENDPOINT,
        "question": question,
        "top_k": 3,
    },
)

# Print the result
for record in result:
    print(f"Name: {record['healthcare_provider.name']}")
    print(f"Bio: {record['healthcare_provider.bio']}")
    print(f"Score: {record['score']}")
    print("---")



# import openai
# import os
# from dotenv import load_dotenv
# from neo4j import GraphDatabase

# # Load environment variables from .env file
# load_dotenv()

# # Set the OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Set up Neo4j connection
# neo4j_uri = os.getenv("NEO4J_URI")
# neo4j_user = os.getenv("NEO4J_USERNAME")
# neo4j_password = os.getenv("NEO4J_PASSWORD")
# driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# def test_openai_api():
#     try:
#         # Use openai.ChatCompletion for the latest API version
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",  # Ensure this model is supported and correctly specified
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": "Translate the following English text to French: 'Hello, world!'"}
#             ],
#             max_tokens=60
#         )
#         print("Response:", response.choices[0].message['content'].strip())
#     except Exception as e:
#         print(f"An error occurred with OpenAI API: {e}")

# def query_neo4j(question):
#     try:
#         with driver.session() as session:
#             result = session.run(
#                 """
#                 WITH genai.vector.encode(
#                     $question,
#                     "OpenAI",
#                     {
#                       token: $openAiApiKey,
#                       endpoint: $openAiEndpoint
#                     }) AS question_embedding
#                 CALL db.index.vector.queryNodes(
#                     'health_providers_embeddings',
#                     $top_k,
#                     question_embedding
#                     ) YIELD node AS healthcare_provider, score
#                 RETURN healthcare_provider.name, healthcare_provider.bio, score
#                 """,
#                 parameters={
#                     "openAiApiKey": openai.api_key,
#                     "openAiEndpoint": os.getenv("OPENAI_ENDPOINT"),
#                     "question": question,
#                     "top_k": 3,
#                 }
#             )
#             for record in result:
#                 print(f"Name: {record['healthcare_provider.name']}")
#                 print(f"Bio: {record['healthcare_provider.bio']}")
#                 print(f"Score: {record['score']}")
#                 print("---")
#     except Exception as e:
#         print(f"An error occurred with Neo4j query: {e}")

# # Test the OpenAI API
# test_openai_api()

# # Test Neo4j query
# query_neo4j("Give me a list of healthcare providers in the area of dermatology")









