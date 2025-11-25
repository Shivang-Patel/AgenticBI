import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
VECTOR_DB_PATH = "./chroma_db_data"
LLM_MODEL = "llama3"  # The model you downloaded via Ollama

class NL2SQLAgent:
    def __init__(self):
        # 1. Initialize the Librarian (Vector DB)
        # We use the same 'all-MiniLM-L6-v2' to ensure we speak the same language as Agent 1
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            persist_directory=VECTOR_DB_PATH, 
            embedding_function=self.embedding_function
        )
        
        # 2. Initialize the Brain (Ollama Local)
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0)

    def get_relevant_schema(self, question: str, k: int = 3) -> str:
        """
        Retrieves the top K most relevant table definitions (DDL) from the Vector DB.
        """
        print(f"🔍 Looking up schema for: '{question}'...")
        results = self.vector_store.similarity_search(question, k=k)
        
        # Combine the DDLs into one big string context
        context = "\n\n".join([doc.page_content for doc in results])
        
        # Log which tables were picked (for debugging)
        table_names = [doc.metadata.get('table_name') for doc in results]
        print(f"   > Found relevant tables: {table_names}")
        
        return context

    def generate_sql(self, question: str) -> str:
        """
        The Core Logic: RAG + LLM Generation
        """
        # Step A: Get Context (RAG)
        schema_context = self.get_relevant_schema(question)
        
        # Step B: Construct the Prompt
        # We give the LLM the role of a PostgreSQL Expert
        template = """
        You are an expert PostgreSQL Data Analyst. 
        Your job is to generate a VALID SQL query to answer the user's question.
        
        ### SCHEMA CONTEXT (Use ONLY these tables):
        {schema}
        
        ### USER QUESTION:
        {question}
        
        ### INSTRUCTIONS:
        1. Return ONLY the SQL query. Do not add markdown markers (like ```sql).
        2. Use table aliases (e.g. 's' for SalesOrderHeader) to make it readable.
        3. If you join tables, ensure the Foreign Keys match.
        4. Do NOT make up columns. Use the schema provided.
        
        ### SQL QUERY:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Step C: Build the Chain
        chain = prompt | self.llm | StrOutputParser()
        
        # Step D: Execute
        print("🤖 Generating SQL...")
        response = chain.invoke({
            "schema": schema_context, 
            "question": question
        })
        
        # Basic cleanup (sometimes models chat a bit)
        cleaned_sql = response.replace("```sql", "").replace("```", "").strip()
        return cleaned_sql

# ---------------------------------------------------------
# TEST EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    agent = NL2SQLAgent()
    
    # Test 1: Simple
    q1 = "Show me the top 5 employees by vacation hours."
    sql1 = agent.generate_sql(q1)
    print(f"\n📝 Generated SQL:\n{sql1}\n")
    
    print("-" * 50)
    
    # Test 2: Complex (Requires JOIN)
    q2 = "List all product names that have inventory quantity greater than 50."
    sql2 = agent.generate_sql(q2)
    print(f"\n📝 Generated SQL:\n{sql2}\n")






    # template = """
    #     You are an expert PostgreSQL Data Analyst. 
    #     Your job is to generate a VALID SQL query to answer the user's question.
        
    #     ### SCHEMA CONTEXT (Use ONLY these tables):
    #     {schema}
        
    #     ### USER QUESTION:
    #     {question}
        
    #     ### INSTRUCTIONS:
    #     1. Return ONLY the SQL query. Do not add markdown markers.
    #     2. Use table aliases (e.g. 'p' for "Product").
    #     3. CRITICAL: The database is Case Sensitive.
    #        - You MUST use Double Quotes for ALL Table Names and Column Names.
    #        - Correct: SELECT "Name" FROM "Production"."Product"
    #        - Incorrect: SELECT Name FROM Production.Product
        
    #     ### SQL QUERY:
    #     """