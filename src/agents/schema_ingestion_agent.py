import psycopg2
import os
import shutil
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DB_CONFIG = {
    "dbname": "postgres", 
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": "5432"
}

VECTOR_DB_PATH = "./chroma_db_data"

class SchemaIngestionAgent:
    def __init__(self, db_config):
        self.db_config = db_config

    def extract_ddl(self):
        """
        Connects to DB and generates 'CREATE TABLE' statements.
        Returns a list of LangChain Documents.
        """
        print(f"🔌 Connecting to database '{self.db_config['dbname']}'...")
        docs = []
        conn = None
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # 1. Get relevant tables (Skipping system stuff)
            cur.execute("""
                SELECT table_schema, table_name 
                FROM information_schema.tables 
                WHERE table_schema IN ('Person', 'Sales', 'Production', 'Purchasing', 'HumanResources') 
                AND table_type = 'BASE TABLE';
            """)
            tables = cur.fetchall()
            print(f"🔍 Found {len(tables)} tables. Generating blueprints...")

            for schema, table in tables:
                full_table_name = f"{schema}.{table}"
                
                # 2. Get columns
                cur.execute(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_schema = '{schema}' AND table_name = '{table}';
                """)
                columns = cur.fetchall()
                
                # 3. Construct DDL (The 'Context' for the AI)
                ddl_lines = [f"CREATE TABLE {full_table_name} ("]
                for col in columns:
                    col_name, dtype, nullable = col
                    null_str = "NULL" if nullable == "YES" else "NOT NULL"
                    ddl_lines.append(f"  {col_name} {dtype} {null_str},")
                ddl_lines.append(");")
                
                ddl_content = "\n".join(ddl_lines)

                # 4. Create Document object
                doc = Document(
                    page_content=ddl_content,
                    metadata={
                        "table_name": full_table_name,
                        "schema": schema
                    }
                )
                docs.append(doc)

            return docs

        except Exception as e:
            print(f"❌ DB Error: {e}")
            return []
        finally:
            if conn: conn.close()

    def build_index(self, documents):
        if not documents:
            print("⚠️ No data to index.")
            return

        print(f"🧠 Vectorizing {len(documents)} tables (using free local model)...")
        
        # 1. Clear old index if it exists (so we don't get duplicates)
        if os.path.exists(VECTOR_DB_PATH):
            shutil.rmtree(VECTOR_DB_PATH)

        # 2. Initialize Free Local Embedding Model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 3. Create Vector DB
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        
        print(f"✅ Success! Schema Index saved to '{VECTOR_DB_PATH}'")

if __name__ == "__main__":
    agent = SchemaIngestionAgent(DB_CONFIG)
    schema_docs = agent.extract_ddl()
    agent.build_index(schema_docs)