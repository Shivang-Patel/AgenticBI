import psycopg2
import pandas as pd
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

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

LLM_MODEL = "llama3"

# ---------------------------------------------------------
# 1. DEFINE THE STATE
#    (The memory passed between the Execute Node and the Fix Node)
# ---------------------------------------------------------
class ExecutionState(TypedDict):
    sql_query: str          # The query we are trying to run
    result_data: Optional[str] # Markdown string of results
    error_message: Optional[str] # DB Error if failed
    retry_count: int        # How many times have we tried?

# ---------------------------------------------------------
# 2. DEFINE THE NODES
# ---------------------------------------------------------

def execute_sql_node(state: ExecutionState):
    """
    Attempts to run the SQL in Postgres.
    """
    query = state['sql_query']
    current_retries = state.get('retry_count', 0)
    
    print(f"\n⚡ Executing SQL (Attempt {current_retries + 1}):")
    print(f"   {query[:60]}...") # Print preview

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        # Using pandas to easily get column names and formatting
        df = pd.read_sql(query, conn)
        
        # If successful, clear errors and save data
        print("   ✅ Success!")
        return {
            "result_data": df.to_markdown(),
            "error_message": None,
            "retry_count": current_retries
        }

    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ DB Error: {error_msg}")
        return {
            "error_message": error_msg,
            "result_data": None,
            "retry_count": current_retries
        }
    finally:
        if conn: conn.close()

def fix_query_node(state: ExecutionState):
    """
    Uses Llama 3 to fix the query based on the Postgres error message.
    """
    broken_query = state['sql_query']
    error_msg = state['error_message']
    retries = state['retry_count']
    
    print("   🔧 Calling LLM to fix query...")

    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    
    template = """
    You are a PostgreSQL Expert. 
    A previous SQL query failed to execute. Fix it based on the error message.

    ### BROKEN QUERY:
    {broken_query}

    ### POSTGRES ERROR:
    {error_msg}

    ### INSTRUCTIONS:
    1. Output ONLY the fixed SQL query.
    2. Do NOT output markdown markers.
    3. Fix syntax errors.
    4. CRITICAL: This database is Case-Sensitive.
       - You MUST wrap ALL Table and Schema names in Double Quotes.
       - Example: Use "Production"."Product" instead of Production.Product.

    ### FIXED SQL:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    fixed_sql = chain.invoke({
        "broken_query": broken_query,
        "error_msg": error_msg
    })
    
    # --- IMPROVED CLEANER ---
    # 1. Remove Markdown
    clean_sql = fixed_sql.replace("```sql", "").replace("```", "").strip()
    
    # 2. Regex to find the actual SQL statement (starts with SELECT, ends with ;)
    # This ignores intro text like "Here is the query:"
    match = re.search(r"(SELECT.*?;)", clean_sql, re.DOTALL | re.IGNORECASE)
    if match:
        clean_sql = match.group(1)
        
    print(f"   💡 LLM suggested fix: {clean_sql[:50]}...")
    
    return {
        "sql_query": clean_sql,
        "retry_count": retries + 1
    }
# ---------------------------------------------------------
# 3. DEFINE THE ROUTER LOGIC
# ---------------------------------------------------------
def should_continue(state: ExecutionState):
    """
    Decides if we are done or need to loop back to fix.
    """
    if state['result_data'] is not None:
        return "success" # We got data!
    
    if state['retry_count'] >= 3:
        print("   🛑 Max retries reached. Giving up.")
        return "give_up"
    
    return "retry" # Loop back to fix_node

# ---------------------------------------------------------
# 4. BUILD THE GRAPH
# ---------------------------------------------------------
def build_execution_graph():
    workflow = StateGraph(ExecutionState)

    # Add Nodes
    workflow.add_node("execute", execute_sql_node)
    workflow.add_node("fix", fix_query_node)

    # Entry Point
    workflow.set_entry_point("execute")

    # Conditional Edges
    workflow.add_conditional_edges(
        "execute",
        should_continue,
        {
            "success": END,
            "give_up": END,
            "retry": "fix"
        }
    )

    # Connect Fix back to Execute (The Loop)
    workflow.add_edge("fix", "execute")

    return workflow.compile()

# ---------------------------------------------------------
# MAIN EXECUTION API
# ---------------------------------------------------------
class SQLExecutor:
    def __init__(self):
        self.app = build_execution_graph()

    def run(self, sql_query: str):
        """
        Takes an SQL query, runs it, fixes it if needed.
        Returns the final DataFrame (markdown) or Error.
        """
        initial_state = {
            "sql_query": sql_query,
            "retry_count": 0,
            "result_data": None,
            "error_message": None
        }
        
        # Invoke the LangGraph workflow
        final_state = self.app.invoke(initial_state)
        
        if final_state['result_data']:
            return True, final_state['result_data']
        else:
            return False, final_state['error_message']

# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------
if __name__ == "__main__":
    executor = SQLExecutor()
    
    print("="*50)
    print("🧪 TEST 1: Broken SQL (Checking Self-Correction)")
    print("="*50)
    # Missing FROM clause on purpose to trigger the Fix Node
    bad_sql = "SELECT Name Production.Product WHERE ProductID = 1;"
    
    success, output = executor.run(bad_sql)
    print(f"\n📝 Final Result:\n{output}\n")


    print("="*50)
    print("🧪 TEST 2: Generated SQL (Checking Happy Path)")
    print("="*50)
    # Corrected SQL with Double Quotes
    generated_sql = """
    SELECT p."Name"
    FROM "Production"."Product" p
    JOIN "Production"."ProductInventory" pi ON p."ProductID" = pi."ProductID"
    WHERE pi."Quantity" > 50;
    """
    
    success, output = executor.run(generated_sql)
    
    if success:
        # We truncate the output just so it doesn't flood your terminal
        print(f"\n📝 Final Result (First 500 chars):\n{output[:500]}...\n") 
    else:
        print(f"\n❌ Error:\n{output}")