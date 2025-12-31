#!/bin/bash
# Verification script to test the setup

set -e

echo "=================================="
echo "HR Assistant RAG Demo - Setup Verification"
echo "=================================="
echo ""

# Check Python
echo "1. Checking Python version..."
python --version || { echo "ERROR: Python not found"; exit 1; }
echo "   ✓ Python found"
echo ""

# Check virtual environment
echo "2. Checking virtual environment..."
if [ -d ".venv" ]; then
    echo "   ✓ Virtual environment exists"
else
    echo "   ✗ Virtual environment not found"
    echo "   Run: python -m venv .venv"
    exit 1
fi
echo ""

# Check if venv is activated
echo "3. Checking if virtual environment is activated..."
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "   ✓ Virtual environment is activated"
else
    echo "   ⚠ Virtual environment not activated"
    echo "   Run: source .venv/bin/activate"
fi
echo ""

# Check dependencies
echo "4. Checking key dependencies..."
python -c "import chainlit" 2>/dev/null && echo "   ✓ chainlit installed" || echo "   ✗ chainlit not installed"
python -c "import langchain" 2>/dev/null && echo "   ✓ langchain installed" || echo "   ✗ langchain not installed"
python -c "import llama_index" 2>/dev/null && echo "   ✓ llama-index installed" || echo "   ✗ llama-index not installed"
python -c "import weaviate" 2>/dev/null && echo "   ✓ weaviate-client installed" || echo "   ✗ weaviate-client not installed"
python -c "import fastmcp" 2>/dev/null && echo "   ✓ fastmcp installed" || echo "   ✗ fastmcp not installed"
echo ""

# Check environment variables
echo "5. Checking environment variables..."
if [ -f ".env" ]; then
    echo "   ✓ .env file exists"
    if grep -q "OPENAI_API_KEY" .env; then
        echo "   ✓ OPENAI_API_KEY is set in .env"
    else
        echo "   ✗ OPENAI_API_KEY not found in .env"
    fi
else
    echo "   ✗ .env file not found"
    echo "   Create .env with OPENAI_API_KEY"
fi
echo ""

# Check Docker
echo "6. Checking Docker..."
if command -v docker &> /dev/null; then
    echo "   ✓ Docker is installed"
    docker ps &> /dev/null && echo "   ✓ Docker daemon is running" || echo "   ✗ Docker daemon not running"
else
    echo "   ✗ Docker not found"
fi
echo ""

# Check Weaviate
echo "7. Checking Weaviate..."
if curl -sf http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
    echo "   ✓ Weaviate is running on port 8080"
else
    echo "   ✗ Weaviate is not running"
    echo "   Run: bash scripts/start_weaviate.sh"
fi
echo ""

# Check MCP server
echo "8. Checking MCP server..."
if curl -sf http://localhost:9000/health > /dev/null 2>&1 || curl -sf http://localhost:9000/ > /dev/null 2>&1; then
    echo "   ✓ MCP server is running on port 9000"
else
    echo "   ⚠ MCP server is not running"
    echo "   Run: python src/mcp_server.py"
fi
echo ""

# Check docs directory
echo "9. Checking policy documents..."
if [ -d "docs" ]; then
    json_count=$(ls -1 docs/*.json 2>/dev/null | wc -l)
    echo "   ✓ docs/ directory exists"
    echo "   Found $json_count JSON policy files"
else
    echo "   ✗ docs/ directory not found"
fi
echo ""

# Check src files
echo "10. Checking source files..."
[ -f "src/chat_rag/chainlit_app.py" ] && echo "   ✓ chainlit_app.py" || echo "   ✗ chainlit_app.py missing"
[ -f "src/chat_rag/agent.py" ] && echo "   ✓ agent.py" || echo "   ✗ agent.py missing"
[ -f "src/chat_rag/mcp_server.py" ] && echo "   ✓ mcp_server.py" || echo "   ✗ mcp_server.py missing"
[ -f "src/chat_rag/ingestion.py" ] && echo "   ✓ ingestion.py" || echo "   ✗ ingestion.py missing"
[ -f "src/chat_rag/retriever.py" ] && echo "   ✓ retriever.py" || echo "   ✗ retriever.py missing"
echo ""

echo "=================================="
echo "Verification Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Ensure Weaviate is running: bash scripts/start_weaviate.sh"
echo "2. Run ingestion: python src/ingestion.py"
echo "3. Start MCP server: python src/mcp_server.py"
echo "4. Start Chainlit app: chainlit run src/chainlit_app.py -w"
echo ""
