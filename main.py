# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import shutil
import subprocess
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from google.cloud import aiplatform
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
import requests
import base64
from typing import Dict, Any

# Load env
load_dotenv()
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION", "us-central1")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GOOGLE_PROJECT_ID:
    raise ValueError("GOOGLE_PROJECT_ID must be set in .env")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN must be set in .env")

# Initialize Vertex AI
aiplatform.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)

def download_repo(repo: str) -> str | None:
    """Downloads the GitHub repository to a temporary local directory."""
    try:
        if "/" not in repo:
            return None
        owner, repo_name = repo.split("/", 1)
        if not owner or not repo_name:
            return None
        url = f"https://{GITHUB_TOKEN}@github.com/{owner}/{repo_name}.git"
        path = f"/tmp/repo-{owner}-{repo_name}"
        if os.path.exists(path):
            shutil.rmtree(path)
        subprocess.run(["git", "clone", url, path], check=True, capture_output=True)
        return path
    except Exception as e:
        print(f"Error downloading repo: {str(e)}")
        return None

def analyze_repo_local(path: str) -> str:
    """Analyzes the local repository to detect tech stack and key files."""
    file_paths = []
    key_files = {}
    tech_stack = set()
    for root, dirs, files in os.walk(path):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), path)
            file_paths.append(rel_path)
            full_path = os.path.join(root, file)
            if file == "package.json":
                tech_stack.add("Node.js")
                with open(full_path, "r", encoding="utf-8") as f:
                    key_files[rel_path] = f.read()
            elif file == "requirements.txt":
                tech_stack.add("Python")
                with open(full_path, "r", encoding="utf-8") as f:
                    key_files[rel_path] = f.read()
            elif file == "go.mod":
                tech_stack.add("Go")
                with open(full_path, "r", encoding="utf-8") as f:
                    key_files[rel_path] = f.read()
            elif file == "pom.xml":
                tech_stack.add("Java (Maven)")
                with open(full_path, "r", encoding="utf-8") as f:
                    key_files[rel_path] = f.read()
            elif file in ["build.gradle", "build.gradle.kts"]:
                tech_stack.add("Java (Gradle)")
                with open(full_path, "r", encoding="utf-8") as f:
                    key_files[rel_path] = f.read()
            elif file == "Cargo.toml":
                tech_stack.add("Rust")
                with open(full_path, "r", encoding="utf-8") as f:
                    key_files[rel_path] = f.read()
    analysis = f"Repository files: {', '.join(file_paths)}\nDetected tech stacks: {', '.join(tech_stack)}\n"
    for key_file, content in key_files.items():
        analysis += f"\nContent of {key_file}:\n{content}\n"
    return analysis

# Define ReadmeSummarizerTool
@tool
def read_and_summarize_readme(repo: str) -> str:
    """Fetches the README.md from a GitHub repository and returns a concise summary.
    
    Args:
        repo: Repository name in 'owner/repo' format (e.g., 'octocat/Hello-World').
    
    Returns:
        A string containing the summary of the README or an error message.
    """
    try:
        # Validate repo format
        if not repo or "/" not in repo:
            return "❌ **Invalid repository format.** Use 'owner/repo' (e.g., 'octocat/Hello-World')."
        
        owner, repo_name = repo.split("/", 1)
        if not owner or not repo_name:
            return "❌ **Invalid repository format.** Both owner and repo name are required."

        # Fetch README from GitHub API
        url = f"https://api.github.com/repos/{owner}/{repo_name}/readme"
        headers = {"Accept": "application/vnd.github.v3.raw"}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return f"❌ **Failed to fetch README:** HTTP {response.status_code}\n\n**Details:** {response.text[:200]}..."

        readme_content = response.text
        if not readme_content.strip():
            return "📄 **README is empty or not found.**"

        # Summarize using LLM
        summary_prompt = (
            f"Summarize the following README content concisely in up to 100 words:\n\n{readme_content}"
        )
        summary = llm.invoke([HumanMessage(content=summary_prompt)]).content
        return f"📄 **README SUMMARY**\n\n{summary}"
    except Exception as e:
        return f"❌ **Error processing README:** {str(e)}"

# Define CreateCIWorkflowTool
@tool
def create_ci_workflow(repo: str) -> str:
    """Creates a GitHub CI workflow for the given repository by analyzing its content and submits a pull request with the workflow file.
    
    Args:
        repo: Repository name in 'owner/repo' format (e.g., 'awwyushh/sample-project').
    
    Returns:
        A string with the PR URL or an error message.
    """
    try:
        owner, repo_name = repo.split("/", 1)
        if not owner or not repo_name:
            return "❌ **Invalid repository format.** Both owner and repo name are required."

        base_url = f"https://api.github.com/repos/{owner}/{repo_name}"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }

        # Get repo tree to analyze structure
        tree_response = requests.get(f"{base_url}/git/trees/main?recursive=1", headers=headers)
        if tree_response.status_code != 200:
            return f"❌ **Failed to get repo tree:** HTTP {tree_response.status_code}"

        tree = tree_response.json()["tree"]
        file_paths = [item["path"] for item in tree if item["type"] == "blob"]

        # Detect tech stack
        tech_stack = []
        key_files = {}
        if "package.json" in file_paths:
            tech_stack.append("Node.js")
            pkg_res = requests.get(f"{base_url}/contents/package.json", headers=headers)
            if pkg_res.status_code == 200:
                pkg_json = pkg_res.json()
                key_files["package.json"] = base64.b64decode(pkg_json["content"]).decode("utf-8")
        if "requirements.txt" in file_paths:
            tech_stack.append("Python")
            req_res = requests.get(f"{base_url}/contents/requirements.txt", headers=headers)
            if req_res.status_code == 200:
                req_json = req_res.json()
                key_files["requirements.txt"] = base64.b64decode(req_json["content"]).decode("utf-8")
        if "go.mod" in file_paths:
            tech_stack.append("Go")
            go_res = requests.get(f"{base_url}/contents/go.mod", headers=headers)
            if go_res.status_code == 200:
                go_json = go_res.json()
                key_files["go.mod"] = base64.b64decode(go_json["content"]).decode("utf-8")
        if "pom.xml" in file_paths:
            tech_stack.append("Java (Maven)")
            pom_res = requests.get(f"{base_url}/contents/pom.xml", headers=headers)
            if pom_res.status_code == 200:
                pom_json = pom_res.json()
                key_files["pom.xml"] = base64.b64decode(pom_json["content"]).decode("utf-8")
        if "build.gradle" in file_paths or "build.gradle.kts" in file_paths:
            tech_stack.append("Java (Gradle)")
            gradle_path = "build.gradle.kts" if "build.gradle.kts" in file_paths else "build.gradle"
            gradle_res = requests.get(f"{base_url}/contents/{gradle_path}", headers=headers)
            if gradle_res.status_code == 200:
                gradle_json = gradle_res.json()
                key_files[gradle_path] = base64.b64decode(gradle_json["content"]).decode("utf-8")
        if "Cargo.toml" in file_paths:
            tech_stack.append("Rust")
            cargo_res = requests.get(f"{base_url}/contents/Cargo.toml", headers=headers)
            if cargo_res.status_code == 200:
                cargo_json = cargo_res.json()
                key_files["Cargo.toml"] = base64.b64decode(cargo_json["content"]).decode("utf-8")

        if not tech_stack:
            return "⚠️ **No supported tech stack detected.** Please add package.json, requirements.txt, or similar files."

        # Build analysis string
        analysis = f"Repository files: {', '.join(file_paths)}\nDetected tech stacks: {', '.join(tech_stack)}\n"
        for key_file, content in key_files.items():
            analysis += f"\nContent of {key_file}:\n{content}\n"

        # Use LLM to generate YAML
        prompt = """
Generate a basic GitHub Actions CI workflow YAML file named ci.yml.
It should trigger on push to main and on pull requests to main.
Include jobs appropriate for the detected tech stacks, such as installing dependencies, running lints, and tests.
Based on this analysis:
{analysis}

Output only the YAML content, without any additional text or code blocks.
"""
        yaml_content = llm.invoke([HumanMessage(content=prompt.format(analysis=analysis))]).content.strip()

        if not yaml_content:
            return "❌ **Failed to generate workflow YAML.**"

        # Get main branch SHA
        main_ref = requests.get(f"{base_url}/git/refs/heads/main", headers=headers)
        if main_ref.status_code != 200:
            return f"❌ **Failed to get main ref:** HTTP {main_ref.status_code}"
        main_sha = main_ref.json()["object"]["sha"]

        # Create new branch
        branch_name = "add-ci-workflow"
        create_branch = requests.post(
            f"{base_url}/git/refs",
            headers=headers,
            json={"ref": f"refs/heads/{branch_name}", "sha": main_sha}
        )
        if create_branch.status_code != 201:
            return f"❌ **Failed to create branch:** HTTP {create_branch.status_code}"

        # Create file on new branch
        content_b64 = base64.b64encode(yaml_content.encode("utf-8")).decode("utf-8")
        create_file = requests.put(
            f"{base_url}/contents/.github/workflows/ci.yml",
            headers=headers,
            json={
                "message": "Add basic CI workflow",
                "content": content_b64,
                "branch": branch_name
            }
        )
        if create_file.status_code != 201:
            return f"❌ **Failed to create file:** HTTP {create_file.status_code}"

        # Create PR
        pr = requests.post(
            f"{base_url}/pulls",
            headers=headers,
            json={
                "title": "Add basic CI workflow",
                "body": f"This PR adds a basic GitHub Actions CI workflow based on the repository's tech stack:\n\n**Detected Tech Stack:** {', '.join(tech_stack)}\n\nThe workflow includes automated testing and linting for your project.",
                "head": branch_name,
                "base": "main"
            }
        )
        if pr.status_code != 201:
            return f"❌ **Failed to create PR:** HTTP {pr.status_code}"

        pr_url = pr.json()["html_url"]
        return f"✅ **CI Workflow Created Successfully!**\n\n**Pull Request:** {pr_url}\n\n**Changes:**\n- Added `.github/workflows/ci.yml`\n- Automated testing for {', '.join(tech_stack)}\n\n**Next Steps:**\n1. Review and merge the PR\n2. Push to main to trigger your first CI run\n3. Monitor the Actions tab in your repository"
    except Exception as e:
        return f"❌ **Error creating CI workflow:** {str(e)}"

@tool
def dockerize_app(repo: str) -> str:
    """Dockerizes the app in the GitHub repository by generating a Dockerfile based on its tech stack and submits a pull request.
    
    Args:
        repo: Repository name in 'owner/repo' format (e.g., 'octocat/Hello-World').
    
    Returns:
        A string with the PR URL or an error message.
    """
    path = download_repo(repo)
    if not path:
        return "❌ **Failed to download repository.** Please check the repository name and access permissions."
    
    try:
        owner, repo_name = repo.split("/", 1)
        base_url = f"https://api.github.com/repos/{owner}/{repo_name}"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        analysis = analyze_repo_local(path)
        if not analysis:
            return "❌ **Failed to analyze repository.** No files detected."
        
        # Use LLM to generate Dockerfile
        prompt = """
Generate a basic Dockerfile for the repository.
It should build and run the app appropriately for the detected tech stacks.
Based on this analysis:
{analysis}

Output only the Dockerfile content, without any additional text or code blocks.
"""
        dockerfile_content = llm.invoke([HumanMessage(content=prompt.format(analysis=analysis))]).content.strip()

        if not dockerfile_content:
            return "❌ **Failed to generate Dockerfile.**"

        # Get main branch SHA
        main_ref = requests.get(f"{base_url}/git/refs/heads/main", headers=headers)
        if main_ref.status_code != 200:
            return f"❌ **Failed to get main ref:** HTTP {main_ref.status_code}"
        main_sha = main_ref.json()["object"]["sha"]

        # Create new branch
        branch_name = "add-dockerfile"
        create_branch = requests.post(
            f"{base_url}/git/refs",
            headers=headers,
            json={"ref": f"refs/heads/{branch_name}", "sha": main_sha}
        )
        if create_branch.status_code != 201:
            return f"❌ **Failed to create branch:** HTTP {create_branch.status_code}"

        # Create file on new branch
        content_b64 = base64.b64encode(dockerfile_content.encode("utf-8")).decode("utf-8")
        create_file = requests.put(
            f"{base_url}/contents/Dockerfile",
            headers=headers,
            json={
                "message": "Add basic Dockerfile",
                "content": content_b64,
                "branch": branch_name
            }
        )
        if create_file.status_code != 201:
            return f"❌ **Failed to create file:** HTTP {create_file.status_code}"

        # Create PR
        pr = requests.post(
            f"{base_url}/pulls",
            headers=headers,
            json={
                "title": "Add basic Dockerfile",
                "body": "This PR adds a basic Dockerfile based on the repository's tech stack. The Dockerfile includes:\n\n- Multi-stage build optimization\n- Security best practices\n- Appropriate base images for the detected tech stack\n\n**Build command:** `docker build -t your-app .`\n**Run command:** `docker run -p 8080:8080 your-app`",
                "head": branch_name,
                "base": "main"
            }
        )
        if pr.status_code != 201:
            return f"❌ **Failed to create PR:** HTTP {pr.status_code}"

        pr_url = pr.json()["html_url"]
        return f"🐳 **Dockerfile Created Successfully!**\n\n**Pull Request:** {pr_url}\n\n**Quick Start:**\n```bash\n# Build the image\ndocker build -t {repo_name} .\n\n# Run the container\ndocker run -p 8080:8080 {repo_name}\n```\n\n**Next Steps:**\n1. Review and merge the PR\n2. Test the Docker build locally\n3. Consider adding Docker Compose for multi-container setups"
    except Exception as e:
        return f"❌ **Error dockerizing app:** {str(e)}"
    finally:
        if path and os.path.exists(path):
            shutil.rmtree(path)

def format_trivy_report(data: Dict[str, Any], repo: str) -> str:
    """Formats the Trivy JSON report into a beautiful, readable string."""
    
    # Extract metadata
    metadata = data.get("Metadata", {})
    results = data.get("Results", [])
    
    # Build the formatted report
    report_parts = []
    
    # Header
    report_parts.append("🔒 **SECURITY SCAN REPORT**")
    report_parts.append("=" * 50)
    report_parts.append("")
    report_parts.append(f"**Repository:** `{repo}`")
    report_parts.append(f"**Scan Type:** Filesystem Analysis")
    report_parts.append(f"**Scan Time:** {data.get('CreatedAt', 'Unknown')}")
    report_parts.append(f"**Artifact:** {data.get('ArtifactName', 'Unknown')}")
    report_parts.append("")
    
    # Repository details
    repo_info = metadata.get("RepoURL", "Unknown")
    if repo_info != "Unknown":
        report_parts.append("📁 **Repository Details**")
        report_parts.append(f"- **URL:** {repo_info}")
        branch = metadata.get("Branch", "Unknown")
        if branch != "Unknown":
            report_parts.append(f"- **Branch:** `{branch}`")
        commit = metadata.get("Commit", "Unknown")
        if commit != "Unknown":
            report_parts.append(f"- **Commit:** `{commit[:7]}...`")
        author = metadata.get("Author", "Unknown")
        if author != "Unknown":
            report_parts.append(f"- **Author:** {author}")
        report_parts.append("")
    
    # Vulnerability Summary
    if not results:
        report_parts.append("🎉 **SECURITY STATUS: CLEAN**")
        report_parts.append("")
        report_parts.append("**No vulnerabilities detected!** Your repository is secure 🚀")
        report_parts.append("")
        report_parts.append("**Recommendations:**")
        report_parts.append("• Keep dependencies up to date")
        report_parts.append("• Regularly run security scans")
        report_parts.append("• Review third-party dependencies")
    else:
        # Count vulnerabilities by severity
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
        total_vulns = 0
        
        for result in results:
            vulns = result.get("Vulnerabilities", [])
            for vuln in vulns:
                severity = vuln.get("Severity", "UNKNOWN")
                severity_counts[severity] += 1
                total_vulns += 1
        
        # Summary table
        report_parts.append("⚠️ **VULNERABILITY SUMMARY**")
        report_parts.append("")
        
        summary_table = f"""
| Severity   | Count | Risk Level                    |
|------------|-------|-------------------------------|
| 🔴 CRITICAL | {severity_counts['CRITICAL']} | Immediate Action Required     |
| 🟠 HIGH     | {severity_counts['HIGH']}     | High Priority                 |
| 🟡 MEDIUM   | {severity_counts['MEDIUM']}   | Review Required               |
| 🟢 LOW      | {severity_counts['LOW']}      | Monitor                       |
| ❓ UNKNOWN  | {severity_counts['UNKNOWN']}  | Investigate                   |
"""
        
        report_parts.append(summary_table.strip())
        report_parts.append("")
        report_parts.append(f"**Total Vulnerabilities:** {total_vulns}")
        
        # Overall risk assessment
        critical_high = severity_counts['CRITICAL'] + severity_counts['HIGH']
        if critical_high == 0:
            risk_level = "🟢 LOW RISK"
        elif critical_high <= 2:
            risk_level = "🟡 MEDIUM RISK"
        elif critical_high <= 5:
            risk_level = "🟠 HIGH RISK"
        else:
            risk_level = "🔴 CRITICAL RISK"
            
        report_parts.append(f"**Overall Risk Level:** {risk_level}")
        report_parts.append("")
        
        # Detailed vulnerabilities by scanner
        for i, result in enumerate(results):
            scanner = result.get("Target", "Unknown Scanner")
            vulns = result.get("Vulnerabilities", [])
            
            if vulns:
                report_parts.append(f"🔍 **{scanner.upper()} SCAN RESULTS**")
                report_parts.append("")
                
                # Create vulnerability table
                vuln_table_header = "| ID | Severity | Package | Version | Fixed Version | Description |"
                vuln_table_separator = "|----|----------|---------|---------|---------------|-------------|"
                
                vuln_table_rows = []
                for vuln in vulns[:10]:  # Limit to top 10 for brevity
                    pkg_name = vuln.get("PkgName", "Unknown")
                    installed_version = vuln.get("InstalledVersion", "Unknown")
                    fixed_version = vuln.get("FixedVersion", "N/A")
                    vuln_id = vuln.get("VulnerabilityID", "Unknown")
                    severity = vuln.get("Severity", "UNKNOWN")
                    title = vuln.get("Title", "No description available")
                    
                    # Severity emoji
                    severity_emoji = {
                        "CRITICAL": "🔴",
                        "HIGH": "🟠", 
                        "MEDIUM": "🟡",
                        "LOW": "🟢",
                        "UNKNOWN": "❓"
                    }.get(severity, "❓")
                    
                    # Truncate long descriptions
                    if len(title) > 60:
                        title = title[:57] + "..."
                    
                    vuln_table_rows.append(
                        f"| {vuln_id} | {severity_emoji} {severity} | {pkg_name} | {installed_version} | {fixed_version} | {title} |"
                    )
                
                if vuln_table_rows:
                    vuln_table = "\n".join([vuln_table_header, vuln_table_separator] + vuln_table_rows)
                    report_parts.append(vuln_table)
                    
                    if len(vulns) > 10:
                        report_parts.append(f"\n*... and {len(vulns) - 10} more vulnerabilities*")
                
                report_parts.append("")
        
        # Recommendations
        report_parts.append("🛡️ **SECURITY RECOMMENDATIONS**")
        report_parts.append("")
        
        if critical_high > 0:
            report_parts.append("**Immediate Actions:**")
            if severity_counts['CRITICAL'] > 0:
                report_parts.append(f"• Address {severity_counts['CRITICAL']} critical vulnerabilities first")
            if severity_counts['HIGH'] > 0:
                report_parts.append(f"• Update {severity_counts['HIGH']} high-risk packages")
            if severity_counts['CRITICAL'] > 0:
                report_parts.append("• Consider pausing deployments until critical issues are resolved")
        else:
            report_parts.append("**Good Practices:**")
            report_parts.append("• Continue monitoring for new vulnerabilities")
            report_parts.append("• Review medium/low severity issues periodically")
        
        report_parts.append("")
        report_parts.append("**Next Steps:**")
        report_parts.append("• Run `trivy fs --format sarif .` for detailed SARIF report")
        report_parts.append("• Set up automated security scans in CI/CD")
        report_parts.append("• Use dependency management tools (Dependabot, Renovate)")
        report_parts.append("")
    
    # Footer
    report_parts.append("─" * 50)
    report_parts.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    report_parts.append("**Tool:** Trivy Security Scanner")
    report_parts.append("**Status:** Scan Complete")
    
    return "\n".join(report_parts)

@tool
def analyze_safety(repo: str) -> str:
    """Analyzes the safety of the app in the GitHub repository using Trivy CLI to scan for vulnerabilities and returns a beautifully formatted report.
    
    Args:
        repo: Repository name in 'owner/repo' format (e.g., 'octocat/Hello-World').
    
    Returns:
        A string containing the beautifully formatted Trivy scan report or an error message.
    """
    path = download_repo(repo)
    if not path:
        return "❌ **Failed to download repository.** Please check the repository name and access permissions."
    
    try:
        # Run Trivy scan (assumes Trivy CLI is installed)
        result = subprocess.run(
            ["trivy", "fs", "--format", "json", "--exit-code", "0", path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            return f"❌ **Trivy scan failed:**\n\n**Error:**\n```\n{result.stderr}\n```"
        
        report = result.stdout
        if not report.strip():
            return f"✅ **Security Scan Complete**\n\n**Repository:** `{repo}`\n**Status:** No vulnerabilities found 🎉\n**Scan Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n**Your repository is secure!** 🚀"
        
        # Parse JSON report
        try:
            trivy_data = json.loads(report)
            formatted_report = format_trivy_report(trivy_data, repo)
            return formatted_report
        except json.JSONDecodeError as e:
            return f"⚠️ **Warning:** Could not parse Trivy JSON output.\n\n**Raw output:**\n```\n{report[:1000]}...\n```"
            
    except subprocess.TimeoutExpired:
        return f"⏰ **Timeout:** Security scan took too long to complete for {repo}\n\n**Recommendation:** Consider scanning individual packages or using Trivy in CI/CD with smaller scopes."
    except FileNotFoundError:
        return "❌ **Trivy not found.** Please install Trivy CLI:\n\n```bash\n# Install Trivy\ncurl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin\n```\n\nThen retry the scan."
    except Exception as e:
        return f"❌ **Error analyzing safety:** {str(e)}"
    finally:
        if path and os.path.exists(path):
            shutil.rmtree(path)

# LLM setup
llm = ChatVertexAI(
    model="gemini-2.5-pro",
    temperature=0.7,
    max_tokens=4096,
    streaming=True
)

# Bind tools to LLM
llm_with_tools = llm.bind_tools([read_and_summarize_readme, create_ci_workflow, dockerize_app, analyze_safety])

app = FastAPI(title="Akira AI Chat API", description="AI-powered GitHub repository assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Akira AI Chat API is running! 🚀", "docs": "/docs"}

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    stream = data.get("stream", False)

    # Debug: Log raw incoming data and messages
    print(f"Raw request data: {data}")
    print(f"Received messages: {messages}")

    # Convert to LangChain messages
    lc_messages = []
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            print(f"Invalid message format: {msg}")
            raise HTTPException(status_code=400, detail="Messages must be a list of objects with 'role' and 'content' fields")
        role = msg.get("role")
        content = str(msg.get("content", ""))
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            print(f"Invalid role in message: {msg}")
            raise HTTPException(status_code=400, detail=f"Invalid role: {role}")

    # Validate messages
    if not lc_messages:
        print("Error: No messages provided")
        raise HTTPException(status_code=400, detail="At least one message is required")
    
    if not any(msg.content.strip() for msg in lc_messages):
        print("Error: No messages with non-empty content")
        raise HTTPException(status_code=400, detail="At least one message with non-empty content is required")

    # Debug: Log LangChain messages
    print(f"LangChain messages: {[{'role': msg.__class__.__name__, 'content': msg.content[:50] + '...' if len(msg.content) > 50 else msg.content} for msg in lc_messages]}")

    if stream:
        def stream_generator():
            try:
                response = llm_with_tools.stream(lc_messages)
                for chunk in response:
                    if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                        # Handle tool call
                        tool_call = chunk.tool_calls[0]
                        tool_name = tool_call.get("name")
                        args = tool_call.get("args", {})
                        result = None
                        
                        print(f"🔧 Executing tool: {tool_name} with args: {args}")
                        
                        if tool_name == "read_and_summarize_readme":
                            repo = args.get("repo")
                            result = read_and_summarize_readme(repo)
                        elif tool_name == "create_ci_workflow":
                            repo = args.get("repo")
                            result = create_ci_workflow(repo)
                        elif tool_name == "dockerize_app":
                            repo = args.get("repo")
                            result = dockerize_app(repo)
                        elif tool_name == "analyze_safety":
                            repo = args.get("repo")
                            result = analyze_safety(repo)
                        else:
                            result = f"❓ **Unknown tool:** {tool_name}"

                        if result:
                            tool_message = ToolMessage(
                                content=result,
                                tool_call_id=tool_call.get("id"),
                                name=tool_name
                            )
                            
                            # For safety analysis, stream in chunks if very long
                            if tool_name == "analyze_safety" and len(result) > 2000:
                                # Stream in smaller chunks for better UX
                                for i in range(0, len(result), 800):
                                    chunk_result = result[i:i+800]
                                    yield f'data: {json.dumps({"choices": [{"delta": {"content": chunk_result}}]})}\n\n'
                            else:
                                # Stream tool result
                                yield f'data: {json.dumps({"choices": [{"delta": {"content": result}}]})}\n\n'
                            
                            # Continue with LLM response after tool call
                            final_response = llm_with_tools.stream([*lc_messages, chunk, tool_message])
                            for final_chunk in final_response:
                                if hasattr(final_chunk, 'content') and final_chunk.content:
                                    yield f'data: {json.dumps({"choices": [{"delta": {"content": final_chunk.content}}]})}\n\n'
                    else:
                        if hasattr(chunk, 'content') and chunk.content:
                            yield f'data: {json.dumps({"choices": [{"delta": {"content": chunk.content}}]})}\n\n'
                yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"Stream error: {str(e)}")
                yield f'data: {json.dumps({"error": f"Stream failed: {str(e)}", "choices": []})}\n\n'
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        try:
            response = llm_with_tools.invoke(lc_messages)
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_name = tool_call["name"]
                args = tool_call.get("args", {})
                result = None
                
                print(f"🔧 Executing tool (non-stream): {tool_name} with args: {args}")
                
                if tool_name == "read_and_summarize_readme":
                    repo = args.get("repo")
                    result = read_and_summarize_readme(repo)
                elif tool_name == "create_ci_workflow":
                    repo = args.get("repo")
                    result = create_ci_workflow(repo)
                elif tool_name == "dockerize_app":
                    repo = args.get("repo")
                    result = dockerize_app(repo)
                elif tool_name == "analyze_safety":
                    repo = args.get("repo")
                    result = analyze_safety(repo)
                else:
                    result = f"❓ **Unknown tool:** {tool_name}"

                if result:
                    tool_message = ToolMessage(
                        content=result,
                        tool_call_id=tool_call.get("id"),
                        name=tool_name
                    )
                    final_response = llm_with_tools.invoke([*lc_messages, response, tool_message])
                    return {"choices": [{"message": {"content": final_response.content}}]}
            return {"choices": [{"message": {"content": response.content}}]}
        except Exception as e:
            print(f"Invoke error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"LLM invocation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "vertex_ai": "available",
            "github_api": "available" if GITHUB_TOKEN else "not_configured"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)