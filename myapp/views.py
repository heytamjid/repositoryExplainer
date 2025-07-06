from django.shortcuts import render
from django.http import HttpResponse
import os
from github import Github
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def get_repo_tree(repo_url):
    try:
        g = Github()
        repo_name = repo_url.split("github.com/")[-1]
        repo = g.get_repo(repo_name)
        tree = repo.get_git_tree(recursive=True).tree
        return [element.path for element in tree]
    except Exception as e:
        print(f"Error getting repo tree: {e}")
        return None


def get_important_files(files):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", google_api_key=os.environ.get("GOOGLE_API_KEY")
    )
    prompt = PromptTemplate(
        input_variables=["files"],
        template="Given the following file list of a repository, identify the most important files to understand the repository's purpose and functionality. Return a comma-separated list of these files.\n\nFiles: {files}",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    important_files_str = chain.run(files=", ".join(files))
    return [file.strip() for file in important_files_str.split(",")]


def get_repo_explanation(repo_url, important_files):
    try:
        g = Github()
        repo_name = repo_url.split("github.com/")[-1]
        repo = g.get_repo(repo_name)

        file_contents = ""
        for file_path in important_files:
            try:
                content = repo.get_contents(file_path).decoded_content.decode("utf-8")
                file_contents += f"--- {file_path} ---\n{content}\n\n"
            except Exception as e:
                print(f"Error getting content for {file_path}: {e}")

        llm = ChatGoogleGenerativeAI(
            model="gemini-pro", google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
        prompt = PromptTemplate(
            input_variables=["file_contents"],
            template="Based on the following file contents, provide a detailed explanation of what this repository does.\n\n{file_contents}",
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        explanation = chain.run(file_contents=file_contents)
        return explanation
    except Exception as e:
        print(f"Error getting repo explanation: {e}")
        return None


def home(request):
    explanation = None
    if request.method == "POST":
        repo_url = request.POST.get("repo_url")
        if repo_url:
            files = get_repo_tree(repo_url)
            if files:
                important_files = get_important_files(files)
                explanation = get_repo_explanation(repo_url, important_files)

    return render(request, "home.html", {"explanation": explanation})
